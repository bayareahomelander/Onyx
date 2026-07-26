"""Bounded offline D36 dual-backend speculative-iteration qualification."""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
import weakref
from dataclasses import dataclass, fields
from typing import Any

from onyx_cuda import (
    BackendStateError,
    CacheCheckpointStateError,
    DEFAULT_TARGET_PROFILE,
    coordinate_speculative_iteration,
    load_torch_cuda_target,
)
from onyx_cuda.torch_backend import select_cuda_argmax
from onyx_cuda.torch_dynamic_cache import inspect_pinned_dynamic_cache


MIB = 1024 * 1024
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 4050 Laptop GPU"
EXPECTED_DEVICE_MEMORY_MIB = 6_141
VRAM_LIMIT_BYTES = EXPECTED_DEVICE_MEMORY_MIB * MIB
POST_FORWARD_ALLOCATED_ENVELOPE_BYTES = 9_443_328
POST_FORWARD_RESERVED_ENVELOPE_BYTES = 501_219_328
QUALIFICATION_PROPOSAL_LENGTH = 3
TRANSACTION_CYCLES = 100
STABLE_WARMUP_CYCLES = 12
DEFAULT_PROMPT = "D31 production batched target verification qualification"


@dataclass(frozen=True, slots=True)
class CudaSnapshot:
    allocated_bytes: int
    reserved_bytes: int
    maximum_allocated_bytes: int
    maximum_reserved_bytes: int
    free_bytes: int
    allocation_count: int
    active_count: int
    process_working_set_bytes: int | None


@dataclass(frozen=True, slots=True)
class ForwardObservation:
    token_ids: tuple[int, ...]
    logits_to_keep: int


@dataclass(frozen=True, slots=True)
class SelectorObservation:
    call_index: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    is_cuda: bool
    data_pointer: int


@dataclass(frozen=True, slots=True)
class CheckpointObservation:
    owner_id: int
    epoch: int
    allocation_id: int
    cache_length: int


@dataclass(frozen=True, slots=True)
class OutcomeEvidence:
    name: str
    proposal_token_ids: tuple[int, ...]
    accepted_count: int
    replacement_token_id: int | None
    final_cache_length: int
    draft_selector_calls: int
    target_selector_calls: int
    draft_forwards: int
    target_forwards: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class StableOutcomeEvidence:
    name: str
    allocated_bytes: int
    reserved_bytes: int
    allocation_count: int
    active_count: int
    minimum_seconds: float
    median_seconds: float
    maximum_seconds: float


@dataclass(frozen=True, slots=True)
class LifecycleEvidence:
    lifecycle: int
    close_order: tuple[str, str]
    prompt_token_ids: tuple[int, ...]
    current_token_id: int
    proposal_token_ids: tuple[int, ...]
    first_load_seconds: float
    second_load_seconds: float
    encoding_seconds: float
    prefill_seconds: float
    first_load: CudaSnapshot
    second_load: CudaSnapshot
    active_rooted: CudaSnapshot
    outcomes: tuple[OutcomeEvidence, ...]
    stable_outcomes: tuple[StableOutcomeEvidence, ...]
    transaction_peak: CudaSnapshot
    first_close_seconds: float
    after_first_close: CudaSnapshot
    second_close_seconds: float
    cleanup_seconds: float
    cleanup: CudaSnapshot


class _ForwardRecorder:
    def __init__(self, model):
        self._model = model
        self.observations: list[ForwardObservation] = []

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        self.observations.append(
            ForwardObservation(
                token_ids=tuple(int(token_id) for token_id in input_ids[0].tolist()),
                logits_to_keep=int(kwargs["logits_to_keep"]),
            )
        )
        return self._model(**kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def detach(self, backend) -> None:
        if self._model is not None:
            backend._model = self._model
            self._model = None


class _CheckpointRecorder:
    def __init__(self, backend):
        self._backend = backend
        self._create = backend.create_cache_checkpoint
        self._release = backend.release_cache_checkpoint
        self.created: list[CheckpointObservation] = []
        self.released: list[CheckpointObservation] = []
        backend.create_cache_checkpoint = self._record_create
        backend.release_cache_checkpoint = self._record_release

    def _record_create(self):
        checkpoint = self._create()
        self.created.append(_checkpoint_observation(checkpoint))
        return checkpoint

    def _record_release(self, checkpoint, /):
        self.released.append(_checkpoint_observation(checkpoint))
        return self._release(checkpoint)

    def mark(self) -> tuple[int, int]:
        return len(self.created), len(self.released)

    def delta(
        self,
        mark: tuple[int, int],
    ) -> tuple[tuple[CheckpointObservation, ...], tuple[CheckpointObservation, ...]]:
        created_at, released_at = mark
        return tuple(self.created[created_at:]), tuple(self.released[released_at:])

    def detach(self) -> None:
        if self._backend is not None:
            self._backend.create_cache_checkpoint = self._create
            self._backend.release_cache_checkpoint = self._release
            self._backend = None


class _RecordingSelector:
    def __init__(self, torch, backend):
        self._torch = torch
        self._backend = backend
        self.observations: list[SelectorObservation] = []
        self.selected_token_ids: list[int] = []
        self.row_references: list[weakref.ReferenceType[Any]] = []
        self.parent_references: list[weakref.ReferenceType[Any]] = []

    def __call__(self, row) -> int:
        call_index = len(self.observations)
        observation = SelectorObservation(
            call_index=call_index,
            shape=tuple(row.shape),
            dtype=str(row.dtype),
            device=str(row.device),
            is_cuda=bool(row.is_cuda),
            data_pointer=int(row.data_ptr()),
        )
        if observation.shape != (self._backend.vocab_size,):
            raise AssertionError(f"selector received row shape {observation.shape}")
        if row.dtype != self._torch.float16:
            raise AssertionError(f"selector received row dtype {row.dtype}")
        if observation.device != "cuda:0" or not observation.is_cuda:
            raise AssertionError(f"selector received row on {observation.device}")
        self.observations.append(observation)
        self.row_references.append(weakref.ref(row))
        parent = row._base
        if parent is not None:
            self.parent_references.append(weakref.ref(parent))
        token_id = self._select(row, call_index)
        self.selected_token_ids.append(token_id)
        return token_id

    def _select(self, row, call_index: int) -> int:
        return select_cuda_argmax(row)


class _TargetSelector(_RecordingSelector):
    def __init__(self, torch, backend, *, proposal_token_ids, mismatch_position):
        super().__init__(torch, backend)
        self._proposal_token_ids = proposal_token_ids
        self._mismatch_position = mismatch_position
        self.real_token_ids: list[int] = []

    def _select(self, row, call_index: int) -> int:
        real_token_id = select_cuda_argmax(row)
        self.real_token_ids.append(real_token_id)
        if self._proposal_token_ids is not None:
            if call_index >= len(self._proposal_token_ids):
                raise AssertionError("target selector received the final verification row")
            expected = self._proposal_token_ids[call_index]
            if real_token_id != expected:
                raise AssertionError(
                    f"real target decision {real_token_id} at {call_index} "
                    f"does not match proposal {expected}"
                )
        if call_index == self._mismatch_position:
            alternative = (real_token_id + 1) % self._backend.vocab_size
            if alternative == real_token_id:
                raise AssertionError("forced target replacement was not distinct")
            return alternative
        return real_token_id


class _InjectedSelectorFailure(RuntimeError):
    pass


class _FailingTargetSelector(_RecordingSelector):
    def __init__(self, torch, backend, failure):
        super().__init__(torch, backend)
        self._failure = failure

    def _select(self, row, call_index: int) -> int:
        if call_index == 1:
            raise self._failure
        return select_cuda_argmax(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.device_index != 0:
        raise ValueError("D36 is qualified only on cuda:0")
    if os.name != "nt":
        raise AssertionError("D36 requires the Windows acceptance workflow")
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise AssertionError(f"{variable}=1 is required before starting D36")

    import bitsandbytes
    import torch
    import transformers

    if torch.__version__ != "2.6.0+cu124":
        raise AssertionError(f"unexpected PyTorch version: {torch.__version__}")
    if transformers.__version__ != "4.57.6":
        raise AssertionError(f"unexpected Transformers version: {transformers.__version__}")
    if bitsandbytes.__version__ != "0.49.2":
        raise AssertionError(f"unexpected bitsandbytes version: {bitsandbytes.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("D36 requires the one-device RTX 4050 acceptance environment")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.name != EXPECTED_DEVICE_NAME:
        raise AssertionError(f"unexpected CUDA device: {properties.name}")
    reported_memory_mib = (properties.total_memory + MIB - 1) // MIB
    if reported_memory_mib != EXPECTED_DEVICE_MEMORY_MIB:
        raise AssertionError(
            f"unexpected dedicated VRAM: {reported_memory_mib} MiB; "
            f"expected {EXPECTED_DEVICE_MEMORY_MIB} MiB"
        )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    starting_cleanup = _cuda_snapshot(torch, device)
    lifecycle_evidence = []
    for lifecycle in (1, 2):
        lifecycle_evidence.append(
            _run_lifecycle(
                torch,
                lifecycle=lifecycle,
                device=device,
                device_index=arguments.device_index,
            )
        )

    first, second = lifecycle_evidence
    comparable_first = (
        first.prompt_token_ids,
        first.current_token_id,
        first.proposal_token_ids,
        tuple(
            (
                outcome.name,
                outcome.accepted_count,
                outcome.replacement_token_id,
                outcome.final_cache_length,
                outcome.draft_selector_calls,
                outcome.target_selector_calls,
                outcome.draft_forwards,
                outcome.target_forwards,
            )
            for outcome in first.outcomes
        ),
    )
    comparable_second = (
        second.prompt_token_ids,
        second.current_token_id,
        second.proposal_token_ids,
        tuple(
            (
                outcome.name,
                outcome.accepted_count,
                outcome.replacement_token_id,
                outcome.final_cache_length,
                outcome.draft_selector_calls,
                outcome.target_selector_calls,
                outcome.draft_forwards,
                outcome.target_forwards,
            )
            for outcome in second.outcomes
        ),
    )
    if comparable_first != comparable_second:
        raise AssertionError("D36 token, call, or cache evidence changed across lifecycles")
    _validate_cleanup(starting_cleanup, first.cleanup, second.cleanup)

    for evidence in lifecycle_evidence:
        for snapshot in (
            evidence.first_load,
            evidence.second_load,
            evidence.active_rooted,
            evidence.transaction_peak,
        ):
            if snapshot.maximum_allocated_bytes >= properties.total_memory:
                raise AssertionError("allocated peak reached physical device memory")
            if snapshot.maximum_reserved_bytes >= properties.total_memory:
                raise AssertionError("reserved peak reached physical device memory")
            if snapshot.maximum_allocated_bytes >= VRAM_LIMIT_BYTES:
                raise AssertionError("allocated peak reached the 6,141 MiB qualification limit")
            if snapshot.maximum_reserved_bytes >= VRAM_LIMIT_BYTES:
                raise AssertionError("reserved peak reached the 6,141 MiB qualification limit")
        _print_lifecycle(evidence)

    print(
        "D36 dual-backend speculative-iteration CUDA qualification passed:",
        "profile=Qwen2.5-0.5B-Instruct+Qwen2.5-0.5B-Instruct",
        "device=cuda:0",
        f"proposal_length_fixture={QUALIFICATION_PROPOSAL_LENGTH}",
        f"transactions={TRANSACTION_CYCLES * 2}",
        "genuine_full_acceptance=True",
        "forced_mismatch_positions=(0,1,2)",
        "close_orders=target-draft,draft-target",
        f"after_allocated_bytes={second.cleanup.allocated_bytes}",
        f"after_reserved_bytes={second.cleanup.reserved_bytes}",
    )


def _run_lifecycle(torch, *, lifecycle: int, device, device_index: int) -> LifecycleEvidence:
    draft = None
    target = None
    draft_forwards = None
    target_forwards = None
    draft_checkpoints = None
    target_checkpoints = None
    close_order = ("target", "draft") if lifecycle == 1 else ("draft", "target")
    closed: set[str] = set()
    try:
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        draft = load_torch_cuda_target(
            DEFAULT_TARGET_PROFILE,
            device_index=device_index,
            local_files_only=True,
        )
        torch.cuda.synchronize(device)
        first_load_seconds = time.perf_counter() - started
        first_load = _cuda_snapshot(torch, device)

        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        target = load_torch_cuda_target(
            DEFAULT_TARGET_PROFILE,
            device_index=device_index,
            local_files_only=True,
        )
        torch.cuda.synchronize(device)
        second_load_seconds = time.perf_counter() - started
        second_load = _cuda_snapshot(torch, device)
        _assert_loaded_ownership(draft, target)

        draft_forwards = _ForwardRecorder(draft._model)
        target_forwards = _ForwardRecorder(target._model)
        draft._model = draft_forwards
        target._model = target_forwards
        draft_checkpoints = _CheckpointRecorder(draft)
        target_checkpoints = _CheckpointRecorder(target)

        started = time.perf_counter()
        draft_prompt = draft.tokenizer.encode(DEFAULT_PROMPT)
        target_prompt = target.tokenizer.encode(DEFAULT_PROMPT)
        encoding_seconds = time.perf_counter() - started
        if draft_prompt != target_prompt or not draft_prompt:
            raise AssertionError("independent tokenizer encodings differ or are empty")

        started = time.perf_counter()
        draft_prefill = draft.prefill(draft_prompt)
        target_prefill = target.prefill(target_prompt)
        torch.cuda.synchronize(device)
        prefill_seconds = time.perf_counter() - started
        draft_current = select_cuda_argmax(draft_prefill.logits)
        target_current = select_cuda_argmax(target_prefill.logits)
        if draft_current != target_current:
            raise AssertionError("independent prefill rows selected different current tokens")
        current_token_id = draft_current
        del draft_prefill
        del target_prefill

        _assert_live_cache_ownership(draft, target)
        prompt_length = len(draft_prompt)
        draft_identity = _cache_identity(draft)
        target_identity = _cache_identity(target)
        draft_epoch = draft._epoch
        target_epoch = target._epoch
        draft_layout = _assert_qualified_layout(
            torch,
            draft,
            expected_length=prompt_length,
            device_index=device_index,
        )
        target_layout = _assert_qualified_layout(
            torch,
            target,
            expected_length=prompt_length,
            device_index=device_index,
        )
        draft_root = draft.create_cache_checkpoint()
        target_root = target.create_cache_checkpoint()
        if draft_root.owner_id != draft._owner_id or target_root.owner_id != target._owner_id:
            raise AssertionError("caller root owner identity differs from its backend")
        if draft_root.owner_id == target_root.owner_id:
            raise AssertionError("dual caller roots share an owner identity")
        _assert_checkpoint_metadata_cpu_only(torch, draft, draft_root)
        _assert_checkpoint_metadata_cpu_only(torch, target, target_root)
        root_snapshots = {
            "draft": _clone_cache(draft),
            "target": _clone_cache(target),
        }
        _assert_foreign_roots_are_atomic(
            draft,
            target,
            draft_root=draft_root,
            target_root=target_root,
        )
        active_rooted = _cuda_snapshot(torch, device)

        common = {
            "torch": torch,
            "draft": draft,
            "target": target,
            "draft_forwards": draft_forwards,
            "target_forwards": target_forwards,
            "draft_checkpoints": draft_checkpoints,
            "target_checkpoints": target_checkpoints,
            "draft_root": draft_root,
            "target_root": target_root,
            "prompt_token_ids": draft_prompt,
            "current_token_id": current_token_id,
            "draft_identity": draft_identity,
            "target_identity": target_identity,
            "draft_epoch": draft_epoch,
            "target_epoch": target_epoch,
            "draft_layout": draft_layout,
            "target_layout": target_layout,
            "root_snapshots": root_snapshots,
            "device": device,
            "device_index": device_index,
        }

        full, draft_full_snapshot, target_full_snapshot = _run_transaction(
            **common,
            expected_proposal=None,
            mismatch_position=None,
            draft_reference=None,
            target_reference=None,
        )
        proposal_token_ids = full.proposal_token_ids
        target_references = _build_target_sequential_references(
            torch,
            target,
            target_root=target_root,
            root_snapshot=root_snapshots["target"],
            prompt_token_ids=draft_prompt,
            current_token_id=current_token_id,
            proposal_token_ids=proposal_token_ids,
        )

        outcomes = [full]
        for mismatch_position in range(QUALIFICATION_PROPOSAL_LENGTH):
            outcome, _, _ = _run_transaction(
                **common,
                expected_proposal=proposal_token_ids,
                mismatch_position=mismatch_position,
                draft_reference=draft_full_snapshot,
                target_reference=target_references[mismatch_position],
            )
            outcomes.append(outcome)

        _run_live_failure_reuse(
            **common,
            proposal_token_ids=proposal_token_ids,
            draft_reference=draft_full_snapshot,
            target_full_reference=target_full_snapshot,
        )

        torch.cuda.reset_peak_memory_stats(device)
        allocator_samples: dict[str, list[tuple[int, int, int, int]]] = {
            "full": [],
            "mismatch-0": [],
            "mismatch-1": [],
            "mismatch-2": [],
        }
        timing_samples = {name: [] for name in allocator_samples}
        for cycle in range(TRANSACTION_CYCLES):
            mismatch_position = (None, 0, 1, 2)[cycle % 4]
            name = "full" if mismatch_position is None else f"mismatch-{mismatch_position}"
            target_reference = (
                target_full_snapshot
                if mismatch_position is None
                else target_references[mismatch_position]
            )
            outcome, _, _ = _run_transaction(
                **common,
                expected_proposal=proposal_token_ids,
                mismatch_position=mismatch_position,
                draft_reference=draft_full_snapshot,
                target_reference=target_reference,
            )
            timing_samples[name].append(outcome.duration_seconds)
            stats = torch.cuda.memory_stats(device)
            allocator_samples[name].append(
                (
                    torch.cuda.memory_allocated(device),
                    torch.cuda.memory_reserved(device),
                    int(stats["allocation.all.current"]),
                    int(stats["active.all.current"]),
                )
            )

        warmup_per_outcome = STABLE_WARMUP_CYCLES // 4
        stable_outcomes = []
        for name in ("full", "mismatch-0", "mismatch-1", "mismatch-2"):
            stable_samples = allocator_samples[name][warmup_per_outcome:]
            if len(set(stable_samples)) != 1:
                raise AssertionError(
                    f"{name} post-root allocator state did not stabilize: "
                    f"{sorted(set(stable_samples))}"
                )
            stable = stable_samples[0]
            durations = timing_samples[name][warmup_per_outcome:]
            stable_outcomes.append(
                StableOutcomeEvidence(
                    name=name,
                    allocated_bytes=stable[0],
                    reserved_bytes=stable[1],
                    allocation_count=stable[2],
                    active_count=stable[3],
                    minimum_seconds=min(durations),
                    median_seconds=statistics.median(durations),
                    maximum_seconds=max(durations),
                )
            )
        transaction_peak = _cuda_snapshot(torch, device)

        draft_full_snapshot = None
        target_full_snapshot = None
        target_references.clear()
        gc.collect()

        backends = {"draft": draft, "target": target}
        forward_recorders = {"draft": draft_forwards, "target": target_forwards}
        checkpoint_recorders = {
            "draft": draft_checkpoints,
            "target": target_checkpoints,
        }
        roots = {"draft": draft_root, "target": target_root}
        identities = {"draft": draft_identity, "target": target_identity}
        epochs = {"draft": draft_epoch, "target": target_epoch}
        layouts = {"draft": draft_layout, "target": target_layout}
        first_name, second_name = close_order

        root_snapshots[first_name] = None
        forward_recorders[first_name].detach(backends[first_name])
        checkpoint_recorders[first_name].detach()
        gc.collect()
        started = time.perf_counter()
        backends[first_name].close()
        torch.cuda.synchronize(device)
        first_close_seconds = time.perf_counter() - started
        closed.add(first_name)
        after_first_close = _cuda_snapshot(torch, device)
        with _expect_closed_backend():
            backends[first_name].decode(current_token_id)

        remaining = backends[second_name]
        if remaining.tokenizer.tokenizer_id != DEFAULT_TARGET_PROFILE.pinned_id:
            raise AssertionError("peer close invalidated the remaining tokenizer")
        remaining.decode(current_token_id)
        remaining.rollback_cache(roots[second_name])
        _assert_root_state(
            torch,
            remaining,
            root=roots[second_name],
            root_snapshot=root_snapshots[second_name],
            prompt_token_ids=draft_prompt,
            cache_identity=identities[second_name],
            epoch=epochs[second_name],
            layout=layouts[second_name],
            device_index=device_index,
        )

        root_snapshots[second_name] = None
        forward_recorders[second_name].detach(remaining)
        checkpoint_recorders[second_name].detach()
        gc.collect()
        started = time.perf_counter()
        remaining.close()
        torch.cuda.synchronize(device)
        second_close_seconds = time.perf_counter() - started
        closed.add(second_name)

        draft = None
        target = None
        draft_forwards = None
        target_forwards = None
        draft_checkpoints = None
        target_checkpoints = None
        backends.clear()
        forward_recorders.clear()
        checkpoint_recorders.clear()
        roots.clear()
        identities.clear()
        epochs.clear()
        layouts.clear()
        root_snapshots.clear()
        started = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        cleanup_seconds = time.perf_counter() - started
        cleanup = _cuda_snapshot(torch, device)

        return LifecycleEvidence(
            lifecycle=lifecycle,
            close_order=close_order,
            prompt_token_ids=draft_prompt,
            current_token_id=current_token_id,
            proposal_token_ids=proposal_token_ids,
            first_load_seconds=first_load_seconds,
            second_load_seconds=second_load_seconds,
            encoding_seconds=encoding_seconds,
            prefill_seconds=prefill_seconds,
            first_load=first_load,
            second_load=second_load,
            active_rooted=active_rooted,
            outcomes=tuple(outcomes),
            stable_outcomes=tuple(stable_outcomes),
            transaction_peak=transaction_peak,
            first_close_seconds=first_close_seconds,
            after_first_close=after_first_close,
            second_close_seconds=second_close_seconds,
            cleanup_seconds=cleanup_seconds,
            cleanup=cleanup,
        )
    finally:
        if draft_checkpoints is not None:
            draft_checkpoints.detach()
        if target_checkpoints is not None:
            target_checkpoints.detach()
        if draft is not None and "draft" not in closed:
            if draft_forwards is not None:
                draft_forwards.detach(draft)
            draft.close()
        if target is not None and "target" not in closed:
            if target_forwards is not None:
                target_forwards.detach(target)
            target.close()


def _run_transaction(
    *,
    torch,
    draft,
    target,
    draft_forwards,
    target_forwards,
    draft_checkpoints,
    target_checkpoints,
    draft_root,
    target_root,
    prompt_token_ids,
    current_token_id,
    draft_identity,
    target_identity,
    draft_epoch,
    target_epoch,
    draft_layout,
    target_layout,
    root_snapshots,
    device,
    device_index,
    expected_proposal,
    mismatch_position,
    draft_reference,
    target_reference,
):
    _assert_root_state(
        torch,
        draft,
        root=draft_root,
        root_snapshot=root_snapshots["draft"],
        prompt_token_ids=prompt_token_ids,
        cache_identity=draft_identity,
        epoch=draft_epoch,
        layout=draft_layout,
        device_index=device_index,
    )
    _assert_root_state(
        torch,
        target,
        root=target_root,
        root_snapshot=root_snapshots["target"],
        prompt_token_ids=prompt_token_ids,
        cache_identity=target_identity,
        epoch=target_epoch,
        layout=target_layout,
        device_index=device_index,
    )
    draft_forward_mark = len(draft_forwards.observations)
    target_forward_mark = len(target_forwards.observations)
    draft_checkpoint_mark = draft_checkpoints.mark()
    target_checkpoint_mark = target_checkpoints.mark()
    target_next_checkpoint = target._next_checkpoint_id
    draft_selector = _RecordingSelector(torch, draft)
    target_selector = _TargetSelector(
        torch,
        target,
        proposal_token_ids=expected_proposal,
        mismatch_position=mismatch_position,
    )

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = coordinate_speculative_iteration(
        draft,
        target,
        current_token_id,
        proposal_length=QUALIFICATION_PROPOSAL_LENGTH,
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    torch.cuda.synchronize(device)
    duration_seconds = time.perf_counter() - started

    proposal_token_ids = result.proposal_token_ids
    if expected_proposal is not None and proposal_token_ids != expected_proposal:
        raise AssertionError("greedy draft proposal changed")
    if tuple(draft_selector.selected_token_ids) != proposal_token_ids:
        raise AssertionError("draft selector decisions differ from the proposal")
    if mismatch_position is None:
        accepted_count = QUALIFICATION_PROPOSAL_LENGTH
        replacement_token_id = None
        if tuple(target_selector.real_token_ids) != proposal_token_ids:
            raise AssertionError("full acceptance was not a genuine greedy target match")
        name = "full"
    else:
        accepted_count = mismatch_position
        replacement_token_id = target_selector.selected_token_ids[mismatch_position]
        if replacement_token_id == proposal_token_ids[mismatch_position]:
            raise AssertionError("forced mismatch did not replace the proposal token")
        name = f"mismatch-{mismatch_position}"

    expected_length = len(prompt_token_ids) + 1 + accepted_count
    expected_prefix = (*prompt_token_ids, current_token_id, *proposal_token_ids[:accepted_count])
    _assert_result(
        torch,
        result,
        proposal_token_ids=proposal_token_ids,
        accepted_count=accepted_count,
        replacement_token_id=replacement_token_id,
        initial_cache_length=len(prompt_token_ids),
        final_cache_length=expected_length,
    )
    _assert_active_state(
        torch,
        draft,
        expected_prefix=expected_prefix,
        cache_identity=draft_identity,
        epoch=draft_epoch,
        layout=draft_layout,
        device_index=device_index,
    )
    _assert_active_state(
        torch,
        target,
        expected_prefix=expected_prefix,
        cache_identity=target_identity,
        epoch=target_epoch,
        layout=target_layout,
        device_index=device_index,
    )
    if draft_reference is not None:
        _assert_cache_equals_prefix(torch, draft, draft_reference, expected_length)
    if target_reference is not None:
        _assert_cache_equals_prefix(torch, target, target_reference, expected_length)

    draft_calls = tuple(draft_forwards.observations[draft_forward_mark:])
    target_calls = tuple(target_forwards.observations[target_forward_mark:])
    expected_draft_inputs = (
        (current_token_id,),
        (proposal_token_ids[0],),
        (proposal_token_ids[1],),
        (proposal_token_ids[2],),
    )
    if tuple(call.token_ids for call in draft_calls) != expected_draft_inputs:
        raise AssertionError(f"draft forward order changed: {draft_calls}")
    if tuple(call.logits_to_keep for call in draft_calls) != (1, 1, 1, 1):
        raise AssertionError("draft forward logits_to_keep changed")
    expected_target_inputs = [
        (current_token_id, *proposal_token_ids),
        *(
            [(token_id,) for token_id in (current_token_id, *proposal_token_ids[:accepted_count])]
            if mismatch_position is not None
            else []
        ),
    ]
    if [call.token_ids for call in target_calls] != expected_target_inputs:
        raise AssertionError(f"target forward order changed: {target_calls}")
    expected_target_keep = [
        QUALIFICATION_PROPOSAL_LENGTH + 1,
        *([1] * (accepted_count + 1) if mismatch_position is not None else []),
    ]
    if [call.logits_to_keep for call in target_calls] != expected_target_keep:
        raise AssertionError("target forward logits_to_keep changed")
    if len(draft_selector.observations) != QUALIFICATION_PROPOSAL_LENGTH:
        raise AssertionError("draft selector call count changed")
    expected_target_selector_calls = (
        QUALIFICATION_PROPOSAL_LENGTH if mismatch_position is None else accepted_count + 1
    )
    if len(target_selector.observations) != expected_target_selector_calls:
        raise AssertionError("target selector call count changed or final row was selected")

    draft_created, draft_released = draft_checkpoints.delta(draft_checkpoint_mark)
    target_created, target_released = target_checkpoints.delta(target_checkpoint_mark)
    expected_checkpoint_lengths = (
        len(prompt_token_ids),
        len(prompt_token_ids) + 1,
        len(prompt_token_ids) + 2,
        len(prompt_token_ids) + 3,
    )
    if tuple(item.cache_length for item in draft_created) != expected_checkpoint_lengths:
        raise AssertionError("D32 checkpoint creation lengths changed")
    if tuple(item.allocation_id for item in draft_created) != tuple(
        range(draft_created[0].allocation_id, draft_created[0].allocation_id + 4)
    ):
        raise AssertionError("D32 checkpoint allocation IDs are not monotonic")
    if draft_released != draft_created:
        raise AssertionError("D32/D35 checkpoint release order changed")
    if any(item.allocation_id == draft_root.allocation_id for item in draft_released):
        raise AssertionError("D35 released the caller-owned draft root")
    if target_created or target_released or target._next_checkpoint_id != target_next_checkpoint:
        raise AssertionError("D35 changed target checkpoint allocation state")
    if tuple(draft._cache_checkpoints) != (draft_root.allocation_id,):
        raise AssertionError("draft registry retained a non-root checkpoint")
    if tuple(target._cache_checkpoints) != (target_root.allocation_id,):
        raise AssertionError("target registry retained a non-root checkpoint")

    captured_draft = _clone_cache(draft) if draft_reference is None else None
    captured_target = _clone_cache(target) if target_reference is None else None
    _assert_transients_released(draft_selector, target_selector)

    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    _assert_root_state(
        torch,
        draft,
        root=draft_root,
        root_snapshot=root_snapshots["draft"],
        prompt_token_ids=prompt_token_ids,
        cache_identity=draft_identity,
        epoch=draft_epoch,
        layout=draft_layout,
        device_index=device_index,
    )
    _assert_root_state(
        torch,
        target,
        root=target_root,
        root_snapshot=root_snapshots["target"],
        prompt_token_ids=prompt_token_ids,
        cache_identity=target_identity,
        epoch=target_epoch,
        layout=target_layout,
        device_index=device_index,
    )
    return (
        OutcomeEvidence(
            name=name,
            proposal_token_ids=proposal_token_ids,
            accepted_count=accepted_count,
            replacement_token_id=replacement_token_id,
            final_cache_length=expected_length,
            draft_selector_calls=len(draft_selector.observations),
            target_selector_calls=len(target_selector.observations),
            draft_forwards=len(draft_calls),
            target_forwards=len(target_calls),
            duration_seconds=duration_seconds,
        ),
        captured_draft,
        captured_target,
    )


def _run_live_failure_reuse(
    *,
    torch,
    draft,
    target,
    draft_forwards,
    target_forwards,
    draft_checkpoints,
    target_checkpoints,
    draft_root,
    target_root,
    prompt_token_ids,
    current_token_id,
    draft_identity,
    target_identity,
    draft_epoch,
    target_epoch,
    draft_layout,
    target_layout,
    root_snapshots,
    device,
    device_index,
    proposal_token_ids,
    draft_reference,
    target_full_reference,
):
    failure = _InjectedSelectorFailure("injected live D36 target-selector failure")
    try:
        coordinate_speculative_iteration(
            draft,
            target,
            current_token_id,
            proposal_length=QUALIFICATION_PROPOSAL_LENGTH,
            draft_select_token=_RecordingSelector(torch, draft),
            target_select_token=_FailingTargetSelector(torch, target, failure),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    except _InjectedSelectorFailure as raised:
        if raised is not failure:
            raise AssertionError("target-selector failure identity changed") from raised
    else:
        raise AssertionError("target-selector failure was not raised")
    _assert_both_roots(
        torch,
        draft,
        target,
        draft_root=draft_root,
        target_root=target_root,
        root_snapshots=root_snapshots,
        prompt_token_ids=prompt_token_ids,
        draft_identity=draft_identity,
        target_identity=target_identity,
        draft_epoch=draft_epoch,
        target_epoch=target_epoch,
        draft_layout=draft_layout,
        target_layout=target_layout,
        device_index=device_index,
    )
    _run_transaction(
        torch=torch,
        draft=draft,
        target=target,
        draft_forwards=draft_forwards,
        target_forwards=target_forwards,
        draft_checkpoints=draft_checkpoints,
        target_checkpoints=target_checkpoints,
        draft_root=draft_root,
        target_root=target_root,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        draft_identity=draft_identity,
        target_identity=target_identity,
        draft_epoch=draft_epoch,
        target_epoch=target_epoch,
        draft_layout=draft_layout,
        target_layout=target_layout,
        root_snapshots=root_snapshots,
        device=device,
        device_index=device_index,
        expected_proposal=proposal_token_ids,
        mismatch_position=None,
        draft_reference=draft_reference,
        target_reference=target_full_reference,
    )

    replay_failure = RuntimeError("injected live D36 pre-mutation replay failure")
    original_decode = target.decode
    replay_calls = 0

    def fail_second_replay(token_id, /):
        nonlocal replay_calls
        replay_calls += 1
        if replay_calls == 2:
            raise replay_failure
        return original_decode(token_id)

    target.decode = fail_second_replay
    try:
        try:
            coordinate_speculative_iteration(
                draft,
                target,
                current_token_id,
                proposal_length=QUALIFICATION_PROPOSAL_LENGTH,
                draft_select_token=_RecordingSelector(torch, draft),
                target_select_token=_TargetSelector(
                    torch,
                    target,
                    proposal_token_ids=proposal_token_ids,
                    mismatch_position=1,
                ),
                draft_root_checkpoint=draft_root,
                target_root_checkpoint=target_root,
            )
        except RuntimeError as raised:
            if raised is not replay_failure:
                raise
        else:
            raise AssertionError("target replay wrapper failure was not raised")
    finally:
        target.decode = original_decode
    _assert_both_roots(
        torch,
        draft,
        target,
        draft_root=draft_root,
        target_root=target_root,
        root_snapshots=root_snapshots,
        prompt_token_ids=prompt_token_ids,
        draft_identity=draft_identity,
        target_identity=target_identity,
        draft_epoch=draft_epoch,
        target_epoch=target_epoch,
        draft_layout=draft_layout,
        target_layout=target_layout,
        device_index=device_index,
    )
    _run_transaction(
        torch=torch,
        draft=draft,
        target=target,
        draft_forwards=draft_forwards,
        target_forwards=target_forwards,
        draft_checkpoints=draft_checkpoints,
        target_checkpoints=target_checkpoints,
        draft_root=draft_root,
        target_root=target_root,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        draft_identity=draft_identity,
        target_identity=target_identity,
        draft_epoch=draft_epoch,
        target_epoch=target_epoch,
        draft_layout=draft_layout,
        target_layout=target_layout,
        root_snapshots=root_snapshots,
        device=device,
        device_index=device_index,
        expected_proposal=proposal_token_ids,
        mismatch_position=None,
        draft_reference=draft_reference,
        target_reference=target_full_reference,
    )


def _build_target_sequential_references(
    torch,
    target,
    *,
    target_root,
    root_snapshot,
    prompt_token_ids,
    current_token_id,
    proposal_token_ids,
):
    references = {}
    for accepted_count in range(QUALIFICATION_PROPOSAL_LENGTH):
        target.rollback_cache(target_root)
        for token_id in (current_token_id, *proposal_token_ids[:accepted_count]):
            target.decode(token_id)
        expected_prefix = (
            *prompt_token_ids,
            current_token_id,
            *proposal_token_ids[:accepted_count],
        )
        if tuple(target._active_token_ids) != expected_prefix:
            raise AssertionError("target sequential reference has the wrong Python prefix")
        references[accepted_count] = _clone_cache(target)
    target.rollback_cache(target_root)
    _assert_cache_equals_prefix(torch, target, root_snapshot, len(prompt_token_ids))
    return references


def _assert_loaded_ownership(draft, target) -> None:
    if draft is target or draft._model is target._model:
        raise AssertionError("dual loader calls shared backend or model ownership")
    if draft.tokenizer is target.tokenizer:
        raise AssertionError("dual loader calls shared tokenizer adapters")
    if draft.tokenizer._tokenizer is target.tokenizer._tokenizer:
        raise AssertionError("dual loader calls shared tokenizer wrapper objects")
    if draft.tokenizer._tokenizer._tokenizer is target.tokenizer._tokenizer._tokenizer:
        raise AssertionError("dual loader calls shared tokenizer runtime objects")
    if draft.tokenizer.compatibility_fingerprint() != target.tokenizer.compatibility_fingerprint():
        raise AssertionError("dual tokenizer compatibility fingerprints differ")
    if draft.vocab_size != target.vocab_size or draft.vocab_size <= 0:
        raise AssertionError("dual backend vocabulary sizes differ or are invalid")
    if draft._owner_id == target._owner_id:
        raise AssertionError("dual backends share an owner identity")
    draft_storage = draft._model.get_input_embeddings().weight.data_ptr()
    target_storage = target._model.get_input_embeddings().weight.data_ptr()
    if draft_storage == target_storage:
        raise AssertionError("representative model embedding storage is shared")


def _assert_live_cache_ownership(draft, target) -> None:
    if draft._cache is target._cache or draft._cache.layers is target._cache.layers:
        raise AssertionError("dual prefill shared DynamicCache ownership")
    if len(draft._cache.layers) != 24 or len(target._cache.layers) != 24:
        raise AssertionError("dual prefill did not create 24-layer caches")
    for layer_index, (draft_layer, target_layer) in enumerate(
        zip(draft._cache.layers, target._cache.layers, strict=True)
    ):
        if draft_layer is target_layer:
            raise AssertionError(f"cache layer {layer_index} is shared")
        if draft_layer.keys is target_layer.keys or draft_layer.values is target_layer.values:
            raise AssertionError(f"cache layer {layer_index} key/value storage is shared")
        if draft_layer.keys.data_ptr() == target_layer.keys.data_ptr():
            raise AssertionError(f"cache layer {layer_index} key storage aliases")
        if draft_layer.values.data_ptr() == target_layer.values.data_ptr():
            raise AssertionError(f"cache layer {layer_index} value storage aliases")


def _assert_foreign_roots_are_atomic(
    draft,
    target,
    *,
    draft_root,
    target_root,
) -> None:
    draft_before = _backend_reference_state(draft)
    target_before = _backend_reference_state(target)
    for backend, foreign_root in ((draft, target_root), (target, draft_root)):
        try:
            backend.rollback_cache(foreign_root)
        except CacheCheckpointStateError:
            pass
        else:
            raise AssertionError("cross-backend root rollback unexpectedly succeeded")
    if _backend_reference_state(draft) != draft_before:
        raise AssertionError("foreign root use mutated the draft backend")
    if _backend_reference_state(target) != target_before:
        raise AssertionError("foreign root use mutated the target backend")


def _assert_both_roots(
    torch,
    draft,
    target,
    *,
    draft_root,
    target_root,
    root_snapshots,
    prompt_token_ids,
    draft_identity,
    target_identity,
    draft_epoch,
    target_epoch,
    draft_layout,
    target_layout,
    device_index,
) -> None:
    _assert_root_state(
        torch,
        draft,
        root=draft_root,
        root_snapshot=root_snapshots["draft"],
        prompt_token_ids=prompt_token_ids,
        cache_identity=draft_identity,
        epoch=draft_epoch,
        layout=draft_layout,
        device_index=device_index,
    )
    _assert_root_state(
        torch,
        target,
        root=target_root,
        root_snapshot=root_snapshots["target"],
        prompt_token_ids=prompt_token_ids,
        cache_identity=target_identity,
        epoch=target_epoch,
        layout=target_layout,
        device_index=device_index,
    )


def _assert_root_state(
    torch,
    backend,
    *,
    root,
    root_snapshot,
    prompt_token_ids,
    cache_identity,
    epoch,
    layout,
    device_index,
) -> None:
    _assert_active_state(
        torch,
        backend,
        expected_prefix=prompt_token_ids,
        cache_identity=cache_identity,
        epoch=epoch,
        layout=layout,
        device_index=device_index,
    )
    if tuple(backend._cache_checkpoints) != (root.allocation_id,):
        raise AssertionError("backend registry does not contain exactly its caller root")
    _assert_cache_equals_prefix(torch, backend, root_snapshot, len(prompt_token_ids))


def _assert_active_state(
    torch,
    backend,
    *,
    expected_prefix,
    cache_identity,
    epoch,
    layout,
    device_index,
) -> None:
    if tuple(backend._active_token_ids) != tuple(expected_prefix):
        raise AssertionError("backend Python token prefix changed")
    if backend.cache_length != len(expected_prefix):
        raise AssertionError("backend logical cache length changed")
    if _cache_identity(backend) != cache_identity:
        raise AssertionError("backend cache object graph identity changed")
    if backend._epoch != epoch or backend._active_cache_layout != layout:
        raise AssertionError("backend epoch or layout signature changed")
    if (
        _assert_qualified_layout(
            torch,
            backend,
            expected_length=len(expected_prefix),
            device_index=device_index,
        )
        != layout
    ):
        raise AssertionError("physical cache layout differs from the epoch signature")


def _assert_result(
    torch,
    result,
    *,
    proposal_token_ids,
    accepted_count,
    replacement_token_id,
    initial_cache_length,
    final_cache_length,
) -> None:
    if [field.name for field in fields(result)] != [
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "initial_cache_length",
        "final_cache_length",
    ]:
        raise AssertionError("SpeculativeIterationResult field surface changed")
    if hasattr(result, "__dict__"):
        raise AssertionError("SpeculativeIterationResult is no longer slotted")
    if any(torch.is_tensor(getattr(result, field.name)) for field in fields(result)):
        raise AssertionError("SpeculativeIterationResult retained a tensor")
    if result.proposal_token_ids != proposal_token_ids:
        raise AssertionError("result proposal differs from the greedy draft")
    if result.accepted_count != accepted_count:
        raise AssertionError("result accepted count changed")
    if result.accepted_token_ids != proposal_token_ids[:accepted_count]:
        raise AssertionError("result accepted prefix changed")
    if result.replacement_token_id != replacement_token_id:
        raise AssertionError("result replacement changed")
    if result.uncached_next_token_id != replacement_token_id:
        raise AssertionError("result uncached replacement changed")
    if result.initial_cache_length != initial_cache_length:
        raise AssertionError("result initial cache length changed")
    if result.final_cache_length != final_cache_length:
        raise AssertionError("result final cache length changed")
    expected_output = (
        proposal_token_ids
        if replacement_token_id is None
        else (*proposal_token_ids[:accepted_count], replacement_token_id)
    )
    if result.output_token_ids != expected_output:
        raise AssertionError("result output token IDs changed")


def _assert_transients_released(*selectors) -> None:
    gc.collect()
    for selector in selectors:
        if any(reference() is not None for reference in selector.row_references):
            raise AssertionError("selector observation retained a native logits row")
        if any(reference() is not None for reference in selector.parent_references):
            raise AssertionError("selector observation retained a parent logits tensor")


def _assert_qualified_layout(torch, backend, *, expected_length: int, device_index: int):
    layout = inspect_pinned_dynamic_cache(
        torch,
        backend._transformers,
        backend._cache,
        expected_length=expected_length,
        device_index=device_index,
    )
    if layout != backend._active_cache_layout:
        raise AssertionError("active cache differs from its epoch layout signature")
    for layer_index, layer in enumerate(backend._cache.layers):
        for label, tensor in (("keys", layer.keys), ("values", layer.values)):
            if tuple(tensor.shape) != (1, 2, expected_length, 64):
                raise AssertionError(
                    f"cache layer {layer_index} {label} shape is {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.float16:
                raise AssertionError(f"cache layer {layer_index} {label} is not FP16")
            if str(tensor.device) != "cuda:0" or not tensor.is_cuda:
                raise AssertionError(f"cache layer {layer_index} {label} is not on cuda:0")
    return layout


def _clone_cache(backend):
    return tuple((layer.keys.clone(), layer.values.clone()) for layer in backend._cache.layers)


def _assert_cache_equals_prefix(torch, backend, snapshot, expected_length: int) -> None:
    if backend.cache_length != expected_length:
        raise AssertionError("cache snapshot comparison received the wrong logical length")
    for layer_index, layer in enumerate(backend._cache.layers):
        expected_keys, expected_values = snapshot[layer_index]
        if not torch.equal(layer.keys, expected_keys[..., :expected_length, :]):
            raise AssertionError(f"cache layer {layer_index} keys differ from role-local reference")
        if not torch.equal(layer.values, expected_values[..., :expected_length, :]):
            raise AssertionError(
                f"cache layer {layer_index} values differ from role-local reference"
            )


def _assert_checkpoint_metadata_cpu_only(torch, backend, checkpoint) -> None:
    snapshot = backend._cache_checkpoints[checkpoint.allocation_id]
    for field in fields(checkpoint):
        if torch.is_tensor(getattr(checkpoint, field.name)):
            raise AssertionError("checkpoint handle retained a tensor")
    if any(torch.is_tensor(token_id) for token_id in snapshot.token_ids):
        raise AssertionError("checkpoint token prefix retained a tensor")
    for field in fields(snapshot.layout):
        if torch.is_tensor(getattr(snapshot.layout, field.name)):
            raise AssertionError("checkpoint layout retained a tensor")


def _checkpoint_observation(checkpoint) -> CheckpointObservation:
    return CheckpointObservation(
        owner_id=checkpoint.owner_id,
        epoch=checkpoint.epoch,
        allocation_id=checkpoint.allocation_id,
        cache_length=checkpoint.cache_length,
    )


def _cache_identity(backend):
    return (
        id(backend._cache),
        id(backend._cache.layers),
        tuple(id(layer) for layer in backend._cache.layers),
    )


def _backend_reference_state(backend):
    return (
        _cache_identity(backend),
        backend._epoch,
        tuple(backend._active_token_ids),
        backend._active_cache_layout,
        backend._next_checkpoint_id,
        tuple(backend._cache_checkpoints),
    )


class _expect_closed_backend:
    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            raise AssertionError("closed backend accepted further work")
        if not isinstance(error, BackendStateError):
            return False
        return True


def _cuda_snapshot(torch, device) -> CudaSnapshot:
    torch.cuda.synchronize(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    try:
        import psutil

        working_set = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        working_set = None
    return CudaSnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        maximum_allocated_bytes=torch.cuda.max_memory_allocated(device),
        maximum_reserved_bytes=torch.cuda.max_memory_reserved(device),
        free_bytes=int(free_bytes),
        allocation_count=int(stats["allocation.all.current"]),
        active_count=int(stats["active.all.current"]),
        process_working_set_bytes=working_set,
    )


def _validate_cleanup(
    starting: CudaSnapshot,
    first: CudaSnapshot,
    second: CudaSnapshot,
) -> None:
    allocated_limit = max(starting.allocated_bytes, POST_FORWARD_ALLOCATED_ENVELOPE_BYTES)
    reserved_limit = max(starting.reserved_bytes, POST_FORWARD_RESERVED_ENVELOPE_BYTES)
    for lifecycle, snapshot in enumerate((first, second), start=1):
        if snapshot.allocated_bytes > allocated_limit:
            raise AssertionError(
                f"lifecycle {lifecycle} allocated cleanup {snapshot.allocated_bytes} "
                f"exceeded {allocated_limit}"
            )
        if snapshot.reserved_bytes > reserved_limit:
            raise AssertionError(
                f"lifecycle {lifecycle} reserved cleanup {snapshot.reserved_bytes} "
                f"exceeded {reserved_limit}"
            )
    if second.allocated_bytes > first.allocated_bytes:
        raise AssertionError("second lifecycle retained additional allocated CUDA memory")
    if second.reserved_bytes > first.reserved_bytes:
        raise AssertionError("second lifecycle retained additional reserved CUDA memory")


def _print_lifecycle(evidence: LifecycleEvidence) -> None:
    print(
        "D36_LIFECYCLE",
        f"lifecycle={evidence.lifecycle}",
        f"close_order={'-'.join(evidence.close_order)}",
        f"prompt_length={len(evidence.prompt_token_ids)}",
        f"prompt_tokens={evidence.prompt_token_ids}",
        f"current={evidence.current_token_id}",
        f"proposal={evidence.proposal_token_ids}",
        f"first_load_seconds={evidence.first_load_seconds:.6f}",
        f"second_load_seconds={evidence.second_load_seconds:.6f}",
        f"encoding_seconds={evidence.encoding_seconds:.6f}",
        f"prefill_seconds={evidence.prefill_seconds:.6f}",
        f"post_first_load_allocated={evidence.first_load.allocated_bytes}",
        f"post_first_load_reserved={evidence.first_load.reserved_bytes}",
        f"post_second_load_allocated={evidence.second_load.allocated_bytes}",
        f"post_second_load_reserved={evidence.second_load.reserved_bytes}",
        f"active_allocated={evidence.active_rooted.allocated_bytes}",
        f"active_reserved={evidence.active_rooted.reserved_bytes}",
        f"transaction_peak_allocated={evidence.transaction_peak.maximum_allocated_bytes}",
        f"transaction_peak_reserved={evidence.transaction_peak.maximum_reserved_bytes}",
        f"first_close_seconds={evidence.first_close_seconds:.6f}",
        f"after_first_close_allocated={evidence.after_first_close.allocated_bytes}",
        f"after_first_close_reserved={evidence.after_first_close.reserved_bytes}",
        f"second_close_seconds={evidence.second_close_seconds:.6f}",
        f"cleanup_seconds={evidence.cleanup_seconds:.6f}",
        f"cleanup_allocated={evidence.cleanup.allocated_bytes}",
        f"cleanup_reserved={evidence.cleanup.reserved_bytes}",
    )
    for outcome in evidence.outcomes:
        print(
            "D36_OUTCOME",
            f"lifecycle={evidence.lifecycle}",
            f"name={outcome.name}",
            f"accepted={outcome.accepted_count}",
            f"replacement={outcome.replacement_token_id}",
            f"cache_length={outcome.final_cache_length}",
            f"draft_selectors={outcome.draft_selector_calls}",
            f"target_selectors={outcome.target_selector_calls}",
            f"draft_forwards={outcome.draft_forwards}",
            f"target_forwards={outcome.target_forwards}",
            f"seconds={outcome.duration_seconds:.6f}",
        )
    for stable in evidence.stable_outcomes:
        print(
            "D36_STABLE",
            f"lifecycle={evidence.lifecycle}",
            f"name={stable.name}",
            f"allocated={stable.allocated_bytes}",
            f"reserved={stable.reserved_bytes}",
            f"allocations={stable.allocation_count}",
            f"active={stable.active_count}",
            f"min_seconds={stable.minimum_seconds:.6f}",
            f"median_seconds={stable.median_seconds:.6f}",
            f"max_seconds={stable.maximum_seconds:.6f}",
        )


if __name__ == "__main__":
    main()
