"""Bounded offline D34 production draft-proposal qualification."""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass, fields

from onyx_cuda import (
    CacheCheckpointStateError,
    CheckpointableAutoregressiveBackend,
    DEFAULT_TARGET_PROFILE,
    TemperatureTopPSelection,
    generate_draft_proposal,
    load_torch_cuda_target,
)
from onyx_cuda.torch_backend import select_cuda_argmax
from onyx_cuda.torch_dynamic_cache import inspect_pinned_dynamic_cache
from onyx_cuda.torch_selection import create_cuda_sampler


MIB = 1024 * 1024
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 4050 Laptop GPU"
EXPECTED_DEVICE_MEMORY_MIB = 6_141
VRAM_LIMIT_BYTES = EXPECTED_DEVICE_MEMORY_MIB * MIB
POST_FORWARD_ALLOCATED_ENVELOPE_BYTES = 8_520_704
POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024
QUALIFICATION_PROPOSAL_LENGTH = 3
PROPOSAL_ROLLBACK_CYCLES = 100
STABLE_WARMUP_CYCLES = 10
BASELINE_TOKEN_COUNT = 4
DEFAULT_PROMPT = "D34 pinned production draft proposal qualification"


@dataclass(frozen=True, slots=True)
class ForwardObservation:
    token_ids: tuple[int, ...]
    logits_to_keep: int


@dataclass(frozen=True, slots=True)
class SelectorObservation:
    forward_count: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    is_cuda: bool


@dataclass(frozen=True, slots=True)
class MatrixEvidence:
    prompt_length: int
    qualification_proposal_length: int
    current_token_id: int
    greedy_proposal_token_ids: tuple[int, ...]
    seeded_proposal_token_ids: tuple[int, ...]
    rejection_checkpoint_lengths: tuple[int, ...]
    final_cache_length: int
    greedy_replay: bool
    seeded_replay: bool
    forward_count: int
    greedy_selector_call_count: int
    seeded_selector_call_count: int
    cycle_count: int
    stable_allocated_bytes: int
    stable_reserved_bytes: int
    stable_allocation_count: int
    stable_active_count: int
    allocation_id_after_cycles: int


@dataclass(frozen=True, slots=True)
class LifecycleEvidence:
    lifecycle: int
    baseline_token_ids: tuple[int, ...]
    matrix: MatrixEvidence
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass(frozen=True, slots=True)
class CleanupSnapshot:
    allocated_bytes: int
    reserved_bytes: int


class _ForwardCounter:
    def __init__(self, model):
        self._model = model
        self.observations = []

    @property
    def calls(self) -> int:
        return len(self.observations)

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


class _RecordingSelector:
    def __init__(self, torch, backend, counter, selector):
        self._torch = torch
        self._backend = backend
        self._counter = counter
        self._selector = selector
        self.observations = []
        self.selected_token_ids = []

    @property
    def calls(self) -> int:
        return len(self.observations)

    def __call__(self, row) -> int:
        observation = SelectorObservation(
            forward_count=self._counter.calls,
            shape=tuple(row.shape),
            dtype=str(row.dtype),
            device=str(row.device),
            is_cuda=bool(row.is_cuda),
        )
        if observation.shape != (self._backend.vocab_size,):
            raise AssertionError(f"selector received row shape {observation.shape}")
        if row.dtype != self._torch.float16:
            raise AssertionError(f"selector received row dtype {row.dtype}")
        if observation.device != "cuda:0" or not observation.is_cuda:
            raise AssertionError(f"selector received row on {observation.device}")
        self.observations.append(observation)
        selected = self._selector(row)
        self.selected_token_ids.append(selected)
        return selected


class _InjectedSelectorFailure(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.device_index != 0:
        raise ValueError("D34 is qualified only on cuda:0")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import transformers

    if torch.__version__ != "2.6.0+cu124":
        raise AssertionError(f"unexpected PyTorch version: {torch.__version__}")
    if transformers.__version__ != "4.57.6":
        raise AssertionError(f"unexpected Transformers version: {transformers.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("D34 requires the one-device RTX 4050 acceptance environment")

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
    starting_cleanup = _cleanup_snapshot(torch, device)

    lifecycle_evidence = []
    cleanup_snapshots = []
    for lifecycle in (1, 2):
        lifecycle_evidence.append(
            _run_lifecycle(
                torch,
                lifecycle=lifecycle,
                device=device,
                device_index=arguments.device_index,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        cleanup_snapshots.append(_cleanup_snapshot(torch, device))

    first, second = lifecycle_evidence
    if first.baseline_token_ids != second.baseline_token_ids:
        raise AssertionError("target-only greedy baseline changed across lifecycles")
    first_matrix = first.matrix
    second_matrix = second.matrix
    if (
        first_matrix.prompt_length,
        first_matrix.current_token_id,
        first_matrix.greedy_proposal_token_ids,
        first_matrix.seeded_proposal_token_ids,
        first_matrix.rejection_checkpoint_lengths,
        first_matrix.final_cache_length,
    ) != (
        second_matrix.prompt_length,
        second_matrix.current_token_id,
        second_matrix.greedy_proposal_token_ids,
        second_matrix.seeded_proposal_token_ids,
        second_matrix.rejection_checkpoint_lengths,
        second_matrix.final_cache_length,
    ):
        raise AssertionError("D34 proposal evidence changed across complete lifecycles")
    if (
        first_matrix.stable_allocated_bytes,
        first_matrix.stable_reserved_bytes,
        first_matrix.stable_allocation_count,
        first_matrix.stable_active_count,
    ) != (
        second_matrix.stable_allocated_bytes,
        second_matrix.stable_reserved_bytes,
        second_matrix.stable_allocation_count,
        second_matrix.stable_active_count,
    ):
        raise AssertionError("D34 stable allocator state changed across lifecycles")

    _validate_cleanup(starting_cleanup, cleanup_snapshots)
    for evidence in lifecycle_evidence:
        if evidence.peak_allocated_bytes >= properties.total_memory:
            raise AssertionError("allocated peak reached the physical device limit")
        if evidence.peak_reserved_bytes >= properties.total_memory:
            raise AssertionError("reserved peak reached the physical device limit")
        if evidence.peak_allocated_bytes >= VRAM_LIMIT_BYTES:
            raise AssertionError("allocated peak reached the 6,141 MiB qualification limit")
        if evidence.peak_reserved_bytes >= VRAM_LIMIT_BYTES:
            raise AssertionError("reserved peak reached the 6,141 MiB qualification limit")

    for evidence in lifecycle_evidence:
        matrix = evidence.matrix
        print(
            "D34_LIFECYCLE",
            f"lifecycle={evidence.lifecycle}",
            f"baseline={evidence.baseline_token_ids}",
            f"prompt_length={matrix.prompt_length}",
            f"current={matrix.current_token_id}",
            f"proposal_length={matrix.qualification_proposal_length}",
            f"greedy={matrix.greedy_proposal_token_ids}",
            f"seeded={matrix.seeded_proposal_token_ids}",
            f"checkpoints={matrix.rejection_checkpoint_lengths}",
            f"cache_length={matrix.final_cache_length}",
            f"greedy_replay={matrix.greedy_replay}",
            f"seeded_replay={matrix.seeded_replay}",
            f"forwards={matrix.forward_count}",
            f"selectors={matrix.greedy_selector_call_count}",
            f"cycles={matrix.cycle_count}",
            f"stable_allocated={matrix.stable_allocated_bytes}",
            f"stable_reserved={matrix.stable_reserved_bytes}",
            f"stable_allocations={matrix.stable_allocation_count}",
            f"stable_active={matrix.stable_active_count}",
            f"peak_allocated={evidence.peak_allocated_bytes}",
            f"peak_reserved={evidence.peak_reserved_bytes}",
        )

    final_cleanup = cleanup_snapshots[-1]
    print(
        "D34 production draft-proposal CUDA qualification passed:",
        "profile=Qwen2.5-0.5B-Instruct",
        "device=cuda:0",
        f"proposal_length_fixture={QUALIFICATION_PROPOSAL_LENGTH}",
        f"cycles={PROPOSAL_ROLLBACK_CYCLES * 2}",
        "greedy_replay=True",
        "seeded_replay=True",
        f"after_allocated_bytes={final_cleanup.allocated_bytes}",
        f"after_reserved_bytes={final_cleanup.reserved_bytes}",
    )


def _run_lifecycle(
    torch,
    *,
    lifecycle: int,
    device,
    device_index: int,
) -> LifecycleEvidence:
    torch.cuda.reset_peak_memory_stats(device)
    backend = load_torch_cuda_target(
        DEFAULT_TARGET_PROFILE,
        device_index=device_index,
        local_files_only=True,
    )
    counter = _ForwardCounter(backend._model)
    backend._model = counter

    def fail_batched_verification(*args, **kwargs):
        raise AssertionError("D34 must not invoke verify_proposal")

    backend.verify_proposal = fail_batched_verification
    try:
        if not isinstance(backend, CheckpointableAutoregressiveBackend):
            raise AssertionError("production backend does not satisfy the checkpoint protocol")
        prompt_token_ids = backend.tokenizer.encode(DEFAULT_PROMPT)
        baseline_before = _run_greedy_baseline(backend, prompt_token_ids)
        matrix = _run_d34_matrix(
            torch,
            backend,
            counter=counter,
            prompt_token_ids=prompt_token_ids,
            device=device,
            device_index=device_index,
        )
        baseline_after = _run_greedy_baseline(backend, prompt_token_ids)
        if baseline_after != baseline_before:
            raise AssertionError("D34 matrix changed the target-only greedy baseline")
        torch.cuda.synchronize(device)
        return LifecycleEvidence(
            lifecycle=lifecycle,
            baseline_token_ids=baseline_before,
            matrix=matrix,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    finally:
        try:
            backend.close()
        finally:
            counter._model = None


def _run_greedy_baseline(backend, prompt_token_ids: tuple[int, ...]) -> tuple[int, ...]:
    step = backend.prefill(prompt_token_ids)
    generated = []
    for token_index in range(BASELINE_TOKEN_COUNT):
        token_id = select_cuda_argmax(step.logits)
        generated.append(token_id)
        if token_index != BASELINE_TOKEN_COUNT - 1:
            step = backend.decode(token_id)
    backend.reset()
    return tuple(generated)


def _run_d34_matrix(
    torch,
    backend,
    *,
    counter: _ForwardCounter,
    prompt_token_ids: tuple[int, ...],
    device,
    device_index: int,
) -> MatrixEvidence:
    prefill_step = backend.prefill(prompt_token_ids)
    prompt_length = len(prompt_token_ids)
    if prompt_length < 1 or backend.cache_length != prompt_length:
        raise AssertionError("qualification prompt did not create the expected cache")
    current_token_id = select_cuda_argmax(prefill_step.logits)
    del prefill_step

    root_cache = backend._cache
    root_epoch = backend._epoch
    root_layout = _assert_qualified_layout(
        torch,
        backend,
        expected_length=prompt_length,
        device_index=device_index,
    )
    root_snapshot = _clone_cache(backend)
    root = backend.create_cache_checkpoint()
    _assert_checkpoint_metadata_cpu_only(torch, backend, root)

    greedy_result, greedy_selector, forward_count = _run_proposal(
        torch,
        backend,
        counter=counter,
        root=root,
        root_cache=root_cache,
        root_layout=root_layout,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        selector=select_cuda_argmax,
        device_index=device_index,
    )
    greedy_proposal_token_ids = greedy_result.proposal_token_ids
    rejection_checkpoint_lengths = tuple(
        checkpoint.cache_length for checkpoint in greedy_result.rollback_checkpoints
    )
    final_cache_length = greedy_result.final_cache_length
    greedy_snapshot = _clone_cache(backend)
    greedy_selector_call_count = greedy_selector.calls

    active_result = greedy_result
    for rejection_index in range(QUALIFICATION_PROPOSAL_LENGTH):
        if rejection_index:
            active_result, _, _ = _run_proposal(
                torch,
                backend,
                counter=counter,
                root=root,
                root_cache=root_cache,
                root_layout=root_layout,
                prompt_token_ids=prompt_token_ids,
                current_token_id=current_token_id,
                selector=select_cuda_argmax,
                device_index=device_index,
            )
            if active_result.proposal_token_ids != greedy_proposal_token_ids:
                raise AssertionError("greedy proposal changed while testing rejection positions")

        target = active_result.rollback_checkpoints[rejection_index]
        backend.rollback_cache(target)
        expected_length = prompt_length + 1 + rejection_index
        expected_prefix = (
            *prompt_token_ids,
            current_token_id,
            *greedy_proposal_token_ids[:rejection_index],
        )
        _assert_cache_equals_snapshot(
            torch,
            backend,
            greedy_snapshot,
            expected_length=expected_length,
        )
        _assert_active_prefix(backend, expected_prefix)
        expected_registry = (
            root.allocation_id,
            *(
                checkpoint.allocation_id
                for checkpoint in active_result.rollback_checkpoints[: rejection_index + 1]
            ),
        )
        if tuple(backend._cache_checkpoints) != expected_registry:
            raise AssertionError("rejection rollback retained the wrong checkpoint handles")
        same_position_state = _cache_reference_state(backend)
        backend.rollback_cache(target)
        if _cache_reference_state(backend) != same_position_state:
            raise AssertionError("same-position rejection rollback changed cache state")
        for checkpoint in active_result.rollback_checkpoints[rejection_index + 1 :]:
            try:
                backend.rollback_cache(checkpoint)
            except CacheCheckpointStateError:
                pass
            else:
                raise AssertionError("rejection rollback retained a deeper checkpoint")

        release_state = _active_cache_reference_state(backend)
        _release_result(backend, active_result)
        _release_result(backend, active_result)
        if _active_cache_reference_state(backend) != release_state:
            raise AssertionError("rejection checkpoint release changed the active cache")
        if tuple(backend._cache_checkpoints) != (root.allocation_id,):
            raise AssertionError("only the external root should remain after rejection cleanup")
        backend.rollback_cache(root)
        _assert_root_state(
            torch,
            backend,
            root_cache=root_cache,
            root_layout=root_layout,
            root_snapshot=root_snapshot,
            prompt_token_ids=prompt_token_ids,
        )

    replay_first, _, _ = _run_proposal(
        torch,
        backend,
        counter=counter,
        root=root,
        root_cache=root_cache,
        root_layout=root_layout,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        selector=select_cuda_argmax,
        device_index=device_index,
    )
    replay_first_allocations = tuple(
        checkpoint.allocation_id for checkpoint in replay_first.rollback_checkpoints
    )
    backend.rollback_cache(root)
    _release_result(backend, replay_first)
    replay_result, _, _ = _run_proposal(
        torch,
        backend,
        counter=counter,
        root=root,
        root_cache=root_cache,
        root_layout=root_layout,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        selector=select_cuda_argmax,
        device_index=device_index,
    )
    replay_allocations = tuple(
        checkpoint.allocation_id for checkpoint in replay_result.rollback_checkpoints
    )
    if replay_result.proposal_token_ids != greedy_proposal_token_ids:
        raise AssertionError("greedy root rollback replay changed the proposal")
    if min(replay_allocations) <= max(replay_first_allocations):
        raise AssertionError("greedy replay reused checkpoint allocation identity")
    _assert_cache_equals_snapshot(
        torch,
        backend,
        greedy_snapshot,
        expected_length=final_cache_length,
    )
    greedy_replay = True

    full_acceptance_state = _active_cache_reference_state(backend)
    _release_result(backend, replay_result)
    _release_result(backend, replay_result)
    if _active_cache_reference_state(backend) != full_acceptance_state:
        raise AssertionError("full-acceptance checkpoint release changed the complete suffix")
    if tuple(backend._cache_checkpoints) != (root.allocation_id,):
        raise AssertionError("full-acceptance release did not leave only the external root")
    backend.rollback_cache(root)
    _assert_root_state(
        torch,
        backend,
        root_cache=root_cache,
        root_layout=root_layout,
        root_snapshot=root_snapshot,
        prompt_token_ids=prompt_token_ids,
    )

    sampling_policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=34)
    seeded_first, seeded_selector_one, _ = _run_proposal(
        torch,
        backend,
        counter=counter,
        root=root,
        root_cache=root_cache,
        root_layout=root_layout,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        selector=create_cuda_sampler(sampling_policy, device_index=device_index),
        device_index=device_index,
    )
    seeded_snapshot = _clone_cache(backend)
    seeded_first_allocations = tuple(
        checkpoint.allocation_id for checkpoint in seeded_first.rollback_checkpoints
    )
    backend.rollback_cache(root)
    _release_result(backend, seeded_first)
    seeded_replay_result, seeded_selector_two, _ = _run_proposal(
        torch,
        backend,
        counter=counter,
        root=root,
        root_cache=root_cache,
        root_layout=root_layout,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        selector=create_cuda_sampler(sampling_policy, device_index=device_index),
        device_index=device_index,
    )
    seeded_replay_allocations = tuple(
        checkpoint.allocation_id
        for checkpoint in seeded_replay_result.rollback_checkpoints
    )
    if seeded_replay_result.proposal_token_ids != seeded_first.proposal_token_ids:
        raise AssertionError("fresh same-seed CUDA sampler changed the proposal replay")
    if min(seeded_replay_allocations) <= max(seeded_first_allocations):
        raise AssertionError("seeded replay reused checkpoint allocation identity")
    _assert_cache_equals_snapshot(
        torch,
        backend,
        seeded_snapshot,
        expected_length=final_cache_length,
    )
    if (
        seeded_selector_one.calls != QUALIFICATION_PROPOSAL_LENGTH
        or seeded_selector_two.calls != QUALIFICATION_PROPOSAL_LENGTH
    ):
        raise AssertionError("seeded CUDA sampler was not borrowed exactly once per proposal row")
    seeded_proposal_token_ids = seeded_first.proposal_token_ids
    seeded_selector_call_count = seeded_selector_one.calls
    seeded_replay = True
    backend.rollback_cache(root)
    _release_result(backend, seeded_replay_result)
    _assert_root_state(
        torch,
        backend,
        root_cache=root_cache,
        root_layout=root_layout,
        root_snapshot=root_snapshot,
        prompt_token_ids=prompt_token_ids,
    )

    failure = _InjectedSelectorFailure("injected D34 selector failure")
    failure_calls = 0

    def fail_second_selection(row):
        nonlocal failure_calls
        failure_calls += 1
        if failure_calls == 2:
            raise failure
        return select_cuda_argmax(row)

    failing_selector = _RecordingSelector(
        torch,
        backend,
        counter,
        fail_second_selection,
    )
    calls_before_failure = counter.calls
    allocation_before_failure = backend._next_checkpoint_id
    try:
        generate_draft_proposal(
            backend,
            current_token_id,
            proposal_length=QUALIFICATION_PROPOSAL_LENGTH,
            select_token=failing_selector,
        )
    except _InjectedSelectorFailure as exc:
        if exc is not failure:
            raise AssertionError("D34 did not preserve the selector exception identity") from exc
    else:
        raise AssertionError("injected D34 selector failure did not propagate")
    if failing_selector.calls != 2 or counter.calls != calls_before_failure + 2:
        raise AssertionError("selector failure used the wrong number of rows or forwards")
    if tuple(backend._cache_checkpoints) != (root.allocation_id,):
        raise AssertionError("selector failure did not clean up D32-owned checkpoints")
    if backend._next_checkpoint_id <= allocation_before_failure:
        raise AssertionError("selector failure rewound checkpoint allocation identity")
    _assert_root_state(
        torch,
        backend,
        root_cache=root_cache,
        root_layout=root_layout,
        root_snapshot=root_snapshot,
        prompt_token_ids=prompt_token_ids,
    )

    reuse_result, _, _ = _run_proposal(
        torch,
        backend,
        counter=counter,
        root=root,
        root_cache=root_cache,
        root_layout=root_layout,
        prompt_token_ids=prompt_token_ids,
        current_token_id=current_token_id,
        selector=select_cuda_argmax,
        device_index=device_index,
    )
    if reuse_result.proposal_token_ids != greedy_proposal_token_ids:
        raise AssertionError("backend was not immediately reusable after selector failure")
    backend.rollback_cache(root)
    _release_result(backend, reuse_result)

    allocator_samples = []
    previous_allocation_id = root.allocation_id
    for _ in range(PROPOSAL_ROLLBACK_CYCLES):
        cycle_result, _, _ = _run_proposal(
            torch,
            backend,
            counter=counter,
            root=root,
            root_cache=root_cache,
            root_layout=root_layout,
            prompt_token_ids=prompt_token_ids,
            current_token_id=current_token_id,
            selector=select_cuda_argmax,
            device_index=device_index,
        )
        if cycle_result.proposal_token_ids != greedy_proposal_token_ids:
            raise AssertionError("bounded D34 cycle changed the greedy proposal")
        cycle_allocation_id = cycle_result.rollback_checkpoints[-1].allocation_id
        if cycle_allocation_id <= previous_allocation_id:
            raise AssertionError("bounded D34 cycle reused checkpoint allocation identity")
        previous_allocation_id = cycle_allocation_id
        backend.rollback_cache(root)
        _release_result(backend, cycle_result)
        if tuple(backend._cache_checkpoints) != (root.allocation_id,):
            raise AssertionError("bounded D34 cycle grew the checkpoint registry")
        if (
            backend._cache is not root_cache
            or backend._epoch != root_epoch
            or backend._active_cache_layout != root_layout
        ):
            raise AssertionError("bounded D34 cycle changed cache identity, epoch, or layout")
        _assert_active_prefix(backend, prompt_token_ids)
        torch.cuda.synchronize(device)
        stats = torch.cuda.memory_stats(device)
        allocator_samples.append(
            (
                torch.cuda.memory_allocated(device),
                torch.cuda.memory_reserved(device),
                stats["allocation.all.current"],
                stats["active.all.current"],
            )
        )

    stable_samples = allocator_samples[STABLE_WARMUP_CYCLES:]
    if len(set(stable_samples)) != 1:
        raise AssertionError(
            f"repeated D34 cycles did not stabilize: {sorted(set(stable_samples))}"
        )
    stable_sample = stable_samples[0]
    allocation_id_after_cycles = backend._next_checkpoint_id
    _assert_cache_equals_snapshot(
        torch,
        backend,
        root_snapshot,
        expected_length=prompt_length,
    )

    backend.release_cache_checkpoint(root)
    if backend._cache_checkpoints:
        raise AssertionError("external root release did not empty the checkpoint registry")
    reset_checkpoint = backend.create_cache_checkpoint()
    backend.reset()
    try:
        backend.rollback_cache(reset_checkpoint)
    except CacheCheckpointStateError:
        pass
    else:
        raise AssertionError("reset did not invalidate the D34-era checkpoint")
    backend.release_cache_checkpoint(reset_checkpoint)

    fresh_step = backend.prefill(prompt_token_ids)
    fresh_current = select_cuda_argmax(fresh_step.logits)
    if fresh_current != current_token_id:
        raise AssertionError("reset/prefill reuse changed the fixed current token")
    fresh_root_cache = backend._cache
    fresh_root_layout = backend._active_cache_layout
    fresh_root = backend.create_cache_checkpoint()
    fresh_result, _, _ = _run_proposal(
        torch,
        backend,
        counter=counter,
        root=fresh_root,
        root_cache=fresh_root_cache,
        root_layout=fresh_root_layout,
        prompt_token_ids=prompt_token_ids,
        current_token_id=fresh_current,
        selector=select_cuda_argmax,
        device_index=device_index,
    )
    if fresh_result.proposal_token_ids != greedy_proposal_token_ids:
        raise AssertionError("reset/prefill reuse changed the greedy proposal")
    _release_result(backend, fresh_result)
    backend.release_cache_checkpoint(fresh_root)
    backend.reset()

    del greedy_snapshot
    del seeded_snapshot
    del root_snapshot
    gc.collect()

    return MatrixEvidence(
        prompt_length=prompt_length,
        qualification_proposal_length=QUALIFICATION_PROPOSAL_LENGTH,
        current_token_id=current_token_id,
        greedy_proposal_token_ids=greedy_proposal_token_ids,
        seeded_proposal_token_ids=seeded_proposal_token_ids,
        rejection_checkpoint_lengths=rejection_checkpoint_lengths,
        final_cache_length=final_cache_length,
        greedy_replay=greedy_replay,
        seeded_replay=seeded_replay,
        forward_count=forward_count,
        greedy_selector_call_count=greedy_selector_call_count,
        seeded_selector_call_count=seeded_selector_call_count,
        cycle_count=PROPOSAL_ROLLBACK_CYCLES,
        stable_allocated_bytes=stable_sample[0],
        stable_reserved_bytes=stable_sample[1],
        stable_allocation_count=stable_sample[2],
        stable_active_count=stable_sample[3],
        allocation_id_after_cycles=allocation_id_after_cycles,
    )


def _run_proposal(
    torch,
    backend,
    *,
    counter: _ForwardCounter,
    root,
    root_cache,
    root_layout,
    prompt_token_ids: tuple[int, ...],
    current_token_id: int,
    selector,
    device_index: int,
):
    calls_before = counter.calls
    recording_selector = _RecordingSelector(
        torch,
        backend,
        counter,
        selector,
    )
    result = generate_draft_proposal(
        backend,
        current_token_id,
        proposal_length=QUALIFICATION_PROPOSAL_LENGTH,
        select_token=recording_selector,
    )
    forward_observations = tuple(counter.observations[calls_before:])
    expected_forward_tokens = (
        (current_token_id,),
        *((token_id,) for token_id in result.proposal_token_ids),
    )
    if tuple(item.token_ids for item in forward_observations) != expected_forward_tokens:
        raise AssertionError("D34 did not use the exact ordered one-token forwards")
    if any(item.logits_to_keep != 1 for item in forward_observations):
        raise AssertionError("D34 used a forward other than the ordinary one-row decode path")
    expected_forward_count = QUALIFICATION_PROPOSAL_LENGTH + 1
    if len(forward_observations) != expected_forward_count:
        raise AssertionError("D34 used the wrong number of model forwards")
    if recording_selector.calls != QUALIFICATION_PROPOSAL_LENGTH:
        raise AssertionError("D34 used the wrong number of selector calls")
    expected_selector_forwards = tuple(
        range(calls_before + 1, calls_before + QUALIFICATION_PROPOSAL_LENGTH + 1)
    )
    if (
        tuple(item.forward_count for item in recording_selector.observations)
        != expected_selector_forwards
    ):
        raise AssertionError("selector rows did not align with the first proposal forwards")
    if result.initial_cache_length != len(prompt_token_ids):
        raise AssertionError("D34 result reported the wrong initial cache length")
    expected_final_length = (
        len(prompt_token_ids) + QUALIFICATION_PROPOSAL_LENGTH + 1
    )
    if result.final_cache_length != expected_final_length:
        raise AssertionError("D34 result reported the wrong final cache length")
    expected_checkpoint_lengths = tuple(
        len(prompt_token_ids) + 1 + position
        for position in range(QUALIFICATION_PROPOSAL_LENGTH)
    )
    if (
        tuple(checkpoint.cache_length for checkpoint in result.rollback_checkpoints)
        != expected_checkpoint_lengths
    ):
        raise AssertionError("D34 result returned misaligned rejection checkpoints")
    expected_prefix = (
        *prompt_token_ids,
        current_token_id,
        *result.proposal_token_ids,
    )
    _assert_active_prefix(backend, expected_prefix)
    if backend._cache is not root_cache:
        raise AssertionError("D34 replaced the active DynamicCache object")
    if backend._active_cache_layout != root_layout:
        raise AssertionError("D34 changed the qualified DynamicCache layout")
    if (
        _assert_qualified_layout(
            torch,
            backend,
            expected_length=expected_final_length,
            device_index=device_index,
        )
        != root_layout
    ):
        raise AssertionError("D34 physical cache layout changed")
    expected_registry = (
        root.allocation_id,
        *(checkpoint.allocation_id for checkpoint in result.rollback_checkpoints),
    )
    if tuple(backend._cache_checkpoints) != expected_registry:
        raise AssertionError("D34 private start checkpoint was not released exactly")
    _assert_result_metadata_cpu_only(torch, backend, result)
    return result, recording_selector, len(forward_observations)


def _assert_root_state(
    torch,
    backend,
    *,
    root_cache,
    root_layout,
    root_snapshot,
    prompt_token_ids: tuple[int, ...],
) -> None:
    if backend._cache is not root_cache:
        raise AssertionError("root rollback replaced the active DynamicCache")
    if backend._active_cache_layout != root_layout:
        raise AssertionError("root rollback changed the qualified cache layout")
    _assert_active_prefix(backend, prompt_token_ids)
    _assert_cache_equals_snapshot(
        torch,
        backend,
        root_snapshot,
        expected_length=len(prompt_token_ids),
    )


def _assert_active_prefix(backend, token_ids: tuple[int, ...]) -> None:
    if tuple(backend._active_token_ids) != tuple(token_ids):
        raise AssertionError("backend Python token prefix differs from the expected sequence")
    if backend.cache_length != len(token_ids):
        raise AssertionError("backend cache length differs from the expected sequence")


def _assert_qualified_layout(
    torch,
    backend,
    *,
    expected_length: int,
    device_index: int,
):
    layout = inspect_pinned_dynamic_cache(
        torch,
        backend._transformers,
        backend._cache,
        expected_length=expected_length,
        device_index=device_index,
    )
    if layout != backend._active_cache_layout:
        raise AssertionError("active cache differs from the epoch layout signature")
    return layout


def _clone_cache(backend):
    return tuple(
        (layer.keys.clone(), layer.values.clone()) for layer in backend._cache.layers
    )


def _assert_cache_equals_snapshot(
    torch,
    backend,
    snapshot,
    *,
    expected_length: int,
) -> None:
    if backend.cache_length != expected_length:
        raise AssertionError("cache has the wrong logical length")
    if len(backend._cache.layers) != 24 or len(snapshot) != 24:
        raise AssertionError("D34 requires the qualified 24-layer cache")
    for layer_index, layer in enumerate(backend._cache.layers):
        expected_keys, expected_values = snapshot[layer_index]
        if not torch.equal(layer.keys, expected_keys[..., :expected_length, :]):
            raise AssertionError(f"cache layer {layer_index} keys differ from the exact prefix")
        if not torch.equal(layer.values, expected_values[..., :expected_length, :]):
            raise AssertionError(f"cache layer {layer_index} values differ from the exact prefix")


def _assert_result_metadata_cpu_only(torch, backend, result) -> None:
    if type(result.proposal_token_ids) is not tuple:
        raise AssertionError("proposal result token IDs are not an exact tuple")
    if type(result.rollback_checkpoints) is not tuple:
        raise AssertionError("proposal result checkpoints are not an exact tuple")
    if any(torch.is_tensor(token_id) for token_id in result.proposal_token_ids):
        raise AssertionError("proposal result retained a token tensor")
    if torch.is_tensor(result.initial_cache_length) or torch.is_tensor(
        result.final_cache_length
    ):
        raise AssertionError("proposal result retained tensor length metadata")
    for checkpoint in result.rollback_checkpoints:
        _assert_checkpoint_metadata_cpu_only(torch, backend, checkpoint)


def _assert_checkpoint_metadata_cpu_only(torch, backend, checkpoint) -> None:
    snapshot = backend._cache_checkpoints[checkpoint.allocation_id]
    for field in fields(checkpoint):
        value = getattr(checkpoint, field.name)
        if torch.is_tensor(value) or not isinstance(value, int):
            raise AssertionError("checkpoint handle retained non-scalar metadata")
    if any(torch.is_tensor(token_id) for token_id in snapshot.token_ids):
        raise AssertionError("checkpoint token prefix retained a tensor")
    if not all(isinstance(token_id, int) for token_id in snapshot.token_ids):
        raise AssertionError("checkpoint token prefix retained non-integer metadata")
    for field in fields(snapshot.layout):
        value = getattr(snapshot.layout, field.name)
        if torch.is_tensor(value) or not _is_plain_metadata(value):
            raise AssertionError("checkpoint layout retained non-scalar metadata")


def _is_plain_metadata(value) -> bool:
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if type(value) is tuple:
        return all(_is_plain_metadata(item) for item in value)
    return False


def _release_result(backend, result) -> None:
    for checkpoint in result.rollback_checkpoints:
        backend.release_cache_checkpoint(checkpoint)


def _cache_reference_state(backend):
    return (
        id(backend._cache),
        id(backend._cache.layers),
        tuple(
            (id(layer), id(layer.keys), id(layer.values))
            for layer in backend._cache.layers
        ),
        tuple(backend._active_token_ids),
        tuple(backend._cache_checkpoints),
        backend._next_checkpoint_id,
    )


def _active_cache_reference_state(backend):
    return (
        id(backend._cache),
        id(backend._cache.layers),
        tuple(
            (id(layer), id(layer.keys), id(layer.values))
            for layer in backend._cache.layers
        ),
        tuple(backend._active_token_ids),
        backend._active_cache_layout,
        backend.cache_length,
    )


def _cleanup_snapshot(torch, device) -> CleanupSnapshot:
    return CleanupSnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
    )


def _validate_cleanup(
    starting: CleanupSnapshot,
    snapshots: list[CleanupSnapshot],
) -> None:
    allocated_limit = max(starting.allocated_bytes, POST_FORWARD_ALLOCATED_ENVELOPE_BYTES)
    reserved_limit = max(starting.reserved_bytes, POST_FORWARD_RESERVED_ENVELOPE_BYTES)
    for lifecycle, snapshot in enumerate(snapshots, start=1):
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
    first, second = snapshots
    if second.allocated_bytes > first.allocated_bytes:
        raise AssertionError("second lifecycle retained additional allocated CUDA memory")
    if second.reserved_bytes > first.reserved_bytes:
        raise AssertionError("second lifecycle retained additional reserved CUDA memory")


if __name__ == "__main__":
    main()
