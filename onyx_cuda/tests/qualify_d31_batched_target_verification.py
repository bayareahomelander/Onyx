"""Bounded offline D31 production batched target-verification qualification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import weakref
from dataclasses import dataclass, fields

from onyx_cuda import (
    BatchedTargetVerificationBackend,
    CacheCheckpointStateError,
    load_torch_cuda_target,
)
from onyx_cuda.torch_backend import select_cuda_argmax
from onyx_cuda.torch_dynamic_cache import inspect_pinned_dynamic_cache


MIB = 1024 * 1024
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 4050 Laptop GPU"
EXPECTED_DEVICE_MEMORY_MIB = 6_141
VRAM_LIMIT_BYTES = EXPECTED_DEVICE_MEMORY_MIB * MIB
POST_FORWARD_ALLOCATED_ENVELOPE_BYTES = 8_520_704
POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024
PROPOSAL_LENGTH = 3
VERIFICATION_CYCLES = 100
STABLE_WARMUP_CYCLES = 10
BASELINE_TOKEN_COUNT = 4
DEFAULT_PROMPT = "D31 production batched target verification qualification"


@dataclass(frozen=True, slots=True)
class MatrixEvidence:
    prompt_length: int
    proposal_token_ids: tuple[int, ...]
    row_count: int
    final_cache_length: int
    sequential_greedy_ids: tuple[int, ...]
    batched_greedy_ids: tuple[int, ...]
    sequential_max_absolute_differences: tuple[float, ...]
    batch_fingerprint: str
    exact_replay: bool
    cycle_count: int
    stable_allocated_bytes: int
    stable_reserved_bytes: int
    stable_allocation_count: int
    stable_active_count: int


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
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self._model(**kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    arguments = parser.parse_args()
    if arguments.device_index != 0:
        raise ValueError("D31 is qualified only on cuda:0")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import transformers

    if torch.__version__ != "2.6.0+cu124":
        raise AssertionError(f"unexpected PyTorch version: {torch.__version__}")
    if transformers.__version__ != "4.57.6":
        raise AssertionError(f"unexpected Transformers version: {transformers.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("D31 requires the one-device RTX 4050 acceptance environment")

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
                prompt=arguments.prompt,
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
    if first.matrix.batch_fingerprint != second.matrix.batch_fingerprint:
        raise AssertionError("exact batched verification rows changed across lifecycles")
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
            "D31_LIFECYCLE",
            f"lifecycle={evidence.lifecycle}",
            f"baseline={evidence.baseline_token_ids}",
            f"prompt_length={matrix.prompt_length}",
            f"proposal={matrix.proposal_token_ids}",
            f"rows={matrix.row_count}",
            f"cache_length={matrix.final_cache_length}",
            f"max_abs_diffs={matrix.sequential_max_absolute_differences}",
            f"exact_replay={matrix.exact_replay}",
            f"stable_allocated={matrix.stable_allocated_bytes}",
            f"stable_reserved={matrix.stable_reserved_bytes}",
            f"peak_allocated={evidence.peak_allocated_bytes}",
            f"peak_reserved={evidence.peak_reserved_bytes}",
        )

    final_cleanup = cleanup_snapshots[-1]
    print(
        "D31 batched target-verification CUDA qualification passed:",
        "profile=Qwen2.5-0.5B-Instruct",
        "device=cuda:0",
        f"rows={PROPOSAL_LENGTH + 1}",
        f"cycles={VERIFICATION_CYCLES * 2}",
        "exact_replay=True",
        f"after_allocated_bytes={final_cleanup.allocated_bytes}",
        f"after_reserved_bytes={final_cleanup.reserved_bytes}",
    )


def _run_lifecycle(
    torch,
    *,
    lifecycle: int,
    prompt: str,
    device,
    device_index: int,
) -> LifecycleEvidence:
    torch.cuda.reset_peak_memory_stats(device)
    backend = load_torch_cuda_target(device_index=device_index, local_files_only=True)
    counter = _ForwardCounter(backend._model)
    backend._model = counter
    try:
        if not isinstance(backend, BatchedTargetVerificationBackend):
            raise AssertionError("production backend does not satisfy the D30 optional protocol")
        prompt_token_ids = backend.tokenizer.encode(prompt)
        baseline_before = _run_greedy_baseline(backend, prompt_token_ids)
        matrix = _run_verification_matrix(
            torch,
            backend,
            counter=counter,
            prompt_token_ids=prompt_token_ids,
            device=device,
            device_index=device_index,
        )
        baseline_after = _run_greedy_baseline(backend, prompt_token_ids)
        if baseline_after != baseline_before:
            raise AssertionError("verification matrix changed the target-only greedy baseline")
        torch.cuda.synchronize(device)
        return LifecycleEvidence(
            lifecycle=lifecycle,
            baseline_token_ids=baseline_before,
            matrix=matrix,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    finally:
        backend.close()
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


def _run_verification_matrix(
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
    root_cache = backend._cache
    root_layout = _assert_qualified_layout(
        torch,
        backend,
        expected_length=prompt_length,
        device_index=device_index,
    )
    root_snapshot = _clone_cache(backend)
    root = backend.create_cache_checkpoint()
    same_position = backend.create_cache_checkpoint()
    _assert_checkpoint_metadata_cpu_only(torch, backend, root)
    _assert_checkpoint_metadata_cpu_only(torch, backend, same_position)

    same_position_state = _cache_reference_state(backend)
    backend.rollback_cache(same_position)
    if _cache_reference_state(backend) != same_position_state:
        raise AssertionError("same-position rollback changed active cache state")

    current_token_id = select_cuda_argmax(prefill_step.logits)
    proposal = []
    sequential_rows = []
    sequential_input = current_token_id
    for row_index in range(PROPOSAL_LENGTH + 1):
        sequential_step = backend.decode(sequential_input)
        sequential_rows.append(sequential_step.logits.clone())
        if row_index < PROPOSAL_LENGTH:
            sequential_input = select_cuda_argmax(sequential_step.logits)
            proposal.append(sequential_input)
    proposal_token_ids = tuple(proposal)

    backend.rollback_cache(root)
    _assert_cache_equals_snapshot(torch, backend, root_snapshot, prompt_length)
    if backend._cache is not root_cache:
        raise AssertionError("rollback replaced the active DynamicCache object")

    calls_before = counter.calls
    first_result = backend.verify_proposal(current_token_id, proposal_token_ids)
    if counter.calls != calls_before + 1:
        raise AssertionError("verify_proposal did not use exactly one model forward")
    row_count = PROPOSAL_LENGTH + 1
    expected_cache_length = prompt_length + row_count
    _assert_verification_result(
        torch,
        backend,
        first_result,
        row_count=row_count,
        expected_cache_length=expected_cache_length,
    )
    if backend._cache is not root_cache:
        raise AssertionError("verification replaced the active DynamicCache object")
    if tuple(backend._active_token_ids) != (
        *prompt_token_ids,
        current_token_id,
        *proposal_token_ids,
    ):
        raise AssertionError("verification did not record the exact input suffix")
    if (
        _assert_qualified_layout(
            torch,
            backend,
            expected_length=expected_cache_length,
            device_index=device_index,
        )
        != root_layout
    ):
        raise AssertionError("verification changed the qualified cache layout")

    sequential_greedy_ids = tuple(select_cuda_argmax(row) for row in sequential_rows)
    batched_greedy_ids = tuple(select_cuda_argmax(row) for row in first_result.logit_rows)
    if batched_greedy_ids != sequential_greedy_ids:
        raise AssertionError("batched row decisions do not align with sequential decoding")
    max_absolute_differences = tuple(
        float((batched - sequential).abs().max().item())
        for batched, sequential in zip(
            first_result.logit_rows,
            sequential_rows,
            strict=True,
        )
    )
    batch_fingerprint = _fingerprint_rows(first_result.logit_rows)
    replay_reference = tuple(row.clone() for row in first_result.logit_rows)
    row_ref = weakref.ref(first_result.logit_rows[0])
    parent_tensor = first_result.logit_rows[0]._base
    parent_ref = weakref.ref(parent_tensor) if parent_tensor is not None else None

    post_batch = backend.create_cache_checkpoint()
    backend.rollback_cache(root)
    _assert_cache_equals_snapshot(torch, backend, root_snapshot, prompt_length)
    try:
        backend.rollback_cache(post_batch)
    except CacheCheckpointStateError:
        pass
    else:
        raise AssertionError("rollback did not invalidate the post-batch checkpoint")

    del first_result
    del parent_tensor
    gc.collect()
    if row_ref() is not None:
        raise AssertionError("backend retained a returned verification row")
    if parent_ref is not None and parent_ref() is not None:
        raise AssertionError("backend retained the parent verification logits tensor")

    calls_before = counter.calls
    replay_result = backend.verify_proposal(current_token_id, proposal_token_ids)
    if counter.calls != calls_before + 1:
        raise AssertionError("verification replay did not use exactly one model forward")
    exact_replay = all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            replay_result.logit_rows,
            replay_reference,
            strict=True,
        )
    )
    if not exact_replay:
        raise AssertionError("same-batch rollback replay changed native logits")

    backend.rollback_cache(root)
    alternative_current = (current_token_id + 1) % backend.vocab_size
    if alternative_current == current_token_id:
        raise AssertionError("alternative current token was not distinct")
    alternative_result = backend.verify_proposal(alternative_current, proposal_token_ids)
    if tuple(backend._active_token_ids) != (
        *prompt_token_ids,
        alternative_current,
        *proposal_token_ids,
    ):
        raise AssertionError("alternative verification suffix was not tracked exactly")
    if backend._active_cache_layout != root_layout:
        raise AssertionError("alternative suffix changed the qualified cache layout")
    del replay_result
    del alternative_result
    del replay_reference
    del sequential_rows

    backend.rollback_cache(root)
    backend.release_cache_checkpoint(same_position)
    backend.release_cache_checkpoint(same_position)
    if tuple(backend._cache_checkpoints) != (root.allocation_id,):
        raise AssertionError("only the reusable root checkpoint should remain")

    allocator_samples = []
    for _ in range(VERIFICATION_CYCLES):
        cycle_checkpoint = backend.create_cache_checkpoint()
        calls_before = counter.calls
        cycle_result = backend.verify_proposal(current_token_id, proposal_token_ids)
        if counter.calls != calls_before + 1:
            raise AssertionError("bounded verification cycle used multiple forwards")
        if tuple(select_cuda_argmax(row) for row in cycle_result.logit_rows) != batched_greedy_ids:
            raise AssertionError("bounded verification cycle changed row order or decisions")
        del cycle_result
        backend.rollback_cache(cycle_checkpoint)
        backend.release_cache_checkpoint(cycle_checkpoint)
        if tuple(backend._cache_checkpoints) != (root.allocation_id,):
            raise AssertionError("bounded cycle grew the checkpoint registry")
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
            f"repeated verification did not stabilize: {sorted(set(stable_samples))}"
        )
    stable_sample = stable_samples[0]

    backend.release_cache_checkpoint(root)
    if backend._cache_checkpoints:
        raise AssertionError("checkpoint registry did not empty after root release")
    reset_checkpoint = backend.create_cache_checkpoint()
    backend.reset()
    try:
        backend.rollback_cache(reset_checkpoint)
    except CacheCheckpointStateError:
        pass
    else:
        raise AssertionError("reset did not invalidate the verification-era checkpoint")
    backend.release_cache_checkpoint(reset_checkpoint)

    fresh_step = backend.prefill(prompt_token_ids)
    fresh_current = select_cuda_argmax(fresh_step.logits)
    fresh_result = backend.verify_proposal(fresh_current, proposal_token_ids)
    _assert_verification_result(
        torch,
        backend,
        fresh_result,
        row_count=row_count,
        expected_cache_length=expected_cache_length,
    )
    del fresh_result
    del root_snapshot
    gc.collect()

    return MatrixEvidence(
        prompt_length=prompt_length,
        proposal_token_ids=proposal_token_ids,
        row_count=row_count,
        final_cache_length=expected_cache_length,
        sequential_greedy_ids=sequential_greedy_ids,
        batched_greedy_ids=batched_greedy_ids,
        sequential_max_absolute_differences=max_absolute_differences,
        batch_fingerprint=batch_fingerprint,
        exact_replay=exact_replay,
        cycle_count=VERIFICATION_CYCLES,
        stable_allocated_bytes=stable_sample[0],
        stable_reserved_bytes=stable_sample[1],
        stable_allocation_count=stable_sample[2],
        stable_active_count=stable_sample[3],
    )


def _assert_verification_result(
    torch,
    backend,
    result,
    *,
    row_count: int,
    expected_cache_length: int,
) -> None:
    if len(result.logit_rows) != row_count:
        raise AssertionError(f"verification returned {len(result.logit_rows)} rows")
    if result.cache_length != expected_cache_length:
        raise AssertionError("verification result reported the wrong cache length")
    if backend.cache_length != expected_cache_length:
        raise AssertionError("verification advanced the cache by the wrong length")
    if backend.padded_vocab_rows != 271:
        raise AssertionError("pinned model/tokenizer padded-vocabulary boundary changed")
    for row_index, row in enumerate(result.logit_rows):
        if tuple(row.shape) != (backend.vocab_size,):
            raise AssertionError(f"verification row {row_index} has shape {tuple(row.shape)}")
        if row.dtype != torch.float16:
            raise AssertionError(f"verification row {row_index} has dtype {row.dtype}")
        if str(row.device) != "cuda:0" or not row.is_cuda:
            raise AssertionError(f"verification row {row_index} is on {row.device}")


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
    return tuple((layer.keys.clone(), layer.values.clone()) for layer in backend._cache.layers)


def _assert_cache_equals_snapshot(torch, backend, snapshot, expected_length: int) -> None:
    if backend.cache_length != expected_length:
        raise AssertionError("cache rollback restored the wrong logical length")
    for layer_index, layer in enumerate(backend._cache.layers):
        expected_keys, expected_values = snapshot[layer_index]
        if not torch.equal(layer.keys, expected_keys[..., :expected_length, :]):
            raise AssertionError(f"cache layer {layer_index} keys differ from the exact prefix")
        if not torch.equal(layer.values, expected_values[..., :expected_length, :]):
            raise AssertionError(f"cache layer {layer_index} values differ from the exact prefix")


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


def _fingerprint_rows(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.detach().contiguous().view(-1).cpu().numpy().tobytes())
    return digest.hexdigest()


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
