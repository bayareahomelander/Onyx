"""Bounded offline D29 production ``DynamicCache`` rollback qualification."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass, fields

from onyx_cuda import (
    CacheCheckpointStateError,
    CheckpointableAutoregressiveBackend,
    load_torch_cuda_target,
)
from onyx_cuda.torch_backend import select_cuda_argmax
from onyx_cuda.torch_dynamic_cache import (
    NATIVE_CROP_ROLLBACK_MECHANISM,
    inspect_pinned_dynamic_cache,
)


VRAM_LIMIT_BYTES = 6_141 * 1024 * 1024
POST_FORWARD_ALLOCATED_ENVELOPE_BYTES = 8_520_704
POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024
ROLLBACK_REPLAY_CYCLES = 100
STABLE_WARMUP_CYCLES = 10
DEFAULT_PROMPT = "D29 production DynamicCache rollback qualification"


@dataclass(frozen=True, slots=True)
class CacheObservation:
    label: str
    length: int
    layer_count: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_stride: tuple[int, ...]
    value_stride: tuple[int, ...]
    storage_offset: int
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class MatrixEvidence:
    observations: tuple[CacheObservation, ...]
    rollback_latency_seconds: float
    rollback_peak_allocated_bytes: int
    rollback_peak_reserved_bytes: int
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    arguments = parser.parse_args()
    if arguments.device_index != 0:
        raise ValueError("D29 is qualified only on cuda:0")

    import torch
    import transformers

    if torch.__version__ != "2.6.0+cu124":
        raise AssertionError(f"unexpected PyTorch version: {torch.__version__}")
    if transformers.__version__ != "4.57.6":
        raise AssertionError(f"unexpected Transformers version: {transformers.__version__}")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("D29 requires the one-device RTX 4050 acceptance environment")
    properties = torch.cuda.get_device_properties(device)
    if properties.name != "NVIDIA GeForce RTX 4050 Laptop GPU":
        raise AssertionError(f"unexpected CUDA device: {properties.name}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    starting_cleanup = CleanupSnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
    )

    lifecycle_evidence = []
    cleanup_snapshots = []
    for lifecycle in (1, 2):
        evidence = _run_lifecycle(
            torch,
            lifecycle=lifecycle,
            prompt=arguments.prompt,
            device=device,
            device_index=arguments.device_index,
        )
        lifecycle_evidence.append(evidence)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        cleanup_snapshots.append(
            CleanupSnapshot(
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
            )
        )

    if lifecycle_evidence[0].baseline_token_ids != lifecycle_evidence[1].baseline_token_ids:
        raise AssertionError("target-only greedy baseline changed across complete lifecycles")
    _validate_cleanup(starting_cleanup, cleanup_snapshots)
    for evidence in lifecycle_evidence:
        if evidence.peak_allocated_bytes >= properties.total_memory:
            raise AssertionError("observed allocated peak reached the physical device limit")
        if evidence.peak_reserved_bytes >= properties.total_memory:
            raise AssertionError("observed reserved peak reached the physical device limit")
        if evidence.peak_allocated_bytes >= VRAM_LIMIT_BYTES:
            raise AssertionError("observed allocated peak reached the 6,141 MiB qualification limit")
        if evidence.peak_reserved_bytes >= VRAM_LIMIT_BYTES:
            raise AssertionError("observed reserved peak reached the 6,141 MiB qualification limit")

    for evidence in lifecycle_evidence:
        matrix = evidence.matrix
        print(
            "D29_LIFECYCLE",
            f"lifecycle={evidence.lifecycle}",
            f"baseline={evidence.baseline_token_ids}",
            f"lengths={tuple(item.length for item in matrix.observations)}",
            f"rollback_ms={matrix.rollback_latency_seconds * 1000:.6f}",
            f"stable_allocated={matrix.stable_allocated_bytes}",
            f"stable_reserved={matrix.stable_reserved_bytes}",
            f"peak_allocated={evidence.peak_allocated_bytes}",
            f"peak_reserved={evidence.peak_reserved_bytes}",
        )

    final_cleanup = cleanup_snapshots[-1]
    median_rollback = statistics.median(
        item.matrix.rollback_latency_seconds for item in lifecycle_evidence
    )
    print(
        "D29 DynamicCache rollback CUDA qualification passed:",
        f"mechanism={NATIVE_CROP_ROLLBACK_MECHANISM}",
        "cache=transformers.cache_utils.DynamicCache",
        "layer=transformers.cache_utils.DynamicLayer",
        "layers=24",
        "shape=(1,2,length,64)",
        f"cycles={ROLLBACK_REPLAY_CYCLES * 2}",
        f"median_rollback_ms={median_rollback * 1000:.6f}",
        f"after_allocated_bytes={final_cleanup.allocated_bytes}",
        f"after_reserved_bytes={final_cleanup.reserved_bytes}",
    )


def _run_lifecycle(torch, *, lifecycle: int, prompt: str, device, device_index: int):
    torch.cuda.reset_peak_memory_stats(device)
    backend = load_torch_cuda_target(device_index=device_index, local_files_only=True)
    try:
        if not isinstance(backend, CheckpointableAutoregressiveBackend):
            raise AssertionError("production backend does not satisfy the D28 optional protocol")
        prompt_token_ids = backend.tokenizer.encode(prompt)
        baseline_before = _run_greedy_baseline(backend, prompt_token_ids)
        matrix = _run_checkpoint_matrix(
            torch,
            backend,
            prompt_token_ids=prompt_token_ids,
            device=device,
            device_index=device_index,
        )
        baseline_after = _run_greedy_baseline(backend, prompt_token_ids)
        if baseline_after != baseline_before:
            raise AssertionError("checkpoint matrix changed the target-only greedy baseline")
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


def _run_greedy_baseline(backend, prompt_token_ids: tuple[int, ...]) -> tuple[int, ...]:
    step = backend.prefill(prompt_token_ids)
    generated = []
    for token_index in range(4):
        token_id = select_cuda_argmax(step.logits)
        generated.append(token_id)
        if token_index != 3:
            step = backend.decode(token_id)
    backend.reset()
    return tuple(generated)


def _run_checkpoint_matrix(
    torch,
    backend,
    *,
    prompt_token_ids: tuple[int, ...],
    device,
    device_index: int,
) -> MatrixEvidence:
    step = backend.prefill(prompt_token_ids)
    before_advance = _observe_cache(
        torch,
        backend,
        label="before_advance",
        device_index=device_index,
    )
    base_length = backend.cache_length
    base_snapshot = _clone_cache(backend)

    torch.cuda.synchronize(device)
    checkpoint_memory_before = _memory_pair(torch, device)
    root = backend.create_cache_checkpoint()
    same_position = backend.create_cache_checkpoint()
    _assert_checkpoint_metadata_cpu_only(torch, backend, root)
    _assert_checkpoint_metadata_cpu_only(torch, backend, same_position)
    torch.cuda.synchronize(device)
    if _memory_pair(torch, device) != checkpoint_memory_before:
        raise AssertionError("checkpoint creation changed CUDA allocator state")

    same_position_state = _cache_reference_state(backend)
    backend.rollback_cache(same_position)
    if _cache_reference_state(backend) != same_position_state:
        raise AssertionError("same-position rollback changed cache references")

    suffix_token_ids = []
    suffix_logits = []
    for suffix_index in range(4):
        token_id = select_cuda_argmax(step.logits)
        suffix_token_ids.append(token_id)
        step = backend.decode(token_id)
        suffix_logits.append(step.logits.clone())
        if suffix_index == 1:
            middle = backend.create_cache_checkpoint()
    deepest = backend.create_cache_checkpoint()
    full_snapshot = _clone_cache(backend)
    before_rollback = _observe_cache(
        torch,
        backend,
        label="before_rollback",
        device_index=device_index,
    )

    backend.rollback_cache(middle)
    _assert_cache_equals_snapshot(torch, backend, full_snapshot, base_length + 2)
    try:
        backend.rollback_cache(deepest)
    except CacheCheckpointStateError:
        pass
    else:
        raise AssertionError("rollback did not invalidate a deeper checkpoint")
    backend.release_cache_checkpoint(deepest)

    torch.cuda.synchronize(device)
    rollback_start = time.perf_counter()
    backend.rollback_cache(root)
    torch.cuda.synchronize(device)
    rollback_latency = time.perf_counter() - rollback_start
    _assert_cache_equals_snapshot(torch, backend, base_snapshot, base_length)
    after_rollback = _observe_cache(
        torch,
        backend,
        label="after_rollback",
        device_index=device_index,
    )
    try:
        backend.rollback_cache(middle)
    except CacheCheckpointStateError:
        pass
    else:
        raise AssertionError("root rollback did not invalidate the middle checkpoint")

    for suffix_index, token_id in enumerate(suffix_token_ids):
        replay = backend.decode(token_id)
        if not torch.equal(replay.logits, suffix_logits[suffix_index]):
            raise AssertionError(f"same-token replay logits changed at suffix index {suffix_index}")
    _assert_cache_equals_snapshot(torch, backend, full_snapshot, base_length + 4)
    after_replay = _observe_cache(
        torch,
        backend,
        label="after_replay",
        device_index=device_index,
    )

    backend.rollback_cache(root)
    alternative_token = (suffix_token_ids[0] + 1) % backend.vocab_size
    if alternative_token == suffix_token_ids[0]:
        raise AssertionError("alternative suffix token was not distinct")
    backend.decode(alternative_token)
    if tuple(backend._active_token_ids) != prompt_token_ids + (alternative_token,):
        raise AssertionError("alternative suffix was not tracked exactly")
    backend.rollback_cache(root)
    _assert_cache_equals_snapshot(torch, backend, base_snapshot, base_length)

    torch.cuda.synchronize(device)
    release_memory_before = _memory_pair(torch, device)
    backend.release_cache_checkpoint(same_position)
    backend.release_cache_checkpoint(same_position)
    torch.cuda.synchronize(device)
    if _memory_pair(torch, device) != release_memory_before:
        raise AssertionError("checkpoint release changed CUDA allocator state")
    if len(backend._cache_checkpoints) != 1:
        raise AssertionError("only the reusable root checkpoint should remain")

    allocator_samples = []
    for _ in range(ROLLBACK_REPLAY_CYCLES):
        cycle_checkpoint = backend.create_cache_checkpoint()
        replay = backend.decode(suffix_token_ids[0])
        if not torch.equal(replay.logits, suffix_logits[0]):
            raise AssertionError("bounded-cycle replay logits changed")
        backend.rollback_cache(cycle_checkpoint)
        backend.release_cache_checkpoint(cycle_checkpoint)
        if len(backend._cache_checkpoints) != 1:
            raise AssertionError("bounded cycle grew the active checkpoint registry")
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
            f"repeated rollback/replay did not stabilize: {sorted(set(stable_samples))}"
        )
    stable_sample = stable_samples[0]
    allocation_id_after_cycles = backend._next_checkpoint_id

    backend.release_cache_checkpoint(root)
    if backend._cache_checkpoints:
        raise AssertionError("checkpoint registry did not empty after final release")
    reset_checkpoint = backend.create_cache_checkpoint()
    backend.reset()
    try:
        backend.rollback_cache(reset_checkpoint)
    except CacheCheckpointStateError:
        pass
    else:
        raise AssertionError("reset did not invalidate the active checkpoint")
    backend.release_cache_checkpoint(reset_checkpoint)

    fresh = backend.prefill(prompt_token_ids)
    if fresh.cache_length != len(prompt_token_ids):
        raise AssertionError("fresh prefill after reset reported the wrong cache length")
    fresh_checkpoint = backend.create_cache_checkpoint()
    backend.release_cache_checkpoint(fresh_checkpoint)

    return MatrixEvidence(
        observations=(before_advance, before_rollback, after_rollback, after_replay),
        rollback_latency_seconds=rollback_latency,
        rollback_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        rollback_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        stable_allocated_bytes=stable_sample[0],
        stable_reserved_bytes=stable_sample[1],
        stable_allocation_count=stable_sample[2],
        stable_active_count=stable_sample[3],
        allocation_id_after_cycles=allocation_id_after_cycles,
    )


def _observe_cache(torch, backend, *, label: str, device_index: int) -> CacheObservation:
    cache = backend._cache
    length = backend.cache_length
    layout = inspect_pinned_dynamic_cache(
        torch,
        backend._transformers,
        cache,
        expected_length=length,
        device_index=device_index,
    )
    if layout != backend._active_cache_layout:
        raise AssertionError(f"{label} cache layout differs from the active epoch signature")
    expected_shape = (1, 2, length, 64)
    first = cache.layers[0]
    for layer_index, layer in enumerate(cache.layers):
        if layer.get_seq_length() != length:
            raise AssertionError(f"{label} layer {layer_index} length disagrees")
        if tuple(layer.keys.shape) != expected_shape or tuple(layer.values.shape) != expected_shape:
            raise AssertionError(f"{label} layer {layer_index} shape disagrees")
        if layer.keys.device != first.keys.device or layer.values.device != first.values.device:
            raise AssertionError(f"{label} layer {layer_index} device disagrees")
        if layer.keys.dtype != first.keys.dtype or layer.values.dtype != first.values.dtype:
            raise AssertionError(f"{label} layer {layer_index} dtype disagrees")
    return CacheObservation(
        label=label,
        length=length,
        layer_count=len(cache.layers),
        key_shape=tuple(first.keys.shape),
        value_shape=tuple(first.values.shape),
        key_stride=tuple(first.keys.stride()),
        value_stride=tuple(first.values.stride()),
        storage_offset=first.keys.storage_offset(),
        device=str(first.keys.device),
        dtype=str(first.keys.dtype),
    )


def _clone_cache(backend):
    return tuple((layer.keys.clone(), layer.values.clone()) for layer in backend._cache.layers)


def _assert_cache_equals_snapshot(torch, backend, snapshot, expected_length: int) -> None:
    if backend.cache_length != expected_length:
        raise AssertionError(
            f"cache length {backend.cache_length} does not equal expected {expected_length}"
        )
    for layer_index, layer in enumerate(backend._cache.layers):
        expected_keys, expected_values = snapshot[layer_index]
        if not torch.equal(layer.keys, expected_keys[..., :expected_length, :]):
            raise AssertionError(f"cache layer {layer_index} keys do not match exact prefix")
        if not torch.equal(layer.values, expected_values[..., :expected_length, :]):
            raise AssertionError(f"cache layer {layer_index} values do not match exact prefix")


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
            (id(layer), id(layer.keys), id(layer.values)) for layer in backend._cache.layers
        ),
        tuple(backend._active_token_ids),
        tuple(backend._cache_checkpoints),
        backend._next_checkpoint_id,
    )


def _memory_pair(torch, device) -> tuple[int, int]:
    return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)


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
