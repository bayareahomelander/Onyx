"""Bounded real-model KV-cache probe for the Windows CUDA prototype."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .real_logits_handoff import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_PROMPT,
    DEFAULT_REGEX,
    QUANTIZATION_NAME,
    MemorySnapshot,
    OneStepSelection,
    _annotate_cleanup_failures,
    _detach_exception_tracebacks,
    _memory_snapshot,
    _require_runtime,
    _run_cleanup_step,
    select_and_advance_one_token,
    validate_real_logits,
)
from .tokenizer_probe import _load_tokenizer_metadata, inspect_loaded_tokenizer


@dataclass(frozen=True)
class KvCacheLayerSnapshot:
    """Shape, placement, and storage metadata for one KV-cache layer."""

    layer_index: int
    key_shape: Tuple[int, ...]
    value_shape: Tuple[int, ...]
    dtype: str
    device: str
    sequence_length: int
    storage_bytes: int


@dataclass(frozen=True)
class KvCacheSnapshot:
    """Validated metadata for a complete model cache at one probe phase."""

    phase: str
    cache_type: str
    layer_count: int
    sequence_length: int
    storage_bytes: int
    layers: Tuple[KvCacheLayerSnapshot, ...]


@dataclass(frozen=True)
class KvCacheProbeTimings:
    """Host-observed timings for setup, prefill, cached decode, and cleanup."""

    metadata_load_s: float
    vocabulary_validation_s: float
    model_load_s: float
    tokenization_s: float
    prefill_s: float
    selection_s: float
    decode_s: float
    cleanup_s: float
    total_s: float


@dataclass(frozen=True)
class KvCacheProbeReport:
    """Machine-readable result for one prefill and one cached decode step."""

    model_id: str
    requested_revision: Optional[str]
    resolved_revision: Optional[str]
    quantization: str
    prompt: str
    regex: str
    input_token_count: int
    expected_layer_count: int
    expected_logits_width: int
    prefill_logits_shape: Tuple[int, ...]
    prefill_logits_dtype: str
    prefill_logits_device: str
    decode_logits_shape: Tuple[int, ...]
    decode_logits_dtype: str
    decode_logits_device: str
    cache_object_reused: bool
    cache_sequence_growth: int
    prefill_cache: KvCacheSnapshot
    decode_cache: KvCacheSnapshot
    selection: OneStepSelection
    peak_cuda_allocated_bytes: int
    timings: KvCacheProbeTimings
    memory_snapshots: Tuple[MemorySnapshot, ...]

    @property
    def passed(self) -> bool:
        """Whether the cache and constrained-decode contracts were demonstrated."""
        return (
            self.prefill_logits_device.startswith("cuda")
            and self.decode_logits_device.startswith("cuda")
            and self.prefill_logits_shape[-1] == self.expected_logits_width
            and self.decode_logits_shape[-1] == self.expected_logits_width
            and self.prefill_cache.layer_count == self.expected_layer_count
            and self.decode_cache.layer_count == self.expected_layer_count
            and self.prefill_cache.sequence_length == self.input_token_count
            and self.decode_cache.sequence_length == self.input_token_count + 1
            and self.cache_sequence_growth == 1
            and self.selection.selected_token_was_valid
            and self.selection.grammar_matched_after_selection
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def inspect_kv_cache(
    cache: Any,
    *,
    phase: str,
    expected_layer_count: int,
    expected_batch_size: int,
    expected_sequence_length: int,
    expected_device: str,
    expected_dtype: Any,
    torch: Any,
) -> KvCacheSnapshot:
    """Validate and snapshot a Transformers-compatible key/value cache."""
    if cache is None:
        raise ValueError(f"{phase} cache is missing")

    try:
        layer_pairs = list(cache)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{phase} cache must expose iterable key/value layers") from exc

    if len(layer_pairs) != expected_layer_count:
        raise ValueError(
            f"{phase} cache has {len(layer_pairs)} layers; expected {expected_layer_count}"
        )

    cache_sequence_length = getattr(cache, "get_seq_length", None)
    if not callable(cache_sequence_length):
        raise TypeError(f"{phase} cache must expose get_seq_length()")
    observed_cache_length = int(cache_sequence_length())
    if observed_cache_length != expected_sequence_length:
        raise ValueError(
            f"{phase} cache sequence length {observed_cache_length}; "
            f"expected {expected_sequence_length}"
        )

    layer_snapshots = []
    total_storage_bytes = 0
    for layer_index, pair in enumerate(layer_pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"{phase} cache layer {layer_index} must contain key/value tensors")
        keys, values = pair
        if not torch.is_tensor(keys) or not torch.is_tensor(values):
            raise TypeError(f"{phase} cache layer {layer_index} entries must be tensors")
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                f"{phase} cache layer {layer_index} must use "
                "[batch, heads, sequence, head_dim] tensors"
            )

        key_shape = tuple(int(size) for size in keys.shape)
        value_shape = tuple(int(size) for size in values.shape)
        if key_shape != value_shape:
            raise ValueError(f"{phase} cache layer {layer_index} key/value shapes differ")
        if key_shape[0] != expected_batch_size:
            raise ValueError(
                f"{phase} cache layer {layer_index} batch size {key_shape[0]}; "
                f"expected {expected_batch_size}"
            )
        if key_shape[-2] != expected_sequence_length:
            raise ValueError(
                f"{phase} cache layer {layer_index} sequence length {key_shape[-2]}; "
                f"expected {expected_sequence_length}"
            )
        if not keys.is_cuda or not values.is_cuda:
            raise ValueError(f"{phase} cache layer {layer_index} must remain on CUDA")
        if str(keys.device) != expected_device or str(values.device) != expected_device:
            raise ValueError(
                f"{phase} cache layer {layer_index} is not on expected device {expected_device}"
            )
        if keys.dtype != expected_dtype or values.dtype != expected_dtype:
            raise ValueError(f"{phase} cache layer {layer_index} must use dtype {expected_dtype}")

        storage_bytes = int(keys.numel()) * int(keys.element_size()) + int(values.numel()) * int(
            values.element_size()
        )
        total_storage_bytes += storage_bytes
        layer_snapshots.append(
            KvCacheLayerSnapshot(
                layer_index=layer_index,
                key_shape=key_shape,
                value_shape=value_shape,
                dtype=str(keys.dtype),
                device=str(keys.device),
                sequence_length=key_shape[-2],
                storage_bytes=storage_bytes,
            )
        )

    return KvCacheSnapshot(
        phase=phase,
        cache_type=type(cache).__name__,
        layer_count=len(layer_snapshots),
        sequence_length=observed_cache_length,
        storage_bytes=total_storage_bytes,
        layers=tuple(layer_snapshots),
    )


def run_kv_cache_probe(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    prompt: str = DEFAULT_PROMPT,
    regex: str = DEFAULT_REGEX,
    local_files_only: bool = False,
    device_index: int = 0,
) -> KvCacheProbeReport:
    """Run one cached decode step with the fixed, tokenizer-validated model."""
    if model_id != DEFAULT_MODEL_ID:
        raise ValueError(f"KV-cache probe only supports the validated model {DEFAULT_MODEL_ID!r}")
    if not prompt:
        raise ValueError("prompt cannot be empty")
    if not regex:
        raise ValueError("regex cannot be empty for the KV-cache probe")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")

    torch, model_loader, quanto_config, grammar_factory = _require_runtime()
    device = torch.device("cuda", device_index)
    device_name = str(device)
    total_start = time.perf_counter()
    snapshots: List[MemorySnapshot] = [_memory_snapshot("baseline", torch, device)]
    baseline_allocated = snapshots[0].cuda_allocated_bytes
    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats()

    config = None
    tokenizer = None
    vocabulary = None
    model = None
    encoded = None
    prefill_outputs = None
    prefill_logits = None
    prefill_last_logits = None
    prefill_cache = None
    decode_input_ids = None
    attention_mask = None
    decode_attention_mask = None
    decode_kwargs = None
    decode_outputs = None
    decode_logits = None
    decode_cache = None
    selection = None
    failure: Optional[BaseException] = None
    cleanup_failures: List[Tuple[str, BaseException]] = []
    resolved_revision = None
    input_token_count = 0
    expected_layer_count = 0
    expected_width = 0
    prefill_logits_shape: Tuple[int, ...] = ()
    prefill_logits_dtype = ""
    prefill_logits_device = ""
    decode_logits_shape: Tuple[int, ...] = ()
    decode_logits_dtype = ""
    decode_logits_device = ""
    prefill_cache_snapshot = None
    decode_cache_snapshot = None
    cache_object_reused = False
    metadata_load_s = 0.0
    vocabulary_validation_s = 0.0
    model_load_s = 0.0
    tokenization_s = 0.0
    prefill_s = 0.0
    selection_s = 0.0
    decode_s = 0.0
    cleanup_s = 0.0
    peak_cuda_allocated_bytes = 0

    try:
        stage_start = time.perf_counter()
        config, tokenizer, resolved_revision = _load_tokenizer_metadata(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        metadata_load_s = time.perf_counter() - stage_start
        expected_width = int(config.vocab_size)
        expected_layer_count = int(config.num_hidden_layers)
        snapshots.append(_memory_snapshot("metadata_loaded", torch, device))

        stage_start = time.perf_counter()
        tokenizer_report, vocabulary = inspect_loaded_tokenizer(
            tokenizer,
            config,
            model_id=model_id,
            grammar_factory=grammar_factory,
            requested_revision=revision,
            resolved_revision=resolved_revision,
        )
        vocabulary_validation_s = time.perf_counter() - stage_start
        if not tokenizer_report.compatible:
            details = "; ".join(tokenizer_report.errors[:3]) or "unknown incompatibility"
            raise ValueError(f"tokenizer compatibility probe failed: {details}")

        model_revision = resolved_revision or revision
        stage_start = time.perf_counter()
        model = model_loader.from_pretrained(
            model_id,
            revision=model_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            device_map=device_name,
            dtype=torch.float16,
            quantization_config=quanto_config(weights="int4"),
        ).eval()
        model_load_s = time.perf_counter() - stage_start
        snapshots.append(_memory_snapshot("model_loaded", torch, device))

        stage_start = time.perf_counter()
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        input_token_count = int(encoded["input_ids"].shape[-1])
        if input_token_count < 1:
            raise ValueError("tokenized prompt cannot be empty")
        tokenization_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        with torch.no_grad():
            prefill_outputs = model(
                **encoded,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        prefill_s = time.perf_counter() - stage_start

        prefill_logits = prefill_outputs.logits
        prefill_logits_shape = validate_real_logits(prefill_logits, expected_width, torch)
        prefill_logits_dtype = str(prefill_logits.dtype)
        prefill_logits_device = str(prefill_logits.device)
        prefill_cache = prefill_outputs.past_key_values
        prefill_cache_snapshot = inspect_kv_cache(
            prefill_cache,
            phase="prefill",
            expected_layer_count=expected_layer_count,
            expected_batch_size=1,
            expected_sequence_length=input_token_count,
            expected_device=device_name,
            expected_dtype=torch.float16,
            torch=torch,
        )
        snapshots.append(_memory_snapshot("prefill_complete", torch, device))

        prefill_last_logits = prefill_logits[:, -1, :].contiguous()
        stage_start = time.perf_counter()
        selection = select_and_advance_one_token(
            prefill_last_logits,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            regex=regex,
            grammar_factory=grammar_factory,
        )
        torch.cuda.synchronize(device)
        selection_s = time.perf_counter() - stage_start

        decode_input_ids = torch.tensor(
            [[selection.selected_token_id]],
            dtype=encoded["input_ids"].dtype,
            device=device,
        )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            decode_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )

        prefill_last_logits = None
        prefill_logits = None
        prefill_outputs = None
        vocabulary = None
        tokenizer = None
        config = None

        decode_kwargs = {
            "input_ids": decode_input_ids,
            "past_key_values": prefill_cache,
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
        if decode_attention_mask is not None:
            decode_kwargs["attention_mask"] = decode_attention_mask

        stage_start = time.perf_counter()
        with torch.no_grad():
            decode_outputs = model(**decode_kwargs)
        torch.cuda.synchronize(device)
        decode_s = time.perf_counter() - stage_start

        decode_logits = decode_outputs.logits
        decode_logits_shape = validate_real_logits(decode_logits, expected_width, torch)
        decode_logits_dtype = str(decode_logits.dtype)
        decode_logits_device = str(decode_logits.device)
        decode_cache = decode_outputs.past_key_values
        cache_object_reused = decode_cache is prefill_cache
        decode_cache_snapshot = inspect_kv_cache(
            decode_cache,
            phase="decode",
            expected_layer_count=expected_layer_count,
            expected_batch_size=1,
            expected_sequence_length=input_token_count + 1,
            expected_device=device_name,
            expected_dtype=torch.float16,
            torch=torch,
        )
        snapshots.append(_memory_snapshot("decode_complete", torch, device))
        peak_cuda_allocated_bytes = max(
            0, int(torch.cuda.max_memory_allocated(device)) - baseline_allocated
        )
    except BaseException as exc:
        _detach_exception_tracebacks(exc)
        failure = exc
    finally:
        cleanup_start = time.perf_counter()
        decode_cache = None
        decode_logits = None
        decode_outputs = None
        decode_kwargs = None
        decode_attention_mask = None
        attention_mask = None
        decode_input_ids = None
        prefill_cache = None
        prefill_last_logits = None
        prefill_logits = None
        prefill_outputs = None
        encoded = None
        model = None
        vocabulary = None
        tokenizer = None
        config = None
        _run_cleanup_step(cleanup_failures, "gc.collect", gc.collect)
        _run_cleanup_step(cleanup_failures, "torch.cuda.empty_cache", torch.cuda.empty_cache)
        _run_cleanup_step(
            cleanup_failures,
            "torch.cuda.synchronize",
            lambda: torch.cuda.synchronize(device),
        )
        cleanup_s = time.perf_counter() - cleanup_start
        _run_cleanup_step(
            cleanup_failures,
            "after_cleanup memory snapshot",
            lambda: snapshots.append(_memory_snapshot("after_cleanup", torch, device)),
        )

    if failure is not None:
        if cleanup_failures:
            _annotate_cleanup_failures(failure, cleanup_failures)
        raise failure
    if cleanup_failures:
        cleanup_failure = cleanup_failures[0][1]
        _annotate_cleanup_failures(cleanup_failure, cleanup_failures)
        raise cleanup_failure
    if prefill_cache_snapshot is None or decode_cache_snapshot is None or selection is None:
        raise RuntimeError("KV-cache probe completed without all required results")

    total_s = time.perf_counter() - total_start
    cache_sequence_growth = (
        decode_cache_snapshot.sequence_length - prefill_cache_snapshot.sequence_length
    )
    return KvCacheProbeReport(
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        quantization=QUANTIZATION_NAME,
        prompt=prompt,
        regex=regex,
        input_token_count=input_token_count,
        expected_layer_count=expected_layer_count,
        expected_logits_width=expected_width,
        prefill_logits_shape=prefill_logits_shape,
        prefill_logits_dtype=prefill_logits_dtype,
        prefill_logits_device=prefill_logits_device,
        decode_logits_shape=decode_logits_shape,
        decode_logits_dtype=decode_logits_dtype,
        decode_logits_device=decode_logits_device,
        cache_object_reused=cache_object_reused,
        cache_sequence_growth=cache_sequence_growth,
        prefill_cache=prefill_cache_snapshot,
        decode_cache=decode_cache_snapshot,
        selection=selection,
        peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
        timings=KvCacheProbeTimings(
            metadata_load_s=metadata_load_s,
            vocabulary_validation_s=vocabulary_validation_s,
            model_load_s=model_load_s,
            tokenization_s=tokenization_s,
            prefill_s=prefill_s,
            selection_s=selection_s,
            decode_s=decode_s,
            cleanup_s=cleanup_s,
            total_s=total_s,
        ),
        memory_snapshots=tuple(snapshots),
    )


def format_kv_cache_report(report: KvCacheProbeReport) -> str:
    """Format a compact console report for the bounded cache milestone."""
    status = "PASS" if report.passed else "INCOMPLETE"
    return "\n".join(
        [
            f"Real-model CUDA KV-cache probe: {status}",
            f"  Model: {report.model_id}",
            (
                "  Revision: "
                f"{report.resolved_revision or report.requested_revision or 'unavailable'}"
            ),
            f"  Quantization: {report.quantization}",
            f"  Prompt tokens: {report.input_token_count}",
            (
                f"  Prefill cache: {report.prefill_cache.cache_type}, "
                f"layers={report.prefill_cache.layer_count}, "
                f"sequence={report.prefill_cache.sequence_length}, "
                f"storage={report.prefill_cache.storage_bytes / (1024 * 1024):.2f} MiB"
            ),
            (
                f"  Decode cache: {report.decode_cache.cache_type}, "
                f"layers={report.decode_cache.layer_count}, "
                f"sequence={report.decode_cache.sequence_length}, "
                f"storage={report.decode_cache.storage_bytes / (1024 * 1024):.2f} MiB"
            ),
            f"  Cache object reused in place: {report.cache_object_reused}",
            f"  Cache sequence growth: {report.cache_sequence_growth}",
            (
                f"  Constrained decode input: id={report.selection.selected_token_id}, "
                f"text={report.selection.selected_token_text!r}"
            ),
            f"  Prefill time: {report.timings.prefill_s * 1000:.2f} ms",
            f"  Cached decode time: {report.timings.decode_s * 1000:.2f} ms",
            (
                "  Peak CUDA allocation increase: "
                f"{report.peak_cuda_allocated_bytes / (1024 * 1024):.1f} MiB"
            ),
            f"  Cleanup time: {report.timings.cleanup_s * 1000:.2f} ms",
        ]
    )


__all__ = [
    "KvCacheLayerSnapshot",
    "KvCacheProbeReport",
    "KvCacheProbeTimings",
    "KvCacheSnapshot",
    "format_kv_cache_report",
    "inspect_kv_cache",
    "run_kv_cache_probe",
]
