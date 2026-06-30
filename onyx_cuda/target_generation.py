"""Bounded target-only CUDA generation for the Windows prototype."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .grammar_handoff import CudaValidIdCache
from .kv_cache_probe import KvCacheSnapshot, inspect_kv_cache
from .masked_argmax import masked_argmax_tensor
from .real_logits_handoff import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    QUANTIZATION_NAME,
    MemorySnapshot,
    _annotate_cleanup_failures,
    _detach_exception_tracebacks,
    _memory_snapshot,
    _require_runtime,
    _run_cleanup_step,
    validate_real_logits,
)
from .tokenizer_probe import _load_tokenizer_metadata, inspect_loaded_tokenizer

DEFAULT_PROMPT = "The next four characters must be digits:"
DEFAULT_REGEX = r"[0-9]{4}"
DEFAULT_MAX_NEW_TOKENS = 4


@dataclass(frozen=True)
class GeneratedToken:
    """One grammar-constrained token selected from real model logits."""

    index: int
    token_id: int
    text: str
    bytes_hex: str
    valid_token_count: int
    grammar_matched_after_token: bool


@dataclass(frozen=True)
class CudaTargetGenerationTimings:
    """Host-observed timings for bounded target-only generation."""

    metadata_load_s: float
    vocabulary_validation_s: float
    model_load_s: float
    tokenization_s: float
    prefill_s: float
    valid_id_lookup_s: float
    selection_s: float
    result_sync_s: float
    grammar_advance_s: float
    detokenization_s: float
    cached_decode_s: float
    cleanup_s: float
    total_s: float


@dataclass(frozen=True)
class CudaTargetGenerationReport:
    """Machine-readable result for bounded target-only constrained generation."""

    model_id: str
    requested_revision: Optional[str]
    resolved_revision: Optional[str]
    quantization: str
    prompt: str
    regex: str
    max_new_tokens: int
    input_token_count: int
    generated_token_ids: Tuple[int, ...]
    output_text: str
    finish_reason: str
    grammar_matched: bool
    expected_layer_count: int
    expected_logits_width: int
    observed_logits_width: int
    prefill_logits_shape: Tuple[int, ...]
    final_logits_shape: Tuple[int, ...]
    final_logits_dtype: str
    final_logits_device: str
    cached_decode_steps: int
    cache_object_reused: bool
    prefill_cache: KvCacheSnapshot
    final_cache: KvCacheSnapshot
    tokens: Tuple[GeneratedToken, ...]
    peak_cuda_allocated_bytes: int
    timings: CudaTargetGenerationTimings
    memory_snapshots: Tuple[MemorySnapshot, ...]

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def passed(self) -> bool:
        """Whether the default correctness boundary completed as intended."""
        return (
            self.final_logits_device.startswith("cuda")
            and self.observed_logits_width == self.expected_logits_width
            and self.finish_reason == "grammar_complete"
            and self.grammar_matched
            and self.generated_tokens <= self.max_new_tokens
            and self.final_cache.sequence_length
            == self.input_token_count + self.cached_decode_steps
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["generated_tokens"] = self.generated_tokens
        result["passed"] = self.passed
        return result


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _release_grammar_states(grammar: Any, states: Sequence[int]) -> None:
    live_states = [int(state) for state in states]
    if not live_states:
        return
    release_many = getattr(grammar, "release_states", None)
    if callable(release_many):
        release_many(live_states)
        return
    release_one = getattr(grammar, "release_state")
    for state in live_states:
        release_one(state)


def _tensor_length(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel())
    return len(value)


def _normalize_stop_strings(stop_strings: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not stop_strings:
        return ()
    return tuple(item for item in stop_strings if item)


def _truncate_at_stop(text: str, stop_strings: Sequence[str]) -> Tuple[str, bool]:
    if not stop_strings:
        return text, False
    positions = [text.find(item) for item in stop_strings if item]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


def _resolve_eos_token_ids(tokenizer: Any) -> Tuple[int, ...]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and not isinstance(eos, bool):
        return (int(eos),)
    if isinstance(eos, (list, tuple)):
        return tuple(
            int(item) for item in eos if isinstance(item, int) and not isinstance(item, bool)
        )
    return ()


def run_target_only_generation(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    prompt: str = DEFAULT_PROMPT,
    regex: str = DEFAULT_REGEX,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    stop_strings: Optional[Sequence[str]] = None,
    local_files_only: bool = False,
    device_index: int = 0,
) -> CudaTargetGenerationReport:
    """Run bounded regex-constrained generation through the fixed CUDA target model."""
    if model_id != DEFAULT_MODEL_ID:
        raise ValueError(
            f"target-only CUDA generation only supports the validated model {DEFAULT_MODEL_ID!r}"
        )
    if not prompt:
        raise ValueError("prompt cannot be empty")
    if not regex:
        raise ValueError("regex cannot be empty for target-only CUDA generation")
    _validate_positive_integer("max_new_tokens", max_new_tokens)
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")

    stop_strings = _normalize_stop_strings(stop_strings)
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
    attention_mask = None
    decode_attention_mask = None
    decode_input_ids = None
    prefill_outputs = None
    decode_outputs = None
    logits = None
    last_logits = None
    current_cache = None
    next_cache = None
    decode_kwargs = None
    grammar = None
    valid_id_cache = None
    live_grammar_states: List[int] = []
    failure: Optional[BaseException] = None
    cleanup_failures: List[Tuple[str, BaseException]] = []

    resolved_revision = None
    input_token_count = 0
    expected_layer_count = 0
    expected_width = 0
    observed_width = 0
    prefill_logits_shape: Tuple[int, ...] = ()
    final_logits_shape: Tuple[int, ...] = ()
    final_logits_dtype = ""
    final_logits_device = ""
    prefill_cache_snapshot = None
    final_cache_snapshot = None
    generated_token_ids: List[int] = []
    generated_tokens: List[GeneratedToken] = []
    output_text = ""
    finish_reason = "length"
    grammar_matched = False
    eos_token_ids: Tuple[int, ...] = ()
    cached_decode_steps = 0
    cache_object_reused = True
    peak_cuda_allocated_bytes = 0

    metadata_load_s = 0.0
    vocabulary_validation_s = 0.0
    model_load_s = 0.0
    tokenization_s = 0.0
    prefill_s = 0.0
    valid_id_lookup_s = 0.0
    selection_s = 0.0
    result_sync_s = 0.0
    grammar_advance_s = 0.0
    detokenization_s = 0.0
    cached_decode_s = 0.0
    cleanup_s = 0.0

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
        attention_mask = encoded.get("attention_mask")
        eos_token_ids = _resolve_eos_token_ids(tokenizer)
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

        logits = prefill_outputs.logits
        prefill_logits_shape = validate_real_logits(logits, expected_width, torch)
        final_logits_shape = prefill_logits_shape
        final_logits_dtype = str(logits.dtype)
        final_logits_device = str(logits.device)
        observed_width = int(logits.shape[-1])
        last_logits = logits[:, -1, :].contiguous()
        current_cache = prefill_outputs.past_key_values
        prefill_cache_snapshot = inspect_kv_cache(
            current_cache,
            phase="prefill",
            expected_layer_count=expected_layer_count,
            expected_batch_size=1,
            expected_sequence_length=input_token_count,
            expected_device=device_name,
            expected_dtype=torch.float16,
            torch=torch,
        )
        snapshots.append(_memory_snapshot("prefill_complete", torch, device))

        grammar = grammar_factory(list(vocabulary))
        grammar.compile_regex(regex)
        grammar_state = int(grammar.init_state())
        live_grammar_states.append(grammar_state)
        valid_id_cache = CudaValidIdCache(grammar)
        grammar_matched = bool(grammar.is_match_state(grammar_state))
        if grammar_matched:
            finish_reason = "grammar_complete"

        while not grammar_matched and len(generated_token_ids) < max_new_tokens:
            if getattr(last_logits, "ndim", None) == 2 and last_logits.shape[0] != 1:
                raise ValueError("target-only CUDA generation requires one logits row")

            stage_start = time.perf_counter()
            valid_ids = valid_id_cache.get(grammar_state, device)
            valid_id_lookup_s += time.perf_counter() - stage_start
            valid_token_count = _tensor_length(valid_ids)

            stage_start = time.perf_counter()
            selected = masked_argmax_tensor(last_logits, valid_ids, check_inputs=True)
            selection_s += time.perf_counter() - stage_start
            if selected.numel() != 1:
                raise ValueError("target-only CUDA generation requires one selected token")

            stage_start = time.perf_counter()
            token_id = int(selected.item())
            result_sync_s += time.perf_counter() - stage_start
            valid_id_cache.discard(grammar_state)

            stage_start = time.perf_counter()
            next_state = int(grammar.advance_state(grammar_state, token_id))
            live_grammar_states.append(next_state)
            _release_grammar_states(grammar, [grammar_state])
            live_grammar_states.remove(grammar_state)
            grammar_state = next_state
            grammar_matched = bool(grammar.is_match_state(grammar_state))
            grammar_advance_s += time.perf_counter() - stage_start

            generated_token_ids.append(token_id)
            stage_start = time.perf_counter()
            next_output_text = tokenizer.decode(generated_token_ids)
            token_text = (
                next_output_text[len(output_text) :]
                if next_output_text.startswith(output_text)
                else tokenizer.decode([token_id])
            )
            output_text = next_output_text
            output_text, stopped = _truncate_at_stop(output_text, stop_strings)
            detokenization_s += time.perf_counter() - stage_start

            generated_tokens.append(
                GeneratedToken(
                    index=len(generated_token_ids) - 1,
                    token_id=token_id,
                    text=token_text,
                    bytes_hex=bytes(vocabulary[token_id]).hex(),
                    valid_token_count=valid_token_count,
                    grammar_matched_after_token=grammar_matched,
                )
            )

            if grammar_matched:
                finish_reason = "grammar_complete"
                break
            if token_id in eos_token_ids:
                finish_reason = "eos_token"
                break
            if stopped:
                finish_reason = "stop"
                break
            if len(generated_token_ids) >= max_new_tokens:
                finish_reason = "length"
                break

            decode_input_ids = torch.tensor(
                [[token_id]],
                dtype=encoded["input_ids"].dtype,
                device=device,
            )
            decode_attention_mask = None
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

            decode_kwargs = {
                "input_ids": decode_input_ids,
                "past_key_values": current_cache,
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
            if decode_attention_mask is not None:
                decode_kwargs["attention_mask"] = decode_attention_mask

            logits = None
            last_logits = None
            prefill_outputs = None
            decode_outputs = None

            stage_start = time.perf_counter()
            with torch.no_grad():
                decode_outputs = model(**decode_kwargs)
            torch.cuda.synchronize(device)
            cached_decode_s += time.perf_counter() - stage_start
            cached_decode_steps += 1

            logits = decode_outputs.logits
            final_logits_shape = validate_real_logits(logits, expected_width, torch)
            final_logits_dtype = str(logits.dtype)
            final_logits_device = str(logits.device)
            observed_width = int(logits.shape[-1])
            last_logits = logits[:, -1, :].contiguous()
            next_cache = decode_outputs.past_key_values
            cache_object_reused = cache_object_reused and next_cache is current_cache
            current_cache = next_cache
            next_cache = None
            attention_mask = decode_attention_mask
            snapshots.append(
                _memory_snapshot(f"decode_{cached_decode_steps}_complete", torch, device)
            )

        if finish_reason == "length":
            grammar_matched = bool(grammar.is_match_state(live_grammar_states[-1]))

        final_cache_snapshot = inspect_kv_cache(
            current_cache,
            phase="final",
            expected_layer_count=expected_layer_count,
            expected_batch_size=1,
            expected_sequence_length=input_token_count + cached_decode_steps,
            expected_device=device_name,
            expected_dtype=torch.float16,
            torch=torch,
        )
        peak_cuda_allocated_bytes = max(
            0, int(torch.cuda.max_memory_allocated(device)) - baseline_allocated
        )
    except BaseException as exc:
        _detach_exception_tracebacks(exc)
        failure = exc
    finally:
        cleanup_start = time.perf_counter()
        if valid_id_cache is not None:
            _run_cleanup_step(cleanup_failures, "valid_id_cache.clear", valid_id_cache.clear)
        if grammar is not None and live_grammar_states:
            states_to_release = tuple(live_grammar_states)
            _run_cleanup_step(
                cleanup_failures,
                "grammar.release_states",
                lambda: _release_grammar_states(grammar, states_to_release),
            )
            live_grammar_states.clear()
        valid_id_cache = None
        grammar = None
        decode_kwargs = None
        next_cache = None
        current_cache = None
        last_logits = None
        logits = None
        decode_outputs = None
        prefill_outputs = None
        decode_attention_mask = None
        attention_mask = None
        decode_input_ids = None
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
    if prefill_cache_snapshot is None or final_cache_snapshot is None:
        raise RuntimeError("target-only CUDA generation completed without cache snapshots")

    total_s = time.perf_counter() - total_start
    return CudaTargetGenerationReport(
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        quantization=QUANTIZATION_NAME,
        prompt=prompt,
        regex=regex,
        max_new_tokens=max_new_tokens,
        input_token_count=input_token_count,
        generated_token_ids=tuple(generated_token_ids),
        output_text=output_text,
        finish_reason=finish_reason,
        grammar_matched=grammar_matched,
        expected_layer_count=expected_layer_count,
        expected_logits_width=expected_width,
        observed_logits_width=observed_width,
        prefill_logits_shape=prefill_logits_shape,
        final_logits_shape=final_logits_shape,
        final_logits_dtype=final_logits_dtype,
        final_logits_device=final_logits_device,
        cached_decode_steps=cached_decode_steps,
        cache_object_reused=cache_object_reused,
        prefill_cache=prefill_cache_snapshot,
        final_cache=final_cache_snapshot,
        tokens=tuple(generated_tokens),
        peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
        timings=CudaTargetGenerationTimings(
            metadata_load_s=metadata_load_s,
            vocabulary_validation_s=vocabulary_validation_s,
            model_load_s=model_load_s,
            tokenization_s=tokenization_s,
            prefill_s=prefill_s,
            valid_id_lookup_s=valid_id_lookup_s,
            selection_s=selection_s,
            result_sync_s=result_sync_s,
            grammar_advance_s=grammar_advance_s,
            detokenization_s=detokenization_s,
            cached_decode_s=cached_decode_s,
            cleanup_s=cleanup_s,
            total_s=total_s,
        ),
        memory_snapshots=tuple(snapshots),
    )


def format_target_generation_report(report: CudaTargetGenerationReport) -> str:
    """Format a compact console report for the bounded generation milestone."""
    status = "PASS" if report.passed else "INCOMPLETE"
    return "\n".join(
        [
            f"Target-only CUDA constrained generation: {status}",
            f"  Model: {report.model_id}",
            f"  Revision: {report.resolved_revision or report.requested_revision or 'unavailable'}",
            f"  Quantization: {report.quantization}",
            f"  Prompt tokens: {report.input_token_count}",
            f"  Generated tokens: {report.generated_tokens}/{report.max_new_tokens}",
            f"  Finish reason: {report.finish_reason}",
            f"  Output: {report.output_text!r}",
            (
                f"  Final cache: {report.final_cache.cache_type}, "
                f"layers={report.final_cache.layer_count}, "
                f"sequence={report.final_cache.sequence_length}, "
                f"storage={report.final_cache.storage_bytes / (1024 * 1024):.2f} MiB"
            ),
            f"  Cached decode steps: {report.cached_decode_steps}",
            f"  Cache object reused in place: {report.cache_object_reused}",
            f"  Prefill time: {report.timings.prefill_s * 1000:.2f} ms",
            f"  Cached decode time: {report.timings.cached_decode_s * 1000:.2f} ms",
            (
                "  Selection time: "
                f"{(report.timings.valid_id_lookup_s + report.timings.selection_s + report.timings.result_sync_s) * 1000:.2f} ms"
            ),
            (
                "  Peak CUDA allocation increase: "
                f"{report.peak_cuda_allocated_bytes / (1024 * 1024):.1f} MiB"
            ),
            f"  Cleanup time: {report.timings.cleanup_s * 1000:.2f} ms",
        ]
    )


__all__ = [
    "CudaTargetGenerationReport",
    "CudaTargetGenerationTimings",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PROMPT",
    "DEFAULT_REGEX",
    "GeneratedToken",
    "format_target_generation_report",
    "run_target_only_generation",
]
