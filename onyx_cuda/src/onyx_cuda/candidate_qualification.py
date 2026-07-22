"""Offline qualification for the separately pinned Qwen2.5 3B target candidate."""

from __future__ import annotations

import gc
import importlib
import time
from dataclasses import dataclass
from typing import Any

from ._torch_install import (
    PYTORCH_CUDA_INSTALL_GUIDANCE,
    torch_cuda_unavailable_message,
)
from .model_profile import DEFAULT_TARGET_PROFILE, QWEN_3B_CANDIDATE_PROFILE
from .production_tokenizer import QwenTokenizerFingerprint, load_qwen_tokenizer
from .torch_backend import load_torch_cuda_target


MAX_CANDIDATE_CACHE_TOKENS = 2048


class CandidateQualificationError(RuntimeError):
    """Base error raised by the bounded 3B candidate qualification path."""


class CandidateQualificationImportError(CandidateQualificationError):
    """Raised when an optional qualification dependency cannot be imported."""


class CandidateQualificationUnavailableError(CandidateQualificationError):
    """Raised when the requested CUDA device cannot run qualification."""


class CandidateQualificationExecutionError(CandidateQualificationError):
    """Raised when tokenizer, model, or forward qualification fails."""


class CandidateQualificationCleanupError(CandidateQualificationError):
    """Raised when a lifecycle cannot clean up or accumulates CUDA state."""


@dataclass(frozen=True, slots=True)
class TokenizerCompatibilityReport:
    """Exact compatibility signals for the default and candidate tokenizers."""

    reference: QwenTokenizerFingerprint
    candidate: QwenTokenizerFingerprint
    reference_load_seconds: float
    candidate_load_seconds: float
    token_ids_equal: bool
    special_tokens_equal: bool
    chat_template_equal: bool

    @property
    def fully_compatible(self) -> bool:
        return (
            self.token_ids_equal
            and self.special_tokens_equal
            and self.chat_template_equal
        )


@dataclass(frozen=True, slots=True)
class CandidateMemorySnapshot:
    """CUDA allocator and Windows process-memory state at one lifecycle boundary."""

    free_memory_bytes: int
    total_memory_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    process_rss_bytes: int
    process_peak_working_set_bytes: int


@dataclass(frozen=True, slots=True)
class CandidateLifecycleMeasurement:
    """One complete load, forward, cache-ceiling, reset, and close lifecycle."""

    lifecycle_number: int
    load_seconds: float
    first_forward_seconds: float
    ceiling_prefill_seconds: float
    ceiling_decode_seconds: float
    close_seconds: float
    tokenizer_vocab_size: int
    model_vocab_size: int
    padded_vocab_rows: int
    model_memory_footprint_bytes: int
    first_forward_prompt_tokens: int
    first_forward_cache_length: int
    ceiling_prefill_tokens: int
    ceiling_prefill_cache_length: int
    ceiling_decode_cache_length: int
    logits_shape: tuple[int, ...]
    logits_dtype: str
    logits_device: str
    before_load: CandidateMemorySnapshot
    after_load: CandidateMemorySnapshot
    after_first_forward: CandidateMemorySnapshot
    at_cache_ceiling: CandidateMemorySnapshot
    after_close: CandidateMemorySnapshot


@dataclass(frozen=True, slots=True)
class CandidateTargetQualification:
    """Two-cycle evidence for the isolated 3B target candidate."""

    model_id: str
    revision: str
    reference_tokenizer_id: str
    quantization: str
    torch_version: str
    transformers_version: str
    bitsandbytes_version: str
    psutil_version: str
    device_index: int
    device_name: str
    total_cache_tokens: int
    tokenizer_compatibility: TokenizerCompatibilityReport
    lifecycles: tuple[CandidateLifecycleMeasurement, ...]

    @property
    def peak_vram_bytes(self) -> int:
        return max(
            max(snapshot.peak_allocated_bytes, snapshot.reserved_bytes)
            for lifecycle in self.lifecycles
            for snapshot in (
                lifecycle.after_load,
                lifecycle.after_first_forward,
                lifecycle.at_cache_ceiling,
            )
        )

    @property
    def peak_process_ram_bytes(self) -> int:
        return max(
            lifecycle.after_close.process_peak_working_set_bytes
            for lifecycle in self.lifecycles
        )


def qualify_qwen_3b_candidate(
    *,
    device_index: int = 0,
    local_files_only: bool = False,
    total_cache_tokens: int = 2048,
) -> CandidateTargetQualification:
    """Qualify the pinned 3B NF4 target twice without changing the production default."""

    _validate_inputs(device_index, local_files_only, total_cache_tokens)
    modules = {}
    for module_name in ("torch", "transformers", "bitsandbytes", "psutil"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            message = f"{module_name} could not be imported: {exc}"
            if module_name == "torch":
                message = f"{message}. {PYTORCH_CUDA_INSTALL_GUIDANCE}"
            raise CandidateQualificationImportError(
                message
            ) from exc

    return _qualify_candidate_modules(
        modules["torch"],
        modules["transformers"],
        modules["bitsandbytes"],
        modules["psutil"],
        device_index=device_index,
        local_files_only=local_files_only,
        total_cache_tokens=total_cache_tokens,
    )


def _qualify_candidate_modules(
    torch_module: Any,
    transformers_module: Any,
    bitsandbytes_module: Any,
    psutil_module: Any,
    *,
    device_index: int,
    local_files_only: bool,
    total_cache_tokens: int,
) -> CandidateTargetQualification:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise CandidateQualificationUnavailableError(torch_cuda_unavailable_message(torch_module))
    device_count = cuda.device_count()
    if device_index >= device_count:
        raise CandidateQualificationUnavailableError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )

    device = torch_module.device(f"cuda:{device_index}")
    properties = cuda.get_device_properties(device)
    compatibility = _compare_tokenizers(local_files_only=local_files_only)

    lifecycles = tuple(
        _run_lifecycle(
            torch_module,
            psutil_module,
            device=device,
            lifecycle_number=lifecycle_number,
            device_index=device_index,
            local_files_only=local_files_only,
            total_cache_tokens=total_cache_tokens,
        )
        for lifecycle_number in (1, 2)
    )
    first_cleanup = lifecycles[0].after_close
    second_cleanup = lifecycles[1].after_close
    if second_cleanup.allocated_bytes > first_cleanup.allocated_bytes:
        raise CandidateQualificationCleanupError(
            "the second candidate lifecycle increased post-close allocated CUDA memory: "
            f"first={first_cleanup.allocated_bytes}, second={second_cleanup.allocated_bytes}"
        )
    if second_cleanup.reserved_bytes > first_cleanup.reserved_bytes:
        raise CandidateQualificationCleanupError(
            "the second candidate lifecycle increased post-close reserved CUDA memory: "
            f"first={first_cleanup.reserved_bytes}, second={second_cleanup.reserved_bytes}"
        )

    return CandidateTargetQualification(
        model_id=QWEN_3B_CANDIDATE_PROFILE.model_id,
        revision=QWEN_3B_CANDIDATE_PROFILE.revision,
        reference_tokenizer_id=DEFAULT_TARGET_PROFILE.pinned_id,
        quantization="bitsandbytes-nf4-double-quant",
        torch_version=str(torch_module.__version__),
        transformers_version=str(transformers_module.__version__),
        bitsandbytes_version=str(bitsandbytes_module.__version__),
        psutil_version=str(psutil_module.__version__),
        device_index=device_index,
        device_name=str(properties.name),
        total_cache_tokens=total_cache_tokens,
        tokenizer_compatibility=compatibility,
        lifecycles=lifecycles,
    )


def _compare_tokenizers(*, local_files_only: bool) -> TokenizerCompatibilityReport:
    reference_load = None
    candidate_load = None
    try:
        reference_load = load_qwen_tokenizer(
            DEFAULT_TARGET_PROFILE,
            local_files_only=local_files_only,
        )
        candidate_load = load_qwen_tokenizer(
            QWEN_3B_CANDIDATE_PROFILE,
            local_files_only=local_files_only,
        )
        reference = reference_load.tokenizer.compatibility_fingerprint()
        candidate = candidate_load.tokenizer.compatibility_fingerprint()
        return TokenizerCompatibilityReport(
            reference=reference,
            candidate=candidate,
            reference_load_seconds=reference_load.load_seconds,
            candidate_load_seconds=candidate_load.load_seconds,
            token_ids_equal=(
                reference.vocab_size == candidate.vocab_size
                and reference.base_vocab_size == candidate.base_vocab_size
                and reference.vocabulary_sha256 == candidate.vocabulary_sha256
            ),
            special_tokens_equal=(
                reference.eos_token_id == candidate.eos_token_id
                and reference.pad_token_id == candidate.pad_token_id
                and reference.special_tokens_sha256 == candidate.special_tokens_sha256
            ),
            chat_template_equal=(
                reference.chat_template_sha256 == candidate.chat_template_sha256
            ),
        )
    except CandidateQualificationError:
        raise
    except Exception as exc:
        raise CandidateQualificationExecutionError(
            f"candidate tokenizer compatibility qualification failed: {exc}"
        ) from exc
    finally:
        reference_load = None
        candidate_load = None
        gc.collect()


def _run_lifecycle(
    torch_module: Any,
    psutil_module: Any,
    *,
    device: Any,
    lifecycle_number: int,
    device_index: int,
    local_files_only: bool,
    total_cache_tokens: int,
) -> CandidateLifecycleMeasurement:
    cuda = torch_module.cuda
    process = psutil_module.Process()
    backend = None
    closed = False
    failure = None
    result_values = None
    before_load = _memory_snapshot(cuda, process, device)

    try:
        cuda.reset_peak_memory_stats(device)
        load_start = time.perf_counter()
        backend = load_torch_cuda_target(
            QWEN_3B_CANDIDATE_PROFILE,
            device_index=device_index,
            local_files_only=local_files_only,
        )
        cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_start
        after_load = _memory_snapshot(cuda, process, device)

        if backend.model_id != QWEN_3B_CANDIDATE_PROFILE.pinned_id:
            raise CandidateQualificationExecutionError(
                f"loaded candidate backend reported unexpected model ID {backend.model_id}"
            )
        if backend.tokenizer.tokenizer_id != QWEN_3B_CANDIDATE_PROFILE.pinned_id:
            raise CandidateQualificationExecutionError(
                "loaded candidate backend reported an unexpected tokenizer identity"
            )

        first_prompt = backend.tokenizer.encode("Hello world")
        if not first_prompt:
            raise CandidateQualificationExecutionError(
                "candidate first-forward fixture encoded to no tokens"
            )
        cuda.reset_peak_memory_stats(device)
        first_start = time.perf_counter()
        first_step = backend.prefill(first_prompt)
        cuda.synchronize(device)
        first_forward_seconds = time.perf_counter() - first_start
        _validate_step(
            first_step,
            backend,
            len(first_prompt),
            backend.vocab_size,
            label="first forward",
        )
        logits_shape, logits_dtype, logits_device = _validate_logits(
            first_step.logits,
            backend.vocab_size,
            expected_device=device,
            label="first forward",
        )
        after_first_forward = _memory_snapshot(cuda, process, device)

        backend.reset()
        if backend.cache_length != 0:
            raise CandidateQualificationExecutionError(
                f"candidate reset retained cache length {backend.cache_length}"
            )

        ceiling_token_id = first_prompt[0]
        ceiling_prompt_tokens = total_cache_tokens - 1
        ceiling_prompt = (ceiling_token_id,) * ceiling_prompt_tokens
        cuda.reset_peak_memory_stats(device)
        prefill_start = time.perf_counter()
        ceiling_step = backend.prefill(ceiling_prompt)
        cuda.synchronize(device)
        ceiling_prefill_seconds = time.perf_counter() - prefill_start
        _validate_step(
            ceiling_step,
            backend,
            ceiling_prompt_tokens,
            backend.vocab_size,
            label="cache-ceiling prefill",
        )
        _validate_logits(
            ceiling_step.logits,
            backend.vocab_size,
            expected_device=device,
            label="cache-ceiling prefill",
        )

        decode_start = time.perf_counter()
        decode_step = backend.decode(ceiling_token_id)
        cuda.synchronize(device)
        ceiling_decode_seconds = time.perf_counter() - decode_start
        _validate_step(
            decode_step,
            backend,
            total_cache_tokens,
            backend.vocab_size,
            label="cache-ceiling decode",
        )
        _validate_logits(
            decode_step.logits,
            backend.vocab_size,
            expected_device=device,
            label="cache-ceiling decode",
        )
        at_cache_ceiling = _memory_snapshot(cuda, process, device)

        backend.reset()
        if backend.cache_length != 0:
            raise CandidateQualificationExecutionError(
                f"candidate final reset retained cache length {backend.cache_length}"
            )

        tokenizer_vocab_size = backend.vocab_size
        model_vocab_size = backend.model_vocab_size
        padded_vocab_rows = backend.padded_vocab_rows
        model_memory_footprint_bytes = backend.model_memory_footprint_bytes
        first_forward_cache_length = first_step.cache_length
        ceiling_prefill_cache_length = ceiling_step.cache_length
        ceiling_decode_cache_length = decode_step.cache_length
        first_step = None
        ceiling_step = None
        decode_step = None

        close_start = time.perf_counter()
        try:
            backend.close()
            closed = True
            backend = None
            gc.collect()
            cuda.empty_cache()
            cuda.synchronize(device)
        except Exception as exc:
            closed = bool(getattr(backend, "is_closed", False))
            raise CandidateQualificationCleanupError(
                f"candidate lifecycle {lifecycle_number} cleanup failed: {exc}"
            ) from exc
        close_seconds = time.perf_counter() - close_start
        after_close = _memory_snapshot(cuda, process, device)
        result_values = {
            "load_seconds": load_seconds,
            "first_forward_seconds": first_forward_seconds,
            "ceiling_prefill_seconds": ceiling_prefill_seconds,
            "ceiling_decode_seconds": ceiling_decode_seconds,
            "close_seconds": close_seconds,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "model_vocab_size": model_vocab_size,
            "padded_vocab_rows": padded_vocab_rows,
            "model_memory_footprint_bytes": model_memory_footprint_bytes,
            "first_forward_prompt_tokens": len(first_prompt),
            "first_forward_cache_length": first_forward_cache_length,
            "ceiling_prefill_tokens": ceiling_prompt_tokens,
            "ceiling_prefill_cache_length": ceiling_prefill_cache_length,
            "ceiling_decode_cache_length": ceiling_decode_cache_length,
            "logits_shape": logits_shape,
            "logits_dtype": logits_dtype,
            "logits_device": logits_device,
            "after_load": after_load,
            "after_first_forward": after_first_forward,
            "at_cache_ceiling": at_cache_ceiling,
            "after_close": after_close,
        }
    except CandidateQualificationError as exc:
        failure = exc
    except Exception as exc:
        failure = CandidateQualificationExecutionError(
            f"candidate lifecycle {lifecycle_number} failed: {exc}"
        )

    cleanup_failure = None
    if backend is not None and not closed:
        try:
            backend.close()
        except Exception as exc:
            cleanup_failure = CandidateQualificationCleanupError(
                f"candidate lifecycle {lifecycle_number} cleanup failed: {exc}"
            )
    backend = None
    gc.collect()

    if failure is not None:
        if cleanup_failure is not None:
            raise CandidateQualificationCleanupError(
                f"{failure}; cleanup also failed: {cleanup_failure}"
            ) from failure
        raise failure
    if cleanup_failure is not None:
        raise cleanup_failure
    if result_values is None:
        raise CandidateQualificationExecutionError(
            f"candidate lifecycle {lifecycle_number} produced no measurements"
        )

    return CandidateLifecycleMeasurement(
        lifecycle_number=lifecycle_number,
        before_load=before_load,
        **result_values,
    )


def _memory_snapshot(cuda: Any, process: Any, device: Any) -> CandidateMemorySnapshot:
    try:
        free_memory, total_memory = cuda.mem_get_info(device)
        process_memory = process.memory_info()
        rss = _nonnegative_int(process_memory.rss, label="process RSS")
        peak_working_set = _nonnegative_int(
            process_memory.peak_wset,
            label="process peak working set",
        )
        return CandidateMemorySnapshot(
            free_memory_bytes=_nonnegative_int(free_memory, label="free CUDA memory"),
            total_memory_bytes=_positive_int(total_memory, label="total CUDA memory"),
            allocated_bytes=_nonnegative_int(
                cuda.memory_allocated(device), label="allocated CUDA memory"
            ),
            reserved_bytes=_nonnegative_int(
                cuda.memory_reserved(device), label="reserved CUDA memory"
            ),
            peak_allocated_bytes=_nonnegative_int(
                cuda.max_memory_allocated(device), label="peak allocated CUDA memory"
            ),
            process_rss_bytes=rss,
            process_peak_working_set_bytes=peak_working_set,
        )
    except CandidateQualificationError:
        raise
    except Exception as exc:
        raise CandidateQualificationExecutionError(
            f"candidate memory measurement failed: {exc}"
        ) from exc


def _validate_step(
    step: Any,
    backend: Any,
    expected_length: int,
    vocab_size: int,
    *,
    label: str,
) -> None:
    if step.cache_length != expected_length:
        raise CandidateQualificationExecutionError(
            f"{label} produced cache length {step.cache_length}; expected {expected_length}"
        )
    if backend.cache_length != expected_length:
        raise CandidateQualificationExecutionError(
            f"{label} backend state reported cache length {backend.cache_length}; "
            f"expected {expected_length}"
        )
    _positive_int(vocab_size, label="candidate vocabulary size")


def _validate_logits(
    logits: Any,
    vocab_size: int,
    *,
    expected_device: Any,
    label: str,
) -> tuple[tuple[int, ...], str, str]:
    try:
        shape = tuple(logits.shape)
        dtype = str(logits.dtype)
        device = str(logits.device)
        is_cuda = bool(logits.is_cuda)
        contains_nan = bool(logits.isnan().any().item())
    except Exception as exc:
        raise CandidateQualificationExecutionError(
            f"{label} logits could not be validated: {exc}"
        ) from exc
    if shape != (vocab_size,):
        raise CandidateQualificationExecutionError(
            f"{label} logits shape {shape}; expected ({vocab_size},)"
        )
    if not is_cuda or device != str(expected_device):
        raise CandidateQualificationExecutionError(
            f"{label} logits are on {device}; expected {expected_device}"
        )
    if dtype not in {"float16", "torch.float16"}:
        raise CandidateQualificationExecutionError(
            f"{label} logits dtype is {dtype}; expected float16"
        )
    if contains_nan:
        raise CandidateQualificationExecutionError(f"{label} logits contain NaN")
    return shape, dtype, device


def _validate_inputs(
    device_index: int,
    local_files_only: bool,
    total_cache_tokens: int,
) -> None:
    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index cannot be negative")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean")
    if isinstance(total_cache_tokens, bool) or not isinstance(total_cache_tokens, int):
        raise TypeError("total_cache_tokens must be an integer")
    if total_cache_tokens < 2:
        raise ValueError("total_cache_tokens must be at least two")
    if total_cache_tokens > MAX_CANDIDATE_CACHE_TOKENS:
        raise ValueError(
            f"total_cache_tokens cannot exceed the candidate ceiling of "
            f"{MAX_CANDIDATE_CACHE_TOKENS}"
        )


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateQualificationExecutionError(f"{label} must be an integer")
    if value <= 0:
        raise CandidateQualificationExecutionError(f"{label} must be greater than zero")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateQualificationExecutionError(f"{label} must be an integer")
    if value < 0:
        raise CandidateQualificationExecutionError(f"{label} cannot be negative")
    return value
