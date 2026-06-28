"""One-step real-model CUDA logits handoff for the Windows prototype."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .masked_argmax import masked_argmax_tensor
from .tokenizer_probe import _load_tokenizer_metadata, inspect_loaded_tokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_PROMPT = "The next character must be a digit:"
DEFAULT_REGEX = r"[0-9]"
QUANTIZATION_NAME = "quanto-int4"


@dataclass(frozen=True)
class MemorySnapshot:
    """Host and CUDA memory observed at one probe phase."""

    phase: str
    rss_bytes: Optional[int]
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int


@dataclass(frozen=True)
class SelectionTimings:
    """Host-observed timings for the grammar-to-CUDA selection boundary."""

    valid_id_lookup_s: float
    selection_call_s: float
    result_sync_s: float
    grammar_advance_s: float


@dataclass(frozen=True)
class OneStepSelection:
    """Scalar result from selecting and advancing one grammar-valid token."""

    valid_token_count: int
    raw_argmax_token_id: int
    raw_argmax_text: str
    raw_argmax_logit: float
    raw_argmax_was_valid: bool
    selected_token_id: int
    selected_token_text: str
    selected_token_bytes_hex: str
    selected_token_logit: float
    selected_token_was_valid: bool
    invalid_raw_argmax_excluded: bool
    grammar_matched_after_selection: bool
    timings: SelectionTimings


@dataclass(frozen=True)
class RealLogitsHandoffTimings:
    """Timings for model setup, one forward pass, selection, and cleanup."""

    metadata_load_s: float
    vocabulary_validation_s: float
    model_load_s: float
    tokenization_s: float
    forward_s: float
    cleanup_s: float
    total_s: float


@dataclass(frozen=True)
class RealLogitsHandoffReport:
    """Complete machine-readable report for the one-step CUDA milestone."""

    model_id: str
    requested_revision: Optional[str]
    resolved_revision: Optional[str]
    quantization: str
    prompt: str
    regex: str
    input_token_count: int
    logits_shape: Tuple[int, ...]
    logits_dtype: str
    logits_device: str
    expected_logits_width: int
    observed_logits_width: int
    peak_cuda_allocated_bytes: int
    selection: OneStepSelection
    timings: RealLogitsHandoffTimings
    memory_snapshots: Tuple[MemorySnapshot, ...]

    @property
    def passed(self) -> bool:
        """Whether the probe demonstrated the complete intended boundary."""
        return (
            self.logits_device.startswith("cuda")
            and self.observed_logits_width == self.expected_logits_width
            and self.selection.selected_token_was_valid
            and self.selection.invalid_raw_argmax_excluded
            and self.selection.grammar_matched_after_selection
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def _require_runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, QuantoConfig
    except ImportError as exc:
        raise RuntimeError(
            "real-logits handoff requires the CUDA extra, including Transformers, "
            "Accelerate, and Optimum Quanto"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("real-logits handoff requires CUDA-enabled PyTorch")

    try:
        from onyx._rust import GrammarConstraint
    except ImportError as exc:
        raise RuntimeError(
            "real-logits handoff requires the Rust extension; run "
            "`python -m maturin develop --release`"
        ) from exc

    return torch, AutoModelForCausalLM, QuantoConfig, GrammarConstraint


def _process_rss_bytes() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _memory_snapshot(phase: str, torch, device) -> MemorySnapshot:
    return MemorySnapshot(
        phase=phase,
        rss_bytes=_process_rss_bytes(),
        cuda_allocated_bytes=int(torch.cuda.memory_allocated(device)),
        cuda_reserved_bytes=int(torch.cuda.memory_reserved(device)),
    )


def _detach_exception_tracebacks(error: BaseException) -> None:
    """Release frames retained by an exception, including chained failures."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        current.__traceback__ = None
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        try:
            nested = getattr(current, "exceptions", ())
        except BaseException:
            nested = ()
        if isinstance(nested, (list, tuple)):
            pending.extend(item for item in nested if isinstance(item, BaseException))


def _annotate_cleanup_failures(
    error: BaseException, failures: Sequence[Tuple[str, BaseException]]
) -> None:
    """Attach cleanup diagnostics without replacing the primary failure."""
    details_list = []
    for step, cleanup_error in failures:
        try:
            message = str(cleanup_error)
        except BaseException:
            message = "<message unavailable>"
        details_list.append(f"{step}: {type(cleanup_error).__name__}: {message}")
    details = tuple(details_list)
    try:
        setattr(error, "_onyx_cleanup_failures", details)
    except BaseException:
        pass
    try:
        add_note = getattr(error, "add_note", None)
    except BaseException:
        add_note = None
    if callable(add_note):
        for detail in details:
            try:
                add_note(f"Onyx cleanup also failed during {detail}")
            except BaseException:
                break


def _run_cleanup_step(
    failures: List[Tuple[str, BaseException]], step: str, action: Callable[[], Any]
) -> None:
    """Run one cleanup action while allowing later cleanup actions to proceed."""
    try:
        action()
    except BaseException as exc:
        _detach_exception_tracebacks(exc)
        failures.append((step, exc))


def validate_real_logits(logits: Any, expected_width: int, torch) -> Tuple[int, ...]:
    """Validate the tensor boundary before grammar-constrained selection."""
    if not torch.is_tensor(logits):
        raise TypeError("model logits must be a torch.Tensor")
    if not logits.is_cuda:
        raise ValueError("model logits must remain on CUDA")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
        raise ValueError("model logits must have shape [1, sequence, vocabulary]")
    if logits.shape[-1] != expected_width:
        raise ValueError(
            f"observed logits width {logits.shape[-1]} does not match "
            f"configured width {expected_width}"
        )
    if logits.dtype not in (torch.float16, torch.float32):
        raise ValueError("model logits must use float16 or float32")
    return tuple(int(size) for size in logits.shape)


def select_and_advance_one_token(
    logits: Any,
    *,
    vocabulary: Sequence[bytes],
    tokenizer: Any,
    regex: str,
    grammar_factory: Callable[[Sequence[bytes]], Any],
    selector: Callable[..., Any] = masked_argmax_tensor,
) -> OneStepSelection:
    """Select one constrained token and release every grammar state it creates."""
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("one-step handoff requires exactly one logits row")
        logits_row = logits[0]
    elif logits.ndim == 1:
        logits_row = logits
    else:
        raise ValueError("selection logits must have shape [vocabulary] or [1, vocabulary]")

    if logits_row.shape[0] != len(vocabulary):
        raise ValueError("logits width and grammar vocabulary length must match")

    grammar = grammar_factory(list(vocabulary))
    grammar.compile_regex(regex)
    initial_state = int(grammar.init_state())
    final_state: Optional[int] = None

    try:
        stage_start = time.perf_counter()
        valid_token_ids = tuple(
            int(token_id) for token_id in grammar.get_valid_token_ids(initial_state)
        )
        valid_id_lookup_s = time.perf_counter() - stage_start
        if not valid_token_ids:
            raise ValueError("grammar produced no valid token IDs for the initial state")

        raw_argmax_token_id = int(logits_row.argmax().item())
        raw_argmax_logit = float(logits_row[raw_argmax_token_id].item())
        raw_argmax_was_valid = raw_argmax_token_id in valid_token_ids

        stage_start = time.perf_counter()
        selected_tensor = selector(logits_row, valid_token_ids, check_inputs=True)
        selection_call_s = time.perf_counter() - stage_start

        if selected_tensor.numel() != 1:
            raise ValueError("one-step handoff requires exactly one selected token")

        stage_start = time.perf_counter()
        selected_token_id = int(selected_tensor.item())
        selected_token_logit = float(logits_row[selected_token_id].item())
        result_sync_s = time.perf_counter() - stage_start
        selected_token_was_valid = selected_token_id in valid_token_ids
        if not selected_token_was_valid:
            raise RuntimeError("CUDA selector returned a grammar-invalid token")

        stage_start = time.perf_counter()
        final_state = int(grammar.advance_state(initial_state, selected_token_id))
        grammar_matched = bool(grammar.is_match_state(final_state))
        grammar_advance_s = time.perf_counter() - stage_start

        return OneStepSelection(
            valid_token_count=len(valid_token_ids),
            raw_argmax_token_id=raw_argmax_token_id,
            raw_argmax_text=tokenizer.decode([raw_argmax_token_id]),
            raw_argmax_logit=raw_argmax_logit,
            raw_argmax_was_valid=raw_argmax_was_valid,
            selected_token_id=selected_token_id,
            selected_token_text=tokenizer.decode([selected_token_id]),
            selected_token_bytes_hex=bytes(vocabulary[selected_token_id]).hex(),
            selected_token_logit=selected_token_logit,
            selected_token_was_valid=selected_token_was_valid,
            invalid_raw_argmax_excluded=(
                not raw_argmax_was_valid and selected_token_id != raw_argmax_token_id
            ),
            grammar_matched_after_selection=grammar_matched,
            timings=SelectionTimings(
                valid_id_lookup_s=valid_id_lookup_s,
                selection_call_s=selection_call_s,
                result_sync_s=result_sync_s,
                grammar_advance_s=grammar_advance_s,
            ),
        )
    finally:
        states: List[int] = [initial_state]
        if final_state is not None and final_state != initial_state:
            states.append(final_state)
        grammar.release_states(states)


def run_real_logits_handoff(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    prompt: str = DEFAULT_PROMPT,
    regex: str = DEFAULT_REGEX,
    local_files_only: bool = False,
    device_index: int = 0,
) -> RealLogitsHandoffReport:
    """Run the fixed Qwen correctness probe through constrained CUDA selection."""
    if model_id != DEFAULT_MODEL_ID:
        raise ValueError(
            f"real-logits handoff only supports the validated model {DEFAULT_MODEL_ID!r}"
        )
    if not prompt:
        raise ValueError("prompt cannot be empty")
    if not regex:
        raise ValueError("regex cannot be empty for the one-step handoff")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")

    torch, model_loader, quanto_config, grammar_factory = _require_runtime()
    device = torch.device("cuda", device_index)
    total_start = time.perf_counter()
    snapshots: List[MemorySnapshot] = [_memory_snapshot("baseline", torch, device)]
    baseline_allocated = snapshots[0].cuda_allocated_bytes
    # PyTorch 2.11 on Windows rejects an explicit device argument for this one
    # memory API even though the related query APIs accept it.
    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats()

    config = None
    tokenizer = None
    vocabulary = None
    model = None
    encoded = None
    outputs = None
    logits = None
    last_logits = None
    selection = None
    failure: Optional[BaseException] = None
    cleanup_failures: List[Tuple[str, BaseException]] = []
    resolved_revision = None
    input_token_count = 0
    logits_shape: Tuple[int, ...] = ()
    logits_dtype = ""
    logits_device = ""
    expected_width = 0
    observed_width = 0
    metadata_load_s = 0.0
    vocabulary_validation_s = 0.0
    model_load_s = 0.0
    tokenization_s = 0.0
    forward_s = 0.0
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
            device_map=str(device),
            dtype=torch.float16,
            quantization_config=quanto_config(weights="int4"),
        ).eval()
        model_load_s = time.perf_counter() - stage_start
        snapshots.append(_memory_snapshot("model_loaded", torch, device))

        stage_start = time.perf_counter()
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        input_token_count = int(encoded["input_ids"].shape[-1])
        tokenization_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        # Quanto 0.2.7 with PyTorch 2.11 is compatible with no_grad(), while
        # inference_mode() rejects its quantized tensor version counters.
        with torch.no_grad():
            outputs = model(**encoded, use_cache=False, return_dict=True)
        torch.cuda.synchronize(device)
        forward_s = time.perf_counter() - stage_start

        logits = outputs.logits
        logits_shape = validate_real_logits(logits, expected_width, torch)
        logits_dtype = str(logits.dtype)
        logits_device = str(logits.device)
        observed_width = int(logits.shape[-1])
        last_logits = logits[:, -1, :].contiguous()
        snapshots.append(_memory_snapshot("forward_complete", torch, device))

        selection = select_and_advance_one_token(
            last_logits,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            regex=regex,
            grammar_factory=grammar_factory,
        )
        torch.cuda.synchronize(device)
        snapshots.append(_memory_snapshot("selection_complete", torch, device))
        peak_cuda_allocated_bytes = max(
            0, int(torch.cuda.max_memory_allocated(device)) - baseline_allocated
        )
    except BaseException as exc:
        # Propagating tracebacks retain frame arguments such as last_logits.
        # Detach the complete chain so tensors can die before empty_cache().
        _detach_exception_tracebacks(exc)
        failure = exc
    finally:
        cleanup_start = time.perf_counter()
        last_logits = None
        logits = None
        outputs = None
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
    if selection is None:
        raise RuntimeError("one-step handoff completed without a selection result")

    total_s = time.perf_counter() - total_start
    return RealLogitsHandoffReport(
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        quantization=QUANTIZATION_NAME,
        prompt=prompt,
        regex=regex,
        input_token_count=input_token_count,
        logits_shape=logits_shape,
        logits_dtype=logits_dtype,
        logits_device=logits_device,
        expected_logits_width=expected_width,
        observed_logits_width=observed_width,
        peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
        selection=selection,
        timings=RealLogitsHandoffTimings(
            metadata_load_s=metadata_load_s,
            vocabulary_validation_s=vocabulary_validation_s,
            model_load_s=model_load_s,
            tokenization_s=tokenization_s,
            forward_s=forward_s,
            cleanup_s=cleanup_s,
            total_s=total_s,
        ),
        memory_snapshots=tuple(snapshots),
    )


def format_real_logits_report(report: RealLogitsHandoffReport) -> str:
    """Format a compact console report for the one-step milestone."""
    status = "PASS" if report.passed else "INCOMPLETE"
    selected = report.selection
    lines = [
        f"Real-model CUDA logits handoff: {status}",
        f"  Model: {report.model_id}",
        f"  Revision: {report.resolved_revision or report.requested_revision or 'unavailable'}",
        f"  Quantization: {report.quantization}",
        f"  Logits: shape={report.logits_shape}, dtype={report.logits_dtype}, device={report.logits_device}",
        (
            f"  Width contract: expected={report.expected_logits_width}, "
            f"observed={report.observed_logits_width}"
        ),
        (
            f"  Raw argmax: id={selected.raw_argmax_token_id}, "
            f"text={selected.raw_argmax_text!r}, valid={selected.raw_argmax_was_valid}"
        ),
        (
            f"  Constrained selection: id={selected.selected_token_id}, "
            f"text={selected.selected_token_text!r}, valid={selected.selected_token_was_valid}"
        ),
        f"  Invalid raw argmax excluded: {selected.invalid_raw_argmax_excluded}",
        f"  Grammar matched after one token: {selected.grammar_matched_after_selection}",
        f"  Forward time: {report.timings.forward_s * 1000:.2f} ms",
        f"  Peak CUDA allocation increase: {report.peak_cuda_allocated_bytes / (1024 * 1024):.1f} MiB",
        f"  Cleanup time: {report.timings.cleanup_s * 1000:.2f} ms",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_PROMPT",
    "DEFAULT_REGEX",
    "MemorySnapshot",
    "OneStepSelection",
    "RealLogitsHandoffReport",
    "RealLogitsHandoffTimings",
    "SelectionTimings",
    "format_real_logits_report",
    "run_real_logits_handoff",
    "select_and_advance_one_token",
    "validate_real_logits",
]
