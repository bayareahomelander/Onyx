"""Framework-neutral target-generation performance and diagnostic metrics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class MetricsError(RuntimeError):
    """Base error raised by target-generation metrics."""


class MetricsStateError(MetricsError):
    """Raised when a metrics session operation is invalid for its state."""


class MetricsInvariantError(MetricsError):
    """Raised when timing or diagnostic data violates the metrics contract."""


@dataclass(frozen=True, slots=True)
class GrammarTimingMetrics:
    """Completed per-request constrained-generation timing totals in seconds."""

    compilation_time: float
    state_scan_time: float
    valid_index_transfer_time: float
    mask_application_time: float

    def __post_init__(self) -> None:
        _validate_finite_nonnegative(
            self.compilation_time,
            label="compilation_time",
        )
        _validate_finite_nonnegative(
            self.state_scan_time,
            label="state_scan_time",
        )
        _validate_finite_nonnegative(
            self.valid_index_transfer_time,
            label="valid_index_transfer_time",
        )
        _validate_finite_nonnegative(
            self.mask_application_time,
            label="mask_application_time",
        )


@dataclass(frozen=True, slots=True)
class TargetGenerationMetrics:
    """Completed target-only performance and device-memory measurements.

    ``ttft`` and ``generation_time`` are measured in seconds. Device-memory peaks are measured in
    bytes; both are ``None`` when the configured backend has no device-memory diagnostics.
    ``grammar_timing`` is present only for successful constrained generation.
    """

    ttft: float
    generation_time: float
    tokens_per_second: float
    cache_mode: str
    peak_allocated_vram_bytes: int | None = None
    peak_reserved_vram_bytes: int | None = None
    grammar_timing: GrammarTimingMetrics | None = None

    def __post_init__(self) -> None:
        ttft = _validate_finite_nonnegative(self.ttft, label="ttft")
        generation_time = _validate_finite_positive(
            self.generation_time,
            label="generation_time",
        )
        _validate_finite_positive(self.tokens_per_second, label="tokens_per_second")
        if ttft > generation_time:
            raise ValueError("ttft cannot exceed generation_time")
        _validate_cache_mode(self.cache_mode)

        allocated = _validate_optional_memory(
            self.peak_allocated_vram_bytes,
            label="peak_allocated_vram_bytes",
        )
        reserved = _validate_optional_memory(
            self.peak_reserved_vram_bytes,
            label="peak_reserved_vram_bytes",
        )
        if (allocated is None) != (reserved is None):
            raise ValueError(
                "peak allocated and reserved VRAM must both be measured or both be unavailable"
            )
        if self.grammar_timing is not None and not isinstance(
            self.grammar_timing,
            GrammarTimingMetrics,
        ):
            raise TypeError("grammar_timing must be GrammarTimingMetrics or None")


@runtime_checkable
class GenerationDiagnosticsSession(Protocol):
    """Backend-specific diagnostics for one target generation."""

    @property
    def cache_mode(self) -> str: ...

    def begin(self) -> None: ...

    def finish(self) -> tuple[int | None, int | None]: ...

    def abort(self) -> None: ...


class _NoDeviceDiagnosticsSession:
    def __init__(self, cache_mode: str) -> None:
        self._cache_mode = _validate_cache_mode(cache_mode)

    @property
    def cache_mode(self) -> str:
        return self._cache_mode

    def begin(self) -> None:
        return None

    def finish(self) -> tuple[None, None]:
        return None, None

    def abort(self) -> None:
        return None


class GrammarTimingSession:
    """Accumulate the four grammar timing components for one constrained request."""

    def __init__(self, *, clock: Callable[[], float]) -> None:
        if not callable(clock):
            raise TypeError("grammar timing clock must be callable")
        self._clock = clock
        self._state = "open"
        self._active_scope: str | None = None
        self._active_start: float | None = None
        self._compilation_attempted = False
        self._compilation_seconds = 0.0
        self._state_scan_seconds = 0.0
        self._valid_index_transfer_seconds = 0.0
        self._mask_application_seconds = 0.0
        self._state_scan_count = 0
        self._mask_call_count = 0

    @contextmanager
    def compilation(self) -> Generator[None, None, None]:
        """Measure the context's single dedicated native compilation call."""

        self._require_open()
        if self._compilation_attempted:
            raise MetricsStateError("grammar compilation can only be measured once")
        self._compilation_attempted = True
        with self._cpu_scope("grammar compilation", self._record_compilation):
            yield

    @contextmanager
    def state_scan(self) -> Generator[None, None, None]:
        """Measure one completed valid-token state scan."""

        with self._cpu_scope("grammar state scan", self._record_state_scan):
            yield

    @contextmanager
    def mask_application(self) -> Generator[None, None, None]:
        """Measure one framework-neutral mask call with zero index-transfer time."""

        with self._cpu_scope("grammar mask application", self._record_fallback_mask):
            yield

    def record_mask_timing(
        self,
        valid_index_transfer_time: float,
        mask_application_time: float,
        /,
    ) -> None:
        """Atomically record one completed transfer/application timing pair."""

        self._require_open()
        if self._active_scope is not None:
            raise MetricsStateError(
                "grammar mask timing cannot be recorded during a CPU timing scope"
            )
        transfer = self._validate_recorded_duration(
            valid_index_transfer_time,
            label="valid-index transfer duration",
        )
        application = self._validate_recorded_duration(
            mask_application_time,
            label="mask application duration",
        )
        transfer_total = self._checked_total(
            self._valid_index_transfer_seconds,
            transfer,
            label="valid-index transfer total",
        )
        application_total = self._checked_total(
            self._mask_application_seconds,
            application,
            label="mask application total",
        )
        self._valid_index_transfer_seconds = transfer_total
        self._mask_application_seconds = application_total
        self._mask_call_count += 1

    def finish(self, generated_tokens: int) -> GrammarTimingMetrics:
        """Validate per-token operation counts and finalize the immutable record."""

        self._require_open()
        if self._active_scope is not None:
            raise MetricsStateError("grammar timing cannot finish during an active scope")
        if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
            raise TypeError("generated_tokens must be an integer")
        if generated_tokens <= 0:
            raise ValueError("generated_tokens must be greater than zero")
        if self._state_scan_count != generated_tokens:
            raise MetricsInvariantError(
                "grammar state-scan count must equal generated token count: "
                f"scans={self._state_scan_count}, generated_tokens={generated_tokens}"
            )
        if self._mask_call_count != generated_tokens:
            raise MetricsInvariantError(
                "grammar mask-call count must equal generated token count: "
                f"mask_calls={self._mask_call_count}, generated_tokens={generated_tokens}"
            )

        metrics = GrammarTimingMetrics(
            compilation_time=self._compilation_seconds,
            state_scan_time=self._state_scan_seconds,
            valid_index_transfer_time=self._valid_index_transfer_seconds,
            mask_application_time=self._mask_application_seconds,
        )
        self._state = "finished"
        return metrics

    def abort(self) -> None:
        """Discard unfinished timing state; repeated calls are safe."""

        if self._state in {"finished", "aborted"}:
            return
        self._active_scope = None
        self._active_start = None
        self._state = "aborted"

    @contextmanager
    def _cpu_scope(
        self,
        operation: str,
        record: Callable[[float], None],
    ) -> Generator[None, None, None]:
        self._start_cpu_scope(operation)
        failure: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                duration = self._stop_cpu_scope(operation)
                if failure is None:
                    record(duration)
            except Exception as timing_exc:
                if failure is not None and not isinstance(failure, GeneratorExit):
                    raise MetricsInvariantError(
                        f"{operation} failed: {failure}; grammar timing also failed: "
                        f"{timing_exc}"
                    ) from failure
                raise

    def _start_cpu_scope(self, operation: str) -> None:
        self._require_open()
        if self._active_scope is not None:
            raise MetricsStateError("grammar CPU timing scopes cannot overlap")
        self._active_scope = operation
        try:
            self._active_start = self._read_clock()
        except Exception:
            self._active_scope = None
            raise

    def _stop_cpu_scope(self, operation: str) -> float:
        if self._active_scope != operation or self._active_start is None:
            raise MetricsStateError(f"no active {operation} timing scope")
        start = self._active_start
        self._active_scope = None
        self._active_start = None
        end = self._read_clock()
        duration = end - start
        if not math.isfinite(duration):
            raise MetricsInvariantError(f"{operation} duration must be finite")
        if duration < 0.0:
            raise MetricsInvariantError(
                f"grammar timing clock moved backwards during {operation}"
            )
        return duration

    def _record_compilation(self, duration: float) -> None:
        self._compilation_seconds = self._checked_total(
            self._compilation_seconds,
            duration,
            label="grammar compilation total",
        )

    def _record_state_scan(self, duration: float) -> None:
        self._state_scan_seconds = self._checked_total(
            self._state_scan_seconds,
            duration,
            label="grammar state-scan total",
        )
        self._state_scan_count += 1

    def _record_fallback_mask(self, duration: float) -> None:
        self.record_mask_timing(0.0, duration)

    def _read_clock(self) -> float:
        try:
            value = self._clock()
        except Exception as exc:
            raise MetricsInvariantError(f"grammar timing clock failed: {exc}") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricsInvariantError("grammar timing clock must return a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise MetricsInvariantError("grammar timing clock must return a finite value")
        return numeric

    def _require_open(self) -> None:
        if self._state != "open":
            raise MetricsStateError("grammar timing session is no longer open")

    @staticmethod
    def _validate_recorded_duration(value: float, *, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricsInvariantError(f"{label} must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise MetricsInvariantError(f"{label} must be finite")
        if numeric < 0.0:
            raise MetricsInvariantError(f"{label} cannot be negative")
        return numeric

    @staticmethod
    def _checked_total(current: float, duration: float, *, label: str) -> float:
        total = current + duration
        if not math.isfinite(total):
            raise MetricsInvariantError(f"{label} must remain finite")
        return total


class TargetMetricsSession:
    """Accumulate active target-generation time and finalize one immutable metric record."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        diagnostics: GenerationDiagnosticsSession,
    ) -> None:
        if not callable(clock):
            raise TypeError("metrics clock must be callable")
        if not isinstance(diagnostics, GenerationDiagnosticsSession):
            raise TypeError("diagnostics must implement GenerationDiagnosticsSession")
        _validate_cache_mode(diagnostics.cache_mode)
        self._clock = clock
        self._diagnostics = diagnostics
        self._state = "new"
        self._active_start: float | None = None
        self._active_seconds = 0.0
        self._ttft: float | None = None

    @property
    def cache_mode(self) -> str:
        return self._diagnostics.cache_mode

    def begin(self) -> None:
        if self._state != "new":
            raise MetricsStateError("metrics session can only begin once")
        self._diagnostics.begin()
        self._state = "running"

    @contextmanager
    def active(self) -> Generator[None, None, None]:
        """Measure one uninterrupted engine-active section."""

        self._start_active()
        failure: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                self._stop_active()
            except Exception as timing_exc:
                if failure is not None and not isinstance(failure, GeneratorExit):
                    raise MetricsInvariantError(
                        f"generation work failed: {failure}; metrics timing also failed: "
                        f"{timing_exc}"
                    ) from failure
                raise

    def mark_first_token(self) -> None:
        if self._state != "running" or self._active_start is None:
            raise MetricsStateError(
                "the first token must be marked during active generation work"
            )
        if self._ttft is not None:
            raise MetricsStateError("the first token has already been marked")
        now = self._read_clock()
        delta = now - self._active_start
        if delta < 0.0:
            raise MetricsInvariantError("metrics clock moved backwards while recording ttft")
        self._ttft = self._active_seconds + delta

    def finish(
        self,
        generated_tokens: int,
        *,
        grammar_timing: GrammarTimingMetrics | None = None,
    ) -> TargetGenerationMetrics:
        if self._state != "running":
            raise MetricsStateError("only a running metrics session can finish")
        if self._active_start is not None:
            raise MetricsStateError("metrics cannot finish during active generation work")
        if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
            raise TypeError("generated_tokens must be an integer")
        if generated_tokens <= 0:
            raise ValueError("generated_tokens must be greater than zero")
        if self._ttft is None:
            raise MetricsInvariantError("metrics cannot finish before the first token")
        if self._active_seconds <= 0.0:
            raise MetricsInvariantError("generation_time must be greater than zero")
        if grammar_timing is not None and not isinstance(
            grammar_timing,
            GrammarTimingMetrics,
        ):
            raise TypeError("grammar_timing must be GrammarTimingMetrics or None")

        allocated, reserved = self._diagnostics.finish()
        metrics = TargetGenerationMetrics(
            ttft=self._ttft,
            generation_time=self._active_seconds,
            tokens_per_second=generated_tokens / self._active_seconds,
            cache_mode=self.cache_mode,
            peak_allocated_vram_bytes=allocated,
            peak_reserved_vram_bytes=reserved,
            grammar_timing=grammar_timing,
        )
        self._state = "finished"
        return metrics

    def abort(self) -> None:
        if self._state in {"finished", "aborted"}:
            return
        try:
            self._diagnostics.abort()
        finally:
            self._active_start = None
            self._state = "aborted"

    def _start_active(self) -> None:
        if self._state != "running":
            raise MetricsStateError("metrics session must be running before timing work")
        if self._active_start is not None:
            raise MetricsStateError("metrics active sections cannot overlap")
        self._active_start = self._read_clock()

    def _stop_active(self) -> None:
        if self._active_start is None:
            raise MetricsStateError("no metrics active section is running")
        end = self._read_clock()
        delta = end - self._active_start
        self._active_start = None
        if delta < 0.0:
            raise MetricsInvariantError("metrics clock moved backwards")
        self._active_seconds += delta

    def _read_clock(self) -> float:
        try:
            value = self._clock()
        except Exception as exc:
            raise MetricsInvariantError(f"metrics clock failed: {exc}") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricsInvariantError("metrics clock must return a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise MetricsInvariantError("metrics clock must return a finite value")
        return numeric


def create_target_metrics_session(
    *,
    cache_mode: str,
    clock: Callable[[], float] = time.perf_counter,
    diagnostics: GenerationDiagnosticsSession | None = None,
) -> TargetMetricsSession:
    """Create one framework-neutral target metrics session."""

    if diagnostics is None:
        diagnostics = _NoDeviceDiagnosticsSession(cache_mode)
    elif cache_mode != diagnostics.cache_mode:
        raise ValueError(
            f"metrics cache mode {cache_mode!r} does not match diagnostics cache mode "
            f"{diagnostics.cache_mode!r}"
        )
    return TargetMetricsSession(clock=clock, diagnostics=diagnostics)


def create_grammar_timing_session(
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> GrammarTimingSession:
    """Create one framework-neutral constrained-request timing session."""

    return GrammarTimingSession(clock=clock)


def _validate_finite_nonnegative(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{label} cannot be negative")
    return numeric


def _validate_finite_positive(value: float, *, label: str) -> float:
    numeric = _validate_finite_nonnegative(value, label=label)
    if numeric <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return numeric


def _validate_cache_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("cache_mode must be a string")
    if not value.strip():
        raise ValueError("cache_mode cannot be empty")
    return value


def _validate_optional_memory(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer or None")
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return value
