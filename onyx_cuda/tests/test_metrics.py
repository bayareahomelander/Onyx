from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import (
    GenerationCleanupError,
    GenerationResult,
    MetricsInvariantError,
    MetricsStateError,
    TargetGenerationMetrics,
    TargetTextEngine,
    TextGenerationComplete,
    TextGenerationDelta,
    create_target_metrics_session,
    select_highest_logit,
)
from onyx_cuda.testing import FakeAutoregressiveBackend, FakeCharacterTokenizer


class ScriptedClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class FakeDiagnostics:
    def __init__(
        self,
        *,
        cache_mode="fake-diagnostics",
        peaks=(123, 456),
        begin_error=None,
        finish_error=None,
        abort_error=None,
    ):
        self._cache_mode = cache_mode
        self.peaks = peaks
        self.begin_error = begin_error
        self.finish_error = finish_error
        self.abort_error = abort_error
        self.calls = []

    @property
    def cache_mode(self):
        return self._cache_mode

    def begin(self):
        self.calls.append("begin")
        if self.begin_error is not None:
            raise self.begin_error

    def finish(self):
        self.calls.append("finish")
        if self.finish_error is not None:
            raise self.finish_error
        return self.peaks

    def abort(self):
        self.calls.append("abort")
        if self.abort_error is not None:
            raise self.abort_error


def test_target_metrics_are_immutable_and_preserve_established_names():
    metrics = TargetGenerationMetrics(
        ttft=0.25,
        generation_time=1.0,
        tokens_per_second=4.0,
        cache_mode="fake",
        peak_allocated_vram_bytes=100,
        peak_reserved_vram_bytes=200,
    )

    assert metrics.ttft == 0.25
    assert metrics.generation_time == 1.0
    assert metrics.tokens_per_second == 4.0
    with pytest.raises(FrozenInstanceError):
        metrics.ttft = 0.5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ttft", -0.1, "cannot be negative"),
        ("ttft", float("nan"), "finite"),
        ("generation_time", 0.0, "greater than zero"),
        ("generation_time", float("inf"), "finite"),
        ("tokens_per_second", 0.0, "greater than zero"),
        ("tokens_per_second", True, "real number"),
        ("cache_mode", "", "cannot be empty"),
        ("cache_mode", None, "must be a string"),
        ("peak_allocated_vram_bytes", -1, "cannot be negative"),
        ("peak_reserved_vram_bytes", 1.5, "integer or None"),
    ],
)
def test_target_metrics_reject_invalid_values(field, value, message):
    values = {
        "ttft": 0.25,
        "generation_time": 1.0,
        "tokens_per_second": 1.0,
        "cache_mode": "fake",
        "peak_allocated_vram_bytes": 100,
        "peak_reserved_vram_bytes": 200,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        TargetGenerationMetrics(**values)


def test_target_metrics_reject_ttft_after_total_and_partial_memory_snapshot():
    with pytest.raises(ValueError, match="cannot exceed"):
        TargetGenerationMetrics(2.0, 1.0, 1.0, "fake")
    with pytest.raises(ValueError, match="both be measured"):
        TargetGenerationMetrics(
            0.5,
            1.0,
            1.0,
            "fake",
            peak_allocated_vram_bytes=1,
        )


def test_metrics_session_accumulates_active_time_and_excludes_idle_gap():
    diagnostics = FakeDiagnostics()
    session = create_target_metrics_session(
        cache_mode="fake-diagnostics",
        clock=ScriptedClock((10.0, 12.0, 15.0, 100.0, 104.0)),
        diagnostics=diagnostics,
    )

    session.begin()
    with session.active():
        session.mark_first_token()
    with session.active():
        pass
    metrics = session.finish(2)

    assert metrics == TargetGenerationMetrics(
        ttft=2.0,
        generation_time=9.0,
        tokens_per_second=2 / 9,
        cache_mode="fake-diagnostics",
        peak_allocated_vram_bytes=123,
        peak_reserved_vram_bytes=456,
    )
    assert diagnostics.calls == ["begin", "finish"]


def test_stream_consumer_delay_does_not_inflate_generation_time():
    backend = FakeAutoregressiveBackend(((0.0, 1.0), (1.0, 0.0)))
    tokenizer = FakeCharacterTokenizer(("P", "a"))

    def create_metrics():
        return create_target_metrics_session(
            cache_mode="fake",
            clock=ScriptedClock((0.0, 1.0, 2.0, 1000.0, 1002.0)),
        )

    engine = TargetTextEngine(
        backend,
        tokenizer,
        select_token=select_highest_logit,
        create_metrics_session=create_metrics,
    )
    stream = engine.stream("P", max_new_tokens=2)

    assert next(stream) == TextGenerationDelta("a")
    assert next(stream) == TextGenerationDelta("P")
    completion = next(stream)

    assert isinstance(completion, TextGenerationComplete)
    assert completion.result.generation.metrics.ttft == 1.0
    assert completion.result.generation.metrics.generation_time == 4.0
    assert completion.result.generation.metrics.tokens_per_second == 0.5


def test_generation_result_validates_metric_throughput_against_sampled_tokens():
    metrics = TargetGenerationMetrics(0.5, 2.0, 1.5, "fake")

    with pytest.raises(ValueError, match="generated tokens divided"):
        GenerationResult("target", (1, 2), "length", 1, 2, metrics)


def test_metrics_session_rejects_invalid_order_and_backwards_clock():
    session = create_target_metrics_session(
        cache_mode="fake",
        clock=ScriptedClock((2.0, 1.0)),
    )

    with pytest.raises(MetricsStateError, match="running"):
        with session.active():
            pass
    session.begin()
    with pytest.raises(MetricsInvariantError, match="backwards"):
        with session.active():
            pass
    session.abort()
    session.abort()


def test_metrics_abort_is_idempotent_and_does_not_finish_diagnostics():
    diagnostics = FakeDiagnostics()
    session = create_target_metrics_session(
        cache_mode="fake-diagnostics",
        diagnostics=diagnostics,
    )

    session.begin()
    session.abort()
    session.abort()

    assert diagnostics.calls == ["begin", "abort"]


def test_generation_reports_execution_and_metrics_abort_failures_together():
    backend = FakeAutoregressiveBackend(((0.0, 1.0),))
    diagnostics = FakeDiagnostics(abort_error=RuntimeError("metrics cleanup failed"))
    session = create_target_metrics_session(
        cache_mode="fake-diagnostics",
        clock=ScriptedClock((0.0, 1.0)),
        diagnostics=diagnostics,
    )
    engine = TargetTextEngine(
        backend,
        FakeCharacterTokenizer(("P", "a")),
        select_token=lambda logits: (_ for _ in ()).throw(RuntimeError("selection failed")),
        create_metrics_session=lambda: session,
    )

    with pytest.raises(
        GenerationCleanupError,
        match="selection failed.*metrics abort also failed.*metrics cleanup failed",
    ):
        engine.generate("P", max_new_tokens=1)

    assert backend.cache_length == 0
    assert diagnostics.calls == ["begin", "abort"]


def test_metrics_factory_rejects_cache_mode_mismatch_and_invalid_session_factory():
    diagnostics = FakeDiagnostics(cache_mode="diagnostics")
    with pytest.raises(ValueError, match="does not match"):
        create_target_metrics_session(cache_mode="other", diagnostics=diagnostics)

    backend = FakeAutoregressiveBackend(((0.0, 1.0),))
    engine = TargetTextEngine(
        backend,
        FakeCharacterTokenizer(("P", "a")),
        select_token=select_highest_logit,
        create_metrics_session=lambda: None,
    )
    with pytest.raises(TypeError, match="must return a TargetMetricsSession"):
        engine.generate("P", max_new_tokens=1)
    assert backend.cache_length == 0


def test_invalid_generation_input_does_not_create_a_metrics_session():
    backend = FakeAutoregressiveBackend(((0.0, 1.0),))
    calls = []

    def create_metrics():
        calls.append("created")
        return create_target_metrics_session(cache_mode="fake")

    engine = TargetTextEngine(
        backend,
        FakeCharacterTokenizer(("P", "a")),
        select_token=select_highest_logit,
        create_metrics_session=create_metrics,
    )

    with pytest.raises(ValueError, match="greater than zero"):
        engine.stream("P", max_new_tokens=0)

    assert calls == []
    assert backend.cache_length == 0
