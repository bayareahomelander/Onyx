from dataclasses import FrozenInstanceError, fields

import pytest

from onyx_cuda import (
    GrammarTimingMetrics,
    MetricsInvariantError,
    MetricsStateError,
    TargetGenerationMetrics,
    create_grammar_timing_session,
    create_target_metrics_session,
)


class ScriptedClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def test_grammar_timing_metrics_are_exact_immutable_and_slotted():
    timing = GrammarTimingMetrics(1.0, 2.0, 3.0, 4.0)

    assert [field.name for field in fields(timing)] == [
        "compilation_time",
        "state_scan_time",
        "valid_index_transfer_time",
        "mask_application_time",
    ]
    assert timing == GrammarTimingMetrics(
        compilation_time=1.0,
        state_scan_time=2.0,
        valid_index_transfer_time=3.0,
        mask_application_time=4.0,
    )
    assert "compilation_time=1.0" in repr(timing)
    assert not hasattr(timing, "__dict__")
    with pytest.raises(FrozenInstanceError):
        timing.compilation_time = 5.0


@pytest.mark.parametrize(
    "value",
    [True, None, "1", -1.0, float("nan"), float("inf"), float("-inf")],
)
@pytest.mark.parametrize(
    "field_name",
    [
        "compilation_time",
        "state_scan_time",
        "valid_index_transfer_time",
        "mask_application_time",
    ],
)
def test_grammar_timing_metrics_reject_invalid_components(field_name, value):
    values = {
        "compilation_time": 0.0,
        "state_scan_time": 0.0,
        "valid_index_transfer_time": 0.0,
        "mask_application_time": 0.0,
    }
    values[field_name] = value

    with pytest.raises((TypeError, ValueError)):
        GrammarTimingMetrics(**values)


def test_target_metrics_append_optional_timing_without_breaking_positional_construction():
    original = TargetGenerationMetrics(0.5, 2.0, 1.0, "fake", 10, 20)
    timing = GrammarTimingMetrics(1.0, 2.0, 3.0, 4.0)
    extended = TargetGenerationMetrics(0.5, 2.0, 1.0, "fake", 10, 20, timing)

    assert original.grammar_timing is None
    assert extended.grammar_timing is timing
    with pytest.raises(TypeError, match="GrammarTimingMetrics or None"):
        TargetGenerationMetrics(0.5, 2.0, 1.0, "fake", grammar_timing=(1, 2, 3, 4))


def test_grammar_timing_session_accumulates_exact_scopes_and_atomic_mask_pairs():
    session = create_grammar_timing_session(
        clock=ScriptedClock((0.0, 2.0, 10.0, 13.0, 20.0, 24.0))
    )

    with session.compilation():
        pass
    with session.state_scan():
        pass
    session.record_mask_timing(0.25, 0.75)
    with session.state_scan():
        pass
    session.record_mask_timing(0.5, 1.25)

    assert session.finish(2) == GrammarTimingMetrics(
        compilation_time=2.0,
        state_scan_time=7.0,
        valid_index_transfer_time=0.75,
        mask_application_time=2.0,
    )


def test_precompiled_context_and_fallback_mask_allow_zero_components():
    session = create_grammar_timing_session(clock=ScriptedClock((1.0, 1.0, 5.0, 8.0)))

    with session.state_scan():
        pass
    with session.mask_application():
        pass

    assert session.finish(1) == GrammarTimingMetrics(
        compilation_time=0.0,
        state_scan_time=0.0,
        valid_index_transfer_time=0.0,
        mask_application_time=3.0,
    )


def test_mask_pair_recording_is_atomic_when_either_duration_is_invalid():
    session = create_grammar_timing_session(clock=ScriptedClock((0.0, 1.0)))
    with session.state_scan():
        pass

    with pytest.raises(MetricsInvariantError, match="cannot be negative"):
        session.record_mask_timing(2.0, -1.0)
    session.record_mask_timing(0.5, 1.5)

    assert session.finish(1).valid_index_transfer_time == 0.5


@pytest.mark.parametrize("durations", [(True, 1.0), (1.0, None), (float("nan"), 1.0)])
def test_mask_pair_recording_rejects_malformed_durations(durations):
    session = create_grammar_timing_session()
    with pytest.raises(MetricsInvariantError):
        session.record_mask_timing(*durations)


def test_finish_requires_one_scan_and_mask_pair_per_generated_token():
    missing_scan = create_grammar_timing_session()
    missing_scan.record_mask_timing(0.0, 0.0)
    with pytest.raises(MetricsInvariantError, match="state-scan count"):
        missing_scan.finish(1)

    missing_mask = create_grammar_timing_session(clock=ScriptedClock((0.0, 1.0)))
    with missing_mask.state_scan():
        pass
    with pytest.raises(MetricsInvariantError, match="mask-call count"):
        missing_mask.finish(1)

    extra_scan = create_grammar_timing_session(clock=ScriptedClock((0.0, 1.0, 2.0, 3.0)))
    with extra_scan.state_scan():
        pass
    with extra_scan.state_scan():
        pass
    extra_scan.record_mask_timing(0.0, 0.0)
    with pytest.raises(MetricsInvariantError, match="state-scan count"):
        extra_scan.finish(1)

    extra_mask = create_grammar_timing_session(clock=ScriptedClock((0.0, 1.0)))
    with extra_mask.state_scan():
        pass
    extra_mask.record_mask_timing(0.0, 0.0)
    extra_mask.record_mask_timing(0.0, 0.0)
    with pytest.raises(MetricsInvariantError, match="mask-call count"):
        extra_mask.finish(1)


@pytest.mark.parametrize("generated_tokens", [True, 1.0, "1", None])
def test_finish_rejects_non_integer_generated_token_counts(generated_tokens):
    with pytest.raises(TypeError, match="integer"):
        create_grammar_timing_session().finish(generated_tokens)


@pytest.mark.parametrize("generated_tokens", [0, -1])
def test_finish_rejects_nonpositive_generated_token_counts(generated_tokens):
    with pytest.raises(ValueError, match="greater than zero"):
        create_grammar_timing_session().finish(generated_tokens)


def test_compilation_can_only_be_measured_once_and_scopes_cannot_overlap():
    session = create_grammar_timing_session(
        clock=ScriptedClock((0.0, 1.0, 2.0, 3.0))
    )
    with session.compilation():
        with pytest.raises(MetricsStateError, match="only be measured once"):
            with session.compilation():
                pass
    with session.state_scan():
        with pytest.raises(MetricsStateError, match="cannot overlap"):
            with session.mask_application():
                pass


@pytest.mark.parametrize(
    ("clock", "message"),
    [
        (ScriptedClock((2.0, 1.0)), "backwards"),
        (ScriptedClock((float("nan"),)), "finite"),
        (ScriptedClock((True,)), "real number"),
        (lambda: (_ for _ in ()).throw(RuntimeError("clock broke")), "clock broke"),
    ],
)
def test_session_maps_invalid_and_failing_clocks_to_metrics_invariants(clock, message):
    session = create_grammar_timing_session(clock=clock)
    with pytest.raises(MetricsInvariantError, match=message):
        with session.state_scan():
            pass
    session.abort()


def test_abort_is_idempotent_and_finished_or_aborted_sessions_cannot_be_reused():
    aborted = create_grammar_timing_session()
    aborted.abort()
    aborted.abort()
    with pytest.raises(MetricsStateError, match="no longer open"):
        with aborted.state_scan():
            pass
    with pytest.raises(MetricsStateError, match="no longer open"):
        aborted.finish(1)

    finished = create_grammar_timing_session(clock=ScriptedClock((0.0, 1.0)))
    with finished.state_scan():
        pass
    finished.record_mask_timing(0.0, 0.0)
    finished.finish(1)
    finished.abort()
    with pytest.raises(MetricsStateError, match="no longer open"):
        finished.record_mask_timing(0.0, 0.0)


def test_target_metrics_session_attaches_timing_without_changing_aggregate_values():
    grammar_timing = GrammarTimingMetrics(7.0, 2.0, 1.0, 1.5)
    session = create_target_metrics_session(
        cache_mode="fake",
        clock=ScriptedClock((10.0, 12.0, 15.0)),
    )
    session.begin()
    with session.active():
        session.mark_first_token()

    metrics = session.finish(1, grammar_timing=grammar_timing)

    assert metrics == TargetGenerationMetrics(
        ttft=2.0,
        generation_time=5.0,
        tokens_per_second=0.2,
        cache_mode="fake",
        grammar_timing=grammar_timing,
    )


def test_target_metrics_session_rejects_invalid_nested_record_before_finalizing():
    session = create_target_metrics_session(
        cache_mode="fake",
        clock=ScriptedClock((0.0, 1.0, 2.0)),
    )
    session.begin()
    with session.active():
        session.mark_first_token()

    with pytest.raises(TypeError, match="GrammarTimingMetrics or None"):
        session.finish(1, grammar_timing=object())
    session.abort()
