import math
from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import (
    ConstrainedGenerationCleanupError,
    ConstrainedGenerationInvariantError,
    GenerationResult,
    GrammarGenerationContext,
    GrammarNoContinuationError,
    GrammarTimingMetrics,
    GrammarTimingSession,
    JsonSchemaGrammar,
    MetricsStateError,
    RegexGrammar,
    TargetMetricsSession,
    generate_constrained_target,
    select_highest_logit,
)
from onyx_cuda.testing import (
    DeterministicMetricsClock,
    FakeAutoregressiveBackend,
    FakeGrammarConstraint,
    FakeGrammarProgram,
    create_deterministic_grammar_timing_session,
    create_deterministic_metrics_session,
    deterministic_target_metrics,
)
from onyx_cuda.constrained_generation import _iterate_constrained_target


VOCABULARY = (b"P", b"a", b"b", b"")
EOS_TOKEN_ID = 3


class RecordingMask:
    def __init__(self, vocab_size, *, failure=None):
        self._vocab_size = vocab_size
        self.failure = failure
        self.calls = []

    @property
    def vocab_size(self):
        return self._vocab_size

    def apply(self, logits, valid_token_ids, /):
        self.calls.append(valid_token_ids)
        if self.failure is not None:
            raise self.failure
        return tuple(
            value if token_id in valid_token_ids else -math.inf
            for token_id, value in enumerate(logits)
        )


class TimedRecordingMask(RecordingMask):
    transport_name = "sparse_valid_indices"

    def __init__(self, vocab_size, *, timing_pairs, failure=None):
        super().__init__(vocab_size, failure=failure)
        self._timing_pairs = iter(timing_pairs)

    def apply_with_timing(self, logits, valid_token_ids, timing_session, /):
        result = self.apply(logits, valid_token_ids)
        transfer_time, application_time = next(self._timing_pairs)
        timing_session.record_mask_timing(transfer_time, application_time)
        return result


class RecordingConstraint(FakeGrammarConstraint):
    def __init__(self, vocabulary, *, grammar_type="regex", program):
        self.calls = []
        self.failures = {}
        self._record_calls = False
        super().__init__(vocabulary, grammar_type=grammar_type, program=program)
        self._record_calls = True

    def _record(self, operation, detail=None):
        if self._record_calls:
            self.calls.append((operation, detail))
        failure = self.failures.get(operation)
        if failure is not None:
            raise failure

    def init_state(self):
        self._record("init_state")
        return super().init_state()

    def advance_state(self, state, token_id, /):
        self._record("advance_state", token_id)
        return super().advance_state(state, token_id)

    def get_valid_token_ids(self, state, /):
        self._record("get_valid_token_ids")
        return super().get_valid_token_ids(state)

    def is_match_state(self, state, /):
        self._record("is_match_state")
        return super().is_match_state(state)

    def is_dead_state(self, state, /):
        self._record("is_dead_state")
        return super().is_dead_state(state)

    def release_state(self, state, /):
        self._record("release_state")
        return super().release_state(state)

    def release_states(self, states, /):
        self._record("release_states", len(tuple(states)))
        return super().release_states(states)

    def reset(self):
        self._record("reset")
        return super().reset()


class BackendWrapper:
    def __init__(self, backend, *, reset_failure=None):
        self.backend = backend
        self.reset_failure = reset_failure
        self.reset_calls = 0

    @property
    def model_id(self):
        return self.backend.model_id

    @property
    def vocab_size(self):
        return self.backend.vocab_size

    @property
    def cache_length(self):
        return self.backend.cache_length

    def prefill(self, prompt_token_ids, /):
        return self.backend.prefill(prompt_token_ids)

    def decode(self, token_id, /):
        return self.backend.decode(token_id)

    def reset(self):
        self.reset_calls += 1
        if self.reset_failure is not None:
            raise self.reset_failure
        self.backend.reset()


class FailingAbortDiagnostics:
    cache_mode = "fake"

    def begin(self):
        return None

    def finish(self):
        return None, None

    def abort(self):
        raise RuntimeError("metrics abort failed")


class RecordingDiagnostics:
    cache_mode = "fake"

    def __init__(self):
        self.calls = []

    def begin(self):
        self.calls.append("begin")

    def finish(self):
        self.calls.append("finish")
        return None, None

    def abort(self):
        self.calls.append("abort")


def _path_program(*, initial_match=False, final_match=True):
    match_states = set()
    if initial_match:
        match_states.add("s0")
    if final_match:
        match_states.add("s2")
    return FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"), ("s1", 2, "s2")),
        valid_token_ids=(("s0", (1,)), ("s1", (2,)), ("s2", ())),
        match_states=frozenset(match_states),
    )


def _context(
    constraint,
    mask=None,
    *,
    eos_token_id=EOS_TOKEN_ID,
    timing_session=None,
):
    return GrammarGenerationContext(
        constraint=constraint,
        logit_mask=mask or RecordingMask(constraint.vocab_size),
        eos_token_id=eos_token_id,
        timing_session=timing_session or create_deterministic_grammar_timing_session(),
    )


def _generate(backend, context, *, max_new_tokens, stops=(), selector=select_highest_logit):
    return generate_constrained_target(
        backend,
        (0,),
        max_new_tokens=max_new_tokens,
        select_token=selector,
        grammar_context=context,
        stop_token_sequences=stops,
        metrics_session=create_deterministic_metrics_session(),
    )


def test_exact_effective_support_is_masked_on_every_position_and_eos_completes():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    mask = RecordingMask(len(VOCABULARY))
    backend = FakeAutoregressiveBackend(
        ((100.0, 1.0, 99.0, -1.0), (9.0, 8.0, 2.0, 1.0), (0.0, 0.0, 0.0, 5.0))
    )

    result = _generate(backend, _context(constraint, mask), max_new_tokens=3)

    assert mask.calls == [(1,), (2,), (EOS_TOKEN_ID,)]
    assert result.token_ids == (1, 2, EOS_TOKEN_ID)
    assert result.visible_token_ids == (1, 2)
    assert result.finish_reason == "grammar_complete"
    assert result.grammar_completion_token_id == EOS_TOKEN_ID
    assert result.final_cache_length == 3
    assert result.metrics.tokens_per_second == pytest.approx(3 / 4)
    assert result.metrics.grammar_timing == GrammarTimingMetrics(
        compilation_time=0.0,
        state_scan_time=3.0,
        valid_index_transfer_time=0.0,
        mask_application_time=3.0,
    )
    assert [detail for operation, detail in constraint.calls if operation == "advance_state"] == [
        1,
        2,
        EOS_TOKEN_ID,
    ]
    assert constraint.active_state_count == 0


def test_timed_mask_records_exact_nonoverlapping_totals_for_every_sampled_token():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    timing_session = create_deterministic_grammar_timing_session()
    with timing_session.compilation():
        pass
    mask = TimedRecordingMask(
        len(VOCABULARY),
        timing_pairs=((0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
    )
    backend = FakeAutoregressiveBackend(
        ((0.0, 3.0, 0.0, 0.0), (0.0, 0.0, 3.0, 0.0), (0.0, 0.0, 0.0, 4.0))
    )

    result = _generate(
        backend,
        _context(constraint, mask, timing_session=timing_session),
        max_new_tokens=3,
    )

    assert mask.calls == [(1,), (2,), (EOS_TOKEN_ID,)]
    assert result.metrics.grammar_timing == GrammarTimingMetrics(
        compilation_time=1.0,
        state_scan_time=3.0,
        valid_index_transfer_time=1.5,
        mask_application_time=2.25,
    )


def test_sparse_index_mask_without_timed_capability_is_a_composition_invariant():
    class UntimedProductionMask(RecordingMask):
        transport_name = "sparse_valid_indices"

    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    backend = BackendWrapper(FakeAutoregressiveBackend(((0.0, 3.0, 0.0, 0.0),)))

    with pytest.raises(
        ConstrainedGenerationInvariantError,
        match="must provide completed transfer/application timing",
    ):
        _generate(
            backend,
            _context(constraint, UntimedProductionMask(len(VOCABULARY))),
            max_new_tokens=1,
        )

    assert backend.reset_calls == 1
    assert constraint.active_state_count == 0


@pytest.mark.parametrize(
    ("eos_token_id", "native_ids", "effective_ids"),
    [
        (0, (1, 3), (0, 1, 3)),
        (2, (1, 3), (1, 2, 3)),
        (4, (1, 3), (1, 3, 4)),
    ],
)
def test_matching_state_inserts_eos_in_sorted_position(
    eos_token_id,
    native_ids,
    effective_ids,
):
    vocabulary = tuple(b"" if token_id == eos_token_id else bytes([65 + token_id]) for token_id in range(5))
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=tuple(("match", token_id, "match") for token_id in native_ids),
        valid_token_ids=(("match", native_ids),),
        match_states=frozenset({"match"}),
    )
    constraint = RecordingConstraint(vocabulary, program=program)
    mask = RecordingMask(5)
    logits = tuple(10.0 if token_id == eos_token_id else 0.0 for token_id in range(5))

    result = _generate(
        FakeAutoregressiveBackend((logits,)),
        _context(constraint, mask, eos_token_id=eos_token_id),
        max_new_tokens=1,
    )

    assert mask.calls == [effective_ids]
    assert result.finish_reason == "grammar_complete"
    assert result.visible_token_ids == ()


def test_matching_state_can_choose_content_continuation_instead_of_immediate_eos():
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=(("match", 2, "extended"),),
        valid_token_ids=(("match", (2,)), ("extended", ())),
        match_states=frozenset({"match", "extended"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)
    mask = RecordingMask(4)

    result = _generate(
        FakeAutoregressiveBackend(((100.0, 0.0, 9.0, 1.0),)),
        _context(constraint, mask),
        max_new_tokens=1,
    )

    assert mask.calls == [(2, EOS_TOKEN_ID)]
    assert result.token_ids == (2,)
    assert result.finish_reason == "length"
    assert result.visible_token_ids == (2,)


def test_nonmatching_empty_support_fails_before_backend_or_mask_work():
    program = FakeGrammarProgram(
        initial_state="empty",
        transitions=(),
        valid_token_ids=(("empty", ()),),
        match_states=frozenset({"empty"}),
    )

    class NonmatchingConstraint(RecordingConstraint):
        def is_match_state(self, state, /):
            super().is_match_state(state)
            return False

    constraint = NonmatchingConstraint(VOCABULARY, program=program)
    mask = RecordingMask(4)
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 2.0, 3.0),))

    with pytest.raises(GrammarNoContinuationError, match="no valid token"):
        _generate(backend, _context(constraint, mask), max_new_tokens=1)

    assert mask.calls == []
    assert backend.cache_length == 0
    assert constraint.active_state_count == 0


def test_dead_current_state_is_an_invariant_failure_before_backend_work():
    program = FakeGrammarProgram(
        initial_state="dead",
        transitions=(),
        valid_token_ids=(("dead", ()),),
        dead_states=frozenset({"dead"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 2.0, 3.0),))

    with pytest.raises(ConstrainedGenerationInvariantError, match="dead grammar state"):
        _generate(backend, _context(constraint), max_new_tokens=1)

    assert backend.cache_length == 0
    assert constraint.active_state_count == 0


def test_unexpected_dead_child_preserves_parent_until_failure_cleanup():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"),),
        valid_token_ids=(("s0", (1,)), ("s1", ())),
        match_states=frozenset({"s1"}),
    )

    class DeadChildConstraint(RecordingConstraint):
        def is_dead_state(self, state, /):
            super().is_dead_state(state)
            return self._state_name(state) == "s1"

    constraint = DeadChildConstraint(VOCABULARY, program=program)
    backend = FakeAutoregressiveBackend(((0.0, 2.0, 1.0, 0.0),))

    with pytest.raises(ConstrainedGenerationInvariantError, match="dead child"):
        _generate(backend, _context(constraint), max_new_tokens=1)

    assert ("release_states", 2) in constraint.calls
    assert not any(operation == "release_state" for operation, _ in constraint.calls)
    assert backend.cache_length == 0
    assert constraint.active_state_count == 0


def test_out_of_support_selection_is_rejected_before_grammar_advancement():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 9.0, 0.0),))

    with pytest.raises(ConstrainedGenerationInvariantError, match="outside the effective"):
        _generate(
            backend,
            _context(constraint),
            max_new_tokens=1,
            selector=lambda _logits: 2,
        )

    assert not any(operation == "advance_state" for operation, _ in constraint.calls)
    assert backend.cache_length == 0
    assert constraint.active_state_count == 0


@pytest.mark.parametrize("invalid_ids", [[1], (2, 1), (1, 1), (True,), (4,)])
def test_malformed_native_support_is_rejected_before_mask_or_backend(invalid_ids):
    class MalformedConstraint(RecordingConstraint):
        def get_valid_token_ids(self, state, /):
            super().get_valid_token_ids(state)
            return invalid_ids

    constraint = MalformedConstraint(VOCABULARY, program=_path_program())
    mask = RecordingMask(4)
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 2.0, 3.0),))

    with pytest.raises(ConstrainedGenerationInvariantError):
        _generate(backend, _context(constraint, mask), max_new_tokens=1)

    assert mask.calls == []
    assert backend.cache_length == 0
    assert constraint.active_state_count == 0


def test_native_support_must_never_advertise_eos():
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=(("match", EOS_TOKEN_ID, "match"),),
        valid_token_ids=(("match", (EOS_TOKEN_ID,)),),
        match_states=frozenset({"match"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)
    backend = FakeAutoregressiveBackend(((0.0, 0.0, 0.0, 1.0),))

    with pytest.raises(ConstrainedGenerationInvariantError, match="must not advertise"):
        _generate(backend, _context(constraint), max_new_tokens=1)

    assert backend.cache_length == 0


def test_eligible_stop_uses_matching_visible_prefix_and_wins_at_length_boundary():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"), ("s1", 2, "s2"), ("s2", 1, "s3")),
        valid_token_ids=(("s0", (1,)), ("s1", (2,)), ("s2", (1,)), ("s3", ())),
        match_states=frozenset({"s1", "s2", "s3"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)
    backend = FakeAutoregressiveBackend(
        ((0.0, 3.0, 0.0, 0.0), (0.0, 0.0, 3.0, 0.0), (0.0, 3.0, 0.0, 0.0))
    )

    result = _generate(
        backend,
        _context(constraint),
        max_new_tokens=3,
        stops=((2, 1), (1,)),
    )

    assert result.finish_reason == "stop"
    assert result.matched_stop_token_ids == (2, 1)
    assert result.token_ids == (1, 2, 1)
    assert result.visible_token_ids == (1,)


def test_ineligible_stop_suffix_remains_visible_grammar_content():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"),),
        valid_token_ids=(("s0", (1,)), ("s1", ())),
        match_states=frozenset({"s1"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)

    result = _generate(
        FakeAutoregressiveBackend(((0.0, 2.0, 0.0, 0.0),)),
        _context(constraint),
        max_new_tokens=1,
        stops=((1,),),
    )

    assert result.finish_reason == "length"
    assert result.matched_stop_token_ids is None
    assert result.visible_token_ids == (1,)


def test_configured_eos_stop_precedes_grammar_completion():
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=(),
        valid_token_ids=(("match", ()),),
        match_states=frozenset({"match"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)

    result = _generate(
        FakeAutoregressiveBackend(((0.0, 0.0, 0.0, 5.0),)),
        _context(constraint),
        max_new_tokens=1,
        stops=((EOS_TOKEN_ID,),),
    )

    assert result.finish_reason == "stop"
    assert result.matched_stop_token_ids == (EOS_TOKEN_ID,)
    assert result.grammar_completion_token_id is None
    assert result.visible_token_ids == ()


def test_context_vocab_mismatch_resets_constraint_without_touching_backend():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    mask = RecordingMask(3)
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 2.0, 3.0),))

    with pytest.raises(ConstrainedGenerationInvariantError, match="must match exactly"):
        _generate(backend, _context(constraint, mask), max_new_tokens=1)

    assert backend.cache_length == 0
    assert constraint.active_state_count == 0
    assert ("reset", None) in constraint.calls


def test_mask_failure_resets_backend_and_releases_all_grammar_state():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    backend = BackendWrapper(FakeAutoregressiveBackend(((0.0, 1.0, 2.0, 3.0),)))
    mask = RecordingMask(4, failure=RuntimeError("mask failed"))

    with pytest.raises(RuntimeError, match="mask failed"):
        _generate(backend, _context(constraint, mask), max_new_tokens=1)

    assert backend.reset_calls == 1
    assert backend.cache_length == 0
    assert constraint.active_state_count == 0


def test_cleanup_error_reports_original_and_every_cleanup_failure_in_order():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    constraint.failures.update(
        {
            "release_states": RuntimeError("state cleanup failed"),
            "reset": RuntimeError("constraint reset failed"),
        }
    )
    backend = BackendWrapper(
        FakeAutoregressiveBackend(((0.0, 1.0, 2.0, 3.0),)),
        reset_failure=RuntimeError("backend reset failed"),
    )
    mask_failure = RuntimeError("mask failed")
    clock_value = -1

    def clock():
        nonlocal clock_value
        clock_value += 1
        return float(clock_value)

    metrics = TargetMetricsSession(clock=clock, diagnostics=FailingAbortDiagnostics())

    with pytest.raises(ConstrainedGenerationCleanupError) as captured:
        generate_constrained_target(
            backend,
            (0,),
            max_new_tokens=1,
            select_token=select_highest_logit,
            grammar_context=_context(
                constraint,
                RecordingMask(4, failure=mask_failure),
            ),
            metrics_session=metrics,
        )

    error = captured.value
    assert error.__cause__ is mask_failure
    assert error.original_failure is mask_failure
    assert [operation for operation, _ in error.cleanup_failures] == [
        "grammar state release",
        "grammar constraint reset",
        "backend reset",
        "metrics abort",
    ]
    assert "state cleanup failed" in str(error)
    assert "constraint reset failed" in str(error)
    assert "backend reset failed" in str(error)
    assert "metrics abort failed" in str(error)


def test_grammar_completion_result_invariants_and_specifications_are_immutable():
    metrics = deterministic_target_metrics(1)
    result = GenerationResult(
        model_id="fake",
        token_ids=(EOS_TOKEN_ID,),
        finish_reason="grammar_complete",
        prompt_tokens=1,
        final_cache_length=1,
        metrics=metrics,
        grammar_completion_token_id=EOS_TOKEN_ID,
    )

    assert result.visible_token_ids == ()
    with pytest.raises(ValueError, match="final sampled"):
        GenerationResult(
            model_id="fake",
            token_ids=(1,),
            finish_reason="grammar_complete",
            prompt_tokens=1,
            final_cache_length=1,
            metrics=metrics,
            grammar_completion_token_id=EOS_TOKEN_ID,
        )
    regex = RegexGrammar("a+")
    schema = JsonSchemaGrammar('{"type":"string"}')
    with pytest.raises(FrozenInstanceError):
        regex.pattern = "b+"
    with pytest.raises(FrozenInstanceError):
        schema.schema = "{}"


def test_constrained_iterator_cancellation_releases_each_owned_layer_once():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    backend = BackendWrapper(
        FakeAutoregressiveBackend(
            ((0.0, 3.0, 0.0, 0.0), (0.0, 0.0, 3.0, 0.0), (0.0, 0.0, 0.0, 4.0))
        )
    )
    diagnostics = RecordingDiagnostics()
    metrics = TargetMetricsSession(
        clock=DeterministicMetricsClock(),
        diagnostics=diagnostics,
    )
    steps = _iterate_constrained_target(
        backend,
        (0,),
        max_new_tokens=3,
        select_token=select_highest_logit,
        grammar_context=_context(constraint),
        stop_token_sequences=(),
        metrics_session=metrics,
        expected_grammar_type=None,
        create_metrics_session=None,
    )

    first = next(steps)
    assert first.token_id == 1
    assert first.result is None
    assert constraint.active_state_count == 1
    assert diagnostics.calls == ["begin"]

    steps.close()
    steps.close()

    assert constraint.active_state_count == 0
    assert [call for call in constraint.calls if call[0] == "release_states"] == [
        ("release_states", 1)
    ]
    assert [call for call in constraint.calls if call[0] == "reset"] == [("reset", None)]
    assert backend.reset_calls == 1
    assert backend.cache_length == 0
    assert diagnostics.calls == ["begin", "abort"]


def test_constrained_iterator_settles_before_terminal_yield_and_close_is_non_destructive():
    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    backend = BackendWrapper(
        FakeAutoregressiveBackend(
            ((0.0, 3.0, 0.0, 0.0), (0.0, 0.0, 3.0, 0.0), (0.0, 0.0, 0.0, 4.0))
        )
    )
    diagnostics = RecordingDiagnostics()
    metrics = TargetMetricsSession(
        clock=DeterministicMetricsClock(),
        diagnostics=diagnostics,
    )
    steps = _iterate_constrained_target(
        backend,
        (0,),
        max_new_tokens=3,
        select_token=select_highest_logit,
        grammar_context=_context(constraint),
        stop_token_sequences=(),
        metrics_session=metrics,
        expected_grammar_type=None,
        create_metrics_session=None,
    )

    assert next(steps).result is None
    assert next(steps).result is None
    terminal = next(steps)

    assert terminal.result is not None
    assert terminal.result.token_ids == (1, 2, EOS_TOKEN_ID)
    assert constraint.active_state_count == 0
    assert [call for call in constraint.calls if call[0] == "reset"] == [("reset", None)]
    assert diagnostics.calls == ["begin", "finish"]
    assert backend.reset_calls == 0
    assert backend.cache_length == 3

    steps.close()

    assert [call for call in constraint.calls if call[0] == "reset"] == [("reset", None)]
    assert diagnostics.calls == ["begin", "finish"]
    assert backend.reset_calls == 0
    assert backend.cache_length == 3


def test_constrained_iterator_aborts_once_when_metrics_begin_partially_fails():
    class BeginFailingDiagnostics(RecordingDiagnostics):
        def begin(self):
            super().begin()
            raise RuntimeError("metrics begin failed")

    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    backend = BackendWrapper(FakeAutoregressiveBackend(((0.0, 3.0, 0.0, 0.0),)))
    diagnostics = BeginFailingDiagnostics()
    steps = _iterate_constrained_target(
        backend,
        (0,),
        max_new_tokens=1,
        select_token=select_highest_logit,
        grammar_context=_context(constraint),
        stop_token_sequences=(),
        metrics_session=TargetMetricsSession(
            clock=DeterministicMetricsClock(),
            diagnostics=diagnostics,
        ),
        expected_grammar_type=None,
        create_metrics_session=None,
    )

    with pytest.raises(RuntimeError, match="metrics begin failed"):
        next(steps)

    assert diagnostics.calls == ["begin", "abort"]
    assert constraint.active_state_count == 0
    assert [call for call in constraint.calls if call[0] == "reset"] == [("reset", None)]
    assert backend.reset_calls == 0


def test_constrained_iterator_aborts_once_when_metrics_finish_fails():
    class FinishFailingDiagnostics(RecordingDiagnostics):
        def finish(self):
            super().finish()
            raise RuntimeError("metrics finish failed")

    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"),),
        valid_token_ids=(("s0", (1,)), ("match", ())),
        match_states=frozenset({"match"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)
    backend = BackendWrapper(FakeAutoregressiveBackend(((0.0, 3.0, 0.0, 0.0),)))
    diagnostics = FinishFailingDiagnostics()
    steps = _iterate_constrained_target(
        backend,
        (0,),
        max_new_tokens=1,
        select_token=select_highest_logit,
        grammar_context=_context(constraint),
        stop_token_sequences=(),
        metrics_session=TargetMetricsSession(
            clock=DeterministicMetricsClock(),
            diagnostics=diagnostics,
        ),
        expected_grammar_type=None,
        create_metrics_session=None,
    )

    with pytest.raises(RuntimeError, match="metrics finish failed"):
        next(steps)

    assert diagnostics.calls == ["begin", "finish", "abort"]
    assert constraint.active_state_count == 0
    assert [call for call in constraint.calls if call[0] == "reset"] == [("reset", None)]
    assert backend.reset_calls == 1
    assert backend.cache_length == 0


def test_grammar_timing_finalization_failure_returns_no_result_and_resets_backend():
    class FinishFailingGrammarTimingSession(GrammarTimingSession):
        def finish(self, generated_tokens):
            raise RuntimeError("grammar timing finish failed")

    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"),),
        valid_token_ids=(("s0", (1,)), ("match", ())),
        match_states=frozenset({"match"}),
    )
    constraint = RecordingConstraint(VOCABULARY, program=program)
    backend = BackendWrapper(FakeAutoregressiveBackend(((0.0, 3.0, 0.0, 0.0),)))
    timing_session = FinishFailingGrammarTimingSession(clock=DeterministicMetricsClock())

    with pytest.raises(RuntimeError, match="grammar timing finish failed"):
        _generate(
            backend,
            _context(constraint, timing_session=timing_session),
            max_new_tokens=1,
        )

    assert backend.reset_calls == 1
    assert backend.cache_length == 0
    assert constraint.active_state_count == 0
    with pytest.raises(MetricsStateError, match="no longer open"):
        timing_session.record_mask_timing(0.0, 0.0)


def test_grammar_timing_abort_failure_joins_existing_cleanup_details_last():
    class AbortFailingGrammarTimingSession(GrammarTimingSession):
        def abort(self):
            super().abort()
            raise RuntimeError("grammar timing abort failed")

    constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    backend = BackendWrapper(FakeAutoregressiveBackend(((0.0, 3.0, 0.0, 0.0),)))
    original = RuntimeError("mask failed")
    timing_session = AbortFailingGrammarTimingSession(clock=DeterministicMetricsClock())

    with pytest.raises(ConstrainedGenerationCleanupError) as captured:
        _generate(
            backend,
            _context(
                constraint,
                RecordingMask(len(VOCABULARY), failure=original),
                timing_session=timing_session,
            ),
            max_new_tokens=1,
        )

    assert captured.value.original_failure is original
    assert [operation for operation, _ in captured.value.cleanup_failures] == [
        "grammar timing abort"
    ]
    assert "grammar timing abort failed" in str(captured.value)


def test_cancellation_aborts_timing_but_terminal_settlement_finishes_it():
    cancelled_timing = create_deterministic_grammar_timing_session()
    cancelled_constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    cancelled_steps = _iterate_constrained_target(
        BackendWrapper(
            FakeAutoregressiveBackend(
                ((0.0, 3.0, 0.0, 0.0), (0.0, 0.0, 3.0, 0.0))
            )
        ),
        (0,),
        max_new_tokens=2,
        select_token=select_highest_logit,
        grammar_context=_context(
            cancelled_constraint,
            timing_session=cancelled_timing,
        ),
        stop_token_sequences=(),
        metrics_session=create_deterministic_metrics_session(),
        expected_grammar_type=None,
        create_metrics_session=None,
    )
    next(cancelled_steps)
    cancelled_steps.close()
    with pytest.raises(MetricsStateError, match="no longer open"):
        cancelled_timing.record_mask_timing(0.0, 0.0)

    finished_timing = create_deterministic_grammar_timing_session()
    finished_constraint = RecordingConstraint(VOCABULARY, program=_path_program())
    finished_steps = _iterate_constrained_target(
        BackendWrapper(
            FakeAutoregressiveBackend(
                ((0.0, 3.0, 0.0, 0.0), (0.0, 0.0, 3.0, 0.0))
            )
        ),
        (0,),
        max_new_tokens=2,
        select_token=select_highest_logit,
        grammar_context=_context(finished_constraint, timing_session=finished_timing),
        stop_token_sequences=(),
        metrics_session=create_deterministic_metrics_session(),
        expected_grammar_type=None,
        create_metrics_session=None,
    )
    next(finished_steps)
    terminal = next(finished_steps)
    assert terminal.result is not None
    finished_steps.close()
    with pytest.raises(MetricsStateError, match="no longer open"):
        finished_timing.record_mask_timing(0.0, 0.0)
