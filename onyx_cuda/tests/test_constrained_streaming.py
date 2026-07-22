import math

import pytest

from onyx_cuda import (
    ConstrainedGenerationError,
    GrammarGenerationContext,
    JsonSchemaGrammar,
    RegexGrammar,
    StreamingInvariantError,
    TargetMetricsSession,
    TargetTextEngine,
    TemperatureTopPSelection,
    TextGenerationComplete,
    TextGenerationDelta,
    select_highest_logit,
)
from onyx_cuda.testing import (
    DeterministicMetricsClock,
    FakeAutoregressiveBackend,
    FakeCharacterTokenizer,
    FakeGrammarConstraint,
    FakeGrammarProgram,
    create_deterministic_grammar_timing_session,
    create_deterministic_metrics_session,
)


VOCABULARY = (b"P", b"a", b"b", b"c", b"x", b"")
TOKENIZER = FakeCharacterTokenizer(("P", "a", "b", "c", "x", "!"))
EOS_TOKEN_ID = 5


class ReferenceMask:
    def __init__(self, vocab_size=len(VOCABULARY)):
        self._vocab_size = vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size

    def apply(self, logits, valid_token_ids, /):
        return tuple(
            value if token_id in valid_token_ids else -math.inf
            for token_id, value in enumerate(logits)
        )


class BackendWrapper:
    def __init__(self, backend):
        self.backend = backend
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

    @property
    def cache_mode(self):
        return self.backend.cache_mode

    def prefill(self, prompt_token_ids, /):
        return self.backend.prefill(prompt_token_ids)

    def decode(self, token_id, /):
        return self.backend.decode(token_id)

    def reset(self):
        self.reset_calls += 1
        self.backend.reset()


class RecordingConstraint(FakeGrammarConstraint):
    def __init__(self, vocabulary, *, grammar_type, program):
        self._record_resets = False
        self.reset_calls = 0
        self.bulk_release_calls = 0
        super().__init__(vocabulary, grammar_type=grammar_type, program=program)
        self._record_resets = True

    def release_states(self, states, /):
        self.bulk_release_calls += 1
        return super().release_states(states)

    def reset(self):
        super().reset()
        if self._record_resets:
            self.reset_calls += 1


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


def _path_program():
    return FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"), ("s1", 2, "match")),
        valid_token_ids=(("s0", (1,)), ("s1", (2,)), ("match", ())),
        match_states=frozenset({"match"}),
    )


def _path_logits():
    return (
        (0.0, 9.0, 8.0, 7.0, 6.0, 1.0),
        (0.0, 1.0, 9.0, 8.0, 7.0, 2.0),
        (0.0, 1.0, 2.0, 3.0, 4.0, 9.0),
    )


def _grammar_type(grammar):
    if isinstance(grammar, RegexGrammar):
        return "regex"
    if isinstance(grammar, JsonSchemaGrammar):
        return "json_schema"
    raise TypeError("unsupported test grammar")


def _make_engine(
    program,
    *,
    backend=None,
    tokenizer=TOKENIZER,
    create_sampling_selector=None,
    create_metrics_session=create_deterministic_metrics_session,
):
    if backend is None:
        backend = FakeAutoregressiveBackend(_path_logits())
    contexts = []
    factory_calls = []

    def create_context(grammar):
        factory_calls.append(grammar)
        constraint = RecordingConstraint(
            VOCABULARY,
            grammar_type=_grammar_type(grammar),
            program=program,
        )
        context = GrammarGenerationContext(
            constraint=constraint,
            logit_mask=ReferenceMask(),
            eos_token_id=EOS_TOKEN_ID,
            timing_session=create_deterministic_grammar_timing_session(),
        )
        contexts.append(context)
        return context

    engine = TargetTextEngine(
        backend,
        tokenizer,
        select_token=select_highest_logit,
        create_sampling_selector=create_sampling_selector,
        create_metrics_session=create_metrics_session,
        create_grammar_context=create_context,
    )
    return engine, contexts, factory_calls


def _collect(stream):
    events = list(stream)
    completions = [event for event in events if isinstance(event, TextGenerationComplete)]
    deltas = [event for event in events if isinstance(event, TextGenerationDelta)]
    assert len(completions) == 1
    assert events[-1] is completions[0]
    assert all(event.text for event in deltas)
    return events, "".join(event.text for event in deltas), completions[0].result


@pytest.mark.parametrize(
    "grammar",
    [RegexGrammar("ab"), JsonSchemaGrammar('{"const":"ab"}')],
)
def test_regex_and_json_streams_are_exactly_equivalent_to_non_streaming(grammar):
    engine, contexts, factory_calls = _make_engine(_path_program())

    expected = engine.generate_constrained(
        "P",
        grammar=grammar,
        max_new_tokens=3,
    )
    events, delta_text, streamed = _collect(
        engine.stream_constrained(
            "P",
            grammar=grammar,
            max_new_tokens=3,
        )
    )

    assert delta_text == expected.text == "ab"
    assert streamed == expected
    assert streamed.sampled_token_ids == (1, 2, EOS_TOKEN_ID)
    assert streamed.output_token_ids == (1, 2)
    assert streamed.generation.finish_reason == "grammar_complete"
    assert streamed.generation.final_cache_length == 3
    assert sum(isinstance(event, TextGenerationComplete) for event in events) == 1
    assert factory_calls == [grammar, grammar]
    assert len(contexts) == 2
    assert contexts[0].constraint is not contexts[1].constraint
    assert all(context.constraint.active_state_count == 0 for context in contexts)


def test_seeded_streams_create_fresh_selectors_and_constraints_and_replay_exactly():
    policies = []

    def create_selector(policy):
        policies.append(policy)
        return select_highest_logit

    engine, contexts, _calls = _make_engine(
        _path_program(),
        create_sampling_selector=create_selector,
    )
    grammar = RegexGrammar("ab")
    policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=17)

    first = _collect(
        engine.stream_constrained(
            "P",
            grammar=grammar,
            max_new_tokens=3,
            selection=policy,
        )
    )
    second = _collect(
        engine.stream_constrained(
            "P",
            grammar=grammar,
            max_new_tokens=3,
            selection=policy,
        )
    )

    assert first == second
    assert policies == [policy, policy]
    assert len(contexts) == 2
    assert contexts[0].constraint is not contexts[1].constraint


def test_initial_match_eos_yields_only_empty_completion_and_never_reaches_decoder():
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=(),
        valid_token_ids=(("match", ()),),
        match_states=frozenset({"match"}),
    )
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 2.0, 3.0, 4.0, 9.0),))
    engine, _contexts, _calls = _make_engine(program, backend=backend)

    events, delta_text, result = _collect(
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("a*"),
            max_new_tokens=1,
        )
    )

    assert len(events) == 1
    assert delta_text == result.text == ""
    assert result.output_token_ids == ()
    assert result.sampled_token_ids == (EOS_TOKEN_ID,)
    assert result.generation.finish_reason == "grammar_complete"


def test_matching_state_can_choose_visible_content_and_length_remains_length():
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=(("match", 1, "extended"),),
        valid_token_ids=(("match", (1,)), ("extended", ())),
        match_states=frozenset({"match", "extended"}),
    )
    backend = FakeAutoregressiveBackend(((0.0, 9.0, 2.0, 3.0, 4.0, 1.0),))
    engine, _contexts, _calls = _make_engine(program, backend=backend)

    _events, delta_text, result = _collect(
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("a*"),
            max_new_tokens=1,
        )
    )

    assert delta_text == result.text == "a"
    assert result.generation.finish_reason == "length"
    assert result.sampled_token_ids == (1,)


def test_eligible_multi_token_stop_is_buffered_and_hidden_at_terminal_boundary():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"), ("match", 2, "s2"), ("s2", 3, "s3")),
        valid_token_ids=(("s0", (1,)), ("match", (2,)), ("s2", (3,)), ("s3", ())),
        match_states=frozenset({"match", "s3"}),
    )
    backend = FakeAutoregressiveBackend(
        (
            (0.0, 9.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 9.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 9.0, 0.0, 0.0),
        )
    )
    engine, _contexts, _calls = _make_engine(program, backend=backend)

    events, delta_text, result = _collect(
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("abc"),
            max_new_tokens=3,
            stop_token_sequences=((2, 3),),
        )
    )

    assert [event.text for event in events if isinstance(event, TextGenerationDelta)] == ["a"]
    assert delta_text == result.text == "a"
    assert result.sampled_token_ids == (1, 2, 3)
    assert result.generation.finish_reason == "stop"
    assert result.generation.matched_stop_token_ids == (2, 3)


def test_grammar_ineligible_stop_suffix_is_flushed_as_visible_length_content():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"),),
        valid_token_ids=(("s0", (1,)), ("match", ())),
        match_states=frozenset({"match"}),
    )
    backend = FakeAutoregressiveBackend(((0.0, 9.0, 0.0, 0.0, 0.0, 0.0),))
    engine, _contexts, _calls = _make_engine(program, backend=backend)

    _events, delta_text, result = _collect(
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=1,
            stop_token_sequences=((1,),),
        )
    )

    assert delta_text == result.text == "a"
    assert result.generation.finish_reason == "length"
    assert result.generation.matched_stop_token_ids is None


def test_incomplete_stop_prefix_is_flushed_at_a_matching_length_boundary():
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=(("match", 2, "extended"),),
        valid_token_ids=(("match", (2,)), ("extended", ())),
        match_states=frozenset({"match", "extended"}),
    )
    backend = FakeAutoregressiveBackend(((0.0, 0.0, 9.0, 0.0, 0.0, 1.0),))
    engine, _contexts, _calls = _make_engine(program, backend=backend)

    _events, delta_text, result = _collect(
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("b*"),
            max_new_tokens=1,
            stop_token_sequences=((2, 3),),
        )
    )

    assert delta_text == result.text == "b"
    assert result.generation.finish_reason == "length"


def test_configured_eos_stop_precedes_grammar_completion_and_remains_hidden():
    program = FakeGrammarProgram(
        initial_state="match",
        transitions=(),
        valid_token_ids=(("match", ()),),
        match_states=frozenset({"match"}),
    )
    backend = FakeAutoregressiveBackend(((0.0, 0.0, 0.0, 0.0, 0.0, 9.0),))
    engine, _contexts, _calls = _make_engine(program, backend=backend)

    events, delta_text, result = _collect(
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("a*"),
            max_new_tokens=1,
            stop_token_sequences=((EOS_TOKEN_ID,),),
        )
    )

    assert len(events) == 1
    assert delta_text == result.text == ""
    assert result.generation.finish_reason == "stop"
    assert result.generation.matched_stop_token_ids == (EOS_TOKEN_ID,)


class SplitUtf8Tokenizer:
    tokenizer_id = "split-utf8"
    vocab_size = 4

    def encode(self, text, /):
        if text != "P":
            raise ValueError("unexpected prompt")
        return (0,)

    def decode(self, token_ids, /):
        values = tuple(token_ids)
        if values == ():
            return ""
        if values == (1,):
            return "\ufffd"
        if values == (1, 2):
            return "é"
        raise ValueError(f"unexpected token IDs: {values}")


def test_split_utf8_remains_stable_and_completion_eos_is_never_decoded():
    vocabulary = (b"P", b"\xc3", b"\xa9", b"")
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"), ("s1", 2, "match")),
        valid_token_ids=(("s0", (1,)), ("s1", (2,)), ("match", ())),
        match_states=frozenset({"match"}),
    )
    contexts = []

    def create_context(_grammar):
        context = GrammarGenerationContext(
            constraint=RecordingConstraint(
                vocabulary,
                grammar_type="regex",
                program=program,
            ),
            logit_mask=ReferenceMask(4),
            eos_token_id=3,
            timing_session=create_deterministic_grammar_timing_session(),
        )
        contexts.append(context)
        return context

    engine = TargetTextEngine(
        FakeAutoregressiveBackend(
            ((0.0, 9.0, 0.0, 0.0), (0.0, 0.0, 9.0, 0.0), (0.0, 0.0, 0.0, 9.0))
        ),
        SplitUtf8Tokenizer(),
        select_token=select_highest_logit,
        create_metrics_session=create_deterministic_metrics_session,
        create_grammar_context=create_context,
    )

    events, delta_text, result = _collect(
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("é"),
            max_new_tokens=3,
        )
    )

    assert [event.text for event in events if isinstance(event, TextGenerationDelta)] == ["é"]
    assert delta_text == result.text == "é"
    assert result.output_token_ids == (1, 2)
    assert result.sampled_token_ids == (1, 2, 3)
    assert contexts[0].constraint.active_state_count == 0


def test_never_started_stream_is_context_metrics_and_backend_lazy():
    metrics_calls = []

    def create_metrics():
        metrics_calls.append("metrics")
        return create_deterministic_metrics_session()

    backend = BackendWrapper(FakeAutoregressiveBackend(_path_logits()))
    engine, contexts, factory_calls = _make_engine(
        _path_program(),
        backend=backend,
        create_metrics_session=create_metrics,
    )

    stream = engine.stream_constrained(
        "P",
        grammar=RegexGrammar("ab"),
        max_new_tokens=3,
    )
    assert contexts == []
    assert factory_calls == []
    assert metrics_calls == []

    stream.close()

    assert contexts == []
    assert factory_calls == []
    assert metrics_calls == []
    assert backend.reset_calls == 0
    assert backend.cache_length == 0


def test_cancellation_after_nonterminal_delta_aborts_and_leaves_engine_reusable():
    diagnostics = []

    def create_metrics():
        session_diagnostics = RecordingDiagnostics()
        diagnostics.append(session_diagnostics)
        return TargetMetricsSession(
            clock=DeterministicMetricsClock(),
            diagnostics=session_diagnostics,
        )

    backend = BackendWrapper(FakeAutoregressiveBackend(_path_logits()))
    engine, contexts, _calls = _make_engine(
        _path_program(),
        backend=backend,
        create_metrics_session=create_metrics,
    )
    grammar = RegexGrammar("ab")
    stream = engine.stream_constrained("P", grammar=grammar, max_new_tokens=3)

    first = next(stream)
    assert first == TextGenerationDelta("a")
    assert contexts[0].constraint.active_state_count == 1

    stream.close()
    stream.close()

    assert contexts[0].constraint.active_state_count == 0
    assert contexts[0].constraint.bulk_release_calls == 1
    assert contexts[0].constraint.reset_calls == 1
    assert backend.reset_calls == 1
    assert diagnostics[0].calls == ["begin", "abort"]

    result = engine.generate_constrained("P", grammar=grammar, max_new_tokens=3)
    assert result.text == "ab"
    assert contexts[1].constraint.active_state_count == 0
    assert diagnostics[1].calls == ["begin", "finish"]


def test_close_after_terminal_delta_resets_only_undelivered_backend_result():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"),),
        valid_token_ids=(("s0", (1,)), ("match", ())),
        match_states=frozenset({"match"}),
    )
    diagnostics = []

    def create_metrics():
        session_diagnostics = RecordingDiagnostics()
        diagnostics.append(session_diagnostics)
        return TargetMetricsSession(
            clock=DeterministicMetricsClock(),
            diagnostics=session_diagnostics,
        )

    backend = BackendWrapper(
        FakeAutoregressiveBackend(((0.0, 9.0, 0.0, 0.0, 0.0, 0.0),))
    )
    engine, contexts, _calls = _make_engine(
        program,
        backend=backend,
        create_metrics_session=create_metrics,
    )
    grammar = RegexGrammar("a")
    stream = engine.stream_constrained("P", grammar=grammar, max_new_tokens=1)

    assert next(stream) == TextGenerationDelta("a")
    assert contexts[0].constraint.active_state_count == 0
    assert contexts[0].constraint.reset_calls == 1
    assert diagnostics[0].calls == ["begin", "finish"]

    stream.close()

    assert contexts[0].constraint.reset_calls == 1
    assert diagnostics[0].calls == ["begin", "finish"]
    assert backend.reset_calls == 1
    assert backend.cache_length == 0
    assert engine.generate_constrained("P", grammar=grammar, max_new_tokens=1).text == "a"


class FailOnSecondDecodeTokenizer(FakeCharacterTokenizer):
    def __init__(self):
        super().__init__(("P", "a", "b", "c", "x", "!"))
        self.decode_calls = 0

    def decode(self, token_ids, /):
        self.decode_calls += 1
        if self.decode_calls == 2:
            raise RuntimeError("final decode failed")
        return super().decode(token_ids)


def test_final_decode_failure_resets_backend_without_double_grammar_or_metrics_cleanup():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"),),
        valid_token_ids=(("s0", (1,)), ("match", ())),
        match_states=frozenset({"match"}),
    )
    diagnostics = []

    def create_metrics():
        session_diagnostics = RecordingDiagnostics()
        diagnostics.append(session_diagnostics)
        return TargetMetricsSession(
            clock=DeterministicMetricsClock(),
            diagnostics=session_diagnostics,
        )

    backend = BackendWrapper(
        FakeAutoregressiveBackend(
            ((0.0, 9.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 9.0))
        )
    )
    engine, contexts, _calls = _make_engine(
        program,
        backend=backend,
        tokenizer=FailOnSecondDecodeTokenizer(),
        create_metrics_session=create_metrics,
    )
    stream = engine.stream_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    )

    assert next(stream) == TextGenerationDelta("a")
    with pytest.raises(RuntimeError, match="final decode failed"):
        next(stream)

    assert contexts[0].constraint.active_state_count == 0
    assert contexts[0].constraint.reset_calls == 1
    assert diagnostics[0].calls == ["begin", "finish"]
    assert backend.reset_calls == 1
    assert backend.cache_length == 0


def test_context_factory_is_deferred_until_first_next_and_failure_is_retryable():
    backend = FakeAutoregressiveBackend(_path_logits())
    calls = []
    program = _path_program()

    def create_context(grammar):
        calls.append(grammar)
        if len(calls) == 1:
            raise RuntimeError("compile failed")
        return GrammarGenerationContext(
            constraint=RecordingConstraint(
                VOCABULARY,
                grammar_type="regex",
                program=program,
            ),
            logit_mask=ReferenceMask(),
            eos_token_id=EOS_TOKEN_ID,
            timing_session=create_deterministic_grammar_timing_session(),
        )

    engine = TargetTextEngine(
        backend,
        TOKENIZER,
        select_token=select_highest_logit,
        create_metrics_session=create_deterministic_metrics_session,
        create_grammar_context=create_context,
    )
    grammar = RegexGrammar("ab")
    failed = engine.stream_constrained("P", grammar=grammar, max_new_tokens=3)
    assert calls == []

    with pytest.raises(RuntimeError, match="compile failed"):
        next(failed)

    assert backend.cache_length == 0
    _events, _text, result = _collect(
        engine.stream_constrained("P", grammar=grammar, max_new_tokens=3)
    )
    assert result.text == "ab"
    assert calls == [grammar, grammar]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"prompt": "", "max_new_tokens": 1}, ValueError),
        ({"prompt": "P", "max_new_tokens": 0}, ValueError),
        ({"prompt": "P", "max_new_tokens": 1, "stop_token_sequences": ((),)}, ValueError),
        ({"prompt": "P", "max_new_tokens": 1, "selection": object()}, TypeError),
    ],
)
def test_invalid_stream_request_fails_synchronously_without_creating_context(kwargs, error_type):
    engine, contexts, calls = _make_engine(_path_program())

    with pytest.raises(error_type):
        engine.stream_constrained(grammar=RegexGrammar("ab"), **kwargs)

    assert contexts == []
    assert calls == []


def test_missing_grammar_factory_fails_synchronously_before_stream_creation():
    backend = FakeAutoregressiveBackend(_path_logits())
    engine = TargetTextEngine(
        backend,
        TOKENIZER,
        select_token=select_highest_logit,
        create_metrics_session=create_deterministic_metrics_session,
    )

    with pytest.raises(ConstrainedGenerationError, match="not configured"):
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("ab"),
            max_new_tokens=3,
        )

    assert backend.cache_length == 0


def test_non_monotonic_constrained_decode_is_typed_and_resets_backend():
    class NonMonotonicTokenizer(FakeCharacterTokenizer):
        def decode(self, token_ids, /):
            values = tuple(token_ids)
            if values == (1,):
                return "a"
            if values == (1, 2):
                return "x"
            return super().decode(values)

    backend = BackendWrapper(FakeAutoregressiveBackend(_path_logits()))
    engine, contexts, _calls = _make_engine(
        _path_program(),
        backend=backend,
        tokenizer=NonMonotonicTokenizer(("P", "a", "b", "c", "x", "!")),
    )
    stream = engine.stream_constrained(
        "P",
        grammar=RegexGrammar("ab"),
        max_new_tokens=3,
    )

    assert next(stream) == TextGenerationDelta("a")
    with pytest.raises(StreamingInvariantError, match="changed text"):
        next(stream)

    assert contexts[0].constraint.active_state_count == 0
    assert backend.reset_calls == 1
    assert backend.cache_length == 0
