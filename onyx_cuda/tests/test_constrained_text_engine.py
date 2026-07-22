import pytest

from onyx_cuda import (
    ConstrainedGenerationError,
    ConstrainedGenerationInvariantError,
    GrammarGenerationContext,
    JsonSchemaGrammar,
    RegexGrammar,
    SelectionError,
    TargetTextEngine,
    TemperatureTopPSelection,
    select_highest_logit,
)
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    FakeCharacterTokenizer,
    FakeGrammarConstraint,
    FakeGrammarProgram,
    create_deterministic_grammar_timing_session,
    create_deterministic_metrics_session,
)


VOCABULARY = (b"P", b"a", b"b", b"")
TOKENIZER = FakeCharacterTokenizer(("P", "a", "b", "!"))
EOS_TOKEN_ID = 3


class ReferenceMask:
    vocab_size = 4

    def apply(self, logits, valid_token_ids, /):
        return tuple(
            value if token_id in valid_token_ids else float("-inf")
            for token_id, value in enumerate(logits)
        )


def _program():
    return FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"),),
        valid_token_ids=(("s0", (1,)), ("match", ())),
        match_states=frozenset({"match"}),
    )


def _context(*, grammar_type="regex"):
    return GrammarGenerationContext(
        constraint=FakeGrammarConstraint(
            VOCABULARY,
            grammar_type=grammar_type,
            program=_program(),
        ),
        logit_mask=ReferenceMask(),
        eos_token_id=EOS_TOKEN_ID,
        timing_session=create_deterministic_grammar_timing_session(),
    )


def _engine(backend, factory, *, create_sampling_selector=None):
    return TargetTextEngine(
        backend,
        TOKENIZER,
        select_token=select_highest_logit,
        create_sampling_selector=create_sampling_selector,
        create_metrics_session=create_deterministic_metrics_session,
        create_grammar_context=factory,
    )


def test_text_engine_passes_exact_regex_specification_and_trims_completion_eos():
    backend = FakeAutoregressiveBackend(
        ((100.0, 2.0, 99.0, 0.0), (0.0, 0.0, 0.0, 5.0))
    )
    calls = []

    def factory(grammar):
        calls.append(grammar)
        return _context()

    engine = _engine(backend, factory)
    grammar = RegexGrammar("a")

    result = engine.generate_constrained(
        "P",
        grammar=grammar,
        max_new_tokens=2,
    )

    assert calls == [grammar]
    assert calls[0] is grammar
    assert result.text == "a"
    assert result.output_token_ids == (1,)
    assert result.sampled_token_ids == (1, EOS_TOKEN_ID)
    assert result.generation.finish_reason == "grammar_complete"


def test_context_and_state_are_created_before_the_metrics_session_begins():
    backend = FakeAutoregressiveBackend(
        ((0.0, 2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 3.0))
    )
    events = []

    class OrderedConstraint(FakeGrammarConstraint):
        def init_state(self):
            events.append("state")
            return super().init_state()

    def factory(_grammar):
        events.append("context")
        return GrammarGenerationContext(
            constraint=OrderedConstraint(
                VOCABULARY,
                grammar_type="regex",
                program=_program(),
            ),
            logit_mask=ReferenceMask(),
            eos_token_id=EOS_TOKEN_ID,
            timing_session=create_deterministic_grammar_timing_session(),
        )

    def create_metrics():
        events.append("metrics")
        return create_deterministic_metrics_session()

    engine = TargetTextEngine(
        backend,
        TOKENIZER,
        select_token=select_highest_logit,
        create_metrics_session=create_metrics,
        create_grammar_context=factory,
    )

    engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    )

    assert events[:3] == ["context", "state", "metrics"]


def test_metrics_factory_failure_releases_the_already_owned_context():
    backend = FakeAutoregressiveBackend(((0.0, 2.0, 0.0, 0.0),))
    context = _context()

    def fail_metrics():
        raise RuntimeError("metrics creation failed")

    engine = TargetTextEngine(
        backend,
        TOKENIZER,
        select_token=select_highest_logit,
        create_metrics_session=fail_metrics,
        create_grammar_context=lambda _grammar: context,
    )

    with pytest.raises(RuntimeError, match="metrics creation failed"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=1,
        )

    assert context.constraint.active_state_count == 0
    assert backend.cache_length == 0


def test_text_engine_passes_exact_json_specification_to_json_constraint_factory():
    backend = FakeAutoregressiveBackend(
        ((0.0, 2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 3.0))
    )
    calls = []

    def factory(grammar):
        calls.append(grammar)
        return _context(grammar_type="json_schema")

    grammar = JsonSchemaGrammar('{"const":"a"}')
    result = _engine(backend, factory).generate_constrained(
        "P",
        grammar=grammar,
        max_new_tokens=2,
    )

    assert calls == [grammar]
    assert calls[0] is grammar
    assert result.text == "a"


def test_missing_grammar_factory_is_typed_and_fails_before_prefill():
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 0.0, 0.0),))
    engine = TargetTextEngine(
        backend,
        TOKENIZER,
        select_token=select_highest_logit,
        create_metrics_session=create_deterministic_metrics_session,
    )

    with pytest.raises(ConstrainedGenerationError, match="not configured"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=1,
        )

    assert backend.cache_length == 0


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"prompt": "", "max_new_tokens": 1}, ValueError),
        ({"prompt": "P", "max_new_tokens": 0}, ValueError),
        (
            {"prompt": "P", "max_new_tokens": 1, "stop_token_sequences": ((),)},
            ValueError,
        ),
        ({"prompt": "P", "max_new_tokens": 1, "selection": object()}, TypeError),
    ],
)
def test_invalid_requests_do_not_create_a_grammar_context(kwargs, error_type):
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 0.0, 0.0),))
    calls = []
    engine = _engine(backend, lambda grammar: calls.append(grammar))

    with pytest.raises(error_type):
        engine.generate_constrained(grammar=RegexGrammar("a"), **kwargs)

    assert calls == []
    assert backend.cache_length == 0


def test_invalid_encoded_prompt_does_not_create_a_grammar_context():
    class InvalidPromptTokenizer:
        tokenizer_id = "invalid-prompt"
        vocab_size = 4

        def encode(self, _text, /):
            return (4,)

        def decode(self, _token_ids, /):
            return ""

    backend = FakeAutoregressiveBackend(((0.0, 1.0, 0.0, 0.0),))
    calls = []
    engine = TargetTextEngine(
        backend,
        InvalidPromptTokenizer(),
        select_token=select_highest_logit,
        create_metrics_session=create_deterministic_metrics_session,
        create_grammar_context=lambda grammar: calls.append(grammar),
    )

    with pytest.raises(ValueError, match="outside vocabulary"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=1,
        )

    assert calls == []
    assert backend.cache_length == 0


def test_seeded_selection_uses_fresh_sessions_and_fresh_constraints():
    backend = FakeAutoregressiveBackend(
        ((0.0, 2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 3.0))
    )
    policies = []
    constraints = []

    def create_selector(policy):
        policies.append(policy)
        return select_highest_logit

    def factory(_grammar):
        context = _context()
        constraints.append(context.constraint)
        return context

    engine = _engine(backend, factory, create_sampling_selector=create_selector)
    policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=17)

    first = engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
        selection=policy,
    )
    second = engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
        selection=policy,
    )

    assert first == second
    assert policies == [policy, policy]
    assert len(constraints) == 2
    assert constraints[0] is not constraints[1]
    assert all(constraint.active_state_count == 0 for constraint in constraints)


def test_sampling_configuration_failure_precedes_grammar_compilation():
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 0.0, 0.0),))
    calls = []
    engine = _engine(backend, lambda grammar: calls.append(grammar))

    with pytest.raises(SelectionError, match="not configured"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=1,
            selection=TemperatureTopPSelection(1.0, 1.0, 0),
        )

    assert calls == []
    assert backend.cache_length == 0


def test_returned_constraint_type_must_match_the_specification_and_is_reset():
    backend = FakeAutoregressiveBackend(((0.0, 1.0, 0.0, 0.0),))
    context = _context(grammar_type="json_schema")
    engine = _engine(backend, lambda _grammar: context)

    with pytest.raises(ConstrainedGenerationInvariantError, match="does not match"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=1,
        )

    assert backend.cache_length == 0
    assert context.constraint.active_state_count == 0


def test_text_result_failure_resets_an_otherwise_successful_backend_sequence():
    class FailingDecodeTokenizer(FakeCharacterTokenizer):
        def decode(self, token_ids, /):
            raise RuntimeError("decode failed")

    tokenizer = FailingDecodeTokenizer(("P", "a", "b", "!"))
    backend = FakeAutoregressiveBackend(
        ((0.0, 2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 3.0))
    )
    contexts = []

    def factory(_grammar):
        context = _context()
        contexts.append(context)
        return context

    engine = TargetTextEngine(
        backend,
        tokenizer,
        select_token=select_highest_logit,
        create_metrics_session=create_deterministic_metrics_session,
        create_grammar_context=factory,
    )

    with pytest.raises(RuntimeError, match="decode failed"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=2,
        )

    assert backend.cache_length == 0
    assert contexts[0].constraint.active_state_count == 0
