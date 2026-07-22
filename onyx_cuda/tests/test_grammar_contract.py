from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import (
    GrammarCompilationError,
    GrammarCompiler,
    GrammarConstraint,
    GrammarStateError,
)
from onyx_cuda.testing import (
    FakeGrammarCompiler,
    FakeGrammarConstraint,
    FakeGrammarProgram,
    FakeGrammarStateHandle,
)


VOCABULARY = (b"{", b'"a"', b'"b"', b":", b'"', b"1")
JSON_SCHEMA = (
    '{"type":"object","properties":{"a":{"type":"string"},'
    '"b":{"type":"number"}}}'
)
REGEX_PATTERN = "(?:a|b):"


def branching_program():
    return FakeGrammarProgram(
        initial_state="initial",
        transitions=(
            ("initial", 1, "after_a"),
            ("initial", 2, "after_b"),
            ("after_a", 3, "a_value"),
            ("after_b", 3, "b_value"),
            ("a_value", 4, "matched"),
            ("b_value", 5, "matched"),
        ),
        valid_token_ids=(
            ("initial", (2, 1)),
            ("after_a", (3,)),
            ("after_b", (3,)),
            ("a_value", (4,)),
            ("b_value", (5,)),
            ("matched", ()),
        ),
        match_states=frozenset({"matched"}),
    )


def dead_program():
    return FakeGrammarProgram(
        initial_state="dead",
        transitions=(),
        valid_token_ids=(("dead", ()),),
        dead_states=frozenset({"dead"}),
    )


def make_compiler():
    program = branching_program()
    return FakeGrammarCompiler(
        regex_programs={REGEX_PATTERN: program},
        json_schema_programs={JSON_SCHEMA: program},
    )


def test_fake_compiler_and_constraint_satisfy_framework_neutral_protocols():
    compiler = make_compiler()
    constraint = compiler.compile_regex(VOCABULARY, REGEX_PATTERN)

    assert isinstance(compiler, GrammarCompiler)
    assert isinstance(constraint, GrammarConstraint)
    assert isinstance(constraint, FakeGrammarConstraint)
    assert constraint.vocab_size == len(VOCABULARY)
    assert constraint.grammar_type == "regex"


def test_regex_and_json_factories_return_fresh_constraints():
    compiler = make_compiler()

    first = compiler.compile_regex(VOCABULARY, REGEX_PATTERN)
    second = compiler.compile_regex(VOCABULARY, REGEX_PATTERN)
    json_constraint = compiler.compile_json_schema(VOCABULARY, JSON_SCHEMA)

    assert first is not second
    assert first.init_state().owner_id != second.init_state().owner_id
    assert json_constraint.grammar_type == "json_schema"


def test_state_advance_branches_without_mutating_parent_and_sorts_valid_ids():
    constraint = make_compiler().compile_json_schema(VOCABULARY, JSON_SCHEMA)
    initial = constraint.init_state()

    after_a = constraint.advance_state(initial, 1)
    after_b = constraint.advance_state(initial, 2)

    assert constraint.get_valid_token_ids(initial) == (1, 2)
    assert constraint.get_valid_token_ids(after_a) == (3,)
    assert constraint.get_valid_token_ids(after_b) == (3,)
    assert initial != after_a != after_b


def test_independent_branches_preserve_typed_value_constraints_and_match_state():
    constraint = make_compiler().compile_json_schema(VOCABULARY, JSON_SCHEMA)
    initial = constraint.init_state()
    after_a_colon = constraint.advance_state(constraint.advance_state(initial, 1), 3)
    after_b_colon = constraint.advance_state(constraint.advance_state(initial, 2), 3)

    assert constraint.get_valid_token_ids(after_a_colon) == (4,)
    assert constraint.get_valid_token_ids(after_b_colon) == (5,)
    matched_a = constraint.advance_state(after_a_colon, 4)
    matched_b = constraint.advance_state(after_b_colon, 5)
    assert constraint.is_match_state(matched_a)
    assert constraint.is_match_state(matched_b)
    assert not constraint.is_dead_state(matched_a)


def test_explicit_dead_state_is_reported_without_valid_continuations():
    compiler = FakeGrammarCompiler(regex_programs={"dead": dead_program()})
    constraint = compiler.compile_regex((b"x",), "dead")
    state = constraint.init_state()

    assert constraint.get_valid_token_ids(state) == ()
    assert constraint.is_dead_state(state)
    assert not constraint.is_match_state(state)


def test_explicit_regex_dead_transitions_create_children_without_changing_parent():
    program = FakeGrammarProgram(
        initial_state="initial",
        transitions=(("initial", 0, "matched"),),
        valid_token_ids=(("initial", (0,)), ("matched", ()), ("dead", ())),
        match_states=frozenset({"matched"}),
        dead_states=frozenset({"dead"}),
        dead_transitions=(("initial", 1, "dead"), ("dead", 0, "dead")),
    )
    constraint = FakeGrammarCompiler(regex_programs={"a": program}).compile_regex(
        (b"a", b"x"),
        "a",
    )
    initial = constraint.init_state()

    dead = constraint.advance_state(initial, 1)
    dead_child = constraint.advance_state(dead, 0)

    assert constraint.get_valid_token_ids(initial) == (0,)
    assert constraint.is_dead_state(dead)
    assert constraint.is_dead_state(dead_child)
    assert dead != dead_child


def test_unregistered_fake_transitions_still_raise_after_dead_mode_is_enabled():
    program = FakeGrammarProgram(
        initial_state="initial",
        transitions=(("initial", 0, "matched"),),
        valid_token_ids=(("initial", (0,)), ("matched", ()), ("dead", ())),
        match_states=frozenset({"matched"}),
        dead_states=frozenset({"dead"}),
        dead_transitions=(("initial", 1, "dead"),),
    )
    constraint = FakeGrammarCompiler(regex_programs={"a": program}).compile_regex(
        (b"a", b"x", b"other"),
        "a",
    )
    state = constraint.init_state()

    with pytest.raises(GrammarStateError, match="not valid"):
        constraint.advance_state(state, 2)


def test_release_is_idempotent_and_released_states_fail_queries_and_advance():
    constraint = make_compiler().compile_regex(VOCABULARY, REGEX_PATTERN)
    initial = constraint.init_state()
    child = constraint.advance_state(initial, 1)

    constraint.release_state(child)
    constraint.release_state(child)

    with pytest.raises(GrammarStateError, match="unknown or has been released"):
        constraint.get_valid_token_ids(child)
    with pytest.raises(GrammarStateError, match="unknown or has been released"):
        constraint.advance_state(child, 3)
    assert constraint.get_valid_token_ids(initial) == (1, 2)


def test_bulk_release_is_atomic_for_cross_constraint_handles():
    first = make_compiler().compile_regex(VOCABULARY, REGEX_PATTERN)
    second = make_compiler().compile_regex(VOCABULARY, REGEX_PATTERN)
    first_state = first.init_state()
    second_state = second.init_state()

    with pytest.raises(GrammarStateError, match="another constraint"):
        first.release_states((first_state, second_state))

    assert first.get_valid_token_ids(first_state) == (1, 2)


def test_reset_invalidates_existing_handles_and_restarts_local_handle_values():
    constraint = make_compiler().compile_regex(VOCABULARY, REGEX_PATTERN)
    old = constraint.init_state()

    constraint.reset()
    new = constraint.init_state()

    assert new.value == old.value == 1
    assert new.epoch != old.epoch
    with pytest.raises(GrammarStateError, match="unknown or has been released"):
        constraint.is_match_state(old)
    constraint.release_state(old)


@pytest.mark.parametrize("token_id", [True, 1.5, "1"])
def test_advance_rejects_non_integer_token_ids(token_id):
    constraint = make_compiler().compile_regex(VOCABULARY, REGEX_PATTERN)
    state = constraint.init_state()

    with pytest.raises(TypeError, match="token ID must be an integer"):
        constraint.advance_state(state, token_id)


def test_advance_rejects_out_of_range_and_invalid_continuations():
    constraint = make_compiler().compile_regex(VOCABULARY, REGEX_PATTERN)
    state = constraint.init_state()

    with pytest.raises(ValueError, match="outside vocabulary"):
        constraint.advance_state(state, len(VOCABULARY))
    with pytest.raises(GrammarStateError, match="not valid"):
        constraint.advance_state(state, 0)


@pytest.mark.parametrize(
    ("vocabulary", "error", "message"),
    [
        ((), ValueError, "cannot be empty"),
        ((b"a", "b"), TypeError, "entry 1 must be bytes"),
        (None, TypeError, "sequence of bytes"),
    ],
)
def test_compiler_rejects_invalid_vocabularies(vocabulary, error, message):
    with pytest.raises(error, match=message):
        make_compiler().compile_regex(vocabulary, REGEX_PATTERN)


@pytest.mark.parametrize(
    ("compile_method", "source", "error", "message"),
    [
        ("compile_regex", "", ValueError, "cannot be empty"),
        ("compile_regex", None, TypeError, "must be a string"),
        ("compile_json_schema", "", ValueError, "cannot be empty"),
        ("compile_json_schema", "not json", ValueError, "valid JSON"),
        ("compile_json_schema", '{"type":NaN}', ValueError, "valid JSON"),
        ("compile_json_schema", "[]", ValueError, "root must be an object"),
        ("compile_json_schema", None, TypeError, "must be a string"),
    ],
)
def test_compiler_rejects_invalid_constraint_source(
    compile_method,
    source,
    error,
    message,
):
    compiler = make_compiler()

    with pytest.raises(error, match=message):
        getattr(compiler, compile_method)(VOCABULARY, source)


def test_compiler_reports_unregistered_valid_source():
    compiler = FakeGrammarCompiler()

    with pytest.raises(GrammarCompilationError, match="no fake regex program"):
        compiler.compile_regex(VOCABULARY, "a")
    with pytest.raises(GrammarCompilationError, match="no fake json_schema program"):
        compiler.compile_json_schema(VOCABULARY, '{"type":"string"}')


def test_program_rejects_inconsistent_transition_and_valid_token_sets():
    with pytest.raises(ValueError, match="is not valid"):
        FakeGrammarProgram(
            initial_state="initial",
            transitions=(("initial", 1, "matched"),),
            valid_token_ids=(("initial", ()),),
        )
    with pytest.raises(ValueError, match="advertised as valid"):
        FakeGrammarProgram(
            initial_state="initial",
            transitions=(),
            valid_token_ids=(("initial", (1,)), ("dead", ())),
            dead_states=frozenset({"dead"}),
            dead_transitions=(("initial", 1, "dead"),),
        )
    with pytest.raises(ValueError, match="must target a dead state"):
        FakeGrammarProgram(
            initial_state="initial",
            transitions=(("initial", 0, "matched"),),
            valid_token_ids=(("initial", (0,)), ("matched", ())),
            match_states=frozenset({"matched"}),
            dead_transitions=(("initial", 1, "matched"),),
        )
    with pytest.raises(ValueError, match="has no transition"):
        FakeGrammarProgram(
            initial_state="initial",
            transitions=(),
            valid_token_ids=(("initial", (1,)),),
        )
    with pytest.raises(ValueError, match="must be matching or dead"):
        FakeGrammarProgram(
            initial_state="initial",
            transitions=(),
            valid_token_ids=(("initial", ()),),
        )


def test_program_and_state_handles_are_immutable():
    program = branching_program()
    constraint = make_compiler().compile_regex(VOCABULARY, REGEX_PATTERN)
    state = constraint.init_state()

    with pytest.raises(FrozenInstanceError):
        program.initial_state = "other"
    with pytest.raises(FrozenInstanceError):
        state.value = 99
    assert isinstance(state, FakeGrammarStateHandle)
