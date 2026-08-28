import pytest

from onyx_cuda import _rust


def test_constructor_and_compilation_errors_are_value_errors():
    assert _rust.__name__ == "onyx_cuda._rust"

    with pytest.raises(ValueError, match="Vocabulary cannot be empty"):
        _rust.GrammarConstraint([])

    constraint = _rust.GrammarConstraint([b"The", b" year", b"x"])
    assert constraint.vocab_size() == 3

    with pytest.raises(ValueError, match="No constraint compiled"):
        constraint.init_state()
    with pytest.raises(ValueError, match="Compilation error"):
        constraint.compile_regex("(")


def test_regex_state_handles_branch_complete_die_and_release():
    constraint = _rust.GrammarConstraint([b"The", b" year", b"x"])
    constraint.compile_regex("The year")

    initial = constraint.init_state()
    after_the = constraint.advance_state(initial, 0)
    complete = constraint.advance_state(after_the, 1)
    dead = constraint.advance_state(initial, 2)

    assert 0 in constraint.get_valid_token_ids(initial)
    assert 1 not in constraint.get_valid_token_ids(initial)
    assert 1 in constraint.get_valid_token_ids(after_the)
    assert 0 not in constraint.get_valid_token_ids(after_the)
    assert constraint.is_match_state(complete)
    assert not constraint.is_dead_state(complete)
    assert constraint.is_dead_state(dead)

    with pytest.raises(ValueError, match="out of range"):
        constraint.advance_state(initial, 3)

    constraint.release_state(after_the)
    with pytest.raises(ValueError, match="Unknown grammar state handle"):
        constraint.get_valid_token_ids(after_the)

    constraint.release_states([initial, complete, dead])
    with pytest.raises(ValueError, match="Unknown grammar state handle"):
        constraint.release_state(initial)


def test_json_state_handles_branch_and_release_from_python():
    vocab = [b"{", b'"a"', b'"b"', b":", b'"', b"1"]
    schema = '{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"number"}}}'
    constraint = _rust.GrammarConstraint(vocab)
    constraint.compile_json_schema(schema)

    initial = constraint.init_state()
    in_object = constraint.advance_state(initial, 0)
    after_a_colon = constraint.advance_state(
        constraint.advance_state(in_object, 1),
        3,
    )
    after_b_colon = constraint.advance_state(
        constraint.advance_state(in_object, 2),
        3,
    )

    valid_for_a = constraint.get_valid_token_ids(after_a_colon)
    valid_for_b = constraint.get_valid_token_ids(after_b_colon)
    assert 4 in valid_for_a
    assert 5 not in valid_for_a
    assert 5 in valid_for_b
    assert 4 not in valid_for_b

    constraint.release_states([after_a_colon, after_b_colon])
    with pytest.raises(ValueError, match="Unknown grammar state handle"):
        constraint.is_dead_state(after_a_colon)

    invalid = _rust.GrammarConstraint(vocab)
    with pytest.raises(ValueError, match="Compilation error"):
        invalid.compile_json_schema("{")
