from onyx._rust import GrammarConstraint


def test_regex_state_handles_are_independent_from_python():
    vocab = [b"The", b" year", b"x"]
    constraint = GrammarConstraint(vocab)
    constraint.compile_regex("The year")

    initial = constraint.init_state()
    after_the = constraint.advance_state(initial, 0)

    initial_valid = constraint.get_valid_token_ids(initial)
    after_the_valid = constraint.get_valid_token_ids(after_the)

    assert 0 in initial_valid
    assert 1 not in initial_valid
    assert 1 in after_the_valid
    assert 0 not in after_the_valid


def test_json_state_handles_can_branch_from_same_parent():
    vocab = [b"{", b'"a"', b'"b"', b":", b'"', b"1"]
    schema = '{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"number"}}}'
    constraint = GrammarConstraint(vocab)
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


def test_unknown_state_handle_raises_value_error():
    constraint = GrammarConstraint([b"The", b" year"])
    constraint.compile_regex("The year")

    try:
        constraint.get_valid_token_ids(999)
    except ValueError as exc:
        assert "Unknown grammar state handle" in str(exc)
    else:
        raise AssertionError("unknown state handle did not raise")
