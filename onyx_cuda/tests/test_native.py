import json

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


def _byte_vocabulary():
    return [bytes([byte]) for byte in range(256)] + [b'"}', b'","', b"ab", "é".encode(), b"\n", b""]


@pytest.mark.parametrize("grammar,document", [
    ({"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string", "maxLength": 9}},
      "required": ["title", "text"]}, '{"title":"ab é","text":"q\\"x"}'),
    ("[a-c]+(de)?", "abcde"),
])
def test_cached_scans_masks_and_ids_match_uncached_scans(grammar, document):
    def compiled(**options):
        constraint = _rust.GrammarConstraint(_byte_vocabulary(), **options)
        if isinstance(grammar, dict):
            constraint.compile_json_schema(json.dumps(grammar))
        else:
            constraint.compile_regex(grammar)
        return constraint, constraint.init_state()

    cached, cached_state = compiled()
    fresh, fresh_state = compiled(scan_cache_entries=0)
    for byte in [*document.encode(), None]:
        expected = fresh.get_valid_token_ids(fresh_state)
        assert cached.get_valid_token_ids(cached_state) == expected
        scan_id, count = cached.scan_valid_tokens(cached_state)
        assert count == len(expected)
        blocked = cached.blocked_token_mask(cached_state)
        assert isinstance(blocked, bytearray) and len(blocked) == cached.vocab_size()
        assert [token for token, value in enumerate(blocked) if not value] == expected
        assert cached.scan_valid_tokens(cached_state)[0] == scan_id
        if byte is not None:
            cached_state = cached.advance_state(cached_state, byte)
            fresh_state = fresh.advance_state(fresh_state, byte)


def test_plain_string_positions_reuse_one_scan_id():
    constraint = _rust.GrammarConstraint(_byte_vocabulary())
    constraint.compile_json_schema('{"type":"object","properties":{"text":{"type":"string"}}}')
    state = constraint.init_state()
    ids = set()
    for index, byte in enumerate(b'{"text":"a longer value'):
        state = constraint.advance_state(state, byte)
        if index >= 8:
            ids.add(constraint.scan_valid_tokens(state)[0])
    assert len(ids) == 1
