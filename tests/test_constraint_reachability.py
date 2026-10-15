"""Constraint states must keep a match reachable.

Schema patterns here must match the whole string, unlike JSON Schema's search
semantics, and string lengths count bytes, so the oracles below use
`re.fullmatch` and keep length cases ASCII.
"""
import itertools
import json
import re

import pytest

from onyx._rust import GrammarConstraint


@pytest.mark.parametrize("pattern", [r"[^\s\S]", "a$b", r"x(?-u:\b)y"])
def test_regex_that_can_never_match_fails_to_compile(pattern):
    with pytest.raises(ValueError, match="can never match"):
        GrammarConstraint([b"a", b"b", b"x"]).compile_regex(pattern)


def test_regex_tokens_into_branches_that_can_never_match_are_invalid():
    # After "b" only "b$c" remains, which no continuation can match.
    constraint = GrammarConstraint([b"a", b"b", b"c", b"bc"])
    constraint.compile_regex("(?:a|b$c)")
    initial = constraint.init_state()
    try:
        assert constraint.get_valid_token_ids(initial) == [0]
    finally:
        constraint.release_state(initial)


def assert_prefixes_match_oracle(schema, alphabet, valid, longest_prefix):
    """Every offered token must lead to a valid word and every such token be offered."""
    constraint = GrammarConstraint([b'"', *(character.encode() for character in alphabet)])
    constraint.compile_json_schema(json.dumps(schema))
    prefixes = ("".join(chars) for length in range(longest_prefix + 1)
                for chars in itertools.product(alphabet, repeat=length))
    for prefix in prefixes:
        if not any(word.startswith(prefix) for word in valid):
            continue
        state = constraint.init_state()
        try:
            for token in [0, *(1 + alphabet.index(character) for character in prefix)]:
                assert token in constraint.get_valid_token_ids(state)
                previous = state
                state = constraint.advance_state(state, token)
                constraint.release_state(previous)
            allowed = constraint.get_valid_token_ids(state)
            assert (0 in allowed) == (prefix in valid), prefix
            for token, character in enumerate(alphabet, 1):
                expected = any(word.startswith(prefix + character) for word in valid)
                assert (token in allowed) == expected, prefix + character
        finally:
            constraint.release_state(state)


@pytest.mark.parametrize("pattern", ["^AB", "AB", "^A?B$", "A$", "^[AB]+$", "^é|🚀$"])
def test_unbounded_pattern_prefixes_match_independent_completion_oracle(pattern):
    # Without maxLength a character must still leave some match reachable:
    # after "AB", "^AB" could never close its string on another character.
    alphabet = "ABXé🚀"
    # Each pattern needs at most two more characters to match when it still can,
    # so five-character words decide every prefix of up to two characters.
    words = ["".join(chars) for length in range(6)
             for chars in itertools.product(alphabet, repeat=length)]
    valid = {word for word in words if re.fullmatch(pattern, word)}
    assert_prefixes_match_oracle({"type": "string", "pattern": pattern}, alphabet, valid, 2)


@pytest.mark.parametrize("pattern,minimum,maximum", [
    ("^(A|BBB)$", 2, 4),
    ("^(A|BBB)$", 2, None),
    ("^[AB]{1,3}$", 3, 4),
    ("^A*$", 2, 4),
    ("^AB.*", 4, None),
    (".*A$", 3, None),
    ("^[AB]+$", 3, None),
    ("^(A|BBBB)X?$", 2, 4),
    ("^(A|BBBB)X?$", 3, 4),
])
def test_min_length_pattern_prefixes_match_independent_completion_oracle(pattern, minimum, maximum):
    # Every character must leave a match reachable at a length minLength allows:
    # for "^(A|BBB)$" with minLength 2, a leading A can never close the string.
    alphabet = "ABX"
    schema = {"type": "string", "pattern": pattern, "minLength": minimum}
    if maximum is not None:
        schema["maxLength"] = maximum
    # Unbounded cases need at most three more characters once a two-character
    # prefix can still match, so five-character words decide those prefixes.
    longest = 5 if maximum is None else maximum
    words = ["".join(chars) for length in range(longest + 1)
             for chars in itertools.product(alphabet, repeat=length)]
    valid = {word for word in words
             if re.fullmatch(pattern, word) and minimum <= len(word) <= (maximum or longest)}
    assert valid
    assert_prefixes_match_oracle(schema, alphabet, valid, 2 if maximum is None else maximum)


@pytest.mark.parametrize("schema,prefix", [
    ({"type": "object", "properties": {"value": {"type": "string", "pattern": "^AB"}},
      "required": ["value"]}, b'{"value":"AB'),
    ({"type": "array", "items": {"type": "string", "pattern": "^AB"}}, b'["AB","AB'),
])
def test_nested_patterns_reject_dead_end_tokens(schema, prefix):
    # After "AB" nothing may follow, not even the escape that starts a character.
    constraint = GrammarConstraint([prefix, b"A", b"B", b"\\", b'"'])
    constraint.compile_json_schema(json.dumps(schema))
    initial = constraint.init_state()
    state = constraint.advance_state(initial, 0)
    try:
        assert constraint.get_valid_token_ids(state) == [4]
    finally:
        constraint.release_states([initial, state])


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"value": {"type": "string", "pattern": "^(A|BB)$", "minLength": 2}},
     "required": ["value"]},
    {"type": "array", "items": {"type": "string", "pattern": "^(A|BB)$", "minLength": 2}},
])
def test_nested_min_length_patterns_reject_short_branches(schema):
    prefix = b'{"value":"' if schema["type"] == "object" else b'["'
    constraint = GrammarConstraint([prefix, b"A", b"B"])
    constraint.compile_json_schema(json.dumps(schema))
    initial = constraint.init_state()
    state = constraint.advance_state(initial, 0)
    try:
        # "A" matches but is shorter than minLength and cannot grow.
        assert constraint.get_valid_token_ids(state) == [2]
    finally:
        constraint.release_states([initial, state])


@pytest.mark.parametrize("schema", [
    {"type": "string", "pattern": "a$b"},
    {"type": "array", "items": {"type": "string", "pattern": "$x"}},
    {"type": "object", "properties": {"value": {"type": "string", "pattern": "a$b"}}},
    {"type": "string", "pattern": "^[0-9]{5}$", "maxLength": 3},
    {"type": "string", "pattern": "^[A-Z]{3}$", "minLength": 5},
    {"type": "string", "pattern": "^(A|BBBB)$", "minLength": 2, "maxLength": 3},
    {"type": "array", "items": {"type": "string", "pattern": "^A?$", "minLength": 2}},
])
def test_unsatisfiable_patterns_fail_to_compile(schema):
    with pytest.raises(ValueError, match="pattern"):
        GrammarConstraint([b'"', b"A"]).compile_json_schema(json.dumps(schema))


@pytest.mark.parametrize("schema", [
    {"type": "string", "pattern": "^(A|BBBB)$", "minLength": 2, "maxLength": 4},
    {"type": "string", "pattern": "^AB.*", "minLength": 5},
    {"type": "string", "pattern": "^[0-9]+$", "minLength": 500},
])
def test_satisfiable_min_length_patterns_are_accepted(schema):
    GrammarConstraint([b'"', b"A"]).compile_json_schema(json.dumps(schema))
