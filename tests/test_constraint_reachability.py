"""Constraint states must keep a match reachable.

Schema patterns follow JSON Schema: a match may occur anywhere in the string,
lengths count Unicode characters, and escapes stand for the characters they
encode. The oracles below therefore use an independent JSON Schema validator.
"""
import itertools
import json

import pytest
from jsonschema import Draft202012Validator

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


def accepts(schema, document, chunk_size=None):
    raw = document.encode("utf-8") if isinstance(document, str) else document
    tokens = (
        [raw]
        if chunk_size is None
        else [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    )
    constraint = GrammarConstraint(tokens)
    constraint.compile_json_schema(json.dumps(schema))
    state = constraint.init_state()
    try:
        for token_id in range(len(tokens)):
            if token_id not in constraint.get_valid_token_ids(state):
                return False
            next_state = constraint.advance_state(state, token_id)
            constraint.release_state(state)
            state = next_state
        return constraint.is_match_state(state)
    finally:
        constraint.release_state(state)


@pytest.mark.parametrize("pattern,maximum,prefix,token,expected", [
    ("AB", 2, b'"', b'X', False),
    ("AB", 2, b'"', b'B', False),
    ("AB", 2, b'"', b'A', True),
    ("AB", 3, b'"', b'X', True),
    ("AB", 3, b'"X', b'X', False),
    ("AB", 2, b'"', b'\\u005', False),
    ("AB", 2, b'"', b'\\u004', True),
    ("AB", 2, b'"', b'\xc2', False),
    ("éB", 2, b'"', b'\xc3', True),
    ("éB", 2, b'"', b'\xc2', False),
    ("éB", 2, b'"', b'\\u00E', True),
    ("éB", 2, b'"', b'\\u00F', False),
    ("🚀B", 2, b'"', b'\xf0\x9f', True),
    ("🚀B", 2, b'"', b'\xf0\x90', False),
    ("🚀B", 2, b'"', b'\\uD83', True),
    ("🚀B", 2, b'"', b'\\uD84', False),
    ("🚀B", 2, b'"', b'\\uD83D\\uDE8', True),
    ("🚀B", 2, b'"', b'\\uD83D\\uDC', False),
    ("A|BC", 1, b'"', b'B', False),
    ("A|BC", 2, b'"', b'B', True),
])
def test_pattern_prefix_can_finish_within_max_length(pattern, maximum, prefix, token, expected):
    # Partial escapes and UTF-8 sequences count as the characters they can
    # still become, and a match must stay reachable within maxLength.
    constraint = GrammarConstraint([prefix, token])
    constraint.compile_json_schema(json.dumps({
        "type": "string", "pattern": pattern, "maxLength": maximum,
    }))
    initial = constraint.init_state()
    state = constraint.advance_state(initial, 0)
    try:
        assert (1 in constraint.get_valid_token_ids(state)) is expected
    finally:
        constraint.release_states([initial, state])


@pytest.mark.parametrize("chunk_size", [None, 1, 2])
@pytest.mark.parametrize("pattern,maximum,document", [
    ("AB", 2, '"AB"'),
    ("AB", 3, '"XAB"'),
    ("AB", 3, '"ABX"'),
    ("éB", 2, '"éB"'),
    ("éB", 2, r'"\u00e9B"'),
    ("🚀B", 2, '"🚀B"'),
    ("🚀B", 2, r'"\uD83d\uDe80B"'),
    ("", 0, '""'),
    ("^A$", 1, '"A"'),
    ("A|BC", 2, '"BC"'),
    (r"\nB", 2, r'"\nB"'),
])
def test_bounded_pattern_preserves_valid_encodings(pattern, maximum, document, chunk_size):
    schema = {"type": "string", "pattern": pattern, "maxLength": maximum}
    assert accepts(schema, document, chunk_size)
    Draft202012Validator(schema).validate(json.loads(document))


@pytest.mark.parametrize("codepoint", [0, 0x7f, 0x80, 0x7ff, 0x800, 0xd7ff,
                                       0xe000, 0xffff, 0x10000, 0x10ffff])
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_bounded_pattern_unicode_boundaries(codepoint, ensure_ascii):
    character = chr(codepoint)
    schema = {"type": "string", "pattern": character + "B", "maxLength": 2}
    document = json.dumps(character + "B", ensure_ascii=ensure_ascii)
    assert accepts(schema, document, 1)


@pytest.mark.parametrize("document", [
    r'"\x"', r'"\u00G0"', r'"\uDC00"', r'"\uD800x"', r'"\uD800\u0041"',
    b'"a\nb"', b'"\x80"', b'"\xc0\xaf"', b'"\xed\xa0\x80"',
])
def test_invalid_string_contents_are_rejected(document):
    # Bad escapes, lone surrogates, raw control characters and invalid UTF-8.
    for chunk_size in [None, 1]:
        assert not accepts({"type": "string"}, document, chunk_size)


ALPHABET = "ABXé🚀"


def assert_prefixes_match_completion_oracle(schema, longest, longest_prefix):
    """Every prefix that can still complete allows exactly the characters an
    independent validator says can still complete it, and the closing quote
    exactly when the prefix itself is valid. Words up to `longest` characters
    decide every prefix up to `longest_prefix` characters."""
    validator = Draft202012Validator(schema)
    words = ["".join(chars) for length in range(longest + 1)
             for chars in itertools.product(ALPHABET, repeat=length)]
    valid = [word for word in words if validator.is_valid(word)]
    assert valid
    vocabulary = [b'"', *(character.encode() for character in ALPHABET)]
    constraint = GrammarConstraint(vocabulary)
    constraint.compile_json_schema(json.dumps(schema))
    for prefix in (word for word in words if len(word) <= longest_prefix):
        if not any(word.startswith(prefix) for word in valid):
            continue
        state = constraint.init_state()
        try:
            for token in [0, *(1 + ALPHABET.index(character) for character in prefix)]:
                assert token in constraint.get_valid_token_ids(state)
                previous = state
                state = constraint.advance_state(state, token)
                constraint.release_state(previous)
            allowed = constraint.get_valid_token_ids(state)
            assert (0 in allowed) == validator.is_valid(prefix), prefix
            for token, character in enumerate(ALPHABET, 1):
                expected = any(word.startswith(prefix + character) for word in valid)
                assert (token in allowed) == expected, prefix + character
        finally:
            constraint.release_state(state)


@pytest.mark.parametrize("pattern", ["AB", "A{2}", "^A?B$", "A$", "A*", "éB|🚀A", "[^A]B"])
def test_bounded_pattern_prefixes_match_independent_completion_oracle(pattern):
    assert_prefixes_match_completion_oracle(
        {"type": "string", "pattern": pattern, "maxLength": 3}, longest=3, longest_prefix=3)


@pytest.mark.parametrize("pattern", ["^AB", "AB", "^A?B$", "A$", "^[AB]+$", "^é|🚀$"])
def test_unbounded_pattern_prefixes_match_independent_completion_oracle(pattern):
    # Without maxLength a character must still leave some match reachable:
    # after a leading X, "^AB" could never close its string.
    # Each pattern needs at most two more characters to match when it still can,
    # so five-character words decide every prefix of up to two characters.
    assert_prefixes_match_completion_oracle(
        {"type": "string", "pattern": pattern}, longest=5, longest_prefix=2)


@pytest.mark.parametrize("pattern,minimum,maximum", [
    ("^(A|BBB)$", 2, 4),
    ("^(A|BBB)$", 2, None),
    ("^[AB]{1,3}$", 3, 4),
    ("^A*$", 2, 4),
    ("^AB", 4, None),
    ("A$", 3, None),
    ("A", 3, 4),
    ("^[AB]+$", 3, None),
    ("^(A|BBBB)X?$", 2, 4),
    ("^(A|BBBB)X?$", 3, 4),
    ("^é|🚀$", 2, 3),
    ("^(é|BB)X?$", 2, 3),
])
def test_min_length_pattern_prefixes_match_independent_completion_oracle(pattern, minimum, maximum):
    # Every character must leave a match reachable at a length minLength allows:
    # for "^(A|BBB)$" with minLength 2, a leading A can never close the string.
    schema = {"type": "string", "pattern": pattern, "minLength": minimum}
    if maximum is not None:
        schema["maxLength"] = maximum
    # Unbounded cases need at most three more characters once a two-character
    # prefix can still match, so five-character words decide those prefixes.
    assert_prefixes_match_completion_oracle(
        schema, longest=5 if maximum is None else maximum,
        longest_prefix=2 if maximum is None else maximum)


@pytest.mark.parametrize("schema,prefix", [
    ({"type": "object", "properties": {"value": {"type": "string", "pattern": "AB", "maxLength": 2}},
      "required": ["value"]}, b'{"value":"'),
    ({"type": "array", "items": {"type": "string", "pattern": "AB", "maxLength": 2}}, b'["AB","'),
    ({"type": "object", "properties": {"value": {"type": "string", "pattern": "^AB"}},
      "required": ["value"]}, b'{"value":"'),
    ({"type": "array", "items": {"type": "string", "pattern": "^AB"}}, b'["AB","'),
])
def test_nested_patterns_reject_dead_end_tokens(schema, prefix):
    # Only "A" can start a match; "\u005" can only become P to _.
    constraint = GrammarConstraint([prefix, b"X", b"A", b"\\u005"])
    constraint.compile_json_schema(json.dumps(schema))
    initial = constraint.init_state()
    state = constraint.advance_state(initial, 0)
    try:
        assert constraint.get_valid_token_ids(state) == [2]
    finally:
        constraint.release_states([initial, state])


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"value": {"type": "string", "pattern": "^(A|BB)$", "minLength": 2}},
     "required": ["value"]},
    {"type": "array", "items": {"type": "string", "pattern": "^(A|BB)$", "minLength": 2}},
])
def test_nested_min_length_patterns_reject_short_branches(schema):
    prefix = b'{"value":"' if schema["type"] == "object" else b'["'
    constraint = GrammarConstraint([prefix, b"A", b"B", b"\\u0041", b"\\u004"])
    constraint.compile_json_schema(json.dumps(schema))
    initial = constraint.init_state()
    state = constraint.advance_state(initial, 0)
    try:
        # "A" matches but is shorter than minLength and cannot grow, however
        # it is encoded; "\u004" can still finish as B.
        assert constraint.get_valid_token_ids(state) == [2, 4]
    finally:
        constraint.release_states([initial, state])


@pytest.mark.parametrize("schema", [
    {"type": "string", "pattern": "a$b"},
    {"type": "array", "items": {"type": "string", "pattern": "$x"}},
    {"type": "object", "properties": {"value": {"type": "string", "pattern": "a$b"}}},
    {"type": "string", "pattern": "^[0-9]{5}$", "maxLength": 3},
    {"type": "string", "pattern": "^[A-Z]{3}$", "minLength": 5},
    {"type": "string", "pattern": "^é{2}$", "minLength": 3},
    {"type": "string", "pattern": "^(A|BBBB)$", "minLength": 2, "maxLength": 3},
    {"type": "array", "items": {"type": "string", "pattern": "^A?$", "minLength": 2}},
])
def test_unsatisfiable_patterns_fail_to_compile(schema):
    with pytest.raises(ValueError, match="pattern"):
        GrammarConstraint([b'"', b"A"]).compile_json_schema(json.dumps(schema))


@pytest.mark.parametrize("pattern", ["(", "a)|(b", r"(?<=a)b"])
def test_uncompilable_patterns_fail_to_compile(pattern):
    # A pattern that cannot compile must not leave the string unconstrained.
    with pytest.raises(ValueError, match="Failed to compile regex"):
        GrammarConstraint([b'"', b"A"]).compile_json_schema(
            json.dumps({"type": "string", "pattern": pattern}))


@pytest.mark.parametrize("schema", [
    {"type": "string", "pattern": "^é{2}$", "minLength": 2},
    {"type": "string", "pattern": "^(A|BBBB)$", "minLength": 2, "maxLength": 4},
    {"type": "string", "pattern": "^AB", "minLength": 5},
    {"type": "string", "pattern": "^[0-9]+$", "minLength": 500},
])
def test_satisfiable_min_length_patterns_are_accepted(schema):
    GrammarConstraint([b'"', b"A"]).compile_json_schema(json.dumps(schema))
