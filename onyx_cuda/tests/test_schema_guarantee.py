"""Permanent regressions for the supported JSON Schema output contract."""

import itertools
import json
import random
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from onyx_cuda import _rust


@pytest.mark.parametrize("key,prefix,token,expected", [
    ("content", b'{"c', b'\xe5\x82', False),  # Qwen token 99267: observed 3B dead end
    ("c傀", b'{"c', b'\xe5\x82', True),
    ("content", b'{"c', b'\xc2', False),
    ("content", b'{"c', b'\\u4', False),
    ("content", b'{"c', b'\\u00', True),
    ("content", b'{"content', b'\\', False),
    ("content", b'{"content', b'\xe5', False),
    ("cé", b'{"c', b'\xc3', True),
    ("cé", b'{"c', b'\xc2', False),
    ("c🚀", b'{"c', b'\\uD83D\\uDE8', True),
    ("c🚀", b'{"c', b'\\ud83d\\udc', False),
    ("c/", b'{"c', b'\\', True),
])
def test_key_pending_sequence_must_match_remaining_key(key, prefix, token, expected):
    constraint = _rust.GrammarConstraint([prefix, token])
    constraint.compile_json_schema(json.dumps({
        "type": "object", "properties": {key: {"type": "integer"}}, "required": [key],
    }))
    initial = constraint.init_state()
    state = constraint.advance_state(initial, 0)
    try:
        assert (1 in constraint.get_valid_token_ids(state)) is expected
    finally:
        constraint.release_state(state)
        constraint.release_state(initial)


def accepts(schema, document, chunk_size=None):
    raw = document.encode("utf-8") if isinstance(document, str) else document
    tokens = (
        [raw]
        if chunk_size is None
        else [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    )
    constraint = _rust.GrammarConstraint(tokens)
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
    constraint = _rust.GrammarConstraint([prefix, token])
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


@pytest.mark.parametrize("pattern", ["AB", "A{2}", "^A?B$", "A$", "A*", "éB|🚀A", "[^A]B"])
def test_bounded_pattern_prefixes_match_independent_completion_oracle(pattern):
    alphabet = "ABXé🚀"
    maximum = 3
    schema = {"type": "string", "pattern": pattern, "maxLength": maximum}
    validator = Draft202012Validator(schema)
    words = ["".join(chars) for length in range(maximum + 1)
             for chars in itertools.product(alphabet, repeat=length)]
    valid = [word for word in words if validator.is_valid(word)]
    vocabulary = [b'"', *(character.encode() for character in alphabet)]
    constraint = _rust.GrammarConstraint(vocabulary)
    constraint.compile_json_schema(json.dumps(schema))
    for prefix in words:
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
            assert (0 in allowed) == validator.is_valid(prefix)
            for token, character in enumerate(alphabet, 1):
                assert (token in allowed) == any(word.startswith(prefix + character) for word in valid)
        finally:
            constraint.release_state(state)


@pytest.mark.parametrize("schema,prefix", [
    ({"type": "object", "properties": {"value": {"type": "string", "pattern": "AB", "maxLength": 2}},
      "required": ["value"]}, b'{"value":"'),
    ({"type": "array", "items": {"type": "string", "pattern": "AB", "maxLength": 2}}, b'["AB","'),
])
def test_nested_bounded_patterns_reject_dead_end_tokens(schema, prefix):
    constraint = _rust.GrammarConstraint([prefix, b'X', b'A', b'\\u005'])
    constraint.compile_json_schema(json.dumps(schema))
    initial = constraint.init_state()
    state = constraint.advance_state(initial, 0)
    try:
        assert constraint.get_valid_token_ids(state) == [2]
    finally:
        constraint.release_states([initial, state])


@pytest.mark.parametrize("chunk_size", [None, 1, 2])
@pytest.mark.parametrize("key,document", [
    ("content", r'{"c\u006Fntent":1}'),
    ("cé", '{"cé":1}'),
    ("cé", r'{"c\u00E9":1}'),
    ("c🚀", '{"c🚀":1}'),
    ("c🚀", r'{"c\uD83d\uDe80":1}'),
    ("c/", r'{"c\/":1}'),
    ('c"', r'{"c\"":1}'),
    ("c\n", r'{"c\n":1}'),
])
def test_key_character_encodings_remain_completable(key, document, chunk_size):
    schema = {"type": "object", "properties": {key: {"type": "integer"}}, "required": [key]}
    assert accepts(schema, document, chunk_size)
    Draft202012Validator(schema).validate(json.loads(document))


CASES = [
    ({"type": "string", "minLength": 1, "maxLength": 1}, '"é"', True),
    ({"type": "string", "minLength": 1, "maxLength": 1}, '"🚀"', True),
    ({"type": "string", "minLength": 2}, '"é"', False),
    ({"type": "string", "maxLength": 1}, r'"a\n"', False),
    ({"type": "string", "maxLength": 0}, r'"\n"', False),
    ({"type": "string", "minLength": 1, "maxLength": 1}, r'"\u00e9"', True),
    ({"type": "string", "minLength": 1, "maxLength": 1}, r'"\ud83d\ude80"', True),
    ({"type": "string", "pattern": "^é🚀$", "maxLength": 2}, r'"\u00e9\ud83d\ude80"', True),
    ({"type": "string", "pattern": "p"}, '"apple"', True),
    ({"type": "string", "pattern": "p"}, '"pear"', True),
    ({"type": "string", "pattern": "p"}, '"orange"', False),
    ({"type": "string", "pattern": "^[A-Z]{2}$"}, '"AB"', True),
    ({"type": "string", "pattern": "^[A-Z]{2}$"}, '"Ab"', False),
    ({"type": "string", "pattern": r"^\d{2}$"}, '"12"', True),
    ({"type": "string"}, r'"\q"', False),
    ({"type": "string"}, '"a\nb"', False),
    ({"type": "string"}, r'"\ud800"', False),
    ({"type": "string"}, b'"\xed\xa0\x80"', False),
    ({"type": "string"}, b'"\xc0\xaf"', False),
    ({"type": "string"}, r'"\"\\\/\b\f\n\r\t"', True),
    ({"type": "number"}, "0", True),
    ({"type": "number"}, "-0.5e+2", True),
    ({"type": "number"}, "01", False),
    ({"type": "number"}, "-.5", False),
    ({"type": "number"}, "1.e2", False),
    ({"type": "number"}, "1e+", False),
    ({"type": ["integer", "null"]}, "null", True),
    ({"type": "array", "items": {"type": "integer"}}, "[1,]", False),
    ({"type": "array", "items": {"enum": [1, 10]}, "minItems": 2}, "[1]", False),
    ({"type": "array", "items": {"enum": [1, 10]}, "minItems": 2}, "[1,10]", True),
    ({"type": "array", "items": {"enum": [1]}, "maxItems": 1}, "[1,1]", False),
    ({"type": "array", "maxItems": 0}, "[]", True),
    (
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        '{"a":1,}',
        False,
    ),
    ({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}, "{}", False),
    (
        {"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": False},
        '{"b":1}',
        False,
    ),
    (
        {"type": "object", "properties": {"é": {"type": "string"}}, "required": ["é"]},
        '{"é":"🚀"}',
        True,
    ),
    (
        {"type": "object", "properties": {"é": {"type": "string"}}, "required": ["é"]},
        r'{"\u00e9":"🚀"}',
        True,
    ),
    ({"type": "object", "properties": {'a"b': {"type": "null"}}}, r'{"a\"b":null}', True),
    ({"type": "object", "properties": {"x": {}}}, '{"x":true}', True),
    ({"type": "string", "enum": [1, "é", "long"], "maxLength": 1}, '"é"', True),
    ({"type": "string", "enum": [1, "é", "long"], "maxLength": 1}, '"long"', False),
    ({"type": "string", "enum": [1, "é", "long"], "maxLength": 1}, "1", False),
    ({"enum": [1, 10]}, "1", True),
    ({"enum": [1, 10]}, "10", True),
    (
        {"type": "object", "enum": [{"a": 1}], "properties": {"a": {"type": "integer"}}},
        '{"a":2}',
        False,
    ),
    (
        {"type": "object", "enum": [{"a": 1}], "properties": {"a": {"type": "integer"}}},
        '{"a":1}',
        True,
    ),
    (
        {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "minItems": 1,
        },
        '[["é"]]',
        True,
    ),
]


@pytest.mark.parametrize("chunk_size", [None, 1, 2])
@pytest.mark.parametrize("schema,document,expected", CASES)
def test_schema_validity_across_token_boundaries(schema, document, expected, chunk_size):
    assert accepts(schema, document, chunk_size) is expected
    if expected:
        Draft202012Validator(schema).validate(json.loads(document))
        _rust.validate_json_output(json.dumps(schema), document)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "integer", "minimum": 18},
        {"type": "object", "properties": {"age": {"type": "integer", "minimum": 18}}},
        {"type": "array", "items": {"type": "number", "maximum": 10}},
        {"type": "string", "format": "email"},
        {"anyOf": [{"type": "string"}]},
        {"$ref": "https://example.invalid/schema"},
        {"type": "typo"},
        {"type": []},
        {"type": ["string", "string"]},
        {"type": "object", "properties": []},
        {"type": "object", "required": ["undeclared"]},
        {"type": "object", "required": "name"},
        {"type": "object", "additionalProperties": {"type": "integer"}},
        {"type": "array", "items": []},
        {"type": "array", "minItems": 2, "maxItems": 1},
        {"type": "string", "minLength": -1},
        {"type": "string", "maxLength": 1.5},
        {"type": "string", "pattern": "["},
        {"type": "object", "properties": {"x": {"type": "string", "pattern": "["}}},
        {"type": "string", "pattern": "(?=x)"},
        {"type": "string", "pattern": r"\p{L}"},
        {"type": "string", "pattern": "[a-z&&[^x]]"},
        {"minLength": 2},
        {"enum": []},
        {"enum": [1, 1]},
        {"type": "integer", "enum": ["x"]},
        {"type": "string", "maxLength": 1, "enum": ["long"]},
        True,
        [],
    ],
)
def test_unsupported_or_malformed_schemas_fail_before_generation(schema):
    raw = json.dumps(schema)
    with pytest.raises(ValueError):
        _rust.validate_json_schema(raw)
    with pytest.raises(ValueError):
        _rust.GrammarConstraint([b"x"]).compile_json_schema(raw)


def test_generated_candidate_corpus_agrees_with_independent_validator():
    rng = random.Random(71)
    schemas = [
        {"type": "string", "minLength": 1, "maxLength": 3},
        {"type": "string", "pattern": "^[A-Zé🚀]+$", "maxLength": 3},
        {
            "type": "array",
            "items": {"type": "string", "maxLength": 2},
            "minItems": 1,
            "maxItems": 2,
        },
        {
            "type": "object",
            "properties": {"value": {"type": ["string", "null"], "maxLength": 2}},
            "required": ["value"],
        },
    ]
    accepted_count = 0
    for _ in range(100):
        text = "".join(rng.choices('ABé🚀\n\\"', k=rng.randrange(5)))
        for schema, value in zip(
            schemas, [text, text, [text] * rng.randrange(4), {"value": rng.choice([text, None])}]
        ):
            document = json.dumps(value, ensure_ascii=rng.choice([True, False]))
            if accepts(schema, document, 1):
                Draft202012Validator(schema).validate(value)
                accepted_count += 1
    assert accepted_count > 50


def test_whitespace_can_share_the_token_that_completes_a_document():
    assert accepts({"type": "string"}, '"ok" \n')


@pytest.mark.parametrize("document", ["9007199254740992.1", "1e-1000", "100e-3"])
def test_integer_validation_does_not_round_fractions(document):
    with pytest.raises(ValueError, match="does not satisfy"):
        _rust.validate_json_output('{"type":"integer"}', document)


@pytest.mark.parametrize(
    "document", ["1.0", "100e-2", "1.20e1", "0e-1000", "999999999999999999999999"]
)
def test_integer_validation_uses_exact_decimal_value(document):
    _rust.validate_json_output('{"type":"integer"}', document)
    assert Decimal(document) == Decimal(document).to_integral_value()


def test_enum_and_compaction_preserve_numeric_precision():
    schema = '{"enum":[0.10000000000000000001]}'
    with pytest.raises(ValueError, match="does not satisfy"):
        _rust.validate_json_output(schema, "0.1")
    output = _rust.validate_json_output(schema, " 0.10000000000000000001 ")
    assert output == "0.10000000000000000001"


def test_invalid_schema_is_rejected_without_prefilling_models():
    from onyx_cuda.generation import generate_tokens
    from onyx_cuda.speculative import generate_speculative
    from onyx_cuda.vocabulary import TokenByteVocabulary

    options = dict(
        prompt_token_ids=[0],
        max_tokens=1,
        eos_token_ids=[],
        token_byte_vocabulary=TokenByteVocabulary([b"0"], 0, 0),
        json_schema='{"type":"integer","minimum":18}',
    )
    with pytest.raises(ValueError, match="unsupported keyword 'minimum'"):
        generate_tokens(object(), **options)
    with pytest.raises(ValueError, match="unsupported keyword 'minimum'"):
        generate_speculative(object(), object(), gamma=1, **options)
