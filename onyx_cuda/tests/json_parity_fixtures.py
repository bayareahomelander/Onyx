"""Frozen 2026-07-15 outcomes from the read-only root JSON grammar runtime."""

from dataclasses import dataclass
from typing import Literal


JsonExpectationDisposition = Literal["windows_parity", "reference_only"]
JsonReferenceCompileOutcome = Literal["accepted", "value_error"]
JsonWindowsSchemaPolicy = Literal["support", "reject_invalid", "reject_unsupported"]


@dataclass(frozen=True, slots=True)
class JsonStateExpectation:
    token_ids: tuple[int, ...]
    valid_token_ids: tuple[int, ...]
    is_match: bool
    is_dead: bool


@dataclass(frozen=True, slots=True)
class JsonRejectedTransitionExpectation:
    prefix_token_ids: tuple[int, ...]
    token_id: int
    reference_error_type: str = "ValueError"
    reference_message_contains: str = "Invalid state: Invalid byte"
    selected_windows_error_type: str = "GrammarStateError"
    allocates_child: bool = False
    preserves_parent: bool = True


@dataclass(frozen=True, slots=True)
class JsonParityCase:
    name: str
    vocabulary: tuple[bytes, ...]
    schema: str
    features: frozenset[str]
    states: tuple[JsonStateExpectation, ...]
    rejected_transitions: tuple[JsonRejectedTransitionExpectation, ...] = ()
    disposition: JsonExpectationDisposition = "windows_parity"


@dataclass(frozen=True, slots=True)
class JsonSchemaObservation:
    name: str
    schema: str
    reference_compile_outcome: JsonReferenceCompileOutcome
    reference_behavior: str
    selected_windows_policy: JsonWindowsSchemaPolicy


SUPPORTED_JSON_FEATURES = frozenset(
    {
        "array",
        "array_max_items",
        "array_min_items",
        "boolean",
        "branching",
        "empty_tokens",
        "enum",
        "integer",
        "multi_byte_tokens",
        "nested_structures",
        "null",
        "number",
        "object",
        "optional_properties",
        "required_properties",
        "string",
        "string_max_length",
        "string_min_length",
        "string_pattern",
        "structural_whitespace",
        "typed_items",
        "union_types",
        "valid_escapes",
    }
)

JSON_INVALID_TRANSITION_POLICY = (
    "raise GrammarStateError without allocating a child and preserve the parent state"
)
JSON_EMPTY_TOKEN_POLICY = (
    "omit empty-byte tokens from valid-token results but return an equivalent independent child "
    "when one is advanced explicitly"
)
JSON_UNSUPPORTED_KEYWORD_POLICY = "reject before native compilation"
JSON_LEXICAL_POLICY = "accept only syntactically valid RFC 8259 JSON values"
JSON_UNICODE_ESCAPE_POLICY = (
    "decode valid surrogate pairs and reject lone or misordered surrogate escapes"
)
JSON_STRING_LENGTH_POLICY = "count Unicode code points rather than UTF-8 bytes"
JSON_PATTERN_POLICY = (
    "apply anchored whole-string regexes to decoded Unicode string values"
)
JSON_TRAILING_WHITESPACE_POLICY = (
    "remain matching after root completion while accepting only structural whitespace bytes"
)
JSON_FUTURE_NATIVE_ABI_VERSION = 3
JSON_FUTURE_FACTORY_NAME = "compile_native_json_schema"
JSON_EXPOSE_COMPLETE_NATIVE_COMPILER = False


JSON_PARITY_CASES = (
    JsonParityCase(
        name="object_required_optional_branching",
        vocabulary=(
            b"{",
            b"}",
            b'"a"',
            b'"b"',
            b":",
            b",",
            b"1",
            b'"x"',
            b" ",
            b'"c"',
            b"",
            b'{"a":1}',
            b"\t\n",
        ),
        schema=(
            '{"type":"object","properties":{"a":{"type":"integer"},'
            '"b":{"type":"string"}},"required":["a"]}'
        ),
        features=frozenset(
            {
                "branching",
                "empty_tokens",
                "integer",
                "multi_byte_tokens",
                "object",
                "optional_properties",
                "required_properties",
                "string",
            }
        ),
        states=(
            JsonStateExpectation((), (0, 8, 11, 12), False, False),
            JsonStateExpectation((0,), (2, 3, 8, 12), False, False),
            JsonStateExpectation((0, 2), (4, 8, 12), False, False),
            JsonStateExpectation((0, 2, 4), (6, 8, 12), False, False),
            JsonStateExpectation((0, 2, 4, 6), (1, 5, 6, 8, 12), False, False),
            JsonStateExpectation((0, 2, 4, 6, 1), (), True, False),
            JsonStateExpectation((0, 3, 4, 7), (5, 8, 12), False, False),
            JsonStateExpectation((0, 3, 4, 7, 5), (2, 8, 12), False, False),
            JsonStateExpectation(
                (0, 3, 4, 7, 5, 2, 4, 6, 1),
                (),
                True,
                False,
            ),
            JsonStateExpectation((10,), (0, 8, 11, 12), False, False),
            JsonStateExpectation((8,), (0, 8, 11, 12), False, False),
            JsonStateExpectation((12,), (0, 8, 11, 12), False, False),
            JsonStateExpectation((11,), (), True, False),
        ),
        rejected_transitions=(
            JsonRejectedTransitionExpectation((0,), 1),
            JsonRejectedTransitionExpectation((0, 2, 4), 7),
            JsonRejectedTransitionExpectation((0,), 9),
        ),
    ),
    JsonParityCase(
        name="root_array_typed_items_and_bounds",
        vocabulary=(b"[", b"]", b'"a"', b'"b"', b",", b" ", b"1", b"", b'["a","b"]'),
        schema=(
            '{"type":"array","minItems":1,"maxItems":2,'
            '"items":{"type":"string"}}'
        ),
        features=frozenset(
            {
                "array",
                "array_max_items",
                "array_min_items",
                "string",
                "typed_items",
            }
        ),
        states=(
            JsonStateExpectation((), (0, 5, 8), False, False),
            JsonStateExpectation((0,), (2, 3, 5), False, False),
            JsonStateExpectation((0, 2), (1, 4, 5), False, False),
            JsonStateExpectation((0, 2, 4), (1, 2, 3, 5), False, False),
            JsonStateExpectation((0, 2, 4, 3), (1, 5), False, False),
            JsonStateExpectation((0, 2, 4, 3, 1), (), True, False),
            JsonStateExpectation((7,), (0, 5, 8), False, False),
            JsonStateExpectation((8,), (), True, False),
        ),
        rejected_transitions=(
            JsonRejectedTransitionExpectation((0,), 1),
            JsonRejectedTransitionExpectation((0,), 6),
            JsonRejectedTransitionExpectation((0, 2, 4, 3), 4),
        ),
    ),
    JsonParityCase(
        name="root_string_pattern_and_lengths",
        vocabulary=(
            b'"AB"',
            b'"A"',
            b'"ABC"',
            b'"ab"',
            b'"',
            b"A",
            b"B",
            b"C",
            b"",
        ),
        schema=(
            '{"type":"string","pattern":"^[A-Z]+$",'
            '"minLength":2,"maxLength":2}'
        ),
        features=frozenset(
            {"string", "string_max_length", "string_min_length", "string_pattern"}
        ),
        states=(
            JsonStateExpectation((), (0, 4), False, False),
            JsonStateExpectation((4,), (5, 6, 7), False, False),
            JsonStateExpectation((4, 5), (5, 6, 7), False, False),
            JsonStateExpectation((4, 5, 6), (4,), False, False),
            JsonStateExpectation((4, 5, 6, 4), (), True, False),
            JsonStateExpectation((8,), (0, 4), False, False),
            JsonStateExpectation((0,), (), True, False),
        ),
        rejected_transitions=(
            JsonRejectedTransitionExpectation((4,), 4),
            JsonRejectedTransitionExpectation((4, 5), 4),
            JsonRejectedTransitionExpectation((4, 5, 6), 7),
            JsonRejectedTransitionExpectation((), 1),
            JsonRejectedTransitionExpectation((), 3),
        ),
    ),
    JsonParityCase(
        name="root_number_fraction_and_exponent",
        vocabulary=(
            b"-",
            b"1",
            b".",
            b"5",
            b"e",
            b"E",
            b"+",
            b"2",
            b" ",
            b"",
            b"-1.5e+2",
        ),
        schema='{"type":"number"}',
        features=frozenset({"number"}),
        states=(
            JsonStateExpectation((), (0, 1, 3, 7, 8, 10), False, False),
            JsonStateExpectation((0,), (1, 2, 3, 4, 5, 7), False, False),
            JsonStateExpectation((0, 1), (1, 2, 3, 4, 5, 7), True, False),
            JsonStateExpectation((1,), (1, 2, 3, 4, 5, 7), True, False),
            JsonStateExpectation((1, 2), (1, 3, 4, 5, 7), False, False),
            JsonStateExpectation((1, 2, 3), (1, 3, 4, 5, 7), True, False),
            JsonStateExpectation((1, 4), (0, 1, 3, 6, 7), False, False),
            JsonStateExpectation((1, 4, 6), (1, 3, 7), False, False),
            JsonStateExpectation((1, 4, 6, 7), (1, 3, 7), True, False),
            JsonStateExpectation((9,), (0, 1, 3, 7, 8, 10), False, False),
            JsonStateExpectation((10,), (1, 3, 7), True, False),
        ),
        rejected_transitions=(
            JsonRejectedTransitionExpectation((), 2),
            JsonRejectedTransitionExpectation((1, 4), 8),
        ),
    ),
    JsonParityCase(
        name="root_integer",
        vocabulary=(b"42", b".", b"e", b"-", b"1", b" ", b"", b"-7"),
        schema='{"type":"integer"}',
        features=frozenset({"integer"}),
        states=(
            JsonStateExpectation((), (0, 3, 4, 5, 7), False, False),
            JsonStateExpectation((0,), (0, 4), True, False),
            JsonStateExpectation((3,), (0, 4), False, False),
            JsonStateExpectation((3, 4), (0, 4), True, False),
            JsonStateExpectation((6,), (0, 3, 4, 5, 7), False, False),
            JsonStateExpectation((7,), (0, 4), True, False),
        ),
        rejected_transitions=(
            JsonRejectedTransitionExpectation((0,), 1),
            JsonRejectedTransitionExpectation((0,), 2),
            JsonRejectedTransitionExpectation((3,), 5),
        ),
    ),
    JsonParityCase(
        name="root_boolean_null_union",
        vocabulary=(
            b"true",
            b"false",
            b"null",
            b"t",
            b"f",
            b"n",
            b"rue",
            b"alse",
            b"ull",
            b"",
            b"1",
        ),
        schema='{"type":["boolean","null"]}',
        features=frozenset({"boolean", "null", "union_types"}),
        states=(
            JsonStateExpectation((), (0, 1, 2, 3, 4, 5), False, False),
            JsonStateExpectation((3,), (6,), False, False),
            JsonStateExpectation((3, 6), (), True, False),
            JsonStateExpectation((4,), (7,), False, False),
            JsonStateExpectation((4, 7), (), True, False),
            JsonStateExpectation((5,), (8,), False, False),
            JsonStateExpectation((5, 8), (), True, False),
            JsonStateExpectation((0,), (), True, False),
            JsonStateExpectation((1,), (), True, False),
            JsonStateExpectation((2,), (), True, False),
            JsonStateExpectation((9,), (0, 1, 2, 3, 4, 5), False, False),
        ),
        rejected_transitions=(JsonRejectedTransitionExpectation((), 10),),
    ),
    JsonParityCase(
        name="root_string_enum",
        vocabulary=(b'"red"', b'"blue"', b'"green"', b'"', b"r", b'ed"', b""),
        schema='{"enum":["red","blue"]}',
        features=frozenset({"enum", "string"}),
        states=(
            JsonStateExpectation((), (0, 1, 3), False, False),
            JsonStateExpectation((0,), (), True, False),
            JsonStateExpectation((1,), (), True, False),
            JsonStateExpectation((3,), (4,), False, False),
            JsonStateExpectation((3, 4), (5,), False, False),
            JsonStateExpectation((3, 4, 5), (), True, False),
            JsonStateExpectation((6,), (0, 1, 3), False, False),
        ),
        rejected_transitions=(JsonRejectedTransitionExpectation((), 2),),
    ),
    JsonParityCase(
        name="nested_object_and_typed_array",
        vocabulary=(
            b"{",
            b"}",
            b'"person"',
            b":",
            b'"name"',
            b'"tags"',
            b'"Ada"',
            b"[",
            b"]",
            b'"x"',
            b",",
            b" ",
            b'{"person":{"name":"Ada","tags":["x"]}}',
            b"",
            b'"other"',
        ),
        schema=(
            '{"type":"object","properties":{"person":{"type":"object",'
            '"properties":{"name":{"type":"string"},"tags":{"type":"array",'
            '"minItems":1,"maxItems":2,"items":{"type":"string"}}},'
            '"required":["name","tags"]}},"required":["person"]}'
        ),
        features=frozenset(
            {
                "array",
                "branching",
                "nested_structures",
                "object",
                "required_properties",
                "string",
                "typed_items",
            }
        ),
        states=(
            JsonStateExpectation((), (0, 11, 12), False, False),
            JsonStateExpectation((0,), (2, 11), False, False),
            JsonStateExpectation((0, 2, 3, 0), (4, 5, 11), False, False),
            JsonStateExpectation((0, 2, 3, 0, 4, 3, 6), (10, 11), False, False),
            JsonStateExpectation((0, 2, 3, 0, 4, 3, 6, 10), (5, 11), False, False),
            JsonStateExpectation(
                (0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7),
                (2, 4, 5, 6, 9, 11, 14),
                False,
                False,
            ),
            JsonStateExpectation(
                (0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7, 9),
                (8, 10, 11),
                False,
                False,
            ),
            JsonStateExpectation(
                (0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7, 9, 8, 1, 1),
                (),
                True,
                False,
            ),
            JsonStateExpectation((12,), (), True, False),
            JsonStateExpectation((13,), (0, 11, 12), False, False),
        ),
        rejected_transitions=(
            JsonRejectedTransitionExpectation((0, 2, 3, 0), 1),
            JsonRejectedTransitionExpectation((0, 2, 3, 0), 14),
            JsonRejectedTransitionExpectation(
                (0, 2, 3, 0, 4, 3, 6, 10, 5, 3, 7),
                8,
            ),
        ),
    ),
    JsonParityCase(
        name="valid_escaped_quote",
        vocabulary=(b'"', b'\\"', b'"\\\""', b"", b"a"),
        schema='{"type":"string","minLength":1,"maxLength":1}',
        features=frozenset(
            {"string", "string_max_length", "string_min_length", "valid_escapes"}
        ),
        states=(
            JsonStateExpectation((), (0, 2), False, False),
            JsonStateExpectation((0,), (1, 4), False, False),
            JsonStateExpectation((0, 1), (0, 1), False, False),
            JsonStateExpectation((0, 1, 0), (), True, False),
            JsonStateExpectation((2,), (), True, False),
            JsonStateExpectation((3,), (0, 2), False, False),
        ),
    ),
    JsonParityCase(
        name="structural_whitespace",
        vocabulary=(b" ", b"\t\n", b"{", b"}", b'"a"', b":", b"1", b'{ "a" : 1 }', b""),
        schema=(
            '{"type":"object","properties":{"a":{"type":"integer"}},'
            '"required":["a"]}'
        ),
        features=frozenset({"object", "structural_whitespace"}),
        states=(
            JsonStateExpectation((), (0, 1, 2, 7), False, False),
            JsonStateExpectation((0,), (0, 1, 2, 7), False, False),
            JsonStateExpectation((1,), (0, 1, 2, 7), False, False),
            JsonStateExpectation((2,), (0, 1, 4), False, False),
            JsonStateExpectation((2, 0), (0, 1, 4), False, False),
            JsonStateExpectation((2, 0, 4, 0, 5, 0, 6, 0, 3), (), True, False),
            JsonStateExpectation((7,), (), True, False),
            JsonStateExpectation((8,), (0, 1, 2, 7), False, False),
        ),
    ),
    JsonParityCase(
        name="reference_utf8_byte_length_counting",
        vocabulary=(
            b'"',
            b"\xc3",
            b"\xa9",
            b"\xc3\xa9",
            b'"\xc3\xa9"',
            b"",
            b'"a"',
            b'"\xf0\x9f\x98\x80"',
        ),
        schema='{"type":"string","minLength":2,"maxLength":2}',
        features=frozenset({"reference_quirk_utf8_byte_lengths"}),
        states=(
            JsonStateExpectation((), (0, 4), False, False),
            JsonStateExpectation((0,), (1, 2, 3), False, False),
            JsonStateExpectation((0, 1), (1, 2), False, False),
            JsonStateExpectation((0, 1, 2), (0,), False, False),
            JsonStateExpectation((0, 1, 2, 0), (), True, False),
            JsonStateExpectation((4,), (), True, False),
            JsonStateExpectation((5,), (0, 4), False, False),
        ),
        rejected_transitions=(
            JsonRejectedTransitionExpectation((), 3),
            JsonRejectedTransitionExpectation((), 6),
            JsonRejectedTransitionExpectation((), 7),
        ),
        disposition="reference_only",
    ),
    JsonParityCase(
        name="reference_invalid_escape_acceptance",
        vocabulary=(b'"', b'\\"', b"\\x", b'"\\\""', b'"\\x"', b"", b"a"),
        schema='{"type":"string","minLength":1,"maxLength":1}',
        features=frozenset({"reference_quirk_invalid_escape"}),
        states=(
            JsonStateExpectation((), (0, 3, 4), False, False),
            JsonStateExpectation((0,), (1, 2, 6), False, False),
            JsonStateExpectation((0, 2), (0, 1, 2), False, False),
            JsonStateExpectation((0, 2, 0), (), True, False),
            JsonStateExpectation((4,), (), True, False),
            JsonStateExpectation((5,), (0, 3, 4), False, False),
        ),
        disposition="reference_only",
    ),
    JsonParityCase(
        name="reference_single_byte_enum_completion",
        vocabulary=(b"2", b"3", b"null", b""),
        schema='{"enum":[2,3,null]}',
        features=frozenset({"reference_quirk_single_byte_enum"}),
        states=(
            JsonStateExpectation((), (0, 1, 2), False, False),
            JsonStateExpectation((0,), (), False, False),
            JsonStateExpectation((1,), (), False, False),
            JsonStateExpectation((2,), (), True, False),
            JsonStateExpectation((3,), (0, 1, 2), False, False),
        ),
        disposition="reference_only",
    ),
    JsonParityCase(
        name="reference_number_lexical_permissiveness",
        vocabulary=(b"0", b"1", b"01", b"-01", b"1.", b"1e", b"1e+", b"1e2", b""),
        schema='{"type":"number"}',
        features=frozenset({"reference_quirk_number_syntax"}),
        states=(
            JsonStateExpectation((), (0, 1, 2, 3, 4, 5, 6, 7), False, False),
            JsonStateExpectation((0,), (0, 1, 2, 4, 5, 6, 7), True, False),
            JsonStateExpectation((0, 1), (0, 1, 2, 4, 5, 6, 7), True, False),
            JsonStateExpectation((2,), (0, 1, 2, 4, 5, 6, 7), True, False),
            JsonStateExpectation((3,), (0, 1, 2, 4, 5, 6, 7), True, False),
            JsonStateExpectation((4,), (0, 1, 2, 5, 6, 7), False, False),
            JsonStateExpectation((5,), (0, 1, 2, 3), False, False),
            JsonStateExpectation((6,), (0, 1, 2), False, False),
            JsonStateExpectation((7,), (0, 1, 2), True, False),
            JsonStateExpectation((8,), (0, 1, 2, 3, 4, 5, 6, 7), False, False),
        ),
        disposition="reference_only",
    ),
)


JSON_SCHEMA_OBSERVATIONS = (
    JsonSchemaObservation(
        name="supported_object",
        schema='{"type":"object","properties":{}}',
        reference_compile_outcome="accepted",
        reference_behavior="enforces the recorded object subset",
        selected_windows_policy="support",
    ),
    JsonSchemaObservation(
        name="malformed_json",
        schema="not json",
        reference_compile_outcome="value_error",
        reference_behavior="compilation fails as an invalid JSON schema",
        selected_windows_policy="reject_invalid",
    ),
    JsonSchemaObservation(
        name="non_object_schema",
        schema="[]",
        reference_compile_outcome="accepted",
        reference_behavior="falls back to an unconstrained Any schema",
        selected_windows_policy="reject_invalid",
    ),
    JsonSchemaObservation(
        name="empty_object_schema",
        schema="{}",
        reference_compile_outcome="accepted",
        reference_behavior="falls back to an unconstrained Any schema",
        selected_windows_policy="reject_unsupported",
    ),
    JsonSchemaObservation(
        name="unsupported_const",
        schema='{"type":"string","const":"x"}',
        reference_compile_outcome="accepted",
        reference_behavior="ignores const and accepts other strings",
        selected_windows_policy="reject_unsupported",
    ),
    JsonSchemaObservation(
        name="unsupported_ref",
        schema='{"$ref":"#/defs/value","defs":{"value":{"type":"string"}}}',
        reference_compile_outcome="accepted",
        reference_behavior="ignores the reference and falls back to Any",
        selected_windows_policy="reject_unsupported",
    ),
    JsonSchemaObservation(
        name="unsupported_one_of",
        schema='{"oneOf":[{"type":"string"},{"type":"null"}]}',
        reference_compile_outcome="accepted",
        reference_behavior="ignores oneOf and falls back to Any",
        selected_windows_policy="reject_unsupported",
    ),
    JsonSchemaObservation(
        name="unknown_type",
        schema='{"type":"mystery"}',
        reference_compile_outcome="accepted",
        reference_behavior="maps the unknown type to Any",
        selected_windows_policy="reject_invalid",
    ),
    JsonSchemaObservation(
        name="invalid_pattern",
        schema='{"type":"string","pattern":"("}',
        reference_compile_outcome="accepted",
        reference_behavior="silently drops the invalid regex constraint",
        selected_windows_policy="reject_invalid",
    ),
    JsonSchemaObservation(
        name="invalid_min_length_type",
        schema='{"type":"string","minLength":"2"}',
        reference_compile_outcome="accepted",
        reference_behavior="silently ignores the malformed length constraint",
        selected_windows_policy="reject_invalid",
    ),
    JsonSchemaObservation(
        name="unsupported_minimum",
        schema='{"type":"number","minimum":10}',
        reference_compile_outcome="accepted",
        reference_behavior="ignores minimum and accepts values below it",
        selected_windows_policy="reject_unsupported",
    ),
    JsonSchemaObservation(
        name="unsupported_additional_properties",
        schema='{"type":"object","properties":{},"additionalProperties":true}',
        reference_compile_outcome="accepted",
        reference_behavior="ignores the keyword and still rejects undeclared keys",
        selected_windows_policy="reject_unsupported",
    ),
    JsonSchemaObservation(
        name="required_key_missing_from_properties",
        schema='{"type":"object","properties":{},"required":["x"]}',
        reference_compile_outcome="accepted",
        reference_behavior="compiles a constraint with no possible completion",
        selected_windows_policy="reject_invalid",
    ),
)


JSON_REFERENCE_QUIRKS = {
    "invalid_escape": (
        "the reference accepts unsupported escapes such as \\x; Windows will reject them under "
        "the RFC 8259 lexical policy"
    ),
    "number_syntax": (
        "the reference accepts leading-zero forms such as 01 and -01; Windows will reject them"
    ),
    "pattern_escapes": (
        "the reference feeds JSON escape designators to string-pattern DFAs; Windows will match "
        "the decoded Unicode string value"
    ),
    "single_byte_enum": (
        "the reference leaves single-byte numeric enum values incomplete; Windows will complete "
        "every fully consumed serialized enum candidate"
    ),
    "string_lengths": (
        "the reference counts UTF-8 bytes; Windows will apply minLength and maxLength to Unicode "
        "code points"
    ),
    "trailing_whitespace": (
        "the reference rejects a token containing a complete root value followed by whitespace; "
        "Windows will accept structural trailing whitespace"
    ),
}
