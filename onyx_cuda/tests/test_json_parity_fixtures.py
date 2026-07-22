import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import json_parity_fixtures as fixtures
from json_parity_fixtures import (
    JSON_EMPTY_TOKEN_POLICY,
    JSON_EXPOSE_COMPLETE_NATIVE_COMPILER,
    JSON_FUTURE_FACTORY_NAME,
    JSON_FUTURE_NATIVE_ABI_VERSION,
    JSON_INVALID_TRANSITION_POLICY,
    JSON_LEXICAL_POLICY,
    JSON_PARITY_CASES,
    JSON_PATTERN_POLICY,
    JSON_REFERENCE_QUIRKS,
    JSON_SCHEMA_OBSERVATIONS,
    JSON_STRING_LENGTH_POLICY,
    JSON_TRAILING_WHITESPACE_POLICY,
    JSON_UNICODE_ESCAPE_POLICY,
    JSON_UNSUPPORTED_KEYWORD_POLICY,
    SUPPORTED_JSON_FEATURES,
    JsonParityCase,
    JsonStateExpectation,
)


def _state_by_path(case):
    return {state.token_ids: state for state in case.states}


def test_windows_parity_cases_cover_the_complete_recorded_json_subset():
    covered = frozenset(
        feature
        for case in JSON_PARITY_CASES
        if case.disposition == "windows_parity"
        for feature in case.features
    )

    assert covered == SUPPORTED_JSON_FEATURES


def test_frozen_matrix_shape_matches_the_recorded_reference_evidence():
    assert len(JSON_PARITY_CASES) == 14
    assert sum(case.disposition == "windows_parity" for case in JSON_PARITY_CASES) == 10
    assert sum(len(case.states) for case in JSON_PARITY_CASES) == 115
    assert sum(len(case.rejected_transitions) for case in JSON_PARITY_CASES) == 24
    assert len(JSON_SCHEMA_OBSERVATIONS) == 13


def test_case_names_and_state_paths_are_unique():
    names = [case.name for case in JSON_PARITY_CASES]
    assert len(names) == len(set(names))

    for case in JSON_PARITY_CASES:
        paths = [state.token_ids for state in case.states]
        assert len(paths) == len(set(paths)), case.name
        assert paths[0] == (), case.name


@pytest.mark.parametrize("case", JSON_PARITY_CASES, ids=lambda case: case.name)
def test_frozen_state_traces_are_well_formed(case):
    assert case.vocabulary
    assert all(isinstance(token_bytes, bytes) for token_bytes in case.vocabulary)
    assert isinstance(json.loads(case.schema), dict)
    assert case.disposition in {"windows_parity", "reference_only"}
    assert case.features

    for state in case.states:
        assert all(
            type(token_id) is int and 0 <= token_id < len(case.vocabulary)
            for token_id in state.token_ids
        )
        assert state.valid_token_ids == tuple(sorted(set(state.valid_token_ids)))
        assert all(
            type(token_id) is int and 0 <= token_id < len(case.vocabulary)
            for token_id in state.valid_token_ids
        )
        assert not state.is_dead


@pytest.mark.parametrize("case", JSON_PARITY_CASES, ids=lambda case: case.name)
def test_empty_tokens_are_never_advertised_but_preserve_logical_state(case):
    empty_token_ids = tuple(
        token_id for token_id, token_bytes in enumerate(case.vocabulary) if not token_bytes
    )
    assert len(empty_token_ids) == 1
    empty_token_id = empty_token_ids[0]

    for state in case.states:
        assert empty_token_id not in state.valid_token_ids

    initial = _state_by_path(case)[()]
    after_empty = _state_by_path(case)[(empty_token_id,)]
    assert after_empty.valid_token_ids == initial.valid_token_ids
    assert after_empty.is_match == initial.is_match
    assert after_empty.is_dead == initial.is_dead


@pytest.mark.parametrize("case", JSON_PARITY_CASES, ids=lambda case: case.name)
def test_rejected_transitions_are_absent_from_the_parent_valid_set(case):
    states = _state_by_path(case)

    for rejected in case.rejected_transitions:
        parent = states[rejected.prefix_token_ids]
        assert rejected.token_id not in parent.valid_token_ids
        assert 0 <= rejected.token_id < len(case.vocabulary)
        assert rejected.reference_error_type == "ValueError"
        assert rejected.reference_message_contains == "Invalid state: Invalid byte"
        assert rejected.selected_windows_error_type == "GrammarStateError"
        assert rejected.allocates_child is False
        assert rejected.preserves_parent is True


def test_json_transition_and_empty_token_policies_are_explicit():
    assert JSON_INVALID_TRANSITION_POLICY == (
        "raise GrammarStateError without allocating a child and preserve the parent state"
    )
    assert JSON_EMPTY_TOKEN_POLICY == (
        "omit empty-byte tokens from valid-token results but return an equivalent independent "
        "child when one is advanced explicitly"
    )
    assert all(not state.is_dead for case in JSON_PARITY_CASES for state in case.states)


def test_reference_only_cases_are_separate_from_windows_parity_expectations():
    reference_only = [
        case for case in JSON_PARITY_CASES if case.disposition == "reference_only"
    ]

    assert {case.name for case in reference_only} == {
        "reference_invalid_escape_acceptance",
        "reference_number_lexical_permissiveness",
        "reference_single_byte_enum_completion",
        "reference_utf8_byte_length_counting",
    }
    assert all(
        all(feature.startswith("reference_quirk_") for feature in case.features)
        for case in reference_only
    )


def test_schema_observations_have_explicit_windows_policies():
    names = [observation.name for observation in JSON_SCHEMA_OBSERVATIONS]
    assert len(names) == len(set(names))

    for observation in JSON_SCHEMA_OBSERVATIONS:
        assert observation.reference_compile_outcome in {"accepted", "value_error"}
        assert observation.selected_windows_policy in {
            "support",
            "reject_invalid",
            "reject_unsupported",
        }
        assert observation.reference_behavior
        if observation.reference_compile_outcome == "accepted":
            json.loads(observation.schema)

    supported = {
        observation.name
        for observation in JSON_SCHEMA_OBSERVATIONS
        if observation.selected_windows_policy == "support"
    }
    assert supported == {"supported_object"}


def test_schema_interface_and_lexical_decisions_are_frozen():
    assert JSON_UNSUPPORTED_KEYWORD_POLICY == "reject before native compilation"
    assert JSON_LEXICAL_POLICY == "accept only syntactically valid RFC 8259 JSON values"
    assert JSON_UNICODE_ESCAPE_POLICY == (
        "decode valid surrogate pairs and reject lone or misordered surrogate escapes"
    )
    assert JSON_STRING_LENGTH_POLICY == "count Unicode code points rather than UTF-8 bytes"
    assert JSON_PATTERN_POLICY == (
        "apply anchored whole-string regexes to decoded Unicode string values"
    )
    assert JSON_TRAILING_WHITESPACE_POLICY == (
        "remain matching after root completion while accepting only structural whitespace bytes"
    )
    assert JSON_FUTURE_NATIVE_ABI_VERSION == 3
    assert JSON_FUTURE_FACTORY_NAME == "compile_native_json_schema"
    assert JSON_EXPOSE_COMPLETE_NATIVE_COMPILER is False


def test_every_selected_reference_quirk_has_an_explicit_windows_behavior():
    assert set(JSON_REFERENCE_QUIRKS) == {
        "invalid_escape",
        "number_syntax",
        "pattern_escapes",
        "single_byte_enum",
        "string_lengths",
        "trailing_whitespace",
    }
    assert all(description for description in JSON_REFERENCE_QUIRKS.values())


def test_fixtures_are_immutable():
    case = JSON_PARITY_CASES[0]
    state = case.states[0]

    with pytest.raises(FrozenInstanceError):
        case.name = "other"
    with pytest.raises(FrozenInstanceError):
        state.is_match = True
    assert isinstance(case, JsonParityCase)
    assert isinstance(state, JsonStateExpectation)


def test_fixture_module_has_no_reference_or_runtime_import_dependency():
    source = Path(fixtures.__file__).read_text(encoding="utf-8")

    assert "import onyx" not in source
    assert "from onyx" not in source
    assert "import mlx" not in source
    assert "import torch" not in source
    assert "_grammar_native" not in source
