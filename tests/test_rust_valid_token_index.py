import json

import pytest

import benchmark_rust_valid_token_index as benchmark

GrammarConstraint = pytest.importorskip("onyx._rust").GrammarConstraint


def _assert_indexed_parity(constraint, state, non_empty_token_count):
    reference = constraint.get_valid_token_ids(state)
    indexed, candidate_count = constraint.get_valid_token_ids_indexed_experimental(state)

    assert indexed == reference
    assert indexed == sorted(indexed)
    assert 0 <= candidate_count <= non_empty_token_count
    return candidate_count


def _assert_path_parity(constraint, initial_state, token_path, non_empty_token_count):
    state = initial_state
    candidate_counts = []
    for token_id in token_path:
        candidate_counts.append(_assert_indexed_parity(constraint, state, non_empty_token_count))
        state = constraint.advance_state(state, token_id)
    candidate_counts.append(_assert_indexed_parity(constraint, state, non_empty_token_count))
    return state, candidate_counts


def test_reference_lookup_does_not_build_experimental_index():
    vocabulary = [b"", b"A", b"B", b"AB", b"-", b"1", b"x", b"A"]
    constraint = GrammarConstraint(vocabulary)
    constraint.compile_regex(r"[A-Z]{2}-[0-9]")
    state = constraint.init_state()

    assert constraint.valid_token_index_built_experimental() is False
    reference = constraint.get_valid_token_ids(state)
    assert reference
    assert constraint.valid_token_index_built_experimental() is False

    indexed, candidate_count = constraint.get_valid_token_ids_indexed_experimental(state)

    assert indexed == reference
    assert candidate_count < len(vocabulary) - 1
    assert constraint.valid_token_index_built_experimental() is True
    non_empty_count, retained_bytes = constraint.build_valid_token_index_experimental()
    assert non_empty_count == len(vocabulary) - 1
    assert retained_bytes < 16 * 1024


def test_indexed_regex_lookup_matches_reference_across_states_and_unicode_splits():
    vocabulary = [b"", b"A", b"B", b"AB", b"-", b"1", b"9", b"x", b"A"]
    constraint = GrammarConstraint(vocabulary)
    constraint.compile_regex(r"[A-Z]{2}-[0-9]")
    initial = constraint.init_state()
    non_empty_count, retained_bytes = constraint.build_valid_token_index_experimental()

    final_state, candidate_counts = _assert_path_parity(
        constraint,
        initial,
        (1, 2, 4, 5),
        non_empty_count,
    )

    assert constraint.is_match_state(final_state) is True
    assert min(candidate_counts) < non_empty_count
    assert retained_bytes < 16 * 1024

    split_vocabulary = [b"", bytes([0xC3]), bytes([0xA9]), "é".encode(), b"e"]
    split_constraint = GrammarConstraint(split_vocabulary)
    split_constraint.compile_regex("é")
    split_initial = split_constraint.init_state()
    split_non_empty, _ = split_constraint.build_valid_token_index_experimental()
    split_final, _ = _assert_path_parity(
        split_constraint,
        split_initial,
        (1, 2),
        split_non_empty,
    )

    assert split_constraint.is_match_state(split_final) is True


@pytest.mark.parametrize(
    ("schema", "vocabulary", "token_path", "expected"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["ok"]},
                        },
                        "required": ["status"],
                    }
                },
                "required": ["profile"],
            },
            [b"", b"{", b'"profile"', b":", b'"status"', b'"ok"', b"}", b"x"],
            (1, 2, 3, 1, 4, 3, 5, 6, 6),
            {"profile": {"status": "ok"}},
        ),
        (
            {"type": ["string", "null"]},
            [b"", b'"', b"ok", b"null", b"true", b"x"],
            (1, 2, 1),
            "ok",
        ),
        (
            {
                "type": "string",
                "pattern": "^é$",
                "minLength": 1,
                "maxLength": 1,
            },
            [b"", b'"', bytes([0xC3]), bytes([0xA9]), b"e", rb"\u00E9"],
            (1, 2, 3, 1),
            "é",
        ),
        (
            {"type": "string", "enum": ["red", "blue"]},
            [b"", b'"red"', b'"blue"', b'"green"'],
            (1,),
            "red",
        ),
    ],
    ids=("nested-object", "type-union", "unicode-pattern-length", "enum"),
)
def test_indexed_json_lookup_matches_reference_across_structural_boundaries(
    schema,
    vocabulary,
    token_path,
    expected,
):
    constraint = GrammarConstraint(vocabulary)
    constraint.compile_json_schema(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
    initial = constraint.init_state()
    non_empty_count, retained_bytes = constraint.build_valid_token_index_experimental()

    final_state, candidate_counts = _assert_path_parity(
        constraint,
        initial,
        token_path,
        non_empty_count,
    )

    assert constraint.is_match_state(final_state) is True
    assert min(candidate_counts) < non_empty_count
    assert retained_bytes < 16 * 1024
    decoded = b"".join(vocabulary[token_id] for token_id in token_path).decode()
    assert json.loads(decoded) == expected


@pytest.mark.parametrize(
    ("prefix_token", "continuation_token", "expected"),
    [
        (3, 4, [1.5]),
        (5, 6, [100.0]),
    ],
    ids=("decimal-prefix", "exponent-prefix"),
)
def test_indexed_json_lookup_matches_reference_for_number_prefixes(
    prefix_token,
    continuation_token,
    expected,
):
    vocabulary = [
        b"",
        b"[",
        b"]",
        b"1.",
        b"5",
        b"1e+",
        b"2",
        b"01",
        b"1.e2",
        b"-e2",
    ]
    schema = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 1,
        "maxItems": 1,
    }
    constraint = GrammarConstraint(vocabulary)
    constraint.compile_json_schema(json.dumps(schema, separators=(",", ":")))
    initial = constraint.init_state()
    non_empty_count, _ = constraint.build_valid_token_index_experimental()
    in_array = constraint.advance_state(initial, 1)

    reference_at_value = constraint.get_valid_token_ids(in_array)
    indexed_at_value, candidate_count = constraint.get_valid_token_ids_indexed_experimental(
        in_array
    )
    assert indexed_at_value == reference_at_value
    assert {3, 5}.issubset(indexed_at_value)
    assert {7, 8, 9}.isdisjoint(indexed_at_value)
    assert candidate_count < non_empty_count

    final_state, _ = _assert_path_parity(
        constraint,
        in_array,
        (prefix_token, continuation_token, 2),
        non_empty_count,
    )

    assert constraint.is_match_state(final_state) is True
    decoded = b"".join(
        vocabulary[token_id] for token_id in (1, prefix_token, continuation_token, 2)
    ).decode()
    assert json.loads(decoded) == expected


def test_experimental_index_survives_reset_and_recompile_without_handle_aliasing():
    vocabulary = [b"", b"A", b"B", b'"', b"ok", b"null", b"x"]
    constraint = GrammarConstraint(vocabulary)
    constraint.compile_regex("AB")
    old_state = constraint.init_state()
    non_empty_count, retained_bytes = constraint.build_valid_token_index_experimental()
    _assert_indexed_parity(constraint, old_state, non_empty_count)

    constraint.reset()
    with pytest.raises(ValueError, match="Unknown grammar state handle"):
        constraint.get_valid_token_ids_indexed_experimental(old_state)

    constraint.compile_json_schema('{"type":["string","null"]}')
    new_state = constraint.init_state()

    assert new_state == old_state
    assert constraint.build_valid_token_index_experimental() == (
        non_empty_count,
        retained_bytes,
    )
    _assert_indexed_parity(constraint, new_state, non_empty_count)


def test_benchmark_cli_writes_machine_readable_report(monkeypatch, tmp_path):
    output_path = tmp_path / "index-report.json"
    calls = []
    report = {
        "model_id": benchmark.DEFAULT_MODEL_ID,
        "requested_revision": benchmark.DEFAULT_MODEL_REVISION,
        "resolved_revision": benchmark.DEFAULT_MODEL_REVISION,
        "vocabulary_size": 100,
        "non_empty_token_count": 90,
        "padded_or_missing_ids": 10,
        "iterations": 2,
        "index_build_ms": 1.0,
        "index_retained_bytes": 1024,
        "index_memory_bound_bytes": benchmark.MAX_INDEX_RETAINED_BYTES,
        "memory_within_bound": True,
        "rss_before_index_bytes": None,
        "rss_after_index_bytes": None,
        "rss_index_delta_bytes": None,
        "json_aggregate_speedup": 2.0,
        "minimum_json_aggregate_speedup": benchmark.MIN_JSON_AGGREGATE_SPEEDUP,
        "recommend_production_followup": True,
        "production_lookup_changed": False,
        "scenarios": [],
    }

    def fake_run_benchmark(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)

    assert (
        benchmark.main(
            [
                "--iterations",
                "2",
                "--local-files-only",
                "--json-output",
                str(output_path),
            ]
        )
        == 0
    )
    assert calls == [{"iterations": 2, "local_files_only": True}]
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_benchmark_cli_reports_runtime_failure(monkeypatch):
    def fail_benchmark(**_kwargs):
        raise RuntimeError("synthetic benchmark failure")

    monkeypatch.setattr(benchmark, "run_benchmark", fail_benchmark)

    assert benchmark.main([]) == 2
