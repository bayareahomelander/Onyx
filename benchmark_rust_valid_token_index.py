"""Benchmark experimental Rust valid-token candidate indexing without a model or CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from onyx.tokenizer_compat import build_grammar_vocabulary
from onyx_cuda.real_logits_handoff import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from onyx_cuda.tokenizer_probe import _load_tokenizer_metadata

MAX_INDEX_RETAINED_BYTES = 2 * 1024 * 1024
MIN_JSON_AGGREGATE_SPEEDUP = 1.25
DEFAULT_ITERATIONS = 5


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    constraint_type: str
    source: str
    prefix: str


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    constraint_type: str
    prefix: str
    valid_token_count: int
    candidate_count: int
    candidate_reduction_fraction: float
    reference_median_ms: float
    reference_p95_ms: float
    indexed_median_ms: float
    indexed_p95_ms: float
    speedup: float


def _benchmark_scenarios() -> Tuple[BenchmarkScenario, ...]:
    nested_schema = json.dumps(
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
        separators=(",", ":"),
    )
    array_schema = json.dumps(
        {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        separators=(",", ":"),
    )
    enum_schema = json.dumps(
        {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok", "error"]}},
            "required": ["status"],
        },
        separators=(",", ":"),
    )
    union_schema = '{"type":["string","null"]}'
    pattern_schema = json.dumps(
        {
            "type": "string",
            "pattern": "^[A-Z]{2}$",
            "minLength": 2,
            "maxLength": 2,
        },
        separators=(",", ":"),
    )
    number_schema = json.dumps(
        {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1,
            "maxItems": 1,
        },
        separators=(",", ":"),
    )
    return (
        BenchmarkScenario("regex_initial", "regex", r"ONY-[0-9]{4}", ""),
        BenchmarkScenario("regex_mid", "regex", r"ONY-[0-9]{4}", "ONY-20"),
        BenchmarkScenario("regex_unicode", "regex", "café", "caf"),
        BenchmarkScenario("json_nested", "json_schema", nested_schema, '{"profile":'),
        BenchmarkScenario("json_typed_array", "json_schema", array_schema, "[1,"),
        BenchmarkScenario("json_enum", "json_schema", enum_schema, '{"status":'),
        BenchmarkScenario("json_union", "json_schema", union_schema, ""),
        BenchmarkScenario("json_pattern_length", "json_schema", pattern_schema, '"O'),
        BenchmarkScenario("json_decimal_prefix", "json_schema", number_schema, "[1."),
        BenchmarkScenario("json_exponent_prefix", "json_schema", number_schema, "[1e+"),
    )


def _process_rss_bytes() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _percentile_95(samples: Sequence[int]) -> float:
    ordered = sorted(samples)
    index = round(0.95 * (len(ordered) - 1))
    return float(ordered[index])


def _time_lookup(call, iterations: int) -> Tuple[Any, float, float]:
    samples_ns = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = call()
        samples_ns.append(time.perf_counter_ns() - started)
    return (
        result,
        statistics.median(samples_ns) / 1_000_000,
        _percentile_95(samples_ns) / 1_000_000,
    )


def _encode_exact_prefix(tokenizer: Any, vocabulary: Sequence[bytes], text: str) -> Tuple[int, ...]:
    token_ids = tuple(int(item) for item in tokenizer.encode(text, add_special_tokens=False))
    reconstructed = b"".join(vocabulary[token_id] for token_id in token_ids)
    if reconstructed != text.encode("utf-8"):
        raise ValueError(
            f"tokenizer bytes did not reconstruct benchmark prefix {text!r}: "
            f"expected={text.encode('utf-8').hex()}, actual={reconstructed.hex()}"
        )
    return token_ids


def _prepare_state(
    constraint: Any, tokenizer: Any, vocabulary: Sequence[bytes], prefix: str
) -> int:
    state = int(constraint.init_state())
    for token_id in _encode_exact_prefix(tokenizer, vocabulary, prefix):
        next_state = int(constraint.advance_state(state, token_id))
        constraint.release_state(state)
        state = next_state
    return state


def run_benchmark(*, iterations: int, local_files_only: bool) -> Dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    try:
        from onyx._rust import GrammarConstraint
    except ImportError as exc:
        raise RuntimeError(
            "benchmark requires the Rust extension; run "
            "`.venv\\Scripts\\python.exe -m maturin develop --release`"
        ) from exc

    config, tokenizer, resolved_revision = _load_tokenizer_metadata(
        DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        local_files_only=local_files_only,
    )
    vocabulary, vocabulary_stats, errors = build_grammar_vocabulary(
        tokenizer,
        int(config.vocab_size),
    )
    if errors:
        raise ValueError(f"tokenizer vocabulary mapping failed: {errors[0]}")

    constraint = GrammarConstraint(vocabulary)
    rss_before_index = _process_rss_bytes()
    build_started = time.perf_counter_ns()
    non_empty_token_count, index_retained_bytes = constraint.build_valid_token_index_experimental()
    index_build_ms = (time.perf_counter_ns() - build_started) / 1_000_000
    rss_after_index = _process_rss_bytes()
    expected_non_empty_token_count = sum(bool(token_bytes) for token_bytes in vocabulary)
    if non_empty_token_count != expected_non_empty_token_count:
        raise AssertionError(
            "candidate index token count mismatch: "
            f"expected={expected_non_empty_token_count}, actual={non_empty_token_count}"
        )

    scenario_results: List[ScenarioResult] = []
    for scenario in _benchmark_scenarios():
        if scenario.constraint_type == "regex":
            constraint.compile_regex(scenario.source)
        else:
            constraint.compile_json_schema(scenario.source)
        state = _prepare_state(constraint, tokenizer, vocabulary, scenario.prefix)
        try:
            reference_once = constraint.get_valid_token_ids(state)
            indexed_once, candidate_count = constraint.get_valid_token_ids_indexed_experimental(
                state
            )
            if indexed_once != reference_once:
                raise AssertionError(
                    f"indexed lookup mismatch for {scenario.name}: "
                    f"reference_count={len(reference_once)}, indexed_count={len(indexed_once)}"
                )
            if not 0 <= candidate_count <= non_empty_token_count:
                raise AssertionError(
                    f"candidate count out of bounds for {scenario.name}: {candidate_count}"
                )

            reference_result, reference_median_ms, reference_p95_ms = _time_lookup(
                lambda: constraint.get_valid_token_ids(state),
                iterations,
            )
            indexed_result, indexed_median_ms, indexed_p95_ms = _time_lookup(
                lambda: constraint.get_valid_token_ids_indexed_experimental(state),
                iterations,
            )
            indexed_ids, repeated_candidate_count = indexed_result
            if indexed_ids != reference_result or repeated_candidate_count != candidate_count:
                raise AssertionError(f"repeated indexed lookup changed for {scenario.name}")

            speedup = reference_median_ms / max(indexed_median_ms, 1e-9)
            scenario_results.append(
                ScenarioResult(
                    name=scenario.name,
                    constraint_type=scenario.constraint_type,
                    prefix=scenario.prefix,
                    valid_token_count=len(reference_once),
                    candidate_count=int(candidate_count),
                    candidate_reduction_fraction=(
                        1.0 - (candidate_count / non_empty_token_count)
                        if non_empty_token_count
                        else 0.0
                    ),
                    reference_median_ms=reference_median_ms,
                    reference_p95_ms=reference_p95_ms,
                    indexed_median_ms=indexed_median_ms,
                    indexed_p95_ms=indexed_p95_ms,
                    speedup=speedup,
                )
            )
        finally:
            constraint.release_state(state)

    json_results = [
        result for result in scenario_results if result.constraint_type == "json_schema"
    ]
    json_reference_total_ms = sum(item.reference_median_ms for item in json_results)
    json_indexed_total_ms = sum(item.indexed_median_ms for item in json_results)
    json_aggregate_speedup = json_reference_total_ms / max(json_indexed_total_ms, 1e-9)
    memory_within_bound = index_retained_bytes <= MAX_INDEX_RETAINED_BYTES
    recommend_followup = (
        memory_within_bound and json_aggregate_speedup >= MIN_JSON_AGGREGATE_SPEEDUP
    )

    return {
        "model_id": DEFAULT_MODEL_ID,
        "requested_revision": DEFAULT_MODEL_REVISION,
        "resolved_revision": resolved_revision,
        "vocabulary_size": len(vocabulary),
        "non_empty_token_count": int(non_empty_token_count),
        "padded_or_missing_ids": int(vocabulary_stats["padded_or_missing_ids"]),
        "iterations": iterations,
        "index_build_ms": index_build_ms,
        "index_retained_bytes": int(index_retained_bytes),
        "index_memory_bound_bytes": MAX_INDEX_RETAINED_BYTES,
        "memory_within_bound": memory_within_bound,
        "rss_before_index_bytes": rss_before_index,
        "rss_after_index_bytes": rss_after_index,
        "rss_index_delta_bytes": (
            rss_after_index - rss_before_index
            if rss_before_index is not None and rss_after_index is not None
            else None
        ),
        "json_aggregate_speedup": json_aggregate_speedup,
        "minimum_json_aggregate_speedup": MIN_JSON_AGGREGATE_SPEEDUP,
        "recommend_production_followup": recommend_followup,
        "production_lookup_changed": False,
        "scenarios": [asdict(item) for item in scenario_results],
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        "Rust valid-token candidate-index experiment",
        f"  Model metadata: {report['model_id']}",
        f"  Revision: {report['resolved_revision'] or report['requested_revision']}",
        (
            "  Vocabulary: "
            f"{report['vocabulary_size']:,} IDs, "
            f"{report['non_empty_token_count']:,} non-empty"
        ),
        f"  Index build: {report['index_build_ms']:.2f} ms",
        f"  Index retained: {report['index_retained_bytes'] / (1024 * 1024):.2f} MiB",
        *(
            [f"  Observed RSS delta: {report['rss_index_delta_bytes'] / (1024 * 1024):.2f} MiB"]
            if report["rss_index_delta_bytes"] is not None
            else []
        ),
        "",
        "  Scenario                 valid  candidates  reduction  reference  indexed  speedup",
    ]
    for item in report["scenarios"]:
        lines.append(
            f"  {item['name']:<24} "
            f"{item['valid_token_count']:>6}  "
            f"{item['candidate_count']:>10}  "
            f"{item['candidate_reduction_fraction'] * 100:>8.1f}%  "
            f"{item['reference_median_ms']:>7.2f} ms  "
            f"{item['indexed_median_ms']:>7.2f} ms  "
            f"{item['speedup']:>6.2f}x"
        )
    recommendation = "GO" if report["recommend_production_followup"] else "NO-GO"
    lines.extend(
        [
            "",
            f"  Aggregate JSON speedup: {report['json_aggregate_speedup']:.2f}x",
            f"  Memory bound satisfied: {report['memory_within_bound']}",
            f"  Production follow-up recommendation: {recommendation}",
            "  Production lookup changed: False",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Rust reference vocabulary scan with an experimental "
            "first-byte candidate index without loading model weights or CUDA."
        )
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only the pinned tokenizer/config files already in the Hugging Face cache",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_benchmark(
            iterations=args.iterations,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print(f"Rust valid-token index benchmark could not run: {type(exc).__name__}: {exc}")
        return 2

    print(format_report(report))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  JSON report: {args.json_output}")
    return 0 if report["memory_within_bound"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
