#!/usr/bin/env python3
"""Benchmark the model-free multi-token Rust grammar/CUDA decode loop."""

from __future__ import annotations

import statistics
import time
from typing import List, Tuple

from onyx_cuda.decode_loop import CudaGrammarDecodeResult, decode_greedy_from_logits

EXPECTED_BYTES = b"ONY-2026"
PATTERN = "[A-Z]{3}-[0-9]{4}"
EXTRA_INVALID_TOKENS = 4_096
REPEATS = 100
WARMUP = 10


def _require_runtime():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the CUDA decode-loop benchmark.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch does not report CUDA availability.")

    try:
        from onyx._rust import GrammarConstraint
    except ImportError as exc:
        raise SystemExit("The onyx Rust grammar extension is required.") from exc

    return torch, GrammarConstraint


def _build_case(torch, GrammarConstraint):
    vocab = [bytes([letter]) for letter in range(ord("A"), ord("Z") + 1)]
    vocab.append(b"-")
    vocab.extend(str(digit).encode("ascii") for digit in range(10))
    vocab.extend(f" invalid_{index}".encode("ascii") for index in range(EXTRA_INVALID_TOKENS))
    token_to_id = {token: index for index, token in enumerate(vocab)}

    constraint = GrammarConstraint(vocab)
    constraint.compile_regex(PATTERN)

    invalid_id = token_to_id[b" invalid_0"]
    expected_ids = tuple(token_to_id[bytes([byte])] for byte in EXPECTED_BYTES)
    logits_steps = []
    for token_id in expected_ids:
        logits = torch.zeros(len(vocab), device="cuda", dtype=torch.float32)
        logits[invalid_id] = 100.0
        logits[token_id] = 10.0
        logits_steps.append(logits)

    return constraint, vocab, logits_steps, expected_ids


def _run_once(constraint, logits_steps) -> Tuple[CudaGrammarDecodeResult, float]:
    constraint.reset()
    state = constraint.init_state()
    start_ns = time.perf_counter_ns()
    result = decode_greedy_from_logits(
        logits_steps,
        constraint,
        state,
        max_steps=len(logits_steps),
        check_inputs=False,
    )
    end_to_end_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    return result, end_to_end_us


def _mean_timings(results: List[CudaGrammarDecodeResult]) -> List[Tuple[str, float]]:
    return [
        (
            "valid_id_lookup",
            statistics.mean(result.timings.valid_id_lookup_us for result in results),
        ),
        (
            "selection_call",
            statistics.mean(result.timings.selection_call_us for result in results),
        ),
        (
            "result_sync",
            statistics.mean(result.timings.result_sync_us for result in results),
        ),
        (
            "grammar_advance",
            statistics.mean(result.timings.grammar_advance_us for result in results),
        ),
        (
            "pre_cleanup",
            statistics.mean(result.timings.pre_cleanup_us for result in results),
        ),
    ]


def main() -> int:
    torch, GrammarConstraint = _require_runtime()
    constraint, vocab, logits_steps, expected_ids = _build_case(torch, GrammarConstraint)

    first, _ = _run_once(constraint, logits_steps)
    if first.token_ids != expected_ids or not first.matched:
        raise AssertionError(
            f"decode mismatch: ids={first.token_ids}, matched={first.matched}, "
            f"reason={first.termination_reason}"
        )

    decoded = b"".join(vocab[token_id] for token_id in first.token_ids)
    if decoded != EXPECTED_BYTES:
        raise AssertionError(f"decoded {decoded!r}, expected {EXPECTED_BYTES!r}")

    for _ in range(WARMUP):
        _run_once(constraint, logits_steps)
    torch.cuda.synchronize()

    measured_runs = [_run_once(constraint, logits_steps) for _ in range(REPEATS)]
    torch.cuda.synchronize()

    results = [result for result, _ in measured_runs]
    end_to_end_us = statistics.mean(elapsed_us for _, elapsed_us in measured_runs)
    mean_timings = _mean_timings(results)
    mean_timings.append(("end_to_end_call", end_to_end_us))
    tokens_per_second = (len(expected_ids) * 1_000_000.0) / end_to_end_us

    print("Onyx CUDA model-free grammar decode-loop benchmark")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"vocab_size: {len(vocab)}")
    print(f"pattern: {PATTERN}")
    print(f"decoded: {decoded!r}")
    print(f"steps: {first.steps}")
    print(f"repeats: {REPEATS}, warmup: {WARMUP}")
    print()
    print(f"{'stage':<24} {'us/decode':>14}")
    for stage, microseconds in mean_timings:
        print(f"{stage:<24} {microseconds:14.2f}")
    print()
    print(f"end_to_end_tokens_per_second: {tokens_per_second:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
