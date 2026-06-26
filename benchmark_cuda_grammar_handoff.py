#!/usr/bin/env python3
"""benchmark the Rust grammar-to-CUDA masked-argmax handoff."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List

from onyx_cuda.grammar_handoff import (
    CudaValidIdCache,
    masked_argmax_from_cached_grammar_state,
    masked_argmax_from_grammar_state,
)
from onyx_cuda.masked_argmax import masked_argmax_tensor

REPEATS = 1_000
WARMUP = 100


@dataclass
class HandoffCase:
    vocab: List[bytes]
    token_to_id: Dict[bytes, int]
    grammar_constraint: object
    grammar_state: int
    valid_token_ids: List[int]
    expected_token_id: int


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Install a CUDA-enabled build, then run "
            "`python -m pip install -e .[cuda]`."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch does not report CUDA availability.")

    return torch


def _require_grammar_constraint():
    try:
        from onyx._rust import GrammarConstraint
    except ImportError as exc:
        raise SystemExit(
            "the Rust grammar extension is required. Run "
            "`python -m pip install -e .` or `python -m maturin develop --release`."
        ) from exc

    return GrammarConstraint


def _build_handoff_case(extra_invalid_tokens: int = 4096) -> HandoffCase:
    GrammarConstraint = _require_grammar_constraint()

    vocab = [b"The", b" year", b" is "]
    vocab.extend(str(digit).encode("ascii") for digit in range(10))
    vocab.extend(f" invalid_{index}".encode("ascii") for index in range(extra_invalid_tokens))

    token_to_id = {token: index for index, token in enumerate(vocab)}
    grammar_constraint = GrammarConstraint(vocab)
    grammar_constraint.compile_regex("The year is [0-9]")

    state = grammar_constraint.init_state()
    for token in (b"The", b" year", b" is "):
        state = grammar_constraint.advance_state(state, token_to_id[token])

    valid_token_ids = grammar_constraint.get_valid_token_ids(state)
    expected_token_id = token_to_id[b"7"]
    if expected_token_id not in valid_token_ids:
        raise AssertionError("expected digit token is not grammar-valid")

    return HandoffCase(
        vocab=vocab,
        token_to_id=token_to_id,
        grammar_constraint=grammar_constraint,
        grammar_state=state,
        valid_token_ids=valid_token_ids,
        expected_token_id=expected_token_id,
    )


def _make_logits(torch, case: HandoffCase):
    logits = torch.zeros(len(case.vocab), device="cuda", dtype=torch.float32)
    invalid_id = case.token_to_id[b" invalid_0"]

    logits[invalid_id] = 100.0
    logits[case.expected_token_id] = 10.0
    return logits


def _time_cpu_us(fn: Callable, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return ((time.perf_counter() - start) * 1_000_000.0) / repeats


def _time_cuda_us(torch, fn: Callable, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()

    return (start.elapsed_time(end) * 1_000.0) / repeats


def _time_synchronized_us(
    torch,
    fn: Callable,
    repeats: int = REPEATS,
    warmup: int = WARMUP,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()

    return ((time.perf_counter() - start) * 1_000_000.0) / repeats


def main() -> int:
    torch = _require_torch()
    case = _build_handoff_case()
    logits = _make_logits(torch, case)

    valid_ids_cuda = torch.as_tensor(case.valid_token_ids, dtype=torch.long, device="cuda")
    valid_id_cache = CudaValidIdCache(case.grammar_constraint)
    selected = int(
        masked_argmax_from_grammar_state(
            logits,
            case.grammar_constraint,
            case.grammar_state,
            check_inputs=False,
        ).item()
    )
    if selected != case.expected_token_id:
        raise AssertionError(f"selected {selected}, expected {case.expected_token_id}")

    cached_selected = int(
        masked_argmax_from_cached_grammar_state(
            logits,
            valid_id_cache,
            case.grammar_state,
            check_inputs=False,
        ).item()
    )
    if cached_selected != case.expected_token_id:
        raise AssertionError(
            f"cached selected {cached_selected}, expected {case.expected_token_id}"
        )

    valid_id_cache.get(case.grammar_state, logits.device)

    rust_valid_us = _time_cpu_us(
        lambda: case.grammar_constraint.get_valid_token_ids(case.grammar_state)
    )
    upload_us = _time_synchronized_us(
        torch,
        lambda: torch.as_tensor(case.valid_token_ids, dtype=torch.long, device="cuda"),
    )
    cuda_select_us = _time_cuda_us(
        torch,
        lambda: masked_argmax_tensor(logits, valid_ids_cuda, check_inputs=False),
    )
    handoff_us = _time_synchronized_us(
        torch,
        lambda: masked_argmax_from_grammar_state(
            logits,
            case.grammar_constraint,
            case.grammar_state,
            check_inputs=False,
        ).item(),
    )
    cached_handoff_us = _time_synchronized_us(
        torch,
        lambda: masked_argmax_from_cached_grammar_state(
            logits,
            valid_id_cache,
            case.grammar_state,
            check_inputs=False,
        ).item(),
    )

    print("Onyx CUDA grammar handoff benchmark")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"vocab_size: {len(case.vocab)}")
    print(f"valid_tokens: {len(case.valid_token_ids)}")
    print(f"selected_token_id: {selected}")
    print(f"selected_token_bytes: {case.vocab[selected]!r}")
    print()
    print(f"{'stage':<24} {'us':>12}")
    print(f"{'rust_valid_ids':<24} {rust_valid_us:12.2f}")
    print(f"{'valid_ids_to_cuda':<24} {upload_us:12.2f}")
    print(f"{'cuda_masked_argmax':<24} {cuda_select_us:12.2f}")
    print(f"{'end_to_end_handoff':<24} {handoff_us:12.2f}")
    print(f"{'cached_handoff':<24} {cached_handoff_us:12.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
