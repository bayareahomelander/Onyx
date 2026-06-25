#!/usr/bin/env python3
"""Microbenchmark the experimental CUDA masked-argmax kernel.

This benchmark uses synthetic logits and valid-token sets. It does not load an
LLM, so it is suitable for small GPUs and for isolating kernel behavior.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Tuple

from onyx_cuda.masked_argmax import masked_argmax_tensor

VOCAB_SIZES = [32_768, 131_072, 262_144]
VALID_COUNTS = [8, 32, 128, 1_024, 8_192]
REPEATS = 1_000
WARMUP = 100


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


def _cpu_masked_argmax(logits_cpu, valid_ids: Iterable[int]) -> int:
    best_id = None
    best_value = None
    for token_id in valid_ids:
        value = float(logits_cpu[token_id])
        if best_id is None or value > best_value or (value == best_value and token_id < best_id):
            best_id = int(token_id)
            best_value = value
    return best_id


def _torch_gather_argmax(torch, logits, valid_ids):
    gathered = logits.index_select(0, valid_ids)
    max_value = gathered.max()
    tied_ids = valid_ids[gathered == max_value]
    return tied_ids.min()


def _time_cuda(torch, fn: Callable, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
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


def _time_cpu(fn: Callable, repeats: int = 100) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return ((time.perf_counter() - start) * 1_000_000.0) / repeats


def _make_case(torch, vocab_size: int, valid_count: int):
    logits = torch.randn(vocab_size, device="cuda", dtype=torch.float32)
    valid_ids = torch.randperm(vocab_size, device="cuda", dtype=torch.long)[:valid_count]
    return logits, valid_ids


def _check_correctness(torch, logits, valid_ids) -> Tuple[int, int]:
    custom = int(masked_argmax_tensor(logits, valid_ids, check_inputs=False).item())
    reference = int(_torch_gather_argmax(torch, logits, valid_ids).item())
    if custom != reference:
        raise AssertionError(f"custom CUDA result {custom} != PyTorch reference {reference}")
    return custom, reference


def main() -> int:
    torch = _require_torch()

    device_name = torch.cuda.get_device_name()
    print("Onyx CUDA masked-argmax benchmark")
    print(f"device: {device_name}")
    print(f"repeats: cuda={REPEATS}, warmup={WARMUP}")
    print()
    print(
        f"{'vocab':>8} {'valid':>8} "
        f"{'cpu_py_us':>12} {'torch_us':>12} {'custom_us':>12} {'custom/torch':>13}"
    )

    for vocab_size in VOCAB_SIZES:
        for valid_count in VALID_COUNTS:
            if valid_count > vocab_size:
                continue

            logits, valid_ids = _make_case(torch, vocab_size, valid_count)
            _check_correctness(torch, logits, valid_ids)

            logits_cpu = logits.cpu()
            valid_list = [int(x) for x in valid_ids.cpu().tolist()]

            cpu_us = _time_cpu(lambda: _cpu_masked_argmax(logits_cpu, valid_list))
            torch_us = _time_cuda(torch, lambda: _torch_gather_argmax(torch, logits, valid_ids))
            custom_us = _time_cuda(
                torch,
                lambda: masked_argmax_tensor(logits, valid_ids, check_inputs=False),
            )
            ratio = custom_us / torch_us if torch_us > 0 else float("inf")

            print(
                f"{vocab_size:8d} {valid_count:8d} "
                f"{cpu_us:12.2f} {torch_us:12.2f} {custom_us:12.2f} {ratio:13.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
