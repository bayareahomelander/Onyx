"""Explicit installed-wheel qualification for the D23 CUDA grammar-logit mask.

This file is intentionally not named ``test_*.py``. Normal test collection remains usable without
PyTorch or CUDA; distribution qualification invokes this script from a clean CUDA environment.
"""

from __future__ import annotations

import gc
import importlib
import statistics
import sys
import time
from pathlib import Path

import onyx_cuda
from onyx_cuda import (
    TemperatureTopPSelection,
    TorchGrammarMaskInvariantError,
    create_cuda_grammar_logit_mask,
    create_cuda_sampler,
    select_cuda_argmax,
)


VOCAB_SIZE = 151_665
EXPECTED_TRANSPORT = "sparse_valid_indices"
VALID_TOKEN_IDS = (7, 64, 512, 4_096, 32_768, 75_832, 151_664)
FORBIDDEN_RUNTIME_PREFIXES = (
    "onyx",
    "mlx",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "bitsandbytes",
    "accelerate",
    "onnxruntime",
    "psutil",
)


def main() -> None:
    _require_installed_package()
    if "torch" in sys.modules:
        raise AssertionError("normal onyx_cuda import loaded PyTorch eagerly")
    if "onyx_cuda._grammar_native" in sys.modules:
        raise AssertionError("normal onyx_cuda import loaded the native grammar extension")
    _require_forbidden_runtimes_absent()

    mask = create_cuda_grammar_logit_mask(VOCAB_SIZE)
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise AssertionError("the installed PyTorch build reports CUDA unavailable")
    if mask.transport_name != EXPECTED_TRANSPORT:
        raise AssertionError(f"unexpected mask transport {mask.transport_name!r}")
    if mask.vocab_size != VOCAB_SIZE or mask.device_index != 0:
        raise AssertionError("mask configuration properties changed")

    logits = torch.linspace(
        -4.0,
        4.0,
        VOCAB_SIZE,
        dtype=torch.float32,
        device="cuda:0",
    ).to(dtype=torch.float16)
    logits[0] = 100.0
    logits[VALID_TOKEN_IDS[-1]] = 9.0
    original = logits.clone()
    before_rng = torch.cuda.get_rng_state(0).clone()
    masked = mask.apply(logits, VALID_TOKEN_IDS)
    if not torch.equal(torch.cuda.get_rng_state(0), before_rng):
        raise AssertionError("mask application advanced the global CUDA RNG")

    indices = torch.tensor(VALID_TOKEN_IDS, dtype=torch.int64, device="cuda:0")
    validity = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device="cuda:0")
    validity.index_fill_(0, indices, True)
    if not _bitwise_equal(masked[indices], original[indices], torch):
        raise AssertionError("allowed logits were not bitwise preserved")
    if not bool(torch.isneginf(masked[~validity]).all().item()):
        raise AssertionError("a disallowed logit was not negative infinity")
    if not _bitwise_equal(logits, original, torch):
        raise AssertionError("mask application mutated the input logits")
    if masked.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr():
        raise AssertionError("mask result aliases the input storage")
    if select_cuda_argmax(masked) != VALID_TOKEN_IDS[-1]:
        raise AssertionError("greedy selection escaped or misread the valid set")

    policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=23)
    first_sampler = create_cuda_sampler(policy)
    second_sampler = create_cuda_sampler(policy)
    first_sequence = tuple(first_sampler(masked) for _ in range(20))
    second_sequence = tuple(second_sampler(masked) for _ in range(20))
    if first_sequence != second_sequence:
        raise AssertionError("fresh seeded CUDA sampling sessions did not replay")
    if not set(first_sequence) <= set(VALID_TOKEN_IDS):
        raise AssertionError("seeded CUDA sampling escaped the valid set")

    try:
        mask.apply(
            torch.full((VOCAB_SIZE,), float("-inf"), dtype=torch.float16, device="cuda:0"),
            VALID_TOKEN_IDS,
        )
    except TorchGrammarMaskInvariantError:
        pass
    else:
        raise AssertionError("mask accepted valid IDs with only negative-infinity support")

    del masked, original, validity, indices
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    baseline_allocated = torch.cuda.memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0)

    full_valid_ids = tuple(range(VOCAB_SIZE))
    block_snapshots = []
    wall_ms = []
    for block in range(5):
        for _ in range(50):
            torch.cuda.synchronize(0)
            started = time.perf_counter_ns()
            result = mask.apply(logits, full_valid_ids)
            torch.cuda.synchronize(0)
            wall_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            del result
        gc.collect()
        torch.cuda.synchronize(0)
        block_snapshots.append(
            (torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0))
        )
    if any(allocated != baseline_allocated for allocated, _ in block_snapshots):
        raise AssertionError(
            f"mask retained CUDA allocations across blocks: {block_snapshots!r}"
        )
    if len({reserved for _, reserved in block_snapshots}) != 1:
        raise AssertionError(
            f"mask reserved-memory cache grew across blocks: {block_snapshots!r}"
        )

    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    median_ms = statistics.median(wall_ms)
    p95_ms = sorted(wall_ms)[int((len(wall_ms) - 1) * 0.95)]
    _require_forbidden_runtimes_absent()
    if "onyx_cuda._grammar_native" in sys.modules:
        raise AssertionError("mask qualification loaded the native grammar extension")

    properties = torch.cuda.get_device_properties(0)
    print(
        "installed CUDA grammar-mask qualification passed: "
        f"transport={EXPECTED_TRANSPORT} vocab_size={VOCAB_SIZE} "
        f"device={properties.name!r} capability={properties.major}.{properties.minor} "
        f"pytorch={torch.__version__} compiled_cuda={torch.version.cuda} "
        f"full_support_median_ms={median_ms:.6f} full_support_p95_ms={p95_ms:.6f} "
        f"peak_allocated_bytes={peak_allocated} peak_reserved_bytes={peak_reserved} "
        f"retained_blocks={block_snapshots!r}"
    )


def _bitwise_equal(first, second, torch) -> bool:
    return bool(
        torch.equal(
            first.contiguous().view(torch.uint8),
            second.contiguous().view(torch.uint8),
        )
    )


def _require_installed_package() -> None:
    package_path = Path(onyx_cuda.__file__).resolve()
    environment = Path(sys.prefix).resolve()
    if not package_path.is_relative_to(environment):
        raise AssertionError(
            f"qualification imported onyx_cuda outside the clean environment: {package_path}"
        )


def _require_forbidden_runtimes_absent() -> None:
    loaded = tuple(sys.modules)
    for prefix in FORBIDDEN_RUNTIME_PREFIXES:
        if any(name == prefix or name.startswith(f"{prefix}.") for name in loaded):
            raise AssertionError(f"qualification imported forbidden runtime {prefix!r}")


if __name__ == "__main__":
    main()
