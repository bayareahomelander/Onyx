"""Reproducible Phase 2 CUDA baseline benchmark."""

import argparse
import gc
import json
import platform
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers

from onyx_cuda.generation import generate_tokens
from onyx_cuda.model import MODEL_ID, load_model
from onyx_cuda.prompt import format_prompt

WARMUPS = 1
REPETITIONS = 3
MAX_TOKENS = 32
PROMPTS = {
    "cuda_ready": "Reply with CUDA ready.",
    "gpu_summary": "In one concise sentence, explain what a GPU does.",
    "number_sequence": "Write the numbers one through ten, separated by commas.",
}
DEFAULT_OUTPUT = Path("benchmarks/results/phase2_baseline.json")


def _run_prompt(model, tokenizer, device, name: str, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": user_prompt},
    ]
    prompt = format_prompt(tokenizer, messages)
    eos_token_id = tokenizer.eos_token_id

    for _ in range(WARMUPS):
        warmup = generate_tokens(
            model,
            prompt.token_ids,
            max_tokens=MAX_TOKENS,
            eos_token_ids=eos_token_id,
        )
        del warmup
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    allocation_baseline = torch.cuda.memory_allocated(device)

    runs = []
    expected_tokens = None
    expected_reason = None
    for _ in range(REPETITIONS):
        torch.cuda.reset_peak_memory_stats(device)
        result = generate_tokens(
            model,
            prompt.token_ids,
            max_tokens=MAX_TOKENS,
            eos_token_ids=eos_token_id,
            measure=True,
        )
        if result.timings is None:
            raise RuntimeError("timing was not recorded")

        token_ids = result.token_ids.copy()
        finish_reason = result.finish_reason
        if expected_tokens is None:
            expected_tokens = token_ids
            expected_reason = finish_reason
        elif token_ids != expected_tokens or finish_reason != expected_reason:
            raise RuntimeError(f"unstable greedy output for {name}")

        run = {
            "output_token_count": len(token_ids),
            "token_ids": token_ids,
            "text": tokenizer.decode(token_ids, skip_special_tokens=True),
            "finish_reason": finish_reason,
            "time_to_first_token_seconds": (
                result.timings.time_to_first_token_seconds
            ),
            "decode_tokens_per_second": (
                result.timings.decode_tokens_per_second
            ),
            "total_seconds": result.timings.total_seconds,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        }
        del result
        gc.collect()
        torch.cuda.empty_cache()
        run["allocated_vram_after_bytes"] = torch.cuda.memory_allocated(device)
        runs.append(run)

    allocations = {run["allocated_vram_after_bytes"] for run in runs}
    if allocations != {allocation_baseline}:
        raise RuntimeError(f"unstable CUDA allocation for {name}: {allocations}")

    decode_rates = [
        run["decode_tokens_per_second"]
        for run in runs
        if run["decode_tokens_per_second"] is not None
    ]
    if not decode_rates:
        raise RuntimeError(f"no decode throughput available for {name}")
    return {
        "name": name,
        "messages": messages,
        "prompt_token_count": len(prompt.token_ids),
        "runs": runs,
        "median_time_to_first_token_seconds": statistics.median(
            run["time_to_first_token_seconds"] for run in runs
        ),
        "median_decode_tokens_per_second": statistics.median(decode_rates),
        "peak_allocated_vram_bytes": max(
            run["peak_allocated_vram_bytes"] for run in runs
        ),
        "stable": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    loaded = load_model()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"id": MODEL_ID, "revision": loaded.revision},
        "dependencies": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "maturin": version("maturin"),
            "pytest": version("pytest"),
            "cuda_runtime": torch.version.cuda,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "settings": {
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "prompts": [
            _run_prompt(loaded.model, loaded.tokenizer, device, name, prompt)
            for name, prompt in PROMPTS.items()
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for prompt in results["prompts"]:
        print(
            f"{prompt['name']}: "
            f"ttft={prompt['median_time_to_first_token_seconds']:.6f}s "
            f"decode={prompt['median_decode_tokens_per_second']:.2f} tok/s "
            f"peak={prompt['peak_allocated_vram_bytes']} bytes "
            f"finish={prompt['runs'][0]['finish_reason']}"
        )


if __name__ == "__main__":
    main()
