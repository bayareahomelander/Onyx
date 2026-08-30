"""Reproducible CUDA baseline and constraint gates."""

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
from onyx_cuda.model import MODEL_ID, TARGET_MODEL_ID, load_model
from onyx_cuda.prompt import format_prompt
from onyx_cuda.vocabulary import build_token_byte_vocabulary

WARMUPS = 1
REPETITIONS = 3
MAX_TOKENS = 32
PROMPTS = {
    "cuda_ready": "Reply with CUDA ready.",
    "gpu_summary": "In one concise sentence, explain what a GPU does.",
    "number_sequence": "Write the numbers one through ten, separated by commas.",
}
DEFAULT_OUTPUT = Path("benchmarks/results/phase2_baseline.json")
CONSTRAINT_OUTPUT = Path("benchmarks/results/phase3_constraint_gate.json")
TARGET_OUTPUT = Path("benchmarks/results/phase4_target_baseline.json")
TARGET_CONSTRAINT_OUTPUT = Path(
    "benchmarks/results/phase4_target_constraint_gate.json"
)


def _run_prompt(
    model,
    tokenizer,
    device,
    name: str,
    user_prompt: str,
    *,
    system_prompt: str = "You are a concise assistant.",
    generation_options: dict | None = None,
) -> dict:
    generation_options = generation_options or {}
    messages = [
        {"role": "system", "content": system_prompt},
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
            **generation_options,
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
            **generation_options,
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
        if result.timings.grammar_compile_seconds is not None:
            run.update(
                {
                    "grammar_compile_seconds": (
                        result.timings.grammar_compile_seconds
                    ),
                    "valid_token_enumeration_seconds": (
                        result.timings.valid_token_enumeration_seconds
                    ),
                    "mask_transfer_seconds": (
                        result.timings.mask_transfer_seconds
                    ),
                    "output_tokens_per_second": (
                        len(token_ids) / result.timings.total_seconds
                    ),
                }
            )
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
    summary = {
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
    if "grammar_compile_seconds" in runs[0]:
        summary.update(
            {
                "median_grammar_compile_seconds": statistics.median(
                    run["grammar_compile_seconds"] for run in runs
                ),
                "median_valid_token_enumeration_seconds": statistics.median(
                    run["valid_token_enumeration_seconds"] for run in runs
                ),
                "median_mask_transfer_seconds": statistics.median(
                    run["mask_transfer_seconds"] for run in runs
                ),
                "median_output_tokens_per_second": statistics.median(
                    run["output_tokens_per_second"] for run in runs
                ),
            }
        )
    return summary


def _assert_baseline(
    current_prompts: list[dict], baseline: dict, label: str
) -> None:
    baseline_prompts = {prompt["name"]: prompt for prompt in baseline["prompts"]}
    for current in current_prompts:
        expected = baseline_prompts.get(current["name"])
        if expected is None:
            raise RuntimeError(f"{label} is missing {current['name']}")
        expected_run = expected["runs"][0]
        current_run = current["runs"][0]
        if (
            current_run["token_ids"] != expected_run["token_ids"]
            or current_run["finish_reason"] != expected_run["finish_reason"]
        ):
            raise RuntimeError(
                f"unconstrained output changed from {label} for {current['name']}"
            )


def _run_constraint_prompts(model, tokenizer, device) -> list[dict]:
    vocabulary = build_token_byte_vocabulary(tokenizer, model.config.vocab_size)
    regex_pattern = "CUDA Ready"
    regex_result = _run_prompt(
        model,
        tokenizer,
        device,
        "regex_cuda_ready",
        "Reply with CUDA ready.",
        generation_options={
            "regex": regex_pattern,
            "token_byte_vocabulary": vocabulary,
        },
    )
    if regex_result["runs"][0]["text"] != regex_pattern:
        raise RuntimeError("regex-constrained output did not match exactly")

    schema = {
        "type": "object",
        "properties": {"content": {"enum": ["CUDA ready", "Ready"]}},
        "required": ["content"],
    }
    json_result = _run_prompt(
        model,
        tokenizer,
        device,
        "json_required_enum",
        "Use no spaces or newlines in the JSON response.",
        system_prompt="Return compact JSON only.",
        generation_options={
            "json_schema": json.dumps(schema, separators=(",", ":")),
            "token_byte_vocabulary": vocabulary,
        },
    )
    parsed = json.loads(json_result["runs"][0]["text"])
    if set(parsed) != {"content"} or parsed["content"] not in {
        "CUDA ready",
        "Ready",
    }:
        raise RuntimeError("JSON-constrained output did not satisfy its schema")
    return [regex_result, json_result]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", action="store_true")
    parser.add_argument("--target", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model_id = TARGET_MODEL_ID if args.target else MODEL_ID
    baseline = args.baseline or (TARGET_OUTPUT if args.target else DEFAULT_OUTPUT)
    if args.target:
        default_output = (
            TARGET_CONSTRAINT_OUTPUT if args.constraints else TARGET_OUTPUT
        )
    else:
        default_output = CONSTRAINT_OUTPUT if args.constraints else DEFAULT_OUTPUT
    output = args.output or default_output

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    loaded = load_model(model_id)
    prompts = [
        _run_prompt(loaded.model, loaded.tokenizer, device, name, prompt)
        for name, prompt in PROMPTS.items()
    ]
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"id": model_id, "revision": loaded.revision},
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
    }
    printed_prompts = prompts
    if args.constraints:
        baseline_label = "target baseline" if args.target else "Phase 2 baseline"
        if not baseline.is_file():
            raise RuntimeError(
                f"{baseline_label} not found at {baseline}; "
                f"run the {'target' if args.target else 'default'} benchmark first"
            )
        baseline_results = json.loads(baseline.read_text(encoding="utf-8"))
        if baseline_results.get("model") != results["model"]:
            raise RuntimeError(f"{baseline_label} model or revision does not match")
        _assert_baseline(prompts, baseline_results, baseline_label)
        constraint_prompts = _run_constraint_prompts(
            loaded.model, loaded.tokenizer, device
        )
        baseline_key = "target_baseline" if args.target else "phase2_baseline"
        results.update(
            {
                baseline_key: str(baseline),
                "unconstrained_prompts": prompts,
                "constraint_prompts": constraint_prompts,
            }
        )
        printed_prompts = constraint_prompts
    else:
        results["prompts"] = prompts

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    if args.constraints:
        print(f"{baseline_label} unconstrained outputs unchanged")
    for prompt in printed_prompts:
        constraint_metrics = ""
        if "median_grammar_compile_seconds" in prompt:
            constraint_metrics = (
                f" compile={prompt['median_grammar_compile_seconds']:.6f}s"
                " enum="
                f"{prompt['median_valid_token_enumeration_seconds']:.6f}s"
                f" mask={prompt['median_mask_transfer_seconds']:.6f}s"
                f" output={prompt['median_output_tokens_per_second']:.2f} tok/s"
            )
        print(
            f"{prompt['name']}: "
            f"ttft={prompt['median_time_to_first_token_seconds']:.6f}s "
            f"decode={prompt['median_decode_tokens_per_second']:.2f} tok/s "
            f"peak={prompt['peak_allocated_vram_bytes']} bytes "
            f"finish={prompt['runs'][0]['finish_reason']}"
            f"{constraint_metrics}"
        )


if __name__ == "__main__":
    main()
