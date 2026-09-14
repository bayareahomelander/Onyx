"""Compare target-only and speculative CUDA generation in one process."""

import argparse
import gc
import json
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers

from onyx_cuda.generation import generate_tokens
from onyx_cuda.config import resolve_greedy_backend
from onyx_cuda.model import load_model_pair
from onyx_cuda.prompt import format_prompt
from onyx_cuda.speculative import generate_speculative
from onyx_cuda.vocabulary import build_token_byte_vocabulary, get_token_byte_vocabulary

WARMUPS = 1
REPETITIONS = 3
MAX_TOKENS = 256
SPECULATIVE_GAMMAS = (1, 2, 4, 8)
COMPARISON_OUTPUT = Path("benchmarks/results/performance_comparison.json")
PROMPTS = {
    "cuda_ready": "Reply with CUDA ready.",
    "gpu_summary": "In one concise sentence, explain what a GPU does.",
    "number_sequence": "Write the numbers one through ten, separated by commas.",
}
@dataclass(frozen=True)
class BenchmarkOptions:
    max_tokens: int = MAX_TOKENS
    enable_thinking: bool | None = False
    require_complete: bool = False


def _comparison_settings(options=BenchmarkOptions()) -> dict:
    backend = resolve_greedy_backend()
    return {
        "warmups": WARMUPS, "repetitions": REPETITIONS, "max_tokens": options.max_tokens,
        "enable_thinking": options.enable_thinking, "require_complete": options.require_complete,
        "temperature": 0.0, "top_p": 1.0, "greedy_backend": backend,
        "dtype": "float16", "timing_contract": "generation-includes-validation-v2",
        "cupy": version("cupy-cuda12x") if backend == "cuda" else None,
    }


def _run_prompt(
    model,
    tokenizer,
    device,
    name: str,
    user_prompt: str,
    *,
    system_prompt: str = "You are a concise assistant.",
    generation_options: dict | None = None,
    draft_model=None,
    gamma: int | None = None,
    options=BenchmarkOptions(),
) -> dict:
    generation_options = generation_options or {}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = format_prompt(tokenizer, messages, enable_thinking=options.enable_thinking)
    eos_token_id = tokenizer.eos_token_id

    def run_generation(measure: bool):
        if draft_model is None:
            return generate_tokens(
                model,
                prompt.token_ids,
                max_tokens=options.max_tokens,
                eos_token_ids=eos_token_id,
                measure=measure,
                **generation_options,
            )
        return generate_speculative(
            draft_model,
            model,
            prompt.token_ids,
            max_tokens=options.max_tokens,
            gamma=gamma,
            eos_token_ids=eos_token_id,
            measure=measure,
            **generation_options,
        )

    for _ in range(WARMUPS):
        warmup = run_generation(False)
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
        torch.cuda.synchronize(device)
        wall_started = time.perf_counter()
        result = run_generation(True)
        torch.cuda.synchronize(device)
        generation_wall_seconds = time.perf_counter() - wall_started
        if result.timings is None:
            raise RuntimeError("timing was not recorded")

        token_ids = result.token_ids.copy()
        finish_reason = result.finish_reason
        if options.require_complete and finish_reason not in ("eos", "stop"):
            raise RuntimeError(f"{name} did not complete within {options.max_tokens} tokens")
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
            "generation_wall_seconds": generation_wall_seconds,
            "output_tokens_per_second": (
                len(token_ids) / result.timings.total_seconds
            ),
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        }
        if result.timings.grammar_compile_seconds is not None:
            run["grammar_compile_seconds"] = (
                result.timings.grammar_compile_seconds
            )
        if result.timings.valid_token_enumeration_seconds is not None:
            run.update(
                {
                    "valid_token_enumeration_seconds": (
                        result.timings.valid_token_enumeration_seconds
                    ),
                    "mask_transfer_seconds": (
                        result.timings.mask_transfer_seconds
                    ),
                }
            )
        if result.timings.proposed_token_count is not None:
            run.update(
                {
                    "proposed_token_count": result.timings.proposed_token_count,
                    "accepted_proposal_count": (
                        result.timings.accepted_proposal_count
                    ),
                    "acceptance_rate": result.timings.acceptance_rate,
                    "speculative_iteration_count": (
                        result.timings.speculative_iteration_count
                    ),
                    "draft_seconds": result.timings.draft_seconds,
                    "verify_seconds": result.timings.verify_seconds,
                    "mask_seconds": result.timings.mask_seconds,
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
        "median_generation_wall_seconds": statistics.median(
            run["generation_wall_seconds"] for run in runs
        ),
        "median_total_seconds": statistics.median(run["total_seconds"] for run in runs),
        "median_output_tokens_per_second": statistics.median(
            run["output_tokens_per_second"] for run in runs
        ),
        "peak_allocated_vram_bytes": max(run["peak_allocated_vram_bytes"] for run in runs),
        "stable": True,
    }
    if "grammar_compile_seconds" in runs[0]:
        summary["median_grammar_compile_seconds"] = statistics.median(
            run["grammar_compile_seconds"] for run in runs
        )
    if "valid_token_enumeration_seconds" in runs[0]:
        summary.update(
            {
                "median_valid_token_enumeration_seconds": statistics.median(
                    run["valid_token_enumeration_seconds"] for run in runs
                ),
                "median_mask_transfer_seconds": statistics.median(
                    run["mask_transfer_seconds"] for run in runs
                ),
            }
        )
    if "proposed_token_count" in runs[0]:
        stable_counts = {
            (
                run["proposed_token_count"],
                run["accepted_proposal_count"],
                run["speculative_iteration_count"],
            )
            for run in runs
        }
        if len(stable_counts) != 1:
            raise RuntimeError(f"unstable speculative counts for {name}")
        summary.update(
            {
                "proposed_token_count": runs[0]["proposed_token_count"],
                "accepted_proposal_count": runs[0][
                    "accepted_proposal_count"
                ],
                "acceptance_rate": runs[0]["acceptance_rate"],
                "speculative_iteration_count": runs[0][
                    "speculative_iteration_count"
                ],
                "median_draft_seconds": statistics.median(
                    run["draft_seconds"] for run in runs
                ),
                "median_verify_seconds": statistics.median(
                    run["verify_seconds"] for run in runs
                ),
                "median_mask_seconds": statistics.median(
                    run["mask_seconds"] for run in runs
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
                f"output changed from {label} for {current['name']}"
            )


def _run_constraint_prompts(
    model,
    tokenizer,
    device,
    *,
    draft_model=None,
    gamma: int | None = None,
    vocabulary=None,
    options=BenchmarkOptions(),
) -> list[dict]:
    vocabulary = vocabulary or build_token_byte_vocabulary(
        tokenizer, model.config.vocab_size
    )
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
        draft_model=draft_model,
        gamma=gamma,
        options=options,
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
        draft_model=draft_model,
        gamma=gamma,
        options=options,
    )
    parsed = json.loads(json_result["runs"][0]["text"])
    if set(parsed) != {"content"} or parsed["content"] not in {
        "CUDA ready",
        "Ready",
    }:
        raise RuntimeError("JSON-constrained output did not satisfy its schema")
    return [regex_result, json_result]


def run_comparison(output: Path, device, selection=None, *, options=BenchmarkOptions()) -> None:
    pair = load_model_pair(selection=selection)
    # Report the API's reusable CPU setup separately from GPU generation.

    cold_setup, warm_setup = [], []
    for _ in range(REPETITIONS):
        get_token_byte_vocabulary.cache_clear()
        started = time.perf_counter()
        vocabulary = get_token_byte_vocabulary(pair.target.tokenizer, pair.target.model.config.vocab_size)
        cold_setup.append(time.perf_counter() - started)
        started = time.perf_counter()
        assert (
            get_token_byte_vocabulary(pair.target.tokenizer, pair.target.model.config.vocab_size)
            is vocabulary
        )
        warm_setup.append(time.perf_counter() - started)
    print("Measuring target baseline in the same process", flush=True)
    target_model = {"id": pair.target.model_id, "revision": pair.target.revision}
    target_prompts = [
        _run_prompt(pair.target.model, pair.target.tokenizer, device, name, prompt, options=options)
        for name, prompt in PROMPTS.items()
    ]
    target_constraints = _run_constraint_prompts(
        pair.target.model, pair.target.tokenizer, device, vocabulary=vocabulary, options=options
    )
    expected_prompts = target_prompts + target_constraints
    expected_by_name = {prompt["name"]: prompt for prompt in expected_prompts}
    gamma_results = []
    for gamma in (0, *SPECULATIVE_GAMMAS):
        print(f"Measuring gamma={gamma}", flush=True)
        if gamma == 0:
            prompts, constraint_prompts = target_prompts, target_constraints
        else:
            prompts = [
                _run_prompt(
                    pair.target.model,
                    pair.target.tokenizer,
                    device,
                    name,
                    prompt,
                    draft_model=pair.draft.model,
                    gamma=gamma,
                    options=options,
                )
                for name, prompt in PROMPTS.items()
            ]
            constraint_prompts = _run_constraint_prompts(
                pair.target.model,
                pair.target.tokenizer,
                device,
                draft_model=pair.draft.model,
                gamma=gamma,
                vocabulary=vocabulary,
                options=options,
            )
        measured_prompts = prompts + constraint_prompts
        _assert_baseline(
            measured_prompts,
            {"prompts": expected_prompts},
            "selected target-only baseline",
        )
        for prompt in measured_prompts:
            target_prompt = expected_by_name[prompt["name"]]
            timing_key = (
                "median_generation_wall_seconds"
                if "median_generation_wall_seconds" in target_prompt
                else "median_total_seconds"
            )
            target_seconds = target_prompt[timing_key]
            prompt["target_median_total_seconds"] = target_seconds
            prompt["speedup_vs_target"] = target_seconds / prompt[timing_key]

        proposed = sum(prompt.get("proposed_token_count", 0) for prompt in measured_prompts)
        accepted = sum(prompt.get("accepted_proposal_count", 0) for prompt in measured_prompts)
        gamma_results.append(
            {
                "gamma": gamma,
                "unconstrained_prompts": prompts,
                "constraint_prompts": constraint_prompts,
                "proposed_token_count": proposed,
                "accepted_proposal_count": accepted,
                "acceptance_rate": accepted / proposed if proposed else 0.0,
                "speculative_iteration_count": sum(
                    prompt.get("speculative_iteration_count", 0) for prompt in measured_prompts
                ),
                "median_output_tokens_per_second": statistics.median(
                    prompt["median_output_tokens_per_second"] for prompt in measured_prompts
                ),
                "median_speedup_vs_target": statistics.median(
                    prompt["speedup_vs_target"] for prompt in measured_prompts
                ),
                "peak_allocated_vram_bytes": max(
                    prompt["peak_allocated_vram_bytes"] for prompt in measured_prompts
                ),
            }
        )

    # Compare total time over the declared corpus. Individual regressions remain
    # visible; this is a workload result, not a guarantee for every request.
    baseline_seconds = sum(p["median_generation_wall_seconds"] for p in expected_prompts)
    for group in gamma_results:
        group["aggregate_speedup_vs_target"] = baseline_seconds / sum(
            p["median_generation_wall_seconds"]
            for p in group["unconstrained_prompts"] + group["constraint_prompts"]
        )
    best = max(gamma_results, key=lambda group: group["aggregate_speedup_vs_target"])
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "draft": {"id": pair.draft.model_id, "revision": pair.draft.revision},
            "target": target_model,
        },
        "dependencies": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "accelerate": version("accelerate"),
            "maturin": version("maturin"),
            "pytest": version("pytest"),
            "cuda_runtime": torch.version.cuda,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "settings": {
            **_comparison_settings(options),
            "gammas": [0, *SPECULATIVE_GAMMAS],
        },
        "target_constraint_baseline": "same-process",
        "vocabulary_setup": {
            "cold_median_seconds": statistics.median(cold_setup),
            "warm_median_seconds": statistics.median(warm_setup),
            "cache_max_entries": get_token_byte_vocabulary.cache_info().maxsize,
        },
        "gamma_results": gamma_results,
        "best_gamma": best["gamma"],
        "best_gamma_criterion": (
            "lowest total generation wall time across this benchmark corpus"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for result in gamma_results:
        print(
            f"gamma={result['gamma']}: "
            f"output={result['median_output_tokens_per_second']:.2f} tok/s "
            f"accepted={result['accepted_proposal_count']}/"
            f"{result['proposed_token_count']} "
            f"rate={result['acceptance_rate']:.1%} "
            f"iterations={result['speculative_iteration_count']} "
            f"target_ratio={result['median_speedup_vs_target']:.3f}x "
            f"peak={result['peak_allocated_vram_bytes']} bytes"
        )
    print(f"recommended_gamma={best['gamma']}: {results['best_gamma_criterion']}")
    print(f"vocabulary_setup={results['vocabulary_setup']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Compare gamma 0/1/2/4/8 (the default)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable model thinking for this benchmark")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Pass enable_thinking=False to the model chat template")
    parser.add_argument("--require-complete", action="store_true",
                        help="Fail rather than report timings for truncated answers")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 131072:
        parser.error("--max-tokens must be between 1 and 131072")
    options = BenchmarkOptions(args.max_tokens, bool(args.enable_thinking and not args.disable_thinking),
                               args.require_complete)
    from onyx_cuda.config import resolve_model_selection

    selection = resolve_model_selection()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    run_comparison(args.output or COMPARISON_OUTPUT, device, selection, options=options)


if __name__ == "__main__":
    main()
