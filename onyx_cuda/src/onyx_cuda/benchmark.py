"""Reproducible CUDA baseline, constraint, and speculative gates."""

import argparse
import gc
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers

from onyx_cuda.generation import generate_tokens
from onyx_cuda.config import resolve_greedy_backend
from onyx_cuda.model import (
    load_model,
    load_model_pair,
)
from onyx_cuda.prompt import format_prompt
from onyx_cuda.speculative import generate_speculative
from onyx_cuda.vocabulary import build_token_byte_vocabulary, get_token_byte_vocabulary

WARMUPS = 1
REPETITIONS = 3
MAX_TOKENS = 32
SPECULATIVE_GAMMAS = (1, 2, 4)
COMPARISON_OUTPUT = Path("benchmarks/results/performance_comparison.json")
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
SPECULATIVE_OUTPUT = Path("benchmarks/results/phase4_speculative_gate.json")


def _comparison_settings() -> dict:
    backend = resolve_greedy_backend()
    return {
        "warmups": WARMUPS, "repetitions": REPETITIONS, "max_tokens": MAX_TOKENS,
        "temperature": 0.0, "top_p": 1.0, "greedy_backend": backend,
        "dtype": "float16", "timing_contract": "generation-includes-validation-v2",
        "cupy": version("cupy-cuda12x") if backend == "cuda" else None,
    }


def _require_matching_settings(baseline: dict) -> None:
    if baseline.get("settings") != _comparison_settings():
        raise RuntimeError(
            "baseline settings, greedy backend, or timing contract differ; regenerate the baseline"
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
    draft_model=None,
    gamma: int | None = None,
) -> dict:
    generation_options = generation_options or {}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = format_prompt(tokenizer, messages)
    eos_token_id = tokenizer.eos_token_id

    def run_generation(measure: bool):
        if draft_model is None:
            return generate_tokens(
                model,
                prompt.token_ids,
                max_tokens=MAX_TOKENS,
                eos_token_ids=eos_token_id,
                measure=measure,
                **generation_options,
            )
        return generate_speculative(
            draft_model,
            model,
            prompt.token_ids,
            max_tokens=MAX_TOKENS,
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
                f"unconstrained output changed from {label} for {current['name']}"
            )


def _run_constraint_prompts(
    model,
    tokenizer,
    device,
    *,
    draft_model=None,
    gamma: int | None = None,
    vocabulary=None,
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
    )
    parsed = json.loads(json_result["runs"][0]["text"])
    if set(parsed) != {"content"} or parsed["content"] not in {
        "CUDA ready",
        "Ready",
    }:
        raise RuntimeError("JSON-constrained output did not satisfy its schema")
    return [regex_result, json_result]


def _run_speculative_gate(baseline_path: Path | None, output: Path, device, selection=None) -> None:
    if baseline_path is not None and not baseline_path.is_file():
        raise RuntimeError(
            f"target constraint baseline not found at {baseline_path}; "
            "run the target and target-constraint benchmarks first"
        )
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path is not None else None
    )
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
    if baseline is None:
        print("Measuring target baseline in the same process", flush=True)
        baseline = {
            "model": {"id": pair.target.model_id, "revision": pair.target.revision},
            "settings": _comparison_settings(),
            "device": {
                "name": torch.cuda.get_device_name(device),
                "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
            },
            "unconstrained_prompts": [
                _run_prompt(pair.target.model, pair.target.tokenizer, device, name, prompt)
                for name, prompt in PROMPTS.items()
            ],
            "constraint_prompts": _run_constraint_prompts(
                pair.target.model, pair.target.tokenizer, device, vocabulary=vocabulary
            ),
        }
    target_model = {
        "id": pair.target.model_id,
        "revision": pair.target.revision,
    }
    if baseline.get("model") != target_model:
        raise RuntimeError("target constraint baseline model or revision does not match")
    _require_matching_settings(baseline)
    baseline_device = baseline.get("device", {})
    if (
        baseline_device.get("name") != torch.cuda.get_device_name(device)
        or baseline_device.get("total_vram_bytes")
        != torch.cuda.get_device_properties(device).total_memory
    ):
        raise RuntimeError("target constraint baseline device does not match")
    target_prompts = baseline.get("unconstrained_prompts")
    target_constraints = baseline.get("constraint_prompts")
    if not isinstance(target_prompts, list) or not isinstance(target_constraints, list):
        raise RuntimeError("target constraint baseline is missing prompt results")
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
            )
        measured_prompts = prompts + constraint_prompts
        _assert_baseline(
            measured_prompts,
            {"prompts": expected_prompts},
            "1.5B target-only baseline",
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

    # A general default must beat target-only across every measured case, with
    # headroom for timing noise. Per-case winners remain visible in the report.
    eligible = [
        group
        for group in gamma_results
        if group["gamma"] == 0
        or all(
            prompt["speedup_vs_target"] >= 1.05
            for prompt in group["unconstrained_prompts"] + group["constraint_prompts"]
        )
    ]
    best = max(eligible, key=lambda group: group["median_speedup_vs_target"])
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
            "maturin": version("maturin"),
            "pytest": version("pytest"),
            "cuda_runtime": torch.version.cuda,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "settings": {
            **_comparison_settings(),
            "gammas": [0, *SPECULATIVE_GAMMAS],
        },
        "target_constraint_baseline": str(baseline_path) if baseline_path else "same-process",
        "vocabulary_setup": {
            "cold_median_seconds": statistics.median(cold_setup),
            "warm_median_seconds": statistics.median(warm_setup),
            "cache_max_entries": get_token_byte_vocabulary.cache_info().maxsize,
        },
        "gamma_results": gamma_results,
        "best_gamma": best["gamma"],
        "best_gamma_criterion": (
            "best median speedup among modes at least 5% faster on every case; "
            "otherwise gamma=0 (target-only)"
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
    parser.add_argument("--constraints", action="store_true")
    parser.add_argument("--target", action="store_true")
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare target-only and speculation in one process; no baseline files required",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from onyx_cuda.config import resolve_model_selection

    selection = resolve_model_selection()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if args.compare:
        if args.target or args.constraints or args.speculative or args.baseline:
            parser.error("--compare cannot be combined with other modes or --baseline")
        _run_speculative_gate(None, args.output or COMPARISON_OUTPUT, device, selection)
        return
    if args.speculative:
        if args.target or args.constraints:
            parser.error("--speculative cannot be combined with --target or --constraints")
        _run_speculative_gate(
            args.baseline or TARGET_CONSTRAINT_OUTPUT,
            args.output or SPECULATIVE_OUTPUT,
            device,
            selection,
        )
        return
    model_id = selection.target_model if args.target else selection.draft_model
    revision = selection.target_revision if args.target else selection.draft_revision
    baseline = args.baseline or (TARGET_OUTPUT if args.target else DEFAULT_OUTPUT)
    if args.target:
        default_output = (
            TARGET_CONSTRAINT_OUTPUT if args.constraints else TARGET_OUTPUT
        )
    else:
        default_output = CONSTRAINT_OUTPUT if args.constraints else DEFAULT_OUTPUT
    output = args.output or default_output

    loaded = load_model(model_id, revision=revision)
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
            **_comparison_settings(),
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
        _require_matching_settings(baseline_results)
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
