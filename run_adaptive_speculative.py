#!/usr/bin/env python3
"""
adaptive speculative decoding benchmark

Compares target-only constrained decoding, fixed-gamma speculation, and the
experimental adaptive-gamma path. This script is intentionally separate from
the existing speculative benchmarks so the stable fixed-gamma path remains
easy to inspect.
"""

import statistics
from typing import Dict, List, Optional

from onyx.adaptive import AdaptiveGammaConfig, AdaptiveSpeculativeEngine
from onyx.engine import get_device_info


DRAFT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
WARMUP_RUNS = 1
TARGET_CONFIGS = [
    ("1.5B target", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
    ("7B target", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
]

TASKS = [
    {
        "name": "short year",
        "prompt": "The year is ",
        "regex": "[0-9]{4}",
        "max_tokens": 10,
        "runs": 3,
    },
    {
        "name": "forced digits",
        "prompt": "Generate digits: ",
        "regex": "[0-9]{32}",
        "max_tokens": 40,
        "runs": 3,
    },
]


def summarize(results: List[dict]) -> Dict[str, float]:
    summary = {
        "tps": statistics.mean(r["tokens_per_second"] for r in results),
        "tokens": statistics.mean(r["generated_tokens"] for r in results),
        "mask_us": statistics.mean(r.get("mask_time_avg", 0.0) for r in results) * 1e6,
    }
    if results and "acceptance_rate" in results[0]:
        summary["acceptance"] = statistics.mean(r["acceptance_rate"] for r in results)
    if results and "gamma_final" in results[0]:
        summary["gamma_final"] = statistics.mean(r["gamma_final"] for r in results)
        summary["gamma_avg"] = statistics.mean(
            statistics.mean(r["gamma_history"]) if r["gamma_history"] else r["gamma_initial"]
            for r in results
        )
        summary["adjustments"] = statistics.mean(r["adaptive_adjustments"] for r in results)
    return summary


def run_many(engine, task: dict, mode: str, gamma: Optional[int] = None) -> Dict[str, float]:
    results = []
    total_runs = WARMUP_RUNS + task["runs"]

    for run_idx in range(total_runs):
        if mode == "baseline":
            output, metrics = engine.generate_baseline(
                prompt=task["prompt"],
                max_tokens=task["max_tokens"],
                regex=task["regex"],
            )
        elif mode == "adaptive":
            output, metrics = engine.generate_adaptive(
                prompt=task["prompt"],
                max_tokens=task["max_tokens"],
                regex=task["regex"],
                controller_config=AdaptiveGammaConfig(initial_gamma=4, min_gamma=1, max_gamma=8),
            )
        else:
            output, metrics = engine.generate(
                prompt=task["prompt"],
                max_tokens=task["max_tokens"],
                gamma=gamma,
                regex=task["regex"],
                draft_grammar_aware=True,
            )

        if run_idx < WARMUP_RUNS:
            print(
                f"    warmup {run_idx + 1}: {metrics['tokens_per_second']:.1f} tok/s, "
                f"tokens={metrics['generated_tokens']}, output={output!r}"
            )
        else:
            measured_idx = run_idx - WARMUP_RUNS + 1
            results.append(metrics)
            print(
                f"    run {measured_idx}: {metrics['tokens_per_second']:.1f} tok/s, "
                f"tokens={metrics['generated_tokens']}, output={output!r}"
            )

    return summarize(results)


def print_result(
    label: str,
    summary: Dict[str, float],
    baseline_tps: float,
    show_gamma: bool = False,
) -> None:
    speedup = summary["tps"] / baseline_tps if baseline_tps > 0 else 0.0
    acceptance = summary.get("acceptance")
    acceptance_text = "n/a" if acceptance is None else f"{acceptance:5.1f}%"
    gamma_text = ""
    if show_gamma:
        gamma_text = (
            f" avg_gamma={summary.get('gamma_avg', 0):4.1f}"
            f" final_gamma={summary.get('gamma_final', 0):4.1f}"
            f" adj={summary.get('adjustments', 0):4.1f}"
        )
    print(
        f"{label:<16} {summary['tps']:>8.1f} tok/s "
        f"{speedup:>6.2f}x acc={acceptance_text} "
        f"mask={summary['mask_us']:>7.1f}us{gamma_text}"
    )


def main() -> int:
    print("adaptive speculative decoding benchmark")
    device_info = get_device_info()
    print(f"device: {device_info['device']}")
    print(f"draft model: {DRAFT_MODEL}")

    for target_label, target_model in TARGET_CONFIGS:
        print(f"\n=== {target_label}: {target_model} ===")
        engine = AdaptiveSpeculativeEngine(
            draft_model_path=DRAFT_MODEL,
            target_model_path=target_model,
            cache_mode="paged",
            use_compile=True,
        )

        for task in TASKS:
            print(f"\n--- task: {task['name']} ({task['regex']}) ---")

            print("  baseline")
            baseline = run_many(engine, task, mode="baseline")
            baseline_tps = baseline["tps"]
            print_result("baseline", baseline, baseline_tps)

            for gamma in [1, 2, 4, 8]:
                print(f"  fixed gamma={gamma}")
                fixed = run_many(engine, task, mode="fixed", gamma=gamma)
                print_result(f"gamma={gamma}", fixed, baseline_tps)

            print("  adaptive")
            adaptive = run_many(engine, task, mode="adaptive")
            print_result("adaptive", adaptive, baseline_tps, show_gamma=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
