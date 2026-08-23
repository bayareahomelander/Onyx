#!/usr/bin/env python3
"""
adaptive speculative decoding benchmark

Compares target-only constrained decoding, fixed-gamma speculation, and the
experimental adaptive-gamma path. This script is intentionally separate from
the existing speculative benchmarks so the stable fixed-gamma path remains
easy to inspect.
"""

from onyx.adaptive import AdaptiveSpeculativeEngine
from onyx.engine import get_device_info
from onyx.evaluator import Evaluator


DRAFT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
WARMUP_RUNS = 1
TARGET_CONFIGS = [
    ("1.5B target", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
    ("8B target", "mlx-community/Qwen3-8B-4bit"),
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
            baseline = Evaluator.run_scenario(engine, task, mode="baseline", warmup_runs=WARMUP_RUNS)
            baseline_tps = baseline.get("tps", 0)
            Evaluator.print_result("baseline", baseline, baseline_tps)

            for gamma in [1, 2, 4, 8]:
                print(f"  fixed gamma={gamma}")
                fixed = Evaluator.run_scenario(engine, task, mode="fixed", gamma=gamma, warmup_runs=WARMUP_RUNS)
                Evaluator.print_result(f"gamma={gamma}", fixed, baseline_tps)

            print("  adaptive")
            adaptive = Evaluator.run_scenario(engine, task, mode="adaptive", warmup_runs=WARMUP_RUNS)
            Evaluator.print_result("adaptive", adaptive, baseline_tps, show_gamma=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
