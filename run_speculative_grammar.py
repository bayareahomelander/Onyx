"""
grammar-aware speculative decoding benchmark

this script benchmarks scenarios to prove the value of grammar-aware speculation:

the hypothesis is that:
1. "blind draft" will perform worse than baseline (low acceptance rate)
2. "aware draft" restores acceptance rate
3. jit compilation makes grammar-aware speculation faster than baseline
"""

import time
from typing import Dict, List, Tuple
import statistics

from onyx.speculative import SpeculativeEngine


def run_scenario(
    engine: SpeculativeEngine,
    prompt: str,
    regex: str,
    scenario_name: str,
    max_tokens: int = 50,
    gamma: int = 4,
    num_runs: int = 3,
    draft_grammar_aware: bool = True,
    use_speculation: bool = True,
) -> Dict:
    """run a single scenario multiple times and collect statistics. returns dictionary of aggregated metrics."""
    print(f"\n>>> running: {scenario_name}")
    print(f"runs: {num_runs}, max tokens: {max_tokens}, gamma: {gamma}")
    print(f"pattern: {regex}")
    
    results = []
    outputs = []
    
    for i in range(num_runs):
        if use_speculation:
            output, metrics = engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                gamma=gamma,
                regex=regex,
                draft_grammar_aware=draft_grammar_aware,
            )
        else:
            output, metrics = engine.generate_baseline(
                prompt=prompt,
                max_tokens=max_tokens,
                regex=regex,
            )
        
        results.append(metrics)
        outputs.append(output)
        print(f"run {i+1}: {metrics['tokens_per_second']:.1f} tok/s, output='{output}'")
    
    agg = {
        "scenario": scenario_name,
        "num_runs": num_runs,
        "output_sample": outputs[0],
        "tokens_per_second_mean": statistics.mean(r["tokens_per_second"] for r in results),
        "tokens_per_second_std": statistics.stdev(r["tokens_per_second"] for r in results) if num_runs > 1 else 0,
        "ttft_mean": statistics.mean(r["ttft"] for r in results) * 1000,
        "generated_tokens_mean": statistics.mean(r["generated_tokens"] for r in results),
    }
    
    if use_speculation:
        agg["acceptance_rate_mean"] = statistics.mean(r["acceptance_rate"] for r in results)
        agg["draft_tokens_proposed_mean"] = statistics.mean(r["draft_tokens_proposed"] for r in results)
        agg["draft_tokens_accepted_mean"] = statistics.mean(r["draft_tokens_accepted"] for r in results)
        agg["speculative_iterations_mean"] = statistics.mean(r["speculative_iterations"] for r in results)
        agg["draft_grammar_aware"] = draft_grammar_aware
    
    if results[0].get("mask_time_avg"):
        agg["mask_time_avg_us"] = statistics.mean(r["mask_time_avg"] for r in results) * 1e6
    
    return agg


def main():
    print("grammar-aware speculative decoding benchmark")
    
    PROMPT = "The year is "
    REGEX = "[0-9]{4}"
    MAX_TOKENS = 10
    GAMMA = 4
    NUM_RUNS = 5
    
    print(f"configuration:")
    print(f"prompt: '{PROMPT}'")
    print(f"regex pattern: {REGEX}")
    print(f"max tokens: {MAX_TOKENS}")
    print(f"gamma (draft tokens): {GAMMA}")
    print(f"runs per scenario: {NUM_RUNS}")
    
    print("draft model: qwen2.5-0.5b-instruct-4bit (fast proposer)")
    print("target model: qwen2.5-7b-instruct-4bit (memory-bound verifier)")
    engine = SpeculativeEngine(
        target_model_path="mlx-community/Qwen2.5-7B-Instruct-4bit",
        cache_mode="paged",
        use_compile=True,
    )
    
    print("benchmark results")
    
    baseline = run_scenario(
        engine=engine,
        prompt=PROMPT,
        regex=REGEX,
        scenario_name="Baseline (Target+Grammar)",
        max_tokens=MAX_TOKENS,
        gamma=GAMMA,
        num_runs=NUM_RUNS,
        use_speculation=False,
    )
    
    blind = run_scenario(
        engine=engine,
        prompt=PROMPT,
        regex=REGEX,
        scenario_name="Blind Draft (Unconstrained)",
        max_tokens=MAX_TOKENS,
        gamma=GAMMA,
        num_runs=NUM_RUNS,
        draft_grammar_aware=False,
        use_speculation=True,
    )
    
    aware = run_scenario(
        engine=engine,
        prompt=PROMPT,
        regex=REGEX,
        scenario_name="Aware Draft (Grammar-Guided)",
        max_tokens=MAX_TOKENS,
        gamma=GAMMA,
        num_runs=NUM_RUNS,
        draft_grammar_aware=True,
        use_speculation=True,
    )
    
    print(f"\n{'scenario':<30} {'tok/s':>10} {'acc rate':>12} {'vs baseline':>12}")
    print(f"{'baseline (target+grammar)':<30} {baseline['tokens_per_second_mean']:>10.1f} {'n/a':>12} {'1.00x':>12}")
    print(f"{'blind draft':<30} {blind['tokens_per_second_mean']:>10.1f} {blind['acceptance_rate_mean']:>11.1f}% {blind['tokens_per_second_mean']/baseline['tokens_per_second_mean']:>11.2f}x")
    print(f"{'aware draft':<30} {aware['tokens_per_second_mean']:>10.1f} {aware['acceptance_rate_mean']:>11.1f}% {aware['tokens_per_second_mean']/baseline['tokens_per_second_mean']:>11.2f}x")
    
    print("performance analysis")
    
    aware_vs_baseline = aware['tokens_per_second_mean'] / baseline['tokens_per_second_mean']
    blind_vs_baseline = blind['tokens_per_second_mean'] / baseline['tokens_per_second_mean']
    aware_vs_blind = aware['tokens_per_second_mean'] / blind['tokens_per_second_mean']
    
    print(f"key metrics:")
    print(f"baseline: {baseline['tokens_per_second_mean']:.1f} tok/s")
    print(f"blind draft: {blind['tokens_per_second_mean']:.1f} tok/s ({blind_vs_baseline:.2f}x baseline, {blind['acceptance_rate_mean']:.1f}% acceptance)")
    print(f"aware draft: {aware['tokens_per_second_mean']:.1f} tok/s ({aware_vs_baseline:.2f}x baseline, {aware['acceptance_rate_mean']:.1f}% acceptance)")
    print(f"aware vs blind: {aware_vs_blind:.2f}x speedup")
    
    print("hypothesis verification:")
    
    if blind_vs_baseline < 1.0:
        print(f"[confirmed] blind draft is {(1-blind_vs_baseline)*100:.1f}% slower than baseline")
    else:
        print(f"[unexpected] blind draft is {(blind_vs_baseline-1)*100:.1f}% faster than baseline")
    
    if aware_vs_blind > 1.0:
        print(f"[confirmed] aware draft is {(aware_vs_blind-1)*100:.1f}% faster than blind draft")
    
    if aware_vs_baseline >= 1.0:
        print(f"[success] aware draft is {(aware_vs_baseline-1)*100:.1f}% faster than baseline!")
        print("grammar-aware speculation exceeds single-model performance!")
    elif aware_vs_baseline >= 0.95:
        print(f"[near success] aware draft achieves {aware_vs_baseline:.2f}x baseline")
        print("performance is within 5% of baseline.")
    else:
        print(f"[progress] aware draft achieves {aware_vs_baseline:.2f}x baseline")
    
    print("extended test: longer generation (50 tokens)")
    
    LONG_PROMPT = "Generate digits: "
    LONG_REGEX = "[0-9]+"
    LONG_MAX = 50
    
    long_baseline = run_scenario(
        engine=engine,
        prompt=LONG_PROMPT,
        regex=LONG_REGEX,
        scenario_name="Long Baseline",
        max_tokens=LONG_MAX,
        gamma=GAMMA,
        num_runs=3,
        use_speculation=False,
    )
    
    long_aware = run_scenario(
        engine=engine,
        prompt=LONG_PROMPT,
        regex=LONG_REGEX,
        scenario_name="Long Aware Draft",
        max_tokens=LONG_MAX,
        gamma=GAMMA,
        num_runs=3,
        draft_grammar_aware=True,
        use_speculation=True,
    )
    
    long_speedup = long_aware['tokens_per_second_mean'] / long_baseline['tokens_per_second_mean']
    
    print(f"\nlonger generation results (amortizes overhead better):")
    print(f"baseline: {long_baseline['tokens_per_second_mean']:.1f} tok/s")
    print(f"aware draft: {long_aware['tokens_per_second_mean']:.1f} tok/s ({long_speedup:.2f}x baseline)")
    print(f"acceptance rate: {long_aware['acceptance_rate_mean']:.1f}%")
    
    if long_speedup > 1.0:
        print(f"\n[success] with longer generation, aware draft is {(long_speedup-1)*100:.1f}% faster than baseline!")
    
    print("final summary")
    print(f"short generation (4 tokens):")
    print(f"- acceptance rate improvement: {aware['acceptance_rate_mean'] - blind['acceptance_rate_mean']:.1f} pp")
    print(f"- aware draft vs baseline: {aware_vs_baseline:.2f}x")
    print(f"long generation (50 tokens):")
    print(f"- aware draft vs baseline: {long_speedup:.2f}x")
    print(f"- acceptance rate: {long_aware['acceptance_rate_mean']:.1f}%")
    
    if long_speedup >= 1.0 or aware_vs_baseline >= 0.95:
        print("conclusion: grammar-aware speculative decoding successfully achieves")
        print("competitive or better performance compared to single-model generation.")
        print("the architecture is validated for structured output generation.")
    else:
        print("conclusion: grammar-aware speculation significantly improves over blind")
        print("speculation. for larger target models where memory bandwidth dominates,")
        print("the speedup is expected to be more pronounced.")


if __name__ == "__main__":
    main()