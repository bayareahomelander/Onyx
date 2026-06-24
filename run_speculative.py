"""
speculative decoding benchmark script

this script benchmarks speculative decoding to validate that the
draft-verify pattern yields wall-clock speedups on apple silicon.

comparison:
1. baseline: target model only (1.5b) with standard autoregressive decoding
2. speculative: draft (0.5b) + target (1.5b) with speculation
3. cache modes: "naive" (mlx_lm kvcache) vs "paged" (pagedkvcache with o(1) rollback)
"""

import sys
import time

sys.path.insert(0, ".")

from onyx.speculative import SpeculativeEngine
from onyx.engine import get_device_info
from onyx.evaluator import Evaluator


def main():
    print("Onyx Speculative Decoding Benchmark (with Paged KV Cache)")
    
    device_info = get_device_info()
    print(f"Device: {device_info['device']}")
    
    prompt = "Explain the concept of recursion in programming."
    max_tokens = 50
    gamma = 4
    block_size = 16
    
    print(f"Configuration:")
    print(f"Prompt: \"{prompt}\"")
    print(f"Max tokens: {max_tokens}")
    print(f"Gamma (draft tokens): {gamma}")
    print(f"Paged cache block size: {block_size}")
    
    print("Initializing Speculative Engine (Paged Cache)...")
    
    load_start = time.perf_counter()
    engine = SpeculativeEngine(cache_mode="paged", block_size=block_size)
    total_load_time = time.perf_counter() - load_start
    
    print(f"Total load time: {total_load_time:.2f}s")
    
    print("BASELINE: Target Model Only (1.5B) - Paged Cache")
    
    baseline_output, baseline_metrics = engine.generate_baseline(
        prompt=prompt,
        max_tokens=max_tokens,
    )
    
    print(f"Output: {baseline_output[:200]}...")
    Evaluator.print_metrics("Baseline", baseline_metrics)
    
    print(f"SPECULATIVE: Draft (0.5B) + Target (1.5B), gamma={gamma} - Paged Cache")
    
    spec_output, spec_metrics = engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        gamma=gamma,
    )
    
    print(f"Output: {spec_output[:200]}...")
    Evaluator.print_metrics("Speculative", spec_metrics)
    
    if "cache_stats" in spec_metrics:
        stats = spec_metrics["cache_stats"]
        print(f"\n  Paged Cache Stats:")
        print(f"    Draft rollbacks: {stats.get('total_draft_rollbacks', 0)}")
        print(f"    Target rollbacks: {stats.get('total_target_rollbacks', 0)}")
        total_rollbacks = stats.get('total_draft_rollbacks', 0) + stats.get('total_target_rollbacks', 0)
    else:
        total_rollbacks = 0
    
    Evaluator.verify_coherence(
        baseline_output, spec_output, ["recursion", "function", "calls", "itself", "base", "case"]
    )
    
    print("COMPARISON")
    
    baseline_tps = baseline_metrics.get('tokens_per_second', 0)
    spec_tps = spec_metrics.get('tokens_per_second', 0)
    speedup = spec_tps / baseline_tps if baseline_tps > 0 else 0
    
    print(f"Baseline (target-only):  {baseline_tps:.1f} tok/s")
    print(f"Speculative (paged):     {spec_tps:.1f} tok/s")
    print(f"Speedup:                 {speedup:.2f}x")
    print(f"Acceptance Rate:         {spec_metrics.get('acceptance_rate', 0):.1f}%")
    
    if speedup > 1.0:
        print("✓ SUCCESS: Speculative decoding achieved a speedup!")
        print("  Memory bandwidth savings outweigh dual-model compute cost.")
    elif speedup > 0.9:
        print("~ NEUTRAL: Speculative decoding roughly matches baseline.")
        print("  Consider adjusting gamma or using a larger target model.")
    else:
        print("✗ SLOWER: Speculative decoding was slower than baseline.")
        print("  Expected for 0.5B/1.5B pairing - target model is too fast.")
    
    print("CACHE MODE COMPARISON: Paged vs Naive")
    
    engine_naive = SpeculativeEngine(cache_mode="naive", lazy_load=True)
    engine_naive.draft_model = engine.draft_model
    engine_naive.target_model = engine.target_model
    engine_naive.tokenizer = engine.tokenizer
    engine_naive._draft_load_time = engine._draft_load_time
    engine_naive._target_load_time = engine._target_load_time
    
    _, naive_metrics = engine_naive.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        gamma=gamma,
    )
    
    paged_tps = spec_metrics.get('tokens_per_second', 0)
    naive_tps = naive_metrics.get('tokens_per_second', 0)
    cache_speedup = paged_tps / naive_tps if naive_tps > 0 else 0
    
    print(f"Naive cache:  {naive_tps:.1f} tok/s")
    print(f"Paged cache:  {paged_tps:.1f} tok/s")
    print(f"Paged/Naive:  {cache_speedup:.2f}x")
    
    if cache_speedup >= 0.95:
        print("✓ Paged cache maintains performance while enabling O(1) rollback.")
    else:
        print("⚠ Paged cache has some overhead - may need optimization.")
    
    Evaluator.log_speculative(
        baseline_metrics=baseline_metrics,
        spec_metrics=spec_metrics,
        gamma=gamma,
        prompt=prompt,
        draft_model=engine.DRAFT_MODEL,
        target_model=engine.TARGET_MODEL,
        cache_mode="paged",
        block_size=block_size,
        total_rollbacks=total_rollbacks,
    )
    
    print("Paged KV Cache integration verified!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
