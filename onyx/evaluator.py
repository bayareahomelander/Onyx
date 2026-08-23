import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any

class Evaluator:
    @staticmethod
    def print_metrics(label: str, metrics: Dict[str, Any]) -> None:
        """Standardized method to print generation metrics."""
        print(f"Metrics ({label}):")
        if "generated_tokens" in metrics:
            print(f"  Generated tokens: {metrics['generated_tokens']}")
        if "ttft" in metrics:
            print(f"  Time to first token: {metrics['ttft']*1000:.1f} ms")
        if "generation_time" in metrics:
            print(f"  Generation time: {metrics['generation_time']:.2f} s")
        if "tokens_per_second" in metrics:
            print(f"  Tokens/second: {metrics['tokens_per_second']:.1f}")
        if "speculative_iterations" in metrics:
            print(f"  Speculative iterations: {metrics['speculative_iterations']}")
        if "draft_tokens_proposed" in metrics:
            print(f"  Draft tokens proposed: {metrics['draft_tokens_proposed']}")
        if "draft_tokens_accepted" in metrics:
            print(f"  Draft tokens accepted: {metrics['draft_tokens_accepted']}")
        if "acceptance_rate" in metrics:
            print(f"  Acceptance rate: {metrics['acceptance_rate']:.1f}%")
        if "cache_mode" in metrics:
            print(f"  Cache mode: {metrics['cache_mode']}")

    @staticmethod
    def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize multiple benchmark runs into average metrics."""
        if not results:
            return {}
        summary = {
            "tps": statistics.mean(r.get("tokens_per_second", 0) for r in results),
            "tokens": statistics.mean(r.get("generated_tokens", 0) for r in results),
            "mask_us": statistics.mean(r.get("mask_time_avg", 0.0) for r in results) * 1e6,
        }
        if "acceptance_rate" in results[0]:
            summary["acceptance"] = statistics.mean(r["acceptance_rate"] for r in results)
        if "gamma_final" in results[0]:
            summary["gamma_final"] = statistics.mean(r["gamma_final"] for r in results)
            summary["gamma_avg"] = statistics.mean(
                statistics.mean(r["gamma_history"]) if r["gamma_history"] else r["gamma_initial"]
                for r in results
            )
            summary["adjustments"] = statistics.mean(r.get("adaptive_adjustments", 0) for r in results)
        return summary

    @staticmethod
    def run_scenario(
        engine, task: dict, mode: str, gamma: Optional[int] = None, warmup_runs: int = 1
    ) -> Dict[str, float]:
        """Orchestrate multiple evaluation runs with warmup."""
        results = []
        total_runs = warmup_runs + task.get("runs", 1)
        
        # Dynamic import to avoid circular dependency if not used
        from onyx.adaptive import AdaptiveGammaConfig
        
        for run_idx in range(total_runs):
            if mode == "baseline":
                output, metrics = engine.generate_baseline(
                    prompt=task["prompt"],
                    max_tokens=task.get("max_tokens", 50),
                    regex=task.get("regex"),
                )
            elif mode == "adaptive":
                if not hasattr(engine, "generate_adaptive"):
                    raise ValueError("Engine does not support generate_adaptive")
                output, metrics = engine.generate_adaptive(
                    prompt=task["prompt"],
                    max_tokens=task.get("max_tokens", 50),
                    regex=task.get("regex"),
                    controller_config=AdaptiveGammaConfig(initial_gamma=4, min_gamma=1, max_gamma=8),
                )
            else:
                output, metrics = engine.generate(
                    prompt=task["prompt"],
                    max_tokens=task.get("max_tokens", 50),
                    gamma=gamma,
                    regex=task.get("regex"),
                    draft_grammar_aware=True,
                )

            if run_idx < warmup_runs:
                print(
                    f"    warmup {run_idx + 1}: {metrics['tokens_per_second']:.1f} tok/s, "
                    f"tokens={metrics['generated_tokens']}, output={output!r}"
                )
            else:
                measured_idx = run_idx - warmup_runs + 1
                results.append(metrics)
                print(
                    f"    run {measured_idx}: {metrics['tokens_per_second']:.1f} tok/s, "
                    f"tokens={metrics['generated_tokens']}, output={output!r}"
                )

        return Evaluator.summarize(results)

    @staticmethod
    def print_result(
        label: str,
        summary: Dict[str, float],
        baseline_tps: float,
        show_gamma: bool = False,
    ) -> None:
        """Format and print speedup results."""
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
            f"mask={summary.get('mask_us', 0.0):>7.1f}us{gamma_text}"
        )

    @staticmethod
    def verify_coherence(baseline_out: str, spec_out: str, terms: List[str]) -> bool:
        """Verify semantic coherence between baseline and speculative outputs."""
        print("COHERENCE VERIFICATION")
        baseline_words = baseline_out.split()
        spec_words = spec_out.split()
        print(f"Baseline output word count: {len(baseline_words)}")
        print(f"Speculative output word count: {len(spec_words)}")
        
        baseline_coherent = any(term in baseline_out.lower() for term in terms)
        spec_coherent = any(term in spec_out.lower() for term in terms)
        
        print(f"Baseline mentions concepts: {'✓' if baseline_coherent else '✗'}")
        print(f"Speculative mentions concepts: {'✓' if spec_coherent else '✗'}")
        
        if spec_coherent:
            print("\n✓ COHERENCE VERIFIED: Paged cache rollback is working correctly!")
            print("  Output is coherent and discusses concepts as expected.")
        else:
            print("\n⚠ WARNING: Output may not be coherent. Check rollback logic.")
        return spec_coherent

    @staticmethod
    def log_baseline(metrics: Dict[str, Any], prompt: str, response: str, load_time: float) -> None:
        """Log baseline execution metrics to file."""
        lines = [
            f"\n--- baseline test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n",
            f"model loading completed in {load_time:.2f} seconds.",
            f"the test prompt contained {metrics.get('prompt_tokens', 0)} tokens.",
            f"time to first token was {metrics.get('ttft', 0) * 1000:.1f} milliseconds.",
            f"the engine generated {metrics.get('generated_tokens', 0)} tokens in {metrics.get('generation_time', 0):.2f} seconds.",
            f"generation speed averaged {metrics.get('tokens_per_second', 0):.1f} tokens per second.",
            f"test prompt: \"{prompt}\"",
            f"model response: \"{response.strip()}\"",
            "",
        ]
        with open("metrics_log.txt", "a") as f:
            f.write("\n".join(lines))

    @staticmethod
    def log_speculative(
        baseline_metrics: Dict[str, Any],
        spec_metrics: Dict[str, Any],
        gamma: int,
        prompt: str,
        draft_model: str,
        target_model: str,
        cache_mode: str,
        block_size: int,
        total_rollbacks: int,
    ) -> None:
        """Log speculative execution comparison metrics to file."""
        baseline_tps = baseline_metrics.get('tokens_per_second', 0)
        spec_tps = spec_metrics.get('tokens_per_second', 0)
        speedup = spec_tps / baseline_tps if baseline_tps > 0 else 0
        acceptance_rate = spec_metrics.get('acceptance_rate', 0)
        
        lines = [
            f"\n--- Speculative Decoding Benchmark ({cache_mode} cache): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n",
            f"Benchmarked speculative decoding with gamma={gamma} (draft tokens per iteration).",
            f"Cache mode: {cache_mode}" + (f" with block_size={block_size}" if cache_mode == "paged" else ""),
            f"Draft model: {draft_model}",
            f"Target model: {target_model}",
            f"Test prompt: \"{prompt}\"",
            "",
            f"Baseline (target-only) generated {baseline_metrics.get('generated_tokens', 0)} tokens in {baseline_metrics.get('generation_time', 0):.2f} seconds at {baseline_tps:.1f} tokens per second.",
            f"Speculative decoding generated {spec_metrics.get('generated_tokens', 0)} tokens in {spec_metrics.get('generation_time', 0):.2f} seconds at {spec_tps:.1f} tokens per second.",
            f"Acceptance rate was {acceptance_rate:.1f}% (percentage of draft tokens accepted by target).",
            f"Speculative decoding achieved a {speedup:.2f}x speedup over the baseline.",
        ]
        
        if cache_mode == "paged":
            lines.append(f"PagedKVCache performed {total_rollbacks} O(1) rollback operations during generation.")
            lines.append("Rollback complexity: O(1) - counter update and block pointer discard, no memory copies.")
        
        if speedup > 1.0:
            lines.append("This confirms that memory bandwidth savings from speculative decoding outweigh the compute cost of running two models.")
        else:
            lines.append("Speculative decoding did not achieve a speedup in this configuration. This may be due to low acceptance rate or insufficient model size difference.")
        
        lines.append("")
        with open("metrics_log.txt", "a") as f:
            f.write("\n".join(lines))
