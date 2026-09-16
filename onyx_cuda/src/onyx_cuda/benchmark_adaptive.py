"""Predetermined, paired comparison of fixed gamma 0/2 and adaptive generation."""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
import random
from pathlib import Path

import torch
import transformers

from onyx_cuda.benchmark_corpus import CORPUS
from onyx_cuda.config import resolve_model_selection
from onyx_cuda.generation import AcceptedTokenEvent
from onyx_cuda.model import load_model_pair
from onyx_cuda.prompt import format_prompt
from onyx_cuda.speculative import generate_speculative_events
from onyx_cuda.vocabulary import get_token_byte_vocabulary
from onyx_cuda._validation_report import evidence

MODES = ("gamma0", "gamma2", "adaptive")
GATES = {"retained_savings_fraction": 0.9, "remaining_regression_fraction": 0.8,
         "aggregate_latency_tolerance": 0.02}


def summarize(cases):
    totals = {mode: sum(c["median_seconds"][mode] for c in cases) for mode in MODES}
    winners = [c for c in cases if c["original"] and c["median_seconds"]["gamma2"] < c["median_seconds"]["gamma0"]]
    losers = [c for c in cases if c["median_seconds"]["gamma2"] > c["median_seconds"]["gamma0"]]
    savings = sum(c["median_seconds"]["gamma0"] - c["median_seconds"]["gamma2"] for c in winners)
    retained = sum(c["median_seconds"]["gamma0"] - c["median_seconds"]["adaptive"] for c in winners)
    excess = {mode: sum(max(c["median_seconds"][mode] - c["median_seconds"]["gamma0"], 0) for c in losers)
              for mode in ("gamma2", "adaptive")}
    return {"total_median_seconds": totals,
            "speedup_vs_target": {m: totals["gamma0"] / totals[m] for m in MODES} if cases else {},
            "original_winner_savings_retained": retained / savings if savings else None,
            "regression_seconds": excess,
            "gates": {
                "retained_gains": retained >= GATES["retained_savings_fraction"] * savings if winners else None,
                "reduced_regressions": excess["adaptive"] <= GATES["remaining_regression_fraction"] * excess["gamma2"] if losers else None,
                "aggregate": totals["adaptive"] <= totals["gamma2"] * (1 + GATES["aggregate_latency_tolerance"]),
                "recovery_observed": any(
                    (r.get("adaptive_stats") or {}).get("recovery_count", 0) > 0
                    for c in cases for r in c["runs"]["adaptive"]),
            }}


def paired_latency_interval(runs, reference="gamma2", draws=1000):
    """Reproducible paired-bootstrap interval for the ratio of median latencies."""
    count = len(runs[reference])
    if count < 2:
        return None
    rng = random.Random(0)
    ratios = []
    for _ in range(draws):
        indices = [rng.randrange(count) for _ in range(count)]
        ratios.append(statistics.median(runs["adaptive"][i]["seconds"] for i in indices) /
                      statistics.median(runs[reference][i]["seconds"] for i in indices))
    ratios.sort()
    return [ratios[int(draws * 0.025)], ratios[int(draws * 0.975)]]


def run(output, *, repetitions=10, split="all", measure=False):
    if output.exists():
        raise ValueError("Choose a new output filename; benchmark evidence is not overwritten")
    corpus_bytes = json.dumps(CORPUS, sort_keys=True).encode()
    selected = [c for c in CORPUS if split == "all" or c["split"] == split]
    device = torch.device("cuda:0")
    pair = load_model_pair(selection=resolve_model_selection())
    vocabulary = get_token_byte_vocabulary(pair.target.tokenizer, pair.target.model.config.vocab_size)
    report = {"corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(), "corpus_version": 1,
              "settings": {"repetitions": repetitions, "warmups": 1, "split": split,
                           "measure": measure, "temperature": 0, "enable_thinking": False,
                           "greedy_backend": "torch", "dtype": "float16", "gates": GATES},
              "models": {role: {"id": getattr(pair, role).model_id, "revision": getattr(pair, role).revision}
                         for role in ("draft", "target")},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name()},
              "cases": [], "complete": False, "correctness_passed": False,
              "eligible_for_promotion": False}
    report["source"] = evidence("adaptive-comparison")["source"]
    report["settings"]["allow_fp16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        for index, case in enumerate(selected):
            prompt = format_prompt(pair.target.tokenizer, case["messages"], enable_thinking=False)
            args = dict(draft_model=pair.draft.model, target_model=pair.target.model,
                        prompt_token_ids=prompt.token_ids, max_tokens=case["max_tokens"],
                        eos_token_ids=pair.target.tokenizer.eos_token_id, greedy_backend="torch")
            if case["regex"] or case["json_schema"]:
                args.update(regex=case["regex"], json_schema=json.dumps(case["json_schema"]) if case["json_schema"] else None,
                            token_byte_vocabulary=vocabulary)
            expected = None

            def generate(mode):
                nonlocal expected
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                first = None
                ids = []
                result = None
                events = generate_speculative_events(**args, gamma=0 if mode == "gamma0" else 2,
                                                     adaptive=mode == "adaptive", measure=measure)
                try:
                    for event in events:
                        if isinstance(event, AcceptedTokenEvent):
                            if first is None:
                                first = time.perf_counter() - started
                            ids.append(event.token_id)
                        else:
                            result = event.result
                finally:
                    events.close()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                assert result is not None and ids == result.token_ids
                signature = (ids, result.finish_reason)
                if expected is None:
                    expected = signature
                if signature != expected:
                    raise RuntimeError(f"Greedy output mismatch: {case['name']} / {mode}; expected={expected}; actual={signature}")
                if result.finish_reason not in ("eos", "stop"):
                    raise RuntimeError(f"Incomplete output: {case['name']} / {mode}")
                return {"seconds": seconds, "ttft_seconds": first, "token_ids": ids,
                        "finish_reason": result.finish_reason,
                        "adaptive_stats": getattr(result.timings, "adaptive_stats", None),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}

            for mode in MODES:  # Warm all modes and establish the target oracle first.
                generate(mode)
            gc.collect()
            torch.cuda.synchronize(device)
            allocation = torch.cuda.memory_allocated(device)
            runs = {mode: [] for mode in MODES}
            for repetition in range(repetitions):
                offset = (index + repetition) % len(MODES)
                for mode in MODES[offset:] + MODES[:offset]:
                    runs[mode].append(generate(mode))
            gc.collect()
            torch.cuda.synchronize(device)
            if torch.cuda.memory_allocated(device) != allocation:
                raise RuntimeError(f"Retained GPU allocation: {case['name']}")
            row = {**case, "prompt_token_count": len(prompt.token_ids), "runs": runs,
                   "median_seconds": {mode: statistics.median(r["seconds"] for r in runs[mode]) for mode in MODES},
                   "adaptive_latency_ratio_95pct_interval": paired_latency_interval(runs),
                   "min_max_seconds": {mode: [min(r["seconds"] for r in runs[mode]),
                                                max(r["seconds"] for r in runs[mode])] for mode in MODES}}
            report["cases"].append(row)
            report["summary"] = summarize(report["cases"])
            save()
            print(case["name"], row["median_seconds"], flush=True)
        report["complete"] = True
        report["correctness_passed"] = True
        report["eligible_for_promotion"] = (split == "all" and repetitions >= 10 and
                                            all(report["summary"]["gates"].values()))
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--split", choices=("development", "heldout", "all"), default="all")
    parser.add_argument("--measure", action="store_true", help="Separate instrumented diagnostic run")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    run(args.output, repetitions=args.repetitions, split=args.split, measure=args.measure)


if __name__ == "__main__":
    main()
