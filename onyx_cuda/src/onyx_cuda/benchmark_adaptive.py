"""Paired speed comparison of target-only and fixed gamma 2 speculative generation.

Every mode runs in one interleaved comparison. On a target that supports graph
recovery, graphs are prepared once and stay resident for all modes, but only
graph modes use them; model loading and graph setup are excluded from timings.
Draft graphs follow the service default (ONYX_DRAFT_BACKEND) for every mode.
"""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers

from onyx_cuda.benchmark_corpus import CORPUS
from onyx_cuda.config import resolve_model_selection
from onyx_cuda.draft_backend import close_draft_backend, prepare_draft_backend, resolve_draft_backend
from onyx_cuda.generation import AcceptedTokenEvent
from onyx_cuda.model import load_model_pair
from onyx_cuda.prompt import format_prompt
from onyx_cuda.replay_backend import close_replay_backend, prepare_replay_backend
from onyx_cuda.speculative import generate_speculative_events
from onyx_cuda.vocabulary import get_token_byte_vocabulary
from onyx_cuda._validation_report import evidence

# Mode: (gamma, graph recovery).
MODES = {
    "gamma0": (0, False),
    "gamma2": (2, False),
    "gamma2_graph": (2, True),
}


def select_modes(replay_configuration):
    """Graph modes run only when graphs are active; otherwise they would repeat scalar timings."""
    graph = replay_configuration["active"] == "graph"
    return [mode for mode, (_, uses_graph) in MODES.items() if graph or not uses_graph]


def use_replay_backend(model, backend):
    """Attach prepared graphs for one generation, or detach them without releasing them."""
    if backend is not None:
        model._onyx_replay_backend = backend
    elif hasattr(model, "_onyx_replay_backend"):
        del model._onyx_replay_backend


def summarize(cases):
    modes = [mode for mode in MODES if cases and mode in cases[0]["median_seconds"]]
    totals = {mode: sum(c["median_seconds"][mode] for c in cases) for mode in modes}
    slower = {mode: [c["name"] for c in cases if c["median_seconds"][mode] > c["median_seconds"]["gamma0"]]
              for mode in modes if mode != "gamma0"}
    summary = {"total_median_seconds": totals,
               "speedup_vs_target": {m: totals["gamma0"] / totals[m] for m in modes},
               "slower_than_target": slower}
    if "gamma2_graph" in totals:
        summary["graph_latency_reduction_vs_scalar"] = 1 - totals["gamma2_graph"] / totals["gamma2"]
    return summary


def run(output, *, repetitions=3, split="all", measure=False):
    if output.exists():
        raise ValueError("Choose a new output filename; benchmark evidence is not overwritten")
    corpus_bytes = json.dumps(CORPUS, sort_keys=True).encode()
    selected = [c for c in CORPUS if split == "all" or c["split"] == split]
    device = torch.device("cuda:0")
    pair = load_model_pair(selection=resolve_model_selection())
    target_model = pair.target.model
    replay_configuration = prepare_replay_backend(target_model, "graph")
    graphs = getattr(target_model, "_onyx_replay_backend", None)
    draft_configuration = prepare_draft_backend(pair.draft.model, resolve_draft_backend())
    modes = select_modes(replay_configuration)
    vocabulary = get_token_byte_vocabulary(pair.target.tokenizer, pair.target.model.config.vocab_size)
    report = {"corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(), "corpus_version": 1,
              "settings": {"repetitions": repetitions, "warmups": 1, "split": split,
                           "measure": measure, "temperature": 0, "enable_thinking": False,
                           "dtype": "float16",
                           "modes": {mode: dict(zip(("gamma", "graph_recovery"), MODES[mode]))
                                     for mode in modes},
                           "replay_backend": replay_configuration,
                           "draft_backend": draft_configuration},
              "models": {role: {"id": getattr(pair, role).model_id, "revision": getattr(pair, role).revision}
                         for role in ("draft", "target")},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name()},
              "cases": [], "complete": False, "correctness_passed": False}
    report["source"] = evidence("speedup-comparison")["source"]
    report["settings"]["allow_fp16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        for index, case in enumerate(selected):
            prompt = format_prompt(pair.target.tokenizer, case["messages"], enable_thinking=False)
            args = dict(draft_model=pair.draft.model, target_model=pair.target.model,
                        prompt_token_ids=prompt.token_ids, max_tokens=case["max_tokens"],
                        eos_token_ids=pair.target.tokenizer.eos_token_id)
            if case["regex"] or case["json_schema"]:
                args.update(regex=case["regex"], json_schema=json.dumps(case["json_schema"]) if case["json_schema"] else None,
                            token_byte_vocabulary=vocabulary)
            expected = None

            def generate(mode):
                nonlocal expected
                gamma, graph = MODES[mode]
                use_replay_backend(target_model, graphs if graph else None)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                first = None
                ids = []
                result = None
                events = generate_speculative_events(**args, gamma=gamma, measure=measure)
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
                if graph and graphs.closed:
                    raise RuntimeError(f"Graph recovery closed during {case['name']}: {graphs.fallback_reason}")
                fallbacks = (getattr(getattr(result, "timings", None), "replay_stats", None) or {}).get("graph_replay_fallbacks", 0)
                if graph and fallbacks:
                    # A scalar fallback (such as memory exhaustion) would mislabel graph timings.
                    raise RuntimeError(f"Graph recovery fell back during {case['name']}")
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
                        "replay_stats": getattr(result.timings, "replay_stats", None),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}

            for mode in modes:  # Warm all modes and establish the target oracle first.
                generate(mode)
            gc.collect()
            torch.cuda.synchronize(device)
            allocation = torch.cuda.memory_allocated(device)
            runs = {mode: [] for mode in modes}
            for repetition in range(repetitions):
                offset = (index + repetition) % len(modes)
                for mode in modes[offset:] + modes[:offset]:
                    runs[mode].append(generate(mode))
            gc.collect()
            torch.cuda.synchronize(device)
            if torch.cuda.memory_allocated(device) != allocation:
                raise RuntimeError(f"Retained GPU allocation: {case['name']}")
            row = {**case, "prompt_token_count": len(prompt.token_ids), "runs": runs,
                   "median_seconds": {mode: statistics.median(r["seconds"] for r in runs[mode]) for mode in modes},
                   "min_max_seconds": {mode: [min(r["seconds"] for r in runs[mode]),
                                                max(r["seconds"] for r in runs[mode])] for mode in modes}}
            report["cases"].append(row)
            report["summary"] = summarize(report["cases"])
            save()
            print(case["name"], row["median_seconds"], flush=True)
        report["complete"] = True
        report["correctness_passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        use_replay_backend(target_model, graphs)
        close_replay_backend(target_model)
        close_draft_backend(pair.draft.model)
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--split", choices=("development", "heldout", "all"), default="all")
    parser.add_argument("--measure", action="store_true", help="Separate instrumented diagnostic run")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    run(args.output, repetitions=args.repetitions, split=args.split, measure=args.measure)


if __name__ == "__main__":
    main()
