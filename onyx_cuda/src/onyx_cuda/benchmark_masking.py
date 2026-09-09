"""Reproducible dense versus sparse selection and optional real-model comparison."""

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

from onyx_cuda.masking import grammar_argmax


def _latency_us(call, *, iterations=32, repetitions=5):
    for _ in range(5):
        call()
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / iterations)
    return statistics.median(samples)


def compare_selection():
    from onyx_cuda._sparse_argmax import sparse_argmax

    results = []
    for dtype in (torch.float16, torch.float32):
        generator = torch.Generator(device="cuda").manual_seed(42)
        logits = torch.randn((1, 152064), device="cuda", dtype=dtype, generator=generator)
        permutation = torch.randperm(152064, device="cuda", generator=generator)
        for count in (1, 16, 256, 1024, 4096, 32768, 152064):
            ids = permutation[:count].tolist()
            device_ids = torch.tensor(ids, device="cuda", dtype=torch.long)

            def dense_uploaded():
                masked = torch.full_like(logits, -torch.inf)
                masked.index_copy_(-1, device_ids, logits.index_select(-1, device_ids))
                return masked.argmax(dim=-1)

            methods = {
                "torch": lambda: grammar_argmax(logits, ids, backend="torch"),
                "cuda": lambda: grammar_argmax(logits, ids, backend="cuda"),
                "torch_preuploaded": dense_uploaded,
                "cuda_preuploaded": lambda: sparse_argmax(logits, device_ids),
            }
            expected = methods["torch"]()
            for method in methods.values():
                if not torch.equal(expected, method()):
                    raise RuntimeError("Sparse selection differs from dense reference")
            timings = {name: _latency_us(method) for name, method in methods.items()}
            row = {"dtype": str(dtype), "vocabulary_size": 152064, "allowed": count,
                   "median_us": timings, "speedup": timings["torch"] / timings["cuda"]}
            results.append(row)
            print(f"{dtype} allowed={count}: {timings['torch']:.1f} -> "
                  f"{timings['cuda']:.1f} us ({row['speedup']:.2f}x)", flush=True)
    return results


def compare_models():
    from onyx_cuda.model import load_model_pair
    from onyx_cuda.prompt import format_prompt
    from onyx_cuda.speculative import generate_speculative
    from onyx_cuda.vocabulary import get_token_byte_vocabulary

    pair = load_model_pair()
    vocabulary = get_token_byte_vocabulary(pair.target.tokenizer, pair.target.model.config.vocab_size)
    schema = json.dumps({"type": "object", "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "count": {"type": "integer", "enum": [3]},
    }, "required": ["status", "count"], "additionalProperties": False})
    cases = [
        ("regex", "Reply with CUDA Ready.", {"regex": "CUDA Ready"}),
        ("json", 'Return {"status":"ok","count":3}.', {"json_schema": schema}),
    ]
    results = []
    previous = os.environ.get("ONYX_GREEDY_BACKEND")
    try:
        for name, prompt, options in cases:
            prompt_ids = format_prompt(pair.target.tokenizer, [
                {"role": "user", "content": prompt}
            ]).token_ids
            expected = None
            for gamma in (0, 2):
                samples = {backend: [] for backend in ("torch", "cuda")}
                for repetition in range(6):
                    # Alternate execution order to reduce clock/temperature bias.
                    order = ("torch", "cuda") if repetition % 2 == 0 else ("cuda", "torch")
                    for backend in order:
                        os.environ["ONYX_GREEDY_BACKEND"] = backend
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        result = generate_speculative(
                            pair.draft.model, pair.target.model, prompt_ids, 64, gamma,
                            pair.target.tokenizer.eos_token_id,
                            token_byte_vocabulary=vocabulary, **options,
                        )
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        signature = (result.token_ids, result.finish_reason)
                        if expected is None:
                            expected = signature
                        if signature != expected:
                            raise RuntimeError(f"Output mismatch for {name}, gamma={gamma}, {backend}")
                        if result.finish_reason != "stop":
                            raise RuntimeError(f"Constraint did not complete for {name}")
                        del result
                        if repetition:
                            samples[backend].append(elapsed)
                medians = {backend: statistics.median(values) for backend, values in samples.items()}
                row = {"case": name, "gamma": gamma, "seconds": samples, "median_seconds": medians,
                       "speedup": medians["torch"] / medians["cuda"],
                       "token_ids": expected[0], "finish_reason": expected[1]}
                results.append(row)
                print(f"{name} gamma={gamma}: {medians['torch']:.4f} -> "
                      f"{medians['cuda']:.4f}s ({row['speedup']:.2f}x)", flush=True)
    finally:
        if previous is None:
            os.environ.pop("ONYX_GREEDY_BACKEND", None)
        else:
            os.environ["ONYX_GREEDY_BACKEND"] = previous
    return {"models": {"draft": {"id": pair.draft.model_id, "revision": pair.draft.revision},
                       "target": {"id": pair.target.model_id, "revision": pair.target.revision}},
            "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="Also compare constrained generation with gamma 0 and 2")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/masking_comparison.json"))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    with torch.inference_mode():
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": torch.cuda.get_device_name(0),
            "dependencies": {"torch": str(torch.__version__), "cupy": version("cupy-cuda12x"),
                             "transformers": version("transformers"), "cuda": torch.version.cuda},
            "settings": {"selection_iterations": 32, "selection_repetitions": 5,
                         "model_warmups": 1, "model_repetitions": 5, "max_tokens": 64},
            "selection": compare_selection(),
        }
        if args.models:
            result["generation"] = compare_models()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
