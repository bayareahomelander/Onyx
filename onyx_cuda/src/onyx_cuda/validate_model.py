"""Validate a selected configuration on CUDA and write portable, scoped evidence."""

import argparse
import hashlib
import json
import re
import sys

import torch

from onyx_cuda._validation_report import (
    add_selection_arguments, check, evidence, reserve_report, selected_models, write_report,
)
from onyx_cuda.config import initialize_greedy_backend
from onyx_cuda.model import describe_model_pair, load_model_pair
from onyx_cuda.preflight import precheck


def validation_cases():
    # Share the existing benchmark's public prompt corpus; no arbitrary user input
    # or generated text is copied into the portable report.
    from onyx_cuda.benchmark import PROMPTS

    cases = [(name, {"messages": [{"role": "system", "content": "You are a concise assistant."},
                                  {"role": "user", "content": prompt}], "max_tokens": 32})
             for name, prompt in PROMPTS.items()]
    cases.extend([
        ("regex", {"messages": [{"role": "user", "content": "Reply with CUDA ready."}],
                   "regex": "CUDA Ready", "max_tokens": 32}),
        ("json_schema", {"messages": [{"role": "user", "content": "Return compact JSON only."}],
                         "json_schema": {"type": "object", "properties": {
                             "content": {"enum": ["CUDA ready", "Ready"]}}, "required": ["content"]},
                         "max_tokens": 64}),
        ("sampled", {"messages": [{"role": "user", "content": "Write the numbers one through ten."}],
                     "temperature": 0.7, "top_p": 0.9, "seed": 17, "max_tokens": 32}),
    ])
    return cases


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_output(payload, text, finish_reason):
    if "regex" in payload:
        require(finish_reason == "stop" and re.fullmatch(payload["regex"], text) is not None,
                "Regex output was incomplete or did not match")
    if "json_schema" in payload:
        from onyx_cuda import _rust

        require(finish_reason == "stop", "JSON output was incomplete")
        _rust.validate_json_output(json.dumps(payload["json_schema"]), text)
    require(finish_reason in ("stop", "eos", "length"), "Unknown finish reason")


def generation_check(pair, payload, gamma, backend):
    from onyx_cuda.generation import GenerationFinishedEvent
    from onyx_cuda.server import ChatCompletionRequest, prepare_generation
    from onyx_cuda.speculative import generate_speculative, generate_speculative_events, decode_speculative_events

    arguments = prepare_generation(ChatCompletionRequest(**payload), pair, gamma=0)
    arguments.update(greedy_backend=backend, measure=True)
    baseline = generate_speculative(**arguments)
    expected_ids, expected_reason = baseline.token_ids, baseline.finish_reason
    # Drop returned caches before running the next mode on a small GPU.
    del baseline
    arguments["gamma"] = gamma
    parts, terminal, timings = [], None, None
    for event in decode_speculative_events(generate_speculative_events(**arguments), pair.target.tokenizer):
        if isinstance(event, GenerationFinishedEvent):
            terminal = (event.result.token_ids, event.result.finish_reason)
            timings = event.result.timings
        else:
            parts.append(event.text)
    require(terminal == (expected_ids, expected_reason), "Selected generation differs from target token oracle")
    text = pair.target.tokenizer.decode(expected_ids, skip_special_tokens=True)
    require("".join(parts) == text, "Incremental decoding differs from collected output")
    validate_output(payload, text, expected_reason)
    return {"prompt_token_count": len(arguments["prompt_token_ids"]),
            "token_count": len(expected_ids), "finish_reason": expected_reason,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "proposed_token_count": timings.proposed_token_count,
            "accepted_proposal_count": timings.accepted_proposal_count}


def parse_sse(lines):
    parts, finishes, done = [], [], False
    for line in lines:
        if not line.startswith("data: "):
            continue
        require(not done, "SSE emitted data after DONE")
        data = line[6:]
        if data == "[DONE]":
            done = True
            continue
        item = json.loads(data)
        require("error" not in item, "SSE reported an error")
        choice = item["choices"][0]
        if choice.get("finish_reason") is not None:
            finishes.append(choice["finish_reason"])
        text = choice["delta"].get("content")
        if text:
            require(not finishes, "SSE emitted content after its finish event")
            parts.append(text)
    require(done and len(finishes) == 1, "SSE did not complete with exactly one finish event and DONE")
    return "".join(parts), finishes[0]


def api_check(client, payload, expected):
    response = client.post("/v1/chat/completions", json={**payload, "compact_json": False})
    require(response.status_code == 200, f"Completion HTTP status {response.status_code}")
    choice = response.json()["choices"][0]
    text, reason = choice["message"]["content"], choice["finish_reason"]
    validate_output(payload, text, reason)
    require(hashlib.sha256(text.encode()).hexdigest() == expected["output_sha256"],
            "API output differs from direct generation")
    require(reason == ("stop" if expected["finish_reason"] == "eos" else expected["finish_reason"]),
            "API finish reason differs from direct generation")
    with client.stream("POST", "/v1/chat/completions", json={**payload, "stream": True}) as stream:
        require(stream.status_code == 200, f"Streaming HTTP status {stream.status_code}")
        streamed = parse_sse(stream.iter_lines())
    require(streamed == (text, reason), "SSE differs from the nonstreaming completion")


def rollback_check(pair):
    """Force a rejected proposal even for identical pairs; compare replay to clean caches."""
    from onyx_cuda.cache import CacheState
    from onyx_cuda.prefill import prefill
    from onyx_cuda.prompt import format_prompt
    from onyx_cuda.speculative import ProposalResult, verify_proposal

    prompt = format_prompt(pair.target.tokenizer, [{"role": "user", "content": "Reply with CUDA ready."}])
    target_prefill = prefill(pair.target.model, prompt.token_ids)
    draft_prefill = prefill(pair.draft.model, prompt.token_ids)
    target_cache = CacheState.from_prefill(target_prefill.past_key_values, target_prefill.logits.device)
    draft_cache = CacheState.from_prefill(draft_prefill.past_key_values, draft_prefill.logits.device)
    device = target_prefill.logits.device
    current = target_prefill.token_id.item()
    start = target_cache.length
    with torch.inference_mode():
        expected = target_cache.extend(pair.target.model, torch.tensor([[current]], device=device))[:, -1, :]
        wrong = (expected.argmax(-1).item() + 1) % expected.shape[-1]
        target_cache.crop(start)
        draft_cache.extend(pair.draft.model, torch.tensor([[current]], device=device))
    proposal = ProposalResult([wrong], start, start + 1)
    verified = verify_proposal(pair.draft.model, draft_cache, pair.target.model, target_cache, current, proposal)
    require(verified.accepted_proposal_count == 0, "Forced proposal was not rejected")
    require(target_cache.length == draft_cache.length == start + 1, "Rejected proposal left incorrect cache lengths")
    replay = verified.token_ids[-1]
    differences = {}
    for role, loaded, cache in (("target", pair.target, target_cache), ("draft", pair.draft, draft_cache)):
        clean = prefill(loaded.model, prompt.token_ids)
        clean_cache = CacheState.from_prefill(clean.past_key_values, device)
        with torch.inference_mode():
            dirty_logits = cache.extend(loaded.model, torch.tensor([[replay]], device=device))[:, -1, :]
            clean_cache.extend(loaded.model, torch.tensor([[current]], device=device))
            clean_logits = clean_cache.extend(loaded.model, torch.tensor([[replay]], device=device))[:, -1, :]
        torch.testing.assert_close(dirty_logits, clean_logits, rtol=1e-2, atol=5e-2)
        require(dirty_logits.argmax(-1).item() == clean_logits.argmax(-1).item(), "Cache replay changed greedy token")
        differences[role] = (dirty_logits - clean_logits).abs().max().item()
    return {"forced_rejections": 1, "max_replay_logit_difference": differences,
            "rtol": 1e-2, "atol": 5e-2}


def context_cache_check(loaded, context_tokens):
    """Exercise the selected context size without spending it on token-by-token decoding."""
    from onyx_cuda.cache import CacheState
    from onyx_cuda.prefill import prefill
    from onyx_cuda.prompt import format_prompt

    prompt = format_prompt(loaded.tokenizer, [{"role": "user", "content": "Reply with CUDA ready."}])
    ids = (prompt.token_ids * ((context_tokens // len(prompt.token_ids)) + 1))[:context_tokens - 2]
    result = prefill(loaded.model, ids)
    cache = CacheState.from_prefill(result.past_key_values, result.logits.device)
    extension = torch.tensor([[result.token_id.item()] * 2], device=result.logits.device)
    with torch.inference_mode():
        expected = cache.extend(loaded.model, extension).clone()
        require(cache.length == context_tokens, "Context probe extended to an incorrect length")
        cache.crop(context_tokens - 2)
        replay = cache.extend(loaded.model, extension)
    require(cache.length == context_tokens, "Context probe replay has an incorrect length")
    torch.testing.assert_close(expected, replay, rtol=1e-3, atol=1e-3)
    require(torch.isfinite(replay).all().item(), "Context probe produced nonfinite logits")
    return {"cache_tokens": cache.length, "batch_size": 1,
            "max_replay_logit_difference": (expected - replay).abs().max().item()}


def validate_selected(selection, *, gamma, backend, report, context_tokens=2048):
    from fastapi.testclient import TestClient
    from onyx_cuda.device import require_cuda
    from onyx_cuda.server import create_app

    device = check(report, "cuda.available", require_cuda)
    report["device"] = {"name": torch.cuda.get_device_name(device),
                        "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
                        "cuda_runtime": torch.version.cuda}
    check(report, "selector.startup", lambda: initialize_greedy_backend(backend))
    torch.cuda.reset_peak_memory_stats(device)
    app = create_app(load_engine=lambda: load_model_pair(include_draft=gamma > 0, selection=selection),
                     gamma=gamma, greedy_backend=backend)
    try:
        with TestClient(app) as client:
            report["support_level"] = "startup-verified"
            pair = app.state.engines["onyx-speculative"]
            report["models"] = describe_model_pair(pair)
            executor = app.state.inference_executor
            for role, loaded in (("target", pair.target), ("draft", pair.draft)):
                if loaded is not None:
                    details = check(report, f"{role}.context_cache", lambda: executor.submit(
                        context_cache_check, loaded, context_tokens).result())
                    report["checks"][-1]["result"] = details
            del loaded
            for name, payload in validation_cases():
                expected = check(report, f"{name}.generation", lambda: executor.submit(
                    generation_check, pair, payload, gamma, backend).result())
                report["checks"][-1]["result"] = expected
                check(report, f"{name}.api_and_sse", lambda: api_check(client, payload, expected))
            if gamma > 0:
                details = check(report, "speculation.forced_rejection_and_replay",
                                lambda: executor.submit(rollback_check, pair).result())
                report["checks"][-1]["result"] = details
            # The caller must not retain the pair after lifespan releases ownership.
            del pair
    finally:
        report["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                            "scope": "Model loading and selected validation cases; excludes driver/desktop memory."}
    report.update(status="passed", support_level="tested")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_arguments(parser)
    parser.add_argument("--greedy-backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--context-tokens", type=int, choices=range(64, 2049), metavar="64..2048", default=2048)
    args = parser.parse_args(argv)
    reserve_report(args.output)
    report = evidence("selected-model-validation")
    report["settings"] = {"gamma": args.gamma, "greedy_backend": args.greedy_backend,
                          "corpus": "onyx-selected-model-v1", "precision": "float16",
                          "max_output_tokens": 64, "sampling_seed": 17, "context_tokens": args.context_tokens}
    report["limitations"] = [
        "Tested means only these cases, model revisions, settings, and recorded OS/GPU.",
        "This is not the full regression suite, an exhaustive schema test, or a performance benchmark.",
        "A batch-1 synthetic cache probe covers the recorded context size; it is not a worst-case memory bound.",
        "In-process HTTP/SSE checks do not establish network latency or disconnect behavior.",
        "Reports omit prompt text, generated text, local paths, and raw exception messages.",
    ]
    try:
        selection = check(report, "selection", lambda: selected_models(args))
        report["preflight"] = evidence("model-preflight")
        selection = check(report, "preflight", lambda: precheck(
            selection, gamma=args.gamma, context_tokens=args.context_tokens, report=report["preflight"]))
        report["support_level"] = "prechecked"
        check(report, "cuda.validation", lambda: validate_selected(
            selection, gamma=args.gamma, backend=args.greedy_backend, context_tokens=args.context_tokens, report=report))
    except Exception as error:
        report["status"] = "failed"
        print(f"Selected model validation failed: {error}", file=sys.stderr)
        return 1
    finally:
        write_report(args.output, report)
    print(f"Selected configuration passed the recorded checks. Report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
