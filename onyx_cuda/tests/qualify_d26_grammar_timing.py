"""Bounded offline D26 integrated-grammar-timing qualification on the 0.5B target."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import FrozenInstanceError, dataclass

from onyx_cuda import (
    GrammarTimingMetrics,
    JsonSchemaGrammar,
    RegexGrammar,
    TemperatureTopPSelection,
    TextGenerationComplete,
    TextGenerationDelta,
    build_qwen_grammar_vocabulary,
    compile_native_json_schema,
    compile_native_regex,
    create_cuda_grammar_logit_mask,
    create_grammar_timing_session,
    load_production_target_engine,
    load_qwen_tokenizer,
)


VRAM_LIMIT_BYTES = 6_141 * 1024 * 1024
POST_FORWARD_ALLOCATED_ENVELOPE_BYTES = 8_520_704
POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024
SECOND_LIFECYCLE_ALLOCATED_GROWTH_TOLERANCE_BYTES = 16 * 1024 * 1024
SECOND_LIFECYCLE_RESERVED_GROWTH_TOLERANCE_BYTES = 64 * 1024 * 1024
TIMING_RECONCILIATION_TOLERANCE_SECONDS = 1e-6
STANDALONE_MASK_ITERATIONS = 10
STANDALONE_MASK_WARMUP_ITERATIONS = 2
DEFAULT_PROMPT = (
    "<|im_start|>system\nReturn only the exact constrained value.<|im_end|>\n"
    "<|im_start|>user\nReturn the requested value.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
SAMPLED_POLICY = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=7)


@dataclass(frozen=True, slots=True)
class ResultSignature:
    text: str
    sampled_token_ids: tuple[int, ...]
    finish_reason: str
    matched_stop_token_ids: tuple[int, ...] | None
    grammar_completion_token_id: int | None
    prompt_tokens: int
    final_cache_length: int


@dataclass(frozen=True, slots=True)
class TimedResult:
    lifecycle: int
    grammar: str
    mode: str
    result: object


@dataclass(frozen=True, slots=True)
class CleanupSnapshot:
    allocated_bytes: int
    reserved_bytes: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    arguments = parser.parse_args()

    import torch

    device = torch.device(f"cuda:{arguments.device_index}")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    standalone_evidence = _measure_standalone_mask_overhead(
        torch,
        device=device,
        device_index=arguments.device_index,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    tokenizer_load = load_qwen_tokenizer(local_files_only=True)
    vocabulary = build_qwen_grammar_vocabulary(tokenizer_load.tokenizer)

    all_results: list[TimedResult] = []
    cleanup_snapshots: list[CleanupSnapshot] = []
    lifecycle_baselines: list[ResultSignature] = []
    for lifecycle in (1, 2):
        baseline, results = _run_lifecycle(
            lifecycle,
            prompt=arguments.prompt,
            vocabulary=vocabulary,
            device_index=arguments.device_index,
        )
        lifecycle_baselines.append(baseline)
        all_results.extend(results)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        cleanup_snapshots.append(
            CleanupSnapshot(
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
            )
        )

    if lifecycle_baselines[0] != lifecycle_baselines[1]:
        raise AssertionError("unconstrained baseline changed across complete engine lifecycles")
    _validate_cleanup(
        cleanup_snapshots,
        baseline_allocated=baseline_allocated,
        baseline_reserved=baseline_reserved,
    )
    _print_result_table(all_results)

    timed_seconds, untimed_seconds, timing = standalone_evidence
    final_cleanup = cleanup_snapshots[-1]
    print(
        "D26 integrated grammar-timing CUDA qualification passed: "
        f"results={len(all_results)} "
        f"standalone_timed_seconds={timed_seconds:.6f} "
        f"standalone_untimed_seconds={untimed_seconds:.6f} "
        f"standalone_transfer_seconds={timing.valid_index_transfer_time:.6f} "
        f"standalone_application_seconds={timing.mask_application_time:.6f} "
        f"after_allocated_bytes={final_cleanup.allocated_bytes} "
        f"after_reserved_bytes={final_cleanup.reserved_bytes}"
    )


def _run_lifecycle(
    lifecycle: int,
    *,
    prompt: str,
    vocabulary: tuple[bytes, ...],
    device_index: int,
) -> tuple[ResultSignature, list[TimedResult]]:
    engine = load_production_target_engine(
        device_index=device_index,
        local_files_only=True,
    )
    try:
        baseline_before = _run_unconstrained_baseline(engine, prompt=prompt)
        results: list[TimedResult] = []
        results.extend(
            _run_grammar_matrix(
                engine,
                lifecycle=lifecycle,
                grammar_name="regex",
                grammar=RegexGrammar("^OK$"),
                prompt=prompt,
                max_new_tokens=8,
                vocabulary=vocabulary,
                source="^OK$",
                expected_value="OK",
            )
        )
        json_schema = '{"type":"string","enum":["ok"]}'
        results.extend(
            _run_grammar_matrix(
                engine,
                lifecycle=lifecycle,
                grammar_name="json_schema",
                grammar=JsonSchemaGrammar(json_schema),
                prompt=prompt,
                max_new_tokens=8,
                vocabulary=vocabulary,
                source=json_schema,
                expected_value="ok",
            )
        )
        baseline_after = _run_unconstrained_baseline(engine, prompt=prompt)
        if baseline_before != baseline_after:
            raise AssertionError(
                "unconstrained generate/stream baseline changed across constrained timing work"
            )
        return baseline_before, results
    finally:
        engine.close()


def _run_unconstrained_baseline(engine, *, prompt: str) -> ResultSignature:
    generated = engine.generate(prompt, max_new_tokens=4)
    streamed = _collect_stream(engine.stream(prompt, max_new_tokens=4))
    if generated.generation.metrics.grammar_timing is not None:
        raise AssertionError("unconstrained generation unexpectedly reported grammar timing")
    if streamed.generation.metrics.grammar_timing is not None:
        raise AssertionError("unconstrained streaming unexpectedly reported grammar timing")
    if _signature(generated) != _signature(streamed):
        raise AssertionError("unconstrained stream did not equal unconstrained generation")
    return _signature(generated)


def _run_grammar_matrix(
    engine,
    *,
    lifecycle: int,
    grammar_name: str,
    grammar,
    prompt: str,
    max_new_tokens: int,
    vocabulary: tuple[bytes, ...],
    source: str,
    expected_value: str,
) -> list[TimedResult]:
    greedy_generated = engine.generate_constrained(
        prompt,
        grammar=grammar,
        max_new_tokens=max_new_tokens,
    )
    greedy_streamed = _collect_stream(
        engine.stream_constrained(
            prompt,
            grammar=grammar,
            max_new_tokens=max_new_tokens,
        )
    )
    sampled_generated = engine.generate_constrained(
        prompt,
        grammar=grammar,
        max_new_tokens=max_new_tokens,
        selection=SAMPLED_POLICY,
    )
    sampled_streamed_first = _collect_stream(
        engine.stream_constrained(
            prompt,
            grammar=grammar,
            max_new_tokens=max_new_tokens,
            selection=SAMPLED_POLICY,
        )
    )
    sampled_streamed_second = _collect_stream(
        engine.stream_constrained(
            prompt,
            grammar=grammar,
            max_new_tokens=max_new_tokens,
            selection=SAMPLED_POLICY,
        )
    )
    reused = _cancel_and_reuse(
        engine,
        grammar=grammar,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        expected=greedy_generated,
    )

    if _signature(greedy_generated) != _signature(greedy_streamed):
        raise AssertionError("greedy constrained streaming changed the semantic result")
    sampled_signature = _signature(sampled_generated)
    if sampled_signature != _signature(sampled_streamed_first):
        raise AssertionError("sampled constrained streaming changed the semantic result")
    if sampled_signature != _signature(sampled_streamed_second):
        raise AssertionError("seeded constrained streaming did not replay exactly")

    named_results = (
        ("greedy_generate", greedy_generated),
        ("greedy_stream", greedy_streamed),
        ("sampled_generate", sampled_generated),
        ("sampled_stream", sampled_streamed_first),
        ("sampled_stream_replay", sampled_streamed_second),
        ("cancel_reuse", reused),
    )
    for _mode, result in named_results:
        _validate_constrained_result(
            result,
            vocabulary=vocabulary,
            grammar_type=grammar_name,
            source=source,
            expected_value=expected_value,
        )
    return [
        TimedResult(lifecycle, grammar_name, mode, result)
        for mode, result in named_results
    ]


def _collect_stream(stream):
    with stream:
        events = list(stream)
    deltas = [event for event in events if isinstance(event, TextGenerationDelta)]
    completions = [event for event in events if isinstance(event, TextGenerationComplete)]
    if len(completions) != 1 or events[-1] is not completions[0]:
        raise AssertionError("stream did not terminate with exactly one completion event")
    if any(not event.text for event in deltas):
        raise AssertionError("stream emitted an empty text delta")
    result = completions[0].result
    if "".join(event.text for event in deltas) != result.text:
        raise AssertionError("stream deltas did not concatenate to terminal text")
    return result


def _cancel_and_reuse(
    engine,
    *,
    grammar,
    prompt: str,
    max_new_tokens: int,
    expected,
):
    stream = engine.stream_constrained(
        prompt,
        grammar=grammar,
        max_new_tokens=max_new_tokens,
    )
    with stream:
        first_event = next(stream)
        if not isinstance(first_event, TextGenerationDelta):
            raise AssertionError("partial constrained stream did not produce visible text")
    if engine.cache_length != 0:
        raise AssertionError("cancelled constrained stream retained backend cache state")

    reused = engine.generate_constrained(
        prompt,
        grammar=grammar,
        max_new_tokens=max_new_tokens,
    )
    if _signature(reused) != _signature(expected):
        raise AssertionError("engine reuse after cancellation changed the semantic result")
    return reused


def _validate_constrained_result(
    result,
    *,
    vocabulary: tuple[bytes, ...],
    grammar_type: str,
    source: str,
    expected_value: str,
) -> None:
    generation = result.generation
    if generation.finish_reason != "grammar_complete":
        raise AssertionError(f"unexpected constrained finish reason {generation.finish_reason}")
    if generation.grammar_completion_token_id != generation.token_ids[-1]:
        raise AssertionError("completion EOS is not the terminal sampled token")
    if result.output_token_ids != generation.token_ids[:-1]:
        raise AssertionError("completion EOS leaked into visible output token IDs")
    if generation.final_cache_length != (
        generation.prompt_tokens + generation.generated_tokens - 1
    ):
        raise AssertionError("terminal constrained token was decoded into the KV cache")

    metrics = generation.metrics
    if metrics.cache_mode != "transformers_dynamic":
        raise AssertionError(f"unexpected cache mode {metrics.cache_mode!r}")
    if metrics.ttft <= 0.0 or metrics.generation_time <= 0.0:
        raise AssertionError("constrained aggregate timings must be positive")
    if metrics.tokens_per_second <= 0.0:
        raise AssertionError("constrained throughput must be positive")
    if metrics.peak_allocated_vram_bytes is None or metrics.peak_reserved_vram_bytes is None:
        raise AssertionError("production constrained result did not report CUDA VRAM peaks")
    if metrics.peak_allocated_vram_bytes >= VRAM_LIMIT_BYTES:
        raise AssertionError("constrained peak allocated VRAM exceeded the RTX 4050 limit")
    if metrics.peak_reserved_vram_bytes >= VRAM_LIMIT_BYTES:
        raise AssertionError("constrained peak reserved VRAM exceeded the RTX 4050 limit")

    timing = metrics.grammar_timing
    if not isinstance(timing, GrammarTimingMetrics):
        raise AssertionError("constrained result did not expose GrammarTimingMetrics")
    for field_name in (
        "compilation_time",
        "state_scan_time",
        "valid_index_transfer_time",
        "mask_application_time",
    ):
        if getattr(timing, field_name) <= 0.0:
            raise AssertionError(f"live production {field_name} must be positive")
    component_time = (
        timing.state_scan_time
        + timing.valid_index_transfer_time
        + timing.mask_application_time
    )
    tolerance = max(
        TIMING_RECONCILIATION_TOLERANCE_SECONDS,
        metrics.generation_time * 1e-6,
    )
    if component_time > metrics.generation_time + tolerance:
        raise AssertionError(
            "grammar active components exceed enclosing generation time: "
            f"components={component_time}, generation_time={metrics.generation_time}, "
            f"tolerance={tolerance}"
        )
    try:
        timing.compilation_time = 0.0
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("grammar timing record is not immutable")

    _replay_native(
        generation.token_ids,
        vocabulary=vocabulary,
        eos_token_id=generation.grammar_completion_token_id,
        grammar_type=grammar_type,
        source=source,
    )
    if grammar_type == "regex":
        if re.fullmatch(source, result.text) is None:
            raise AssertionError(f"regex output did not fully match: {result.text!r}")
    elif json.loads(result.text) != expected_value:
        raise AssertionError(f"JSON output did not equal {expected_value!r}: {result.text!r}")


def _replay_native(
    sampled_token_ids: tuple[int, ...],
    *,
    vocabulary: tuple[bytes, ...],
    eos_token_id: int,
    grammar_type: str,
    source: str,
) -> None:
    constraint = (
        compile_native_regex(vocabulary, source)
        if grammar_type == "regex"
        else compile_native_json_schema(vocabulary, source)
    )
    state = constraint.init_state()
    try:
        for token_id in sampled_token_ids:
            valid_token_ids = constraint.get_valid_token_ids(state)
            if token_id == eos_token_id:
                if token_id in valid_token_ids or not constraint.is_match_state(state):
                    raise AssertionError("EOS replay did not begin from injected matching support")
            elif token_id not in valid_token_ids:
                raise AssertionError(f"sampled token {token_id} was not native-valid during replay")
            child = constraint.advance_state(state, token_id)
            if constraint.is_dead_state(child):
                raise AssertionError("native replay reached a dead state")
            constraint.release_state(state)
            state = child
        if not constraint.is_match_state(state):
            raise AssertionError("terminal native replay state was not matching")
    finally:
        constraint.release_state(state)
        constraint.reset()


def _measure_standalone_mask_overhead(torch, *, device, device_index: int):
    vocab_size = 1_024
    valid_token_ids = tuple(range(0, vocab_size, 8))
    logits = torch.linspace(-1.0, 1.0, vocab_size, device=device)
    mask = create_cuda_grammar_logit_mask(vocab_size, device_index=device_index)

    warmup_timing_session = create_grammar_timing_session()
    for _ in range(STANDALONE_MASK_WARMUP_ITERATIONS):
        result = mask.apply(logits, valid_token_ids)
        with warmup_timing_session.state_scan():
            pass
        result = mask.apply_with_timing(logits, valid_token_ids, warmup_timing_session)
    torch.cuda.synchronize(device)
    del result, warmup_timing_session

    torch.cuda.synchronize(device)
    untimed_start = time.perf_counter()
    for _ in range(STANDALONE_MASK_ITERATIONS):
        result = mask.apply(logits, valid_token_ids)
    torch.cuda.synchronize(device)
    untimed_seconds = time.perf_counter() - untimed_start
    del result

    timing_session = create_grammar_timing_session()
    torch.cuda.synchronize(device)
    timed_start = time.perf_counter()
    for _ in range(STANDALONE_MASK_ITERATIONS):
        with timing_session.state_scan():
            pass
        result = mask.apply_with_timing(logits, valid_token_ids, timing_session)
    torch.cuda.synchronize(device)
    timed_seconds = time.perf_counter() - timed_start
    timing = timing_session.finish(STANDALONE_MASK_ITERATIONS)
    del result, mask, logits
    return timed_seconds, untimed_seconds, timing


def _validate_cleanup(
    snapshots: list[CleanupSnapshot],
    *,
    baseline_allocated: int,
    baseline_reserved: int,
) -> None:
    allocated_limit = max(baseline_allocated, POST_FORWARD_ALLOCATED_ENVELOPE_BYTES)
    reserved_limit = max(baseline_reserved, POST_FORWARD_RESERVED_ENVELOPE_BYTES)
    for lifecycle, snapshot in enumerate(snapshots, start=1):
        if snapshot.allocated_bytes > allocated_limit:
            raise AssertionError(
                f"lifecycle {lifecycle} allocated memory missed the cleanup envelope: "
                f"after={snapshot.allocated_bytes}, limit={allocated_limit}"
            )
        if snapshot.reserved_bytes > reserved_limit:
            raise AssertionError(
                f"lifecycle {lifecycle} reserved memory missed the cleanup envelope: "
                f"after={snapshot.reserved_bytes}, limit={reserved_limit}"
            )

    first, second = snapshots
    if second.allocated_bytes > (
        first.allocated_bytes + SECOND_LIFECYCLE_ALLOCATED_GROWTH_TOLERANCE_BYTES
    ):
        raise AssertionError("second lifecycle allocated-memory growth exceeded tolerance")
    if second.reserved_bytes > (
        first.reserved_bytes + SECOND_LIFECYCLE_RESERVED_GROWTH_TOLERANCE_BYTES
    ):
        raise AssertionError("second lifecycle reserved-memory growth exceeded tolerance")


def _print_result_table(results: list[TimedResult]) -> None:
    print(
        "life grammar mode tokens compile_s scan_s transfer_s mask_s "
        "scan_per_token transfer_per_token mask_per_token ttft_s generation_s tok_s "
        "peak_allocated peak_reserved"
    )
    for item in results:
        generation = item.result.generation
        metrics = generation.metrics
        timing = metrics.grammar_timing
        tokens = generation.generated_tokens
        print(
            f"{item.lifecycle} {item.grammar} {item.mode} {tokens} "
            f"{timing.compilation_time:.6f} {timing.state_scan_time:.6f} "
            f"{timing.valid_index_transfer_time:.6f} "
            f"{timing.mask_application_time:.6f} "
            f"{timing.state_scan_time / tokens:.6f} "
            f"{timing.valid_index_transfer_time / tokens:.6f} "
            f"{timing.mask_application_time / tokens:.6f} "
            f"{metrics.ttft:.6f} {metrics.generation_time:.6f} "
            f"{metrics.tokens_per_second:.3f} "
            f"{metrics.peak_allocated_vram_bytes} {metrics.peak_reserved_vram_bytes}"
        )


def _signature(result) -> ResultSignature:
    generation = result.generation
    return ResultSignature(
        text=result.text,
        sampled_token_ids=result.sampled_token_ids,
        finish_reason=generation.finish_reason,
        matched_stop_token_ids=generation.matched_stop_token_ids,
        grammar_completion_token_id=generation.grammar_completion_token_id,
        prompt_tokens=generation.prompt_tokens,
        final_cache_length=generation.final_cache_length,
    )


if __name__ == "__main__":
    main()
