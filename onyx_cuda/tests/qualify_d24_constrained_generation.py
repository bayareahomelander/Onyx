"""Bounded offline D24 qualification on the pinned 0.5B CUDA target."""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass

from onyx_cuda import (
    JsonSchemaGrammar,
    RegexGrammar,
    TemperatureTopPSelection,
    build_qwen_grammar_vocabulary,
    compile_native_json_schema,
    compile_native_regex,
    load_production_target_engine,
    load_qwen_tokenizer,
)


VRAM_LIMIT_BYTES = 6_141 * 1024 * 1024
POST_FORWARD_ALLOCATED_ENVELOPE_BYTES = 8_520_704
POST_FORWARD_RESERVED_ENVELOPE_BYTES = 497_025_024
DEFAULT_PROMPT = (
    "<|im_start|>system\nReturn only the exact constrained value.<|im_end|>\n"
    "<|im_start|>user\nReturn the JSON string ok.<|im_end|>\n"
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

    tokenizer_load = load_qwen_tokenizer(local_files_only=True)
    vocabulary = build_qwen_grammar_vocabulary(tokenizer_load.tokenizer)
    engine = load_production_target_engine(
        device_index=arguments.device_index,
        local_files_only=True,
    )
    try:
        baseline_before = engine.generate(arguments.prompt, max_new_tokens=4)

        regex_results = _run_constrained_matrix(
            engine,
            grammar=RegexGrammar("^OK$"),
            prompt=arguments.prompt,
            max_new_tokens=8,
        )
        _validate_matrix(
            regex_results,
            vocabulary=vocabulary,
            grammar_type="regex",
            source="^OK$",
            expected_value="OK",
        )

        json_schema = '{"type":"string","enum":["ok"]}'
        json_results = _run_constrained_matrix(
            engine,
            grammar=JsonSchemaGrammar(json_schema),
            prompt=arguments.prompt,
            max_new_tokens=8,
        )
        _validate_matrix(
            json_results,
            vocabulary=vocabulary,
            grammar_type="json_schema",
            source=json_schema,
            expected_value="ok",
        )

        baseline_after = engine.generate(arguments.prompt, max_new_tokens=4)
        if _signature(baseline_before) != _signature(baseline_after):
            raise AssertionError(
                "unconstrained greedy baseline changed across the D24 constrained matrix"
            )
    finally:
        engine.close()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)
    if after_allocated > max(baseline_allocated, POST_FORWARD_ALLOCATED_ENVELOPE_BYTES):
        raise AssertionError(
            "allocated CUDA memory did not return to the D24 cleanup envelope: "
            f"baseline={baseline_allocated}, after={after_allocated}"
        )
    if after_reserved > max(baseline_reserved, POST_FORWARD_RESERVED_ENVELOPE_BYTES):
        raise AssertionError(
            "reserved CUDA memory did not return to the D24 cleanup envelope: "
            f"baseline={baseline_reserved}, after={after_reserved}"
        )

    constrained_results = (*regex_results, *json_results)
    peak_allocated = max(
        result.generation.metrics.peak_allocated_vram_bytes
        for result in constrained_results
    )
    peak_reserved = max(
        result.generation.metrics.peak_reserved_vram_bytes
        for result in constrained_results
    )

    print(
        "D24 constrained CUDA qualification passed: "
        f"regex_greedy={regex_results[0].sampled_token_ids} "
        f"regex_sampled={regex_results[1].sampled_token_ids} "
        f"json_greedy={json_results[0].sampled_token_ids} "
        f"json_sampled={json_results[1].sampled_token_ids} "
        f"peak_allocated_bytes={peak_allocated} peak_reserved_bytes={peak_reserved} "
        f"after_allocated_bytes={after_allocated} after_reserved_bytes={after_reserved}"
    )


def _run_constrained_matrix(engine, *, grammar, prompt: str, max_new_tokens: int):
    greedy = engine.generate_constrained(
        prompt,
        grammar=grammar,
        max_new_tokens=max_new_tokens,
    )
    sampled_first = engine.generate_constrained(
        prompt,
        grammar=grammar,
        max_new_tokens=max_new_tokens,
        selection=SAMPLED_POLICY,
    )
    sampled_second = engine.generate_constrained(
        prompt,
        grammar=grammar,
        max_new_tokens=max_new_tokens,
        selection=SAMPLED_POLICY,
    )
    if _signature(sampled_first) != _signature(sampled_second):
        raise AssertionError("seeded constrained CUDA sampling did not replay exactly")
    return greedy, sampled_first, sampled_second


def _validate_matrix(
    results,
    *,
    vocabulary: tuple[bytes, ...],
    grammar_type: str,
    source: str,
    expected_value: str,
) -> None:
    for result in results:
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
        if metrics.ttft < 0.0 or metrics.generation_time <= 0.0 or metrics.tokens_per_second <= 0.0:
            raise AssertionError("constrained aggregate metrics must be positive")
        if metrics.peak_allocated_vram_bytes is None or metrics.peak_reserved_vram_bytes is None:
            raise AssertionError("production constrained generation did not report CUDA VRAM peaks")
        if metrics.peak_allocated_vram_bytes >= VRAM_LIMIT_BYTES:
            raise AssertionError("constrained peak allocated VRAM exceeded the RTX 4050 limit")
        if metrics.peak_reserved_vram_bytes >= VRAM_LIMIT_BYTES:
            raise AssertionError("constrained peak reserved VRAM exceeded the RTX 4050 limit")

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
