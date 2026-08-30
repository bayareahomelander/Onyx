"""Cached token generation."""

import json
import math
import time
from typing import Literal, NamedTuple

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from onyx_cuda.cache import CacheState
from onyx_cuda.masking import apply_grammar_mask
from onyx_cuda.prefill import prefill
from onyx_cuda.vocabulary import TokenByteVocabulary


class GenerationResult(NamedTuple):
    token_ids: list[int]
    past_key_values: Cache
    finish_reason: Literal["eos", "stop", "length"]
    timings: "GenerationTimings | None" = None


class GenerationTimings(NamedTuple):
    time_to_first_token_seconds: float
    decode_tokens_per_second: float | None
    total_seconds: float
    grammar_compile_seconds: float | None = None
    valid_token_enumeration_seconds: float | None = None
    mask_transfer_seconds: float | None = None


def _matched_stop_length(
    token_ids: list[int], stop_sequences: list[list[int]]
) -> int:
    return max(
        (
            len(sequence)
            for sequence in stop_sequences
            if sequence and token_ids[-len(sequence) :] == sequence
        ),
        default=0,
    )


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)

    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(
            dim=-1, descending=True
        )
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
        sorted_probabilities.masked_fill_(
            cumulative_probabilities - sorted_probabilities >= top_p, 0
        )
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_index = torch.multinomial(
            sorted_probabilities, 1, generator=generator
        )
        return sorted_indices.gather(-1, sampled_index).squeeze(-1)

    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def _validate_generation_options(
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> None:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def generate_tokens(
    model: PreTrainedModel,
    prompt_token_ids: list[int],
    max_tokens: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    measure: bool = False,
    regex: str | None = None,
    token_byte_vocabulary: TokenByteVocabulary | None = None,
    json_schema: str | None = None,
) -> GenerationResult:
    """Generate at most max_tokens with greedy or top-p sampling."""
    _validate_generation_options(max_tokens, temperature, top_p, seed)
    grammar_requested = regex is not None or json_schema is not None
    if grammar_requested and token_byte_vocabulary is None:
        raise ValueError("token_byte_vocabulary is required when a grammar is set")

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []

    generated: list[int] = []
    finish_reason: Literal["eos", "stop", "length"] = "length"
    started_at = None
    time_to_first_token = None
    grammar_compile_seconds = None
    valid_token_enumeration_seconds = None
    mask_transfer_seconds = None
    if measure:
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()

    constraint = None
    grammar_state = None
    try:
        with torch.inference_mode():
            result = prefill(model, prompt_token_ids)
            logits = result.logits
            cache = CacheState.from_prefill(
                result.past_key_values, result.logits.device
            )
            if grammar_requested:
                from onyx_cuda import _rust

                token_bytes = token_byte_vocabulary.token_bytes
                if len(token_bytes) != logits.shape[-1]:
                    raise ValueError(
                        "token_byte_vocabulary must match the model logits width"
                    )
                compile_started_at = time.perf_counter() if measure else None
                constraint = _rust.GrammarConstraint(token_bytes)
                if json_schema is not None:
                    constraint.compile_json_schema(json_schema)
                else:
                    constraint.compile_regex(regex)
                grammar_state = constraint.init_state()
                if compile_started_at is not None:
                    grammar_compile_seconds = (
                        time.perf_counter() - compile_started_at
                    )
                    valid_token_enumeration_seconds = 0.0
                    mask_transfer_seconds = 0.0

            generator = None
            if temperature > 0 and seed is not None:
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(seed)

            for step in range(max_tokens):
                if constraint is not None:
                    if constraint.is_match_state(grammar_state):
                        finish_reason = "stop"
                        break
                    enumeration_started_at = (
                        time.perf_counter() if measure else None
                    )
                    valid_token_ids = constraint.get_valid_token_ids(grammar_state)
                    if enumeration_started_at is not None:
                        valid_token_enumeration_seconds += (
                            time.perf_counter() - enumeration_started_at
                        )
                    if not valid_token_ids:
                        raise ValueError(
                            "Grammar constraint has no valid token continuation"
                        )
                    if measure:
                        torch.cuda.synchronize(logits.device)
                        mask_started_at = time.perf_counter()
                        logits = apply_grammar_mask(logits, valid_token_ids)
                        torch.cuda.synchronize(logits.device)
                        mask_transfer_seconds += (
                            time.perf_counter() - mask_started_at
                        )
                    else:
                        logits = apply_grammar_mask(logits, valid_token_ids)

                token_id = _sample_token(logits, temperature, top_p, generator)
                token = token_id.item()
                if started_at is not None and time_to_first_token is None:
                    time_to_first_token = time.perf_counter() - started_at
                generated.append(token)

                if constraint is not None:
                    previous_state = grammar_state
                    grammar_state = constraint.advance_state(grammar_state, token)
                    constraint.release_state(previous_state)
                    if constraint.is_match_state(grammar_state):
                        finish_reason = "stop"
                        break

                matched_stop_length = _matched_stop_length(
                    generated, stop_sequences
                )
                if matched_stop_length:
                    del generated[-matched_stop_length:]
                    finish_reason = "stop"
                    break
                if token in eos_token_ids:
                    finish_reason = "eos"
                    break
                if step + 1 == max_tokens:
                    break

                logits = cache.extend(model, token_id[:, None])[:, -1, :]
    finally:
        if constraint is not None and grammar_state is not None:
            constraint.release_state(grammar_state)

    timings = None
    if started_at is not None and time_to_first_token is not None:
        torch.cuda.synchronize(result.logits.device)
        total_seconds = time.perf_counter() - started_at
        decode_seconds = total_seconds - time_to_first_token
        decode_token_count = max(len(generated) - 1, 0)
        decode_tokens_per_second = (
            decode_token_count / decode_seconds
            if decode_token_count and decode_seconds > 0
            else None
        )
        timings = GenerationTimings(
            time_to_first_token_seconds=time_to_first_token,
            decode_tokens_per_second=decode_tokens_per_second,
            total_seconds=total_seconds,
            grammar_compile_seconds=grammar_compile_seconds,
            valid_token_enumeration_seconds=(
                valid_token_enumeration_seconds
            ),
            mask_transfer_seconds=mask_transfer_seconds,
        )

    if json_schema is not None:
        json.loads(
            b"".join(
                token_byte_vocabulary.token_bytes[token_id]
                for token_id in generated
            ).decode("utf-8")
        )

    return GenerationResult(
        generated, cache.past_key_values, finish_reason, timings
    )
