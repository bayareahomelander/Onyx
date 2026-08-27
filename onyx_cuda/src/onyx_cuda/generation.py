"""Cached token generation."""

import math
import time
from typing import Literal, NamedTuple

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from onyx_cuda.prefill import prefill


class GenerationResult(NamedTuple):
    token_ids: list[int]
    past_key_values: Cache
    finish_reason: Literal["eos", "stop", "length"]
    timings: "GenerationTimings | None" = None


class GenerationTimings(NamedTuple):
    time_to_first_token_seconds: float
    decode_tokens_per_second: float | None
    total_seconds: float


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
) -> GenerationResult:
    """Generate at most max_tokens with greedy or top-p sampling."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []

    generated: list[int] = []
    finish_reason: Literal["eos", "stop", "length"] = "length"
    started_at = None
    time_to_first_token = None
    if measure:
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()

    with torch.inference_mode():
        result = prefill(model, prompt_token_ids)
        cache = result.past_key_values
        generator = None
        if temperature > 0 and seed is not None:
            generator = torch.Generator(device=result.logits.device)
            generator.manual_seed(seed)
        token_id = _sample_token(result.logits, temperature, top_p, generator)

        for step in range(max_tokens):
            token = token_id.item()
            if started_at is not None and time_to_first_token is None:
                time_to_first_token = time.perf_counter() - started_at
            generated.append(token)

            matched_stop_length = _matched_stop_length(generated, stop_sequences)
            if matched_stop_length:
                del generated[-matched_stop_length:]
                finish_reason = "stop"
                break
            if token in eos_token_ids:
                finish_reason = "eos"
                break
            if step + 1 == max_tokens:
                break

            output = model(
                input_ids=token_id[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            token_id = _sample_token(
                output.logits[:, -1, :], temperature, top_p, generator
            )

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
            time_to_first_token,
            decode_tokens_per_second,
            total_seconds,
        )

    return GenerationResult(generated, cache, finish_reason, timings)
