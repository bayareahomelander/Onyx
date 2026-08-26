"""Cached greedy generation."""

from typing import Literal, NamedTuple

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from onyx_cuda.prefill import prefill


class GenerationResult(NamedTuple):
    token_ids: list[int]
    past_key_values: Cache
    finish_reason: Literal["eos", "stop", "length"]


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


def generate_greedy(
    model: PreTrainedModel,
    prompt_token_ids: list[int],
    max_tokens: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
) -> GenerationResult:
    """Generate at most max_tokens with a batch-one dynamic cache."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []

    generated: list[int] = []
    finish_reason: Literal["eos", "stop", "length"] = "length"
    with torch.inference_mode():
        result = prefill(model, prompt_token_ids)
        cache = result.past_key_values
        token_id = result.token_id

        for step in range(max_tokens):
            token = token_id.item()
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
            token_id = output.logits[:, -1, :].argmax(dim=-1)

    return GenerationResult(generated, cache, finish_reason)
