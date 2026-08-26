"""Cached greedy generation."""

from typing import NamedTuple

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from onyx_cuda.prefill import prefill


class GenerationResult(NamedTuple):
    token_ids: list[int]
    past_key_values: Cache


def generate_greedy(
    model: PreTrainedModel, prompt_token_ids: list[int], max_tokens: int
) -> GenerationResult:
    """Generate at most max_tokens with a batch-one dynamic cache."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    eos_token_ids = model.generation_config.eos_token_id or []
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    generated: list[int] = []
    with torch.inference_mode():
        result = prefill(model, prompt_token_ids)
        cache = result.past_key_values
        token_id = result.token_id

        for step in range(max_tokens):
            token = token_id.item()
            generated.append(token)
            if token in eos_token_ids or step + 1 == max_tokens:
                break

            output = model(
                input_ids=token_id[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            token_id = output.logits[:, -1, :].argmax(dim=-1)

    return GenerationResult(generated, cache)
