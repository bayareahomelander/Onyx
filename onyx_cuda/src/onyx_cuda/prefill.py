"""Cached model prefill with bounded attention workspace."""

from typing import NamedTuple
import inspect

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache


class PrefillResult(NamedTuple):
    logits: torch.Tensor
    past_key_values: Cache
    token_id: torch.Tensor


def prefill(model: PreTrainedModel, prompt_token_ids: list[int], *, chunk_size: int = 1024) -> PrefillResult:
    """Prefill the complete prompt in cached chunks and select the next token."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Onyx CUDA prefill requires a model on CUDA")

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("prefill chunk_size must be a positive integer")
    if not prompt_token_ids:
        raise ValueError("prefill requires a nonempty prompt")
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        # Avoid retaining prompt_length x vocabulary logits for long contexts.
        # Preserve support for causal LMs without the optional Transformers API.
        options = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
        cache = None
        # Bound attention workspace on GPUs whose SDPA path materializes scores.
        # Causal cached chunks preserve the full context; no prompt truncation.
        for offset in range(0, input_ids.shape[1], chunk_size):
            output = model(input_ids=input_ids[:, offset:offset + chunk_size],
                           use_cache=True, past_key_values=cache, **options)
            cache = output.past_key_values

    logits = output.logits[:, -1, :].clone()
    token_id = logits.argmax(dim=-1)
    return PrefillResult(logits, output.past_key_values, token_id)
