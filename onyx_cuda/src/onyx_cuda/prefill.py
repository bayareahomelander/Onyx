"""Single cached model prefill."""

from typing import NamedTuple

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache


class PrefillResult(NamedTuple):
    logits: torch.Tensor
    past_key_values: Cache
    token_id: torch.Tensor


def prefill(model: PreTrainedModel, prompt_token_ids: list[int]) -> PrefillResult:
    """Run one cached prompt forward and select the greedy next token."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Onyx CUDA prefill requires a model on CUDA")

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)

    logits = output.logits[:, -1, :]
    token_id = logits.argmax(dim=-1)
    return PrefillResult(logits, output.past_key_values, token_id)
