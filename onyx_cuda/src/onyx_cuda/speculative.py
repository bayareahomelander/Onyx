"""One greedy draft proposal step."""

from typing import NamedTuple

import torch
from transformers import PreTrainedModel

from onyx_cuda.cache import CacheState
from onyx_cuda.generation import _matched_stop_length


class ProposalResult(NamedTuple):
    token_ids: list[int]
    draft_cache_length_before: int
    draft_cache_length_after: int


def propose_tokens(
    draft_model: PreTrainedModel,
    draft_cache: CacheState,
    generated_token_ids: list[int],
    gamma: int,
    remaining_tokens: int,
    eos_token_ids: int | list[int],
    stop_sequences: list[list[int]] | None = None,
) -> ProposalResult:
    """Greedily propose tokens after the target-selected current token."""
    if not generated_token_ids:
        raise ValueError("draft proposal requires a target-selected token")
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    stop_sequences = stop_sequences or []
    start_length = draft_cache.length
    if generated_token_ids[-1] in eos_token_ids or _matched_stop_length(
        generated_token_ids, stop_sequences
    ):
        return ProposalResult([], start_length, start_length)

    proposed: list[int] = []
    input_ids = torch.tensor(
        [[generated_token_ids[-1]]],
        device=draft_cache.attention_mask.device,
    )
    with torch.inference_mode():
        for _ in range(min(gamma, remaining_tokens)):
            logits = draft_cache.extend(draft_model, input_ids)[:, -1, :]
            token_id = logits.argmax(dim=-1)
            token = token_id.item()
            proposed.append(token)
            if token in eos_token_ids or _matched_stop_length(
                generated_token_ids + proposed, stop_sequences
            ):
                break
            input_ids = token_id[:, None]

    return ProposalResult(proposed, start_length, draft_cache.length)
