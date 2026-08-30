"""One greedy draft proposal step."""

from typing import NamedTuple

import torch
from transformers import PreTrainedModel

from onyx_cuda.cache import CacheState
from onyx_cuda.generation import _matched_stop_length
from onyx_cuda.prefill import prefill


class ProposalResult(NamedTuple):
    token_ids: list[int]
    draft_cache_length_before: int
    draft_cache_length_after: int


class VerificationResult(NamedTuple):
    token_ids: list[int]
    accepted_proposal_count: int


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


def verify_proposal(
    draft_model: PreTrainedModel,
    draft_cache: CacheState,
    target_model: PreTrainedModel,
    target_cache: CacheState,
    current_token_id: int,
    proposal: ProposalResult,
) -> VerificationResult:
    """Verify every proposal position in one target forward."""
    input_ids = torch.tensor(
        [[current_token_id, *proposal.token_ids]],
        device=target_cache.attention_mask.device,
    )
    target_length_before = target_cache.length
    with torch.inference_mode():
        target_token_ids = (
            target_cache.extend(target_model, input_ids)
            .argmax(dim=-1)[0]
            .tolist()
        )

    accepted = 0
    for proposed, target in zip(proposal.token_ids, target_token_ids):
        if proposed != target:
            break
        accepted += 1

    if accepted == len(proposal.token_ids):
        with torch.inference_mode():
            draft_cache.extend(
                draft_model,
                torch.tensor(
                    [[proposal.token_ids[-1]]],
                    device=draft_cache.attention_mask.device,
                ),
            )

    target_cache.crop(target_length_before + accepted + 1)
    draft_cache.crop(proposal.draft_cache_length_before + accepted + 1)
    return VerificationResult(
        proposal.token_ids[:accepted] + [target_token_ids[accepted]], accepted
    )


def generate_speculative(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    prompt_token_ids: list[int],
    max_tokens: int,
    gamma: int,
    eos_token_ids: int | list[int],
) -> list[int]:
    """Run the greedy fixed-gamma draft/verify loop."""
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    draft_prefill = prefill(draft_model, prompt_token_ids)
    target_prefill = prefill(target_model, prompt_token_ids)
    draft_cache = CacheState.from_prefill(
        draft_prefill.past_key_values, draft_prefill.logits.device
    )
    target_cache = CacheState.from_prefill(
        target_prefill.past_key_values, target_prefill.logits.device
    )
    generated = [target_prefill.token_id.item()]

    while len(generated) < max_tokens and generated[-1] not in eos_token_ids:
        proposal = propose_tokens(
            draft_model,
            draft_cache,
            generated,
            gamma,
            max_tokens - len(generated),
            eos_token_ids,
        )
        if not proposal.token_ids:
            break
        verified = verify_proposal(
            draft_model,
            draft_cache,
            target_model,
            target_cache,
            generated[-1],
            proposal,
        )
        for token_id in verified.token_ids:
            generated.append(token_id)
            if token_id in eos_token_ids:
                return generated

    return generated
