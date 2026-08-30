import pytest
import torch

from onyx_cuda.cache import CacheState
from onyx_cuda.generation import generate_tokens
from onyx_cuda.model import load_model_pair
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt
from onyx_cuda.speculative import ProposalResult, propose_tokens


class ScriptedCache:
    def __init__(self, token_ids):
        self.token_ids = iter(token_ids)
        self.inputs = []
        self.length = 4
        self.attention_mask = torch.ones((1, 4), dtype=torch.long)

    def extend(self, model, input_ids):
        self.inputs.extend(input_ids.flatten().tolist())
        self.length += input_ids.shape[1]
        logits = torch.full((1, 1, 16), -torch.inf)
        logits[0, 0, next(self.token_ids)] = 0
        return logits


@pytest.mark.parametrize(
    (
        "generated",
        "scripted",
        "gamma",
        "remaining",
        "eos",
        "stops",
        "expected_tokens",
        "expected_inputs",
    ),
    [
        ([1], [2, 3, 4], 2, 5, 15, [], [2, 3], [1, 2]),
        ([1], [2, 3], 4, 1, 15, [], [2], [1]),
        ([1], [15, 3], 4, 4, 15, [], [15], [1]),
        ([7], [8, 9], 4, 4, 15, [[7, 8]], [8], [7]),
        ([15], [2], 4, 4, 15, [], [], []),
    ],
)
def test_scripted_draft_proposal_bounds_order_and_termination(
    generated,
    scripted,
    gamma,
    remaining,
    eos,
    stops,
    expected_tokens,
    expected_inputs,
):
    cache = ScriptedCache(scripted)
    result = propose_tokens(
        object(), cache, generated, gamma, remaining, eos, stops
    )

    assert result == ProposalResult(
        expected_tokens, 4, 4 + len(expected_tokens)
    )
    assert cache.inputs == expected_inputs


def test_real_draft_proposal_matches_direct_greedy_prefix():
    pair = load_model_pair()
    prompt = format_prompt(
        pair.draft.tokenizer,
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with CUDA ready."},
        ],
    )
    draft_prefill = prefill(pair.draft.model, prompt.token_ids)
    target_prefill = prefill(pair.target.model, prompt.token_ids)
    draft_cache = CacheState.from_prefill(
        draft_prefill.past_key_values, draft_prefill.logits.device
    )
    target_token = target_prefill.token_id.item()
    proposal = propose_tokens(
        pair.draft.model,
        draft_cache,
        [target_token],
        gamma=4,
        remaining_tokens=4,
        eos_token_ids=pair.draft.tokenizer.eos_token_id,
    )
    direct = generate_tokens(
        pair.draft.model,
        prompt.token_ids,
        max_tokens=5,
        eos_token_ids=pair.draft.tokenizer.eos_token_id,
    )

    assert direct.token_ids[0] == target_token
    assert proposal.token_ids == direct.token_ids[1:]
    assert proposal == ProposalResult(
        [30982, 151645], len(prompt.token_ids), len(prompt.token_ids) + 2
    )
    assert draft_cache.attention_mask.shape == (1, len(prompt.token_ids) + 2)
    print(f"target_first_token_id={target_token}")
    print(f"draft_proposal_token_ids={proposal.token_ids}")
