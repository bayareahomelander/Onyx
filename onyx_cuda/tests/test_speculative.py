from types import SimpleNamespace

import pytest
import torch

import onyx_cuda.speculative as speculative_module
from onyx_cuda.benchmark import MAX_TOKENS, PROMPTS
from onyx_cuda.cache import CacheState
from onyx_cuda.generation import generate_tokens
from onyx_cuda.model import load_model_pair
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt
from onyx_cuda.speculative import (
    ProposalResult,
    VerificationResult,
    generate_speculative,
    propose_tokens,
    verify_proposal,
)


class ScriptedCache:
    def __init__(self, token_ids):
        self.token_ids = iter(token_ids)
        self.inputs = []
        self.length = 4
        self.attention_mask = torch.ones((1, 4), dtype=torch.long)
        self.past_key_values = self

    def extend(self, model, input_ids):
        self.inputs.extend(input_ids.flatten().tolist())
        self.length += input_ids.shape[1]
        logits = torch.full((1, input_ids.shape[1], 16), -torch.inf)
        for position in range(input_ids.shape[1]):
            logits[0, position, next(self.token_ids)] = 0
        return logits

    def crop(self, length):
        self.length = length


def _run_scripted_speculation(
    monkeypatch,
    *,
    first_token,
    draft_tokens,
    target_tokens,
    max_tokens,
    eos=15,
    stops=None,
):
    draft_model = object()
    target_model = object()
    draft_cache = ScriptedCache(draft_tokens)
    target_cache = ScriptedCache(target_tokens)

    def scripted_prefill(model, prompt_token_ids):
        cache = draft_cache if model is draft_model else target_cache
        return SimpleNamespace(
            logits=torch.zeros((1, 16)),
            past_key_values=cache,
            token_id=torch.tensor([first_token]),
        )

    monkeypatch.setattr(speculative_module, "prefill", scripted_prefill)
    monkeypatch.setattr(
        speculative_module.CacheState,
        "from_prefill",
        classmethod(lambda cls, cache, device: cache),
    )
    result = generate_speculative(
        draft_model,
        target_model,
        [0],
        max_tokens=max_tokens,
        gamma=4,
        eos_token_ids=eos,
        stop_sequences=stops,
    )
    return result, draft_cache, target_cache


def test_speculative_options_fail_before_prefill(monkeypatch):
    monkeypatch.setattr(
        speculative_module,
        "prefill",
        lambda *args: pytest.fail("prefill must not run"),
    )

    with pytest.raises(ValueError, match="gamma"):
        generate_speculative(object(), object(), [0], 1, 0, [])
    with pytest.raises(ValueError, match="max_tokens"):
        generate_speculative(object(), object(), [0], 0, 1, [])


@pytest.mark.parametrize(
    (
        "max_tokens",
        "draft_tokens",
        "target_tokens",
        "expected_tokens",
    ),
    [
        (1, [], [], [1]),
        (2, [0], [2], [1, 2]),
        (3, [2, 0], [2, 3], [1, 2, 3]),
    ],
)
def test_speculative_final_capacity_is_exact(
    monkeypatch,
    max_tokens,
    draft_tokens,
    target_tokens,
    expected_tokens,
):
    result, draft_cache, target_cache = _run_scripted_speculation(
        monkeypatch,
        first_token=1,
        draft_tokens=draft_tokens,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )

    assert result.token_ids == expected_tokens
    assert result.finish_reason == "length"
    assert len(result.token_ids) == max_tokens
    if max_tokens == 1:
        assert draft_cache.inputs == target_cache.inputs == []


@pytest.mark.parametrize(
    (
        "draft_tokens",
        "target_tokens",
        "eos",
        "stops",
        "expected_tokens",
        "expected_reason",
    ),
    [
        ([15, 0], [15, 9], 15, [], [1, 15], "eos"),
        ([8, 0], [8, 9], 15, [[8], [7, 8]], [], "stop"),
    ],
)
def test_speculative_eos_and_overlapping_stops_ignore_plus_one(
    monkeypatch,
    draft_tokens,
    target_tokens,
    eos,
    stops,
    expected_tokens,
    expected_reason,
):
    result, _, _ = _run_scripted_speculation(
        monkeypatch,
        first_token=7 if stops else 1,
        draft_tokens=draft_tokens,
        target_tokens=target_tokens,
        max_tokens=4,
        eos=eos,
        stops=stops,
    )

    assert result.token_ids == expected_tokens
    assert result.finish_reason == expected_reason


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


@pytest.mark.parametrize(
    ("target_tokens", "expected_tokens", "accepted"),
    [
        ([2, 3, 4, 5], [2, 3, 4, 5], 3),
        ([9, 3, 4, 5], [9], 0),
        ([2, 9, 4, 5], [2, 9], 1),
        ([2, 3, 9, 5], [2, 3, 9], 2),
    ],
)
def test_scripted_target_verification_acceptance_and_cache_lengths(
    target_tokens, expected_tokens, accepted
):
    proposal = ProposalResult([2, 3, 4], 4, 7)
    draft_cache = ScriptedCache([0])
    draft_cache.length = proposal.draft_cache_length_after
    target_cache = ScriptedCache(target_tokens)
    target_cache.length = 10

    result = verify_proposal(
        object(), draft_cache, object(), target_cache, 1, proposal
    )

    assert result == VerificationResult(expected_tokens, accepted)
    assert target_cache.inputs == [1, 2, 3, 4]
    assert target_cache.length == 10 + accepted + 1
    assert draft_cache.length == 4 + accepted + 1
    assert draft_cache.inputs == ([4] if accepted == 3 else [])


def test_real_speculation_matches_target_oracle_and_rejects_cleanly():
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
    target_cache = CacheState.from_prefill(
        target_prefill.past_key_values, target_prefill.logits.device
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
    verified = verify_proposal(
        pair.draft.model,
        draft_cache,
        pair.target.model,
        target_cache,
        target_token,
        proposal,
    )
    assert verified == VerificationResult([5527], 0)

    replay_token = verified.token_ids[-1]
    max_replay_differences = {}
    for name, loaded, dirty_cache in (
        ("draft", pair.draft, draft_cache),
        ("target", pair.target, target_cache),
    ):
        with torch.inference_mode():
            dirty_logits = dirty_cache.extend(
                loaded.model,
                torch.tensor([[replay_token]], device=draft_prefill.logits.device),
            )[:, -1, :]
        clean_prefill = prefill(loaded.model, prompt.token_ids)
        clean_cache = CacheState.from_prefill(
            clean_prefill.past_key_values, clean_prefill.logits.device
        )
        with torch.inference_mode():
            clean_cache.extend(
                loaded.model,
                torch.tensor([[target_token]], device=clean_prefill.logits.device),
            )
            clean_logits = clean_cache.extend(
                loaded.model,
                torch.tensor([[replay_token]], device=clean_prefill.logits.device),
            )[:, -1, :]
        torch.testing.assert_close(
            dirty_logits, clean_logits, rtol=1e-2, atol=5e-2
        )
        assert dirty_logits.argmax(dim=-1).item() == (
            clean_logits.argmax(dim=-1).item()
        )
        max_replay_differences[name] = (
            (dirty_logits - clean_logits).abs().max().item()
        )

    eos_token_id = pair.target.tokenizer.eos_token_id
    oracle_ids = {}
    for name, user_prompt in PROMPTS.items():
        formatted = format_prompt(
            pair.target.tokenizer,
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": user_prompt},
            ],
        )
        oracle = generate_tokens(
            pair.target.model,
            formatted.token_ids,
            max_tokens=MAX_TOKENS,
            eos_token_ids=eos_token_id,
        )
        oracle_ids[name] = oracle.token_ids
        for gamma in (1, 4):
            speculative = generate_speculative(
                pair.draft.model,
                pair.target.model,
                formatted.token_ids,
                max_tokens=MAX_TOKENS,
                gamma=gamma,
                eos_token_ids=eos_token_id,
            )
            assert speculative.token_ids == oracle.token_ids
            assert speculative.finish_reason == oracle.finish_reason

    sampled_oracle = generate_tokens(
        pair.target.model,
        prompt.token_ids,
        max_tokens=8,
        eos_token_ids=eos_token_id,
        temperature=0.8,
        top_p=0.9,
        seed=1234,
    )
    sampled_speculative = generate_speculative(
        pair.draft.model,
        pair.target.model,
        prompt.token_ids,
        max_tokens=8,
        gamma=4,
        eos_token_ids=eos_token_id,
        temperature=0.8,
        top_p=0.9,
        seed=1234,
    )
    assert sampled_speculative.token_ids == sampled_oracle.token_ids
    assert sampled_speculative.finish_reason == sampled_oracle.finish_reason

    print(f"speculative_oracle_token_ids={oracle_ids}")
    print(f"rejected_replay_max_abs_differences={max_replay_differences}")
