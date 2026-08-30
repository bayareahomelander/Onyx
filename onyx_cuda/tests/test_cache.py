import pytest
import torch

from onyx_cuda.cache import CacheState
from onyx_cuda.model import load_model_pair
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = length


def test_cache_state_length_and_invalid_rollback():
    state = CacheState(
        FakeCache(4),
        torch.ones((1, 4), dtype=torch.long),
        torch.arange(4),
    )

    assert state.length == 4
    for invalid_length in (-1, 5):
        with pytest.raises(ValueError, match="between 0 and 4"):
            state.crop(invalid_length)
    assert state.length == 4

    state.crop(2)
    assert state.length == 2
    assert state.attention_mask.tolist() == [[1, 1]]
    assert state.cache_position.tolist() == [0, 1]


def test_draft_and_target_cache_crop_replay_matches_clean_logits():
    pair = load_model_pair()
    prompt = format_prompt(
        pair.draft.tokenizer,
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with CUDA ready."},
        ],
    )

    for loaded in (pair.draft, pair.target):
        device = next(loaded.model.parameters()).device
        result = prefill(loaded.model, prompt.token_ids)
        state = CacheState.from_prefill(result.past_key_values, device)
        accepted_token = result.token_id.item()
        with torch.inference_mode():
            state.extend(
                loaded.model,
                torch.tensor([[accepted_token, 0]], device=device),
            )
        state.crop(len(prompt.token_ids) + 1)
        replay_token = 11
        with torch.inference_mode():
            replay_logits = state.extend(
                loaded.model, torch.tensor([[replay_token]], device=device)
            )[:, -1, :]

        clean_result = prefill(loaded.model, prompt.token_ids)
        clean_state = CacheState.from_prefill(
            clean_result.past_key_values, device
        )
        with torch.inference_mode():
            clean_state.extend(
                loaded.model, torch.tensor([[accepted_token]], device=device)
            )
            clean_logits = clean_state.extend(
                loaded.model, torch.tensor([[replay_token]], device=device)
            )[:, -1, :]

        max_difference = (replay_logits - clean_logits).abs().max().item()
        torch.testing.assert_close(
            replay_logits, clean_logits, rtol=1e-2, atol=5e-2
        )
        assert replay_logits.argmax(dim=-1).item() == (
            clean_logits.argmax(dim=-1).item()
        )
        expected_length = len(prompt.token_ids) + 2
        assert state.length == clean_state.length == expected_length
        assert state.attention_mask.shape == (
            1,
            expected_length,
        )
        assert state.cache_position.tolist() == list(range(expected_length))
        print(
            f"{loaded.model.config._name_or_path} "
            f"cache_replay_max_abs_difference={max_difference}"
        )
