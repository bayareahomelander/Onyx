import pytest
import torch

from onyx_cuda.cache import CacheState
from onyx_cuda.cache import snapshot_cache
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


def test_dynamic_snapshot_shares_storage_but_append_and_crop_are_independent():
    from transformers.cache_utils import DynamicCache

    kv = DynamicCache()
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    kv.update(keys, keys + 100, 0)
    state = CacheState.from_prefill(kv, torch.device("cpu"))
    fork = snapshot_cache(state)
    assert fork.past_key_values is not kv
    assert fork.past_key_values.layers[0] is not kv.layers[0]
    assert fork.past_key_values.layers[0].keys.data_ptr() == kv.layers[0].keys.data_ptr()

    kv.update(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4), 0)
    assert state.length == 4 and fork.length == 3
    fork.crop(2)
    fork.past_key_values.update(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), 0)
    assert state.length == 4 and fork.length == 3
    torch.testing.assert_close(kv.layers[0].keys[..., :3, :], keys)
    assert fork.past_key_values.layers[0].keys[..., -1, :].eq(0).all()


def test_unknown_mutable_cache_snapshot_owns_its_tensors():
    state = CacheState(FakeCache(3), torch.ones(1, 3), torch.arange(3))
    fork = snapshot_cache(state)
    state.attention_mask.zero_()
    state.past_key_values.crop(1)
    assert fork.length == 3
    assert fork.attention_mask.eq(1).all()


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


@pytest.mark.gpu
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
