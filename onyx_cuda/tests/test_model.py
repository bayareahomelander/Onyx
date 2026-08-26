import torch
from transformers.cache_utils import DynamicCache

from onyx_cuda.model import load_model
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt

MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Reply with CUDA ready."},
]
EXPECTED_PROMPT_TEXT = (
    "<|im_start|>system\n"
    "You are a concise assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Reply with CUDA ready.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
EXPECTED_PROMPT_TOKEN_IDS = [
    151644,
    8948,
    198,
    2610,
    525,
    264,
    63594,
    17847,
    13,
    151645,
    198,
    151644,
    872,
    198,
    20841,
    448,
    54809,
    5527,
    13,
    151645,
    198,
    151644,
    77091,
    198,
]


def test_load_model_prompt_and_prefill_on_cuda():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    loaded = load_model()

    assert loaded.revision == loaded.model.config._commit_hash
    assert len(loaded.revision) == 40
    assert not loaded.model.training
    assert all(parameter.device == device for parameter in loaded.model.parameters())
    assert all(parameter.dtype == torch.float16 for parameter in loaded.model.parameters())

    prompt = format_prompt(loaded.tokenizer, MESSAGES)
    assert prompt.text == EXPECTED_PROMPT_TEXT
    assert prompt.token_ids == EXPECTED_PROMPT_TOKEN_IDS
    assert loaded.tokenizer.decode(prompt.token_ids) == EXPECTED_PROMPT_TEXT
    assert loaded.tokenizer.decode(prompt.token_ids, skip_special_tokens=True) == (
        "system\nYou are a concise assistant.\n"
        "user\nReply with CUDA ready.\n"
        "assistant\n"
    )
    assert loaded.tokenizer.bos_token_id is None
    assert (loaded.tokenizer.eos_token, loaded.tokenizer.eos_token_id) == (
        "<|im_end|>",
        151645,
    )
    assert (loaded.tokenizer.pad_token, loaded.tokenizer.pad_token_id) == (
        "<|endoftext|>",
        151643,
    )
    assert prompt.token_ids[-3:] == [151644, 77091, 198]
    assert prompt.token_ids[-1] != loaded.tokenizer.eos_token_id

    result = prefill(loaded.model, prompt.token_ids)
    torch.cuda.synchronize(device)

    assert result.logits.shape == (1, loaded.model.config.vocab_size)
    assert result.logits.device == device
    assert result.logits.dtype == torch.float16
    assert not result.logits.requires_grad
    assert result.token_id.shape == (1,)
    assert result.token_id.device == device
    assert result.token_id.dtype == torch.int64
    assert result.token_id.item() == 80285
    assert loaded.tokenizer.decode(result.token_id.tolist()) == "CUDA"

    cache = result.past_key_values
    assert isinstance(cache, DynamicCache)
    assert len(cache.layers) == loaded.model.config.num_hidden_layers == 24
    assert cache.get_seq_length() == len(prompt.token_ids)
    assert cache.get_max_cache_shape() == -1
    assert callable(cache.crop)

    head_dim = loaded.model.config.hidden_size // loaded.model.config.num_attention_heads
    expected_cache_shape = (
        1,
        loaded.model.config.num_key_value_heads,
        len(prompt.token_ids),
        head_dim,
    )
    for layer in cache.layers:
        assert layer.is_initialized
        assert layer.keys.shape == expected_cache_shape
        assert layer.values.shape == expected_cache_shape
        assert layer.keys.device == layer.values.device == device
        assert layer.keys.dtype == layer.values.dtype == torch.float16

    assert torch.cuda.max_memory_allocated(device) < torch.cuda.get_device_properties(device).total_memory

    print(f"model_revision={loaded.revision}")
    print(f"prompt_token_count={len(prompt.token_ids)}")
    print(f"greedy_token={result.token_id.item()}:{loaded.tokenizer.decode(result.token_id.tolist())}")
    print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
