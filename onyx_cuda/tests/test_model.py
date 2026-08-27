import gc

import torch
from transformers.cache_utils import DynamicCache

from onyx_cuda.generation import generate_tokens
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


def test_load_model_prompt_prefill_and_generation_on_cuda():
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

    del result, cache
    gc.collect()
    torch.cuda.empty_cache()

    input_ids = torch.tensor([prompt.token_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        reference = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        limited_reference = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=2,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    expected = reference[0, len(prompt.token_ids) :].tolist()
    limited_expected = limited_reference[0, len(prompt.token_ids) :].tolist()
    eos_token_id = loaded.tokenizer.eos_token_id
    generated = generate_tokens(
        loaded.model,
        prompt.token_ids,
        max_tokens=16,
        eos_token_ids=eos_token_id,
        temperature=0.0,
        top_p=0.1,
        measure=True,
    )
    limited = generate_tokens(
        loaded.model,
        prompt.token_ids,
        max_tokens=2,
        eos_token_ids=eos_token_id,
    )

    assert generated.token_ids == expected == [80285, 30982, 151645]
    assert generated.token_ids[-1] == eos_token_id
    assert generated.finish_reason == "eos"
    assert generated.timings is not None
    assert generated.timings.time_to_first_token_seconds > 0
    assert generated.timings.decode_tokens_per_second > 0
    assert generated.timings.total_seconds >= (
        generated.timings.time_to_first_token_seconds
    )
    assert generated.past_key_values.get_seq_length() == (
        len(prompt.token_ids) + len(generated.token_ids) - 1
    )
    assert limited.token_ids == limited_expected == [80285, 30982]
    assert len(limited.token_ids) == 2
    assert limited.finish_reason == "length"
    assert limited.timings is None
    assert limited.past_key_values.get_seq_length() == len(prompt.token_ids) + 1

    one_token_stop_ids = loaded.tokenizer.encode(" Ready", add_special_tokens=False)
    multi_token_stop_ids = loaded.tokenizer.encode(
        "CUDA Ready", add_special_tokens=False
    )
    assert one_token_stop_ids == [30982]
    assert multi_token_stop_ids == [80285, 30982]

    one_token_stop = generate_tokens(
        loaded.model,
        prompt.token_ids,
        max_tokens=16,
        eos_token_ids=eos_token_id,
        stop_sequences=[one_token_stop_ids],
    )
    multi_token_stop = generate_tokens(
        loaded.model,
        prompt.token_ids,
        max_tokens=16,
        eos_token_ids=eos_token_id,
        stop_sequences=[multi_token_stop_ids],
    )
    overlapping_stops = generate_tokens(
        loaded.model,
        prompt.token_ids,
        max_tokens=16,
        eos_token_ids=[eos_token_id],
        stop_sequences=[one_token_stop_ids, multi_token_stop_ids],
    )

    assert one_token_stop.token_ids == [80285]
    assert loaded.tokenizer.decode(one_token_stop.token_ids) == "CUDA"
    assert one_token_stop.finish_reason == "stop"
    assert multi_token_stop.token_ids == []
    assert multi_token_stop.finish_reason == "stop"
    assert overlapping_stops.token_ids == []
    assert overlapping_stops.finish_reason == "stop"

    sampled_a = generate_tokens(
        loaded.model,
        prompt.token_ids,
        max_tokens=8,
        eos_token_ids=eos_token_id,
        temperature=0.8,
        top_p=0.9,
        seed=1234,
    )
    sampled_b = generate_tokens(
        loaded.model,
        prompt.token_ids,
        max_tokens=8,
        eos_token_ids=eos_token_id,
        temperature=0.8,
        top_p=0.9,
        seed=1234,
    )
    assert sampled_a.token_ids == sampled_b.token_ids
    assert sampled_a.finish_reason == sampled_b.finish_reason
    sampled_token_ids = sampled_a.token_ids

    del (
        generated,
        limited,
        one_token_stop,
        multi_token_stop,
        overlapping_stops,
        sampled_a,
        sampled_b,
        reference,
        limited_reference,
    )
    gc.collect()
    torch.cuda.empty_cache()
    allocation_baseline = torch.cuda.memory_allocated(device)
    allocations = []
    for _ in range(3):
        completed = generate_tokens(
            loaded.model,
            prompt.token_ids,
            max_tokens=16,
            eos_token_ids=eos_token_id,
        )
        torch.cuda.synchronize(device)
        del completed
        gc.collect()
        torch.cuda.empty_cache()
        allocations.append(torch.cuda.memory_allocated(device))

    assert allocations == [allocation_baseline] * 3
    assert torch.cuda.max_memory_allocated(device) < torch.cuda.get_device_properties(device).total_memory

    print(f"model_revision={loaded.revision}")
    print(f"prompt_token_count={len(prompt.token_ids)}")
    print(f"generated_token_ids={expected}")
    print(f"generated_text={loaded.tokenizer.decode(expected)!r}")
    print(f"sampled_token_ids={sampled_token_ids}")
    print("finish_reasons=eos,stop,length")
    print(f"allocation_baseline_bytes={allocation_baseline}")
    print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
