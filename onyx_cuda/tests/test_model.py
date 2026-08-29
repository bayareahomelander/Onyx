import gc
import json

import pytest
import torch
from transformers.cache_utils import DynamicCache

from onyx_cuda import _rust
import onyx_cuda.generation as generation_module
from onyx_cuda.generation import generate_tokens
from onyx_cuda.model import load_model
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt
from onyx_cuda.vocabulary import build_token_byte_vocabulary

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


def test_load_model_prompt_prefill_and_generation_on_cuda(monkeypatch):
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

    vocabulary = build_token_byte_vocabulary(
        loaded.tokenizer, result.logits.shape[-1]
    )
    repeated_vocabulary = build_token_byte_vocabulary(
        loaded.tokenizer, result.logits.shape[-1]
    )
    assert vocabulary == repeated_vocabulary
    assert len(vocabulary.token_bytes) == result.logits.shape[-1]
    assert vocabulary.special_token_count == 14
    assert vocabulary.empty_token_count == 1728
    assert vocabulary.token_bytes[11] == b","
    assert vocabulary.token_bytes[13] == b"."
    assert vocabulary.token_bytes[198] == b"\n"
    assert vocabulary.token_bytes[220] == b" "
    assert vocabulary.token_bytes[80285] == b"CUDA"
    assert vocabulary.token_bytes[151643] == b"<|endoftext|>"
    assert vocabulary.token_bytes[94] == b""
    assert vocabulary.token_bytes[151665] == b""
    assert vocabulary.token_bytes[-1] == b""

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
    assert generated.timings.grammar_compile_seconds is None
    assert generated.timings.valid_token_enumeration_seconds is None
    assert generated.timings.mask_transfer_seconds is None
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

    native_constraint = _rust.GrammarConstraint
    tracked_constraints = []

    class TrackingConstraint:
        def __init__(self, token_bytes):
            self.inner = native_constraint(token_bytes)
            self.active_states = set()
            self.compile_count = 0
            self.compile_kind = None
            self.valid_token_ids = None
            tracked_constraints.append(self)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def compile_regex(self, pattern):
            self.compile_count += 1
            self.compile_kind = "regex"
            return self.inner.compile_regex(pattern)

        def compile_json_schema(self, schema):
            self.compile_count += 1
            self.compile_kind = "json"
            return self.inner.compile_json_schema(schema)

        def get_valid_token_ids(self, state):
            self.valid_token_ids = self.inner.get_valid_token_ids(state)
            return self.valid_token_ids

        def init_state(self):
            state = self.inner.init_state()
            self.active_states.add(state)
            return state

        def advance_state(self, state, token_id):
            next_state = self.inner.advance_state(state, token_id)
            self.active_states.add(next_state)
            return next_state

        def release_state(self, state):
            self.inner.release_state(state)
            self.active_states.remove(state)

    monkeypatch.setattr(_rust, "GrammarConstraint", TrackingConstraint)
    sample_token = generation_module._sample_token

    def checked_sample_token(logits, temperature, top_p, generator):
        if not tracked_constraints or not tracked_constraints[-1].active_states:
            return sample_token(logits, temperature, top_p, generator)
        valid_token_ids = tracked_constraints[-1].valid_token_ids
        assert valid_token_ids is not None
        assert torch.isneginf(logits).sum().item() == (
            logits.numel() - len(valid_token_ids)
        )
        return sample_token(logits, temperature, top_p, generator)

    monkeypatch.setattr(generation_module, "_sample_token", checked_sample_token)
    constrained = []
    for pattern in ["CUDA", "Ready"]:
        completed = generate_tokens(
            loaded.model,
            prompt.token_ids,
            max_tokens=8,
            eos_token_ids=eos_token_id,
            regex=pattern,
            token_byte_vocabulary=vocabulary,
            measure=pattern == "CUDA",
        )
        assert loaded.tokenizer.decode(completed.token_ids) == pattern
        assert completed.finish_reason == "stop"
        if pattern == "CUDA":
            assert completed.timings is not None
            assert completed.timings.grammar_compile_seconds > 0
            assert completed.timings.valid_token_enumeration_seconds > 0
            assert completed.timings.mask_transfer_seconds > 0
        constrained.append(completed)

    with pytest.raises(ValueError, match="no valid token continuation"):
        generate_tokens(
            loaded.model,
            prompt.token_ids,
            max_tokens=8,
            eos_token_ids=eos_token_id,
            regex=r"\u{10FFFF}",
            token_byte_vocabulary=vocabulary,
        )

    json_prompt = format_prompt(
        loaded.tokenizer,
        [
            {"role": "system", "content": "Return compact JSON only."},
            {
                "role": "user",
                "content": "Use no spaces or newlines in the JSON response.",
            },
        ],
    )
    json_cases = [
        (
            "object",
            {
                "type": "object",
                "properties": {"content": {"type": "boolean"}},
            },
        ),
        (
            "required",
            {
                "type": "object",
                "properties": {"content": {"type": "boolean"}},
                "required": ["content"],
            },
        ),
        (
            "enum",
            {
                "type": "object",
                "properties": {
                    "content": {"enum": ["CUDA ready", "Ready"]}
                },
                "required": ["content"],
            },
        ),
        (
            "nested",
            {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "properties": {},
                    }
                },
                "required": ["content"],
            },
        ),
        (
            "bounded_array",
            {
                "type": "array",
                "items": {"enum": ["a", "b"]},
                "minItems": 2,
                "maxItems": 2,
            },
        ),
    ]
    parsed_json = {}
    for name, schema in json_cases:
        completed = generate_tokens(
            loaded.model,
            json_prompt.token_ids,
            max_tokens=64,
            eos_token_ids=eos_token_id,
            regex="Ready" if name == "object" else None,
            token_byte_vocabulary=vocabulary,
            json_schema=json.dumps(schema),
            measure=name == "object",
        )
        parsed_json[name] = json.loads(
            loaded.tokenizer.decode(completed.token_ids)
        )
        assert completed.finish_reason == "stop"
        if name == "object":
            assert tracked_constraints[-1].compile_kind == "json"
            assert completed.timings is not None
            assert completed.timings.grammar_compile_seconds > 0
            assert completed.timings.valid_token_enumeration_seconds > 0
            assert completed.timings.mask_transfer_seconds > 0
        del completed

    assert isinstance(parsed_json["object"], dict)
    assert set(parsed_json["required"]) == {"content"}
    assert isinstance(parsed_json["required"]["content"], bool)
    assert parsed_json["enum"]["content"] in {"CUDA ready", "Ready"}
    assert set(parsed_json["nested"]) == {"content"}
    assert parsed_json["nested"]["content"] == {}
    assert len(parsed_json["bounded_array"]) == 2
    assert set(parsed_json["bounded_array"]) <= {"a", "b"}

    assert all(item.compile_count == 1 for item in tracked_constraints)
    assert all(not item.active_states for item in tracked_constraints)

    del (
        generated,
        limited,
        one_token_stop,
        multi_token_stop,
        overlapping_stops,
        sampled_a,
        sampled_b,
        constrained,
        parsed_json,
        reference,
        limited_reference,
    )
    tracked_constraints.clear()
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
    print(f"vocabulary_special_token_count={vocabulary.special_token_count}")
    print(f"vocabulary_empty_token_count={vocabulary.empty_token_count}")
    print(f"generated_token_ids={expected}")
    print(f"generated_text={loaded.tokenizer.decode(expected)!r}")
    print(f"sampled_token_ids={sampled_token_ids}")
    print("finish_reasons=eos,stop,length")
    print(f"allocation_baseline_bytes={allocation_baseline}")
    print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")
