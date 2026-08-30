import gc
import json

import torch

from onyx_cuda.benchmark import MAX_TOKENS, PROMPTS
from onyx_cuda.generation import generate_tokens
from onyx_cuda.model import TARGET_MODEL_ID, load_model
from onyx_cuda.prompt import format_prompt
from onyx_cuda.vocabulary import build_token_byte_vocabulary


def test_target_generation_oracle_constraints_and_memory():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loaded = load_model(TARGET_MODEL_ID)
    eos_token_id = loaded.tokenizer.eos_token_id
    oracle_token_ids = {}

    for name, user_prompt in PROMPTS.items():
        prompt = format_prompt(
            loaded.tokenizer,
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": user_prompt},
            ],
        )
        input_ids = torch.tensor([prompt.token_ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            reference = loaded.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=MAX_TOKENS,
                use_cache=True,
                pad_token_id=loaded.tokenizer.pad_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
                # Match Onyx's plain argmax contract, not the checkpoint's 1.1 default.
                repetition_penalty=1.0,
            )
        expected = reference[0, len(prompt.token_ids) :].tolist()
        generated = generate_tokens(
            loaded.model,
            prompt.token_ids,
            max_tokens=MAX_TOKENS,
            eos_token_ids=eos_token_id,
            measure=True,
        )
        expected_reason = (
            "eos" if expected and expected[-1] == eos_token_id else "length"
        )
        assert generated.token_ids == expected
        assert generated.finish_reason == expected_reason
        assert generated.timings is not None
        assert generated.timings.time_to_first_token_seconds > 0
        assert generated.timings.total_seconds >= (
            generated.timings.time_to_first_token_seconds
        )
        oracle_token_ids[name] = expected
        del generated, reference, input_ids, attention_mask

    vocabulary = build_token_byte_vocabulary(
        loaded.tokenizer, loaded.model.config.vocab_size
    )
    regex_result = generate_tokens(
        loaded.model,
        format_prompt(
            loaded.tokenizer,
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with CUDA ready."},
            ],
        ).token_ids,
        max_tokens=MAX_TOKENS,
        eos_token_ids=eos_token_id,
        regex="CUDA Ready",
        token_byte_vocabulary=vocabulary,
    )
    assert loaded.tokenizer.decode(regex_result.token_ids) == "CUDA Ready"
    assert regex_result.finish_reason == "stop"

    schema = {
        "type": "object",
        "properties": {"content": {"enum": ["CUDA ready", "Ready"]}},
        "required": ["content"],
    }
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
    json_result = generate_tokens(
        loaded.model,
        json_prompt.token_ids,
        max_tokens=64,
        eos_token_ids=eos_token_id,
        json_schema=json.dumps(schema, separators=(",", ":")),
        token_byte_vocabulary=vocabulary,
    )
    parsed = json.loads(loaded.tokenizer.decode(json_result.token_ids))
    assert set(parsed) == {"content"}
    assert parsed["content"] in {"CUDA ready", "Ready"}
    assert json_result.finish_reason == "stop"

    del regex_result, json_result
    gc.collect()
    torch.cuda.empty_cache()
    allocation_baseline = torch.cuda.memory_allocated(device)
    allocations = []
    memory_prompt = format_prompt(
        loaded.tokenizer,
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with CUDA ready."},
        ],
    )
    for _ in range(3):
        completed = generate_tokens(
            loaded.model,
            memory_prompt.token_ids,
            max_tokens=16,
            eos_token_ids=eos_token_id,
        )
        torch.cuda.synchronize(device)
        del completed
        gc.collect()
        torch.cuda.empty_cache()
        allocations.append(torch.cuda.memory_allocated(device))

    assert allocations == [allocation_baseline] * 3
    peak_allocated = torch.cuda.max_memory_allocated(device)
    assert peak_allocated < torch.cuda.get_device_properties(device).total_memory

    print(f"target_model_revision={loaded.revision}")
    print(f"target_oracle_token_ids={json.dumps(oracle_token_ids)}")
    print(f"target_allocation_baseline_bytes={allocation_baseline}")
    print(f"target_peak_allocated_bytes={peak_allocated}")
