"""Comparison settings, completed answers, and target-output equivalence."""

import pytest

from onyx_cuda import benchmark
from onyx_cuda.prompt import format_prompt
from types import SimpleNamespace


def test_report_records_answer_settings():
    settings = benchmark._comparison_settings(benchmark.BenchmarkOptions(256, False, True))
    assert settings["max_tokens"] == 256
    assert settings["enable_thinking"] is False
    assert settings["require_complete"] is True


def test_thinking_override_applies_to_text_and_tokens_only_when_requested():
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: calls.append(kw))
    format_prompt(tokenizer, [], enable_thinking=None)
    assert all("enable_thinking" not in call for call in calls)
    calls.clear()
    format_prompt(tokenizer, [], enable_thinking=False)
    assert calls == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False},
    ]


@pytest.mark.parametrize("gamma", [None, 2])
def test_completed_answer_gate_rejects_truncation_and_passes_token_budget(monkeypatch, gamma):
    monkeypatch.setattr(benchmark, "WARMUPS", 0)
    monkeypatch.setattr(benchmark, "format_prompt", lambda *a, **kw: SimpleNamespace(token_ids=[1]))
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(benchmark.torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(benchmark.torch.cuda, "memory_allocated", lambda *a: 0)
    def generate(*args, **kwargs):
        assert kwargs["max_tokens"] == 256
        return SimpleNamespace(timings=object(), token_ids=[2], finish_reason="length")
    monkeypatch.setattr(benchmark, "generate_tokens", generate)
    monkeypatch.setattr(benchmark, "generate_speculative", generate)
    with pytest.raises(RuntimeError, match="did not complete within 256 tokens"):
        benchmark._run_prompt(object(), SimpleNamespace(eos_token_id=0), None, "case", "hello",
                              draft_model=object() if gamma else None, gamma=gamma,
                              options=benchmark.BenchmarkOptions(256, False, True))


@pytest.mark.parametrize("field,value", [("token_ids", [2]), ("finish_reason", "length")])
def test_comparison_rejects_output_different_from_target(field, value):
    baseline = {"prompts": [{"name": "case", "runs": [{"token_ids": [1], "finish_reason": "eos"}]}]}
    current = {"name": "case", "runs": [{"token_ids": [1], "finish_reason": "eos", field: value}]}
    with pytest.raises(RuntimeError, match="output changed"):
        benchmark._assert_baseline([current], baseline, "target")
