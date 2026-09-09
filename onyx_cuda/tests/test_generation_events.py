"""Permanent checks for sampled streaming, cancellation, and token exhaustion."""

from types import SimpleNamespace

import pytest
import torch

import onyx_cuda.generation as generation
from onyx_cuda.generation import AcceptedTokenEvent, GenerationFinishedEvent
from onyx_cuda.speculative import decode_speculative_events, generate_speculative_events
from onyx_cuda.vocabulary import TokenByteVocabulary


def scripted_target(monkeypatch, tokens):
    calls = []

    def logits(token):
        scores = torch.full((1, 8), -torch.inf)
        scores[0, token] = 0
        return scores

    def extend(model, token_ids):
        assert torch.is_inference_mode_enabled()
        calls.append(token_ids.item())
        return logits(tokens[len(calls)])[:, None, :]

    cache = SimpleNamespace(past_key_values=object(), extend=extend)
    monkeypatch.setattr(
        generation,
        "prefill",
        lambda *_: SimpleNamespace(logits=logits(tokens[0]), past_key_values=cache),
    )
    monkeypatch.setattr(generation.CacheState, "from_prefill", lambda *args: cache)
    monkeypatch.setattr(
        generation,
        "apply_grammar_mask",
        lambda scores, valid: scores.masked_fill(
            ~torch.tensor([[index in valid for index in range(8)]]), -torch.inf
        ),
    )
    monkeypatch.setattr(
        generation, "grammar_argmax",
        lambda scores, valid, **_: generation.apply_grammar_mask(scores, valid).argmax(dim=-1),
    )
    return calls


def test_sampled_text_arrives_before_next_forward_and_close_stops_work(monkeypatch):
    calls = scripted_target(monkeypatch, [0, 1, 2])
    tokenizer = SimpleNamespace(decode=lambda ids, **_: "".join("ABC"[i] for i in ids))
    events = decode_speculative_events(
        generate_speculative_events(
            object(),
            object(),
            [0],
            max_tokens=3,
            gamma=4,
            eos_token_ids=[],
            temperature=0.8,
            top_p=0.9,
            seed=42,
        ),
        tokenizer,
    )
    assert next(events).text == "A"
    assert calls == []  # A buffered implementation would have computed all three tokens.
    assert not torch.is_inference_mode_enabled()
    assert next(events).text == "B"
    assert calls == [0]
    events.close()
    assert calls == [0]
    assert not torch.is_inference_mode_enabled()


def test_sampled_stop_tokens_are_never_emitted(monkeypatch):
    scripted_target(monkeypatch, [0, 1, 2])
    events = list(
        generation.generate_token_events(
            object(),
            [0],
            3,
            [],
            temperature=0.8,
            stop_sequences=[[1, 2]],
        )
    )
    assert [event.token_id for event in events if isinstance(event, AcceptedTokenEvent)] == [0]
    assert events[-1].result.token_ids == [0]
    assert events[-1].result.finish_reason == "stop"


@pytest.mark.parametrize("budget,reason", [(1, "length"), (4, "stop")])
def test_sampled_json_budget_and_split_unicode(monkeypatch, budget, reason):
    scripted_target(monkeypatch, [0, 1, 2, 3])
    monkeypatch.setattr(
        generation, "grammar_argmax",
        lambda *_, **__: pytest.fail("Sampling must not call the greedy CUDA selector"),
    )
    raw = [b'"', b"\xc3", b"\xa9", b'"', b"", b"", b"", b""]
    tokenizer = SimpleNamespace(
        decode=lambda ids, **_: b"".join(raw[i] for i in ids).decode("utf-8", errors="replace")
    )
    events = list(
        decode_speculative_events(
            generate_speculative_events(
                object(),
                object(),
                [0],
                max_tokens=budget,
                gamma=4,
                eos_token_ids=[],
                temperature=0.8,
                seed=42,
                json_schema='{"type":"string","enum":["é"]}',
                token_byte_vocabulary=TokenByteVocabulary(raw, 0, 4),
            ),
            tokenizer,
        )
    )
    assert isinstance(events[-1], GenerationFinishedEvent)
    assert events[-1].result.finish_reason == reason
    text = "".join(event.text for event in events[:-1])
    assert text == ('"' if budget == 1 else '"é"')


def test_closing_sampled_json_stream_releases_grammar_state(monkeypatch):
    calls = scripted_target(monkeypatch, [0, 1])
    states = []
    initialize = generation._initialize_grammar_constraint

    def track(*args):
        constraint, state = initialize(*args)
        states.append((constraint, state))
        return constraint, state

    monkeypatch.setattr(generation, "_initialize_grammar_constraint", track)
    events = generation.generate_token_events(
        object(),
        [0],
        2,
        [],
        temperature=0.8,
        json_schema='{"type":"string"}',
        token_byte_vocabulary=TokenByteVocabulary([b'"', b"a"] + [b""] * 6, 0, 6),
    )
    assert next(events).token_id == 0
    constraint, initial = states[0]
    # Native state handles advance monotonically: both old and current handles must be released.
    events.close()
    for handle in (initial, initial + 1):
        with pytest.raises(ValueError):
            constraint.get_valid_token_ids(handle)
    assert calls == []


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_gamma_zero_uses_the_target_stream_without_a_draft(monkeypatch, temperature):
    calls = scripted_target(monkeypatch, [0, 1, 2])
    events = generate_speculative_events(
        None,
        object(),
        [0],
        max_tokens=3,
        gamma=0,
        eos_token_ids=[],
        temperature=temperature,
    )
    assert next(events).token_id == 0
    assert calls == []
    remaining = list(events)
    assert [event.token_id for event in remaining[:-1]] == [1, 2]
    assert remaining[-1].result.token_ids == [0, 1, 2]
    assert remaining[-1].result.finish_reason == "length"


def test_target_timing_includes_final_schema_validation(monkeypatch):
    scripted_target(monkeypatch, [0, 1, 2, 3])
    clock = [0.0]

    def now():
        clock[0] += 0.001
        return clock[0]

    original = generation._validate_json_result

    def validate(*args):
        original(*args)
        clock[0] += 10.0

    monkeypatch.setattr(generation.time, "perf_counter", now)
    monkeypatch.setattr(generation.torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(generation, "_validate_json_result", validate)
    model = SimpleNamespace(parameters=lambda: iter([torch.tensor(0)]))
    result = generation.generate_tokens(
        model, [0], 4, [], measure=True,
        json_schema='{"type":"string","enum":["é"]}',
        token_byte_vocabulary=TokenByteVocabulary(
            [b'"', b"\xc3", b"\xa9", b'"'] + [b""] * 4, 0, 4
        ),
    )
    assert result.finish_reason == "stop"
    assert result.timings.total_seconds >= 10.0
