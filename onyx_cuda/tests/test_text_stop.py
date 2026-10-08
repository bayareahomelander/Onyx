"""CPU checks for decoded stops through the real generation and SSE loops."""

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import onyx_cuda.generation as generation
import onyx_cuda.speculative as speculative
from onyx_cuda.server import MODEL_ID, create_app
from onyx_cuda.vocabulary import TokenByteVocabulary


@pytest.fixture
def runtime(monkeypatch):
    forwards, releases = [], []
    backend = SimpleNamespace(active=False)

    def scores(token):
        logits = torch.full((1, 32), -torch.inf)
        logits[0, token] = 0
        return logits

    class Cache:
        device = torch.device("cpu")

        def __init__(self, ids):
            self.history = list(ids)
            self.prompt_length = len(ids)
            self.attention_mask = torch.ones((1, len(ids)), dtype=torch.long)
            self.past_key_values = self

        @property
        def length(self):
            return len(self.history)

        def extend(self, model, ids):
            assert torch.is_inference_mode_enabled()
            forwards.append((model.role, ids[0].tolist()))
            result = []
            for token in ids[0].tolist():
                self.history.append(token)
                result.append(scores(self.length - self.prompt_length + 1))
            return torch.stack(result, dim=1)

        def crop(self, length):
            assert 0 <= length <= self.length
            del self.history[length:]

        @classmethod
        def from_prefill(cls, cache, device):
            return cache

    def prefill(model, ids):
        forwards.append((model.role, list(ids)))
        cache = Cache(ids)
        return SimpleNamespace(logits=scores(1), token_id=torch.tensor([1]), past_key_values=cache)

    def start(cache, capacity):
        assert not backend.active
        backend.active = True

        def release():
            assert backend.active
            backend.active = False
            releases.append(True)

        cache.release = release
        return cache

    backend.start = start

    def close():
        assert not backend.active

    backend.close = close
    draft = SimpleNamespace(role="draft", _onyx_draft_backend=backend)
    target = SimpleNamespace(role="target", config=SimpleNamespace(vocab_size=32))
    for model in (draft, target):
        model.parameters = lambda: iter([torch.tensor(0)])
    for module in (generation, speculative):
        monkeypatch.setattr(module, "prefill", prefill)
        monkeypatch.setattr(module, "CacheState", Cache)
    # The measured target-only path synchronizes even for a CPU test double.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    return SimpleNamespace(
        draft=draft, target=target, forwards=forwards,
        backend=backend, releases=releases,
    )


def arguments(runtime, mode, **kwargs):
    return dict(
        draft_model=runtime.draft, target_model=runtime.target,
        prompt_token_ids=[0] * 4, max_tokens=8, eos_token_ids=[],
        gamma=0 if mode == "target" else 2,
        temperature=0.8 if mode == "sampled" else 0.0,
        measure=True, **kwargs,
    )


@pytest.mark.parametrize("mode", ["target", "sampled", "fixed"])
@pytest.mark.parametrize("pieces,expected", [
    ([b"Hello END trailing"], "Hello "),
    ([b"Hello ", b"END trailing"], "Hello "),
    ([b"Hello E", b"N", b"D trailing"], "Hello "),
    ([b"caf\xc3", b"\xa9 EN", b"D trailing"], "café "),
])
def test_text_stop_finalizes_without_another_forward(runtime, monkeypatch, mode, pieces, expected):
    calls_at_stop = []

    def decode(ids, **kwargs):
        text = b"".join(pieces[i - 1] if i <= len(pieces) else b"wasted" for i in ids).decode(
            "utf-8", errors="replace",
        )
        if "END" in text:
            calls_at_stop.append(list(runtime.forwards))
        return text

    tokenizer = SimpleNamespace(decode=decode)
    # A second request must be able to reacquire the shared draft backend.
    for _ in range(2):
        calls_at_stop.clear()
        events = list(speculative.decode_speculative_events(
            speculative.generate_speculative_events(**arguments(runtime, mode)), tokenizer, ["END"],
        ))
        assert runtime.forwards == calls_at_stop[0]
        assert "".join(e.text for e in events if isinstance(e, speculative.TextDeltaEvent)) == expected
        terminals = [e for e in events if isinstance(e, generation.GenerationFinishedEvent)]
        assert len(terminals) == 1
        result = terminals[0].result
        assert result.token_ids == list(range(1, len(pieces) + 1))
        assert result.finish_reason == "stop"
        assert result.past_key_values.history == [0] * 4 + result.token_ids[:-1]
        assert result.timings.total_seconds >= result.timings.time_to_first_token_seconds >= 0
        if len(pieces) == 1:
            assert result.timings.decode_tokens_per_second is None
        else:
            assert result.timings.decode_tokens_per_second > 0
        assert not runtime.backend.active
        assert not torch.is_inference_mode_enabled()
    if mode == "fixed" and len(pieces) > 1:
        assert runtime.releases == [True, True]
    elif mode == "fixed":
        # A stop in the first token ends generation before the draft prefill.
        assert runtime.releases == []
        assert {role for role, _ in runtime.forwards} == {"target"}


@pytest.mark.parametrize("mode", ["target", "sampled", "fixed"])
@pytest.mark.parametrize("budget,stop_tokens", [
    (2, None), (3, [[29, 30, 31]]), (8, [[29, 30, 31]]),
])
def test_text_stop_during_pending_token_flush(runtime, mode, budget, stop_tokens):
    args = arguments(runtime, mode, stop_sequences=stop_tokens)
    args["max_tokens"] = budget
    seen_forwards = []

    def decode(ids, **kwargs):
        text = "".join({1: "Hello ", 2: "END"}.get(i, "wasted") for i in ids)
        if "END" in text:
            seen_forwards.append(list(runtime.forwards))
        return text

    events = list(speculative.decode_speculative_events(
        speculative.generate_speculative_events(**args), SimpleNamespace(decode=decode), ["END"],
    ))
    assert runtime.forwards == seen_forwards[0]
    assert events[-1].result.token_ids == [1, 2]
    assert events[-1].result.finish_reason == "stop"
    assert events[-1].result.past_key_values.history == [0] * 4 + [1]
    assert not runtime.backend.active


@pytest.mark.parametrize("mode", ["target", "sampled", "fixed"])
@pytest.mark.parametrize("finish", ["eos", "stop_sequence", "grammar"])
def test_finished_cache_excludes_the_last_returned_token(runtime, monkeypatch, mode, finish):
    # Every model predicts 1, 2, 3, ...; speculation verifies [1, 2, 3] in one
    # round, consuming the accepted EOS, stop sequence, or completing token.
    args = arguments(runtime, mode)
    if finish == "eos":
        args["eos_token_ids"] = [3]
    elif finish == "stop_sequence":
        args["stop_sequences"] = [[2, 3]]
    else:
        for module in (generation, speculative):
            monkeypatch.setattr(module, "grammar_argmax", lambda logits, valid, **_: logits.argmax(dim=-1))
        monkeypatch.setattr(generation, "apply_grammar_mask", lambda logits, valid: logits)
        args.update(regex="xxx", token_byte_vocabulary=TokenByteVocabulary([b"x"] * 32, 0, 32))
    result = speculative.generate_speculative(**args)
    expected = [1] if finish == "stop_sequence" else [1, 2, 3]
    assert result.token_ids == expected
    assert result.finish_reason == ("eos" if finish == "eos" else "stop")
    assert result.past_key_values.history == [0] * 4 + expected[:-1]
    assert not runtime.backend.active


@pytest.mark.parametrize("mode", ["target", "sampled", "fixed"])
def test_text_stop_releases_native_grammar_states(runtime, monkeypatch, mode):
    constraints = []
    original = generation._initialize_grammar_constraint

    def initialize(*args):
        constraint, state = original(*args)
        constraints.append(constraint)
        return constraint, state

    for module in (generation, speculative):
        monkeypatch.setattr(module, "_initialize_grammar_constraint", initialize)
        monkeypatch.setattr(module, "grammar_argmax", lambda logits, valid, **_: logits.argmax(dim=-1))
    monkeypatch.setattr(generation, "apply_grammar_mask", lambda logits, valid: logits)
    # Direct decoder use also needs cleanup; HTTP rejects custom stops with grammar.
    events = list(speculative.decode_speculative_events(
        speculative.generate_speculative_events(**arguments(
            runtime, mode, regex="x+",
            token_byte_vocabulary=TokenByteVocabulary([b"x"] * 32, 0, 32),
        )), SimpleNamespace(decode=lambda ids, **_: "".join("x" for _ in ids)), ["xx"],
    ))
    assert events[-1].result.token_ids == [1, 2]
    assert events[-1].result.finish_reason == "stop"
    for constraint in constraints:
        # Handles advance monotonically; no previously allocated state may survive.
        next_state = constraint.init_state()
        for state in range(next_state):
            with pytest.raises(ValueError):
                constraint.get_valid_token_ids(state)
        constraint.release_state(next_state)
    assert not runtime.backend.active


@pytest.mark.parametrize("mode", ["target", "sampled", "fixed"])
def test_sse_text_stop_finishes_once_and_allows_next_request(runtime, monkeypatch, mode):
    import onyx_cuda.draft_backend as draft_backend

    monkeypatch.setattr(draft_backend, "prepare_draft_backend", lambda *_, **__: {"active": "eager"})
    tokenizer = SimpleNamespace(
        eos_token_id=31,
        apply_chat_template=lambda *_, **kwargs: [0] * 4,
        encode=lambda *_, **kwargs: [30],  # Different tokenization from generated text.
        decode=lambda ids, **_: "".join({1: "Hello ", 2: "END"}.get(i, "wasted") for i in ids),
    )
    engine = SimpleNamespace(
        target=SimpleNamespace(model=runtime.target, tokenizer=tokenizer),
        draft=SimpleNamespace(model=runtime.draft, tokenizer=tokenizer),
    )
    app = create_app(
        engine=engine, gamma=0 if mode == "target" else 2,
    )
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hi"}], "stream": True,
                "stop": "END", "max_tokens": 8, "temperature": 0.8 if mode == "sampled" else 0,
            })
            assert response.status_code == 200
            payloads = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert payloads.count("[DONE]") == 1 and payloads[-1] == "[DONE]"
            chunks = [json.loads(payload) for payload in payloads[:-1]]
            assert all("error" not in chunk for chunk in chunks)
            choices = [chunk["choices"][0] for chunk in chunks]
            assert "".join(choice["delta"].get("content") or "" for choice in choices) == "Hello "
            assert [choice["finish_reason"] for choice in choices if choice["finish_reason"]] == ["stop"]
            assert not app.state.engine_locks[MODEL_ID].locked()
            assert not runtime.backend.active


@pytest.mark.parametrize("mode", ["target", "sampled", "fixed"])
@pytest.mark.parametrize("n", [1, 2])
def test_non_streaming_text_stop_ends_inference(runtime, monkeypatch, mode, n):
    import onyx_cuda.draft_backend as draft_backend

    monkeypatch.setattr(draft_backend, "prepare_draft_backend", lambda *_, **__: {"active": "eager"})
    forwards_at_stop = []

    def decode(ids, **_):
        text = "".join({1: "Hello ", 2: "END"}.get(i, "wasted") for i in ids)
        if "END" in text:
            forwards_at_stop.append(list(runtime.forwards))
        return text

    tokenizer = SimpleNamespace(
        eos_token_id=31,
        apply_chat_template=lambda *_, **kwargs: [0] * 4,
        encode=lambda *_, **kwargs: [30],  # Different tokenization from generated text.
        decode=decode,
    )
    engine = SimpleNamespace(
        target=SimpleNamespace(model=runtime.target, tokenizer=tokenizer),
        draft=SimpleNamespace(model=runtime.draft, tokenizer=tokenizer),
    )
    app = create_app(
        engine=engine, gamma=0 if mode == "target" else 2,
    )
    with TestClient(app) as client:
        for _ in range(2):
            forwards_at_stop.clear()
            runtime.forwards.clear()
            response = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hi"}], "n": n,
                "stop": "END", "max_tokens": 8, "temperature": 0.8 if mode == "sampled" else 0,
            })
            assert response.status_code == 200
            body = response.json()
            assert [choice["message"]["content"] for choice in body["choices"]] == ["Hello "] * n
            assert [choice["finish_reason"] for choice in body["choices"]] == ["stop"] * n
            # Only the consumed prefix through the stop-completing token was generated.
            assert body["usage"] == {"prompt_tokens": 4, "completion_tokens": 2 * n,
                                     "total_tokens": 4 + 2 * n}
            # Per choice, the decoder reaches the stop and the handler then decodes
            # the result. Equal snapshots mean no forward ran after the stop.
            assert len(forwards_at_stop) == 2 * n
            assert forwards_at_stop[0::2] == forwards_at_stop[1::2]
            assert runtime.forwards == forwards_at_stop[-1]
            assert not app.state.engine_locks[MODEL_ID].locked()
            assert not runtime.backend.active


@pytest.mark.parametrize("mode", ["target", "sampled", "fixed"])
@pytest.mark.parametrize("stop_in_first_token", [False, True])
def test_closing_decoder_around_text_stop_releases_producer(runtime, mode, stop_in_first_token):
    def decode(ids, **_):
        if stop_in_first_token:
            return "Hello END"
        # Text first appears at the second token, after the draft cache has started.
        return "Hello " if len(ids) > 1 else ""

    tokenizer = SimpleNamespace(decode=decode)
    events = speculative.decode_speculative_events(
        speculative.generate_speculative_events(**arguments(runtime, mode)), tokenizer, ["END"],
    )
    assert next(events) == speculative.TextDeltaEvent("Hello ")
    forwards = list(runtime.forwards)
    if stop_in_first_token:
        assert not runtime.backend.active  # Final text may block on the client's queue.
    events.close()
    assert runtime.forwards == forwards
    assert not runtime.backend.active
    # A stop in the first token finishes before the draft cache is acquired.
    assert runtime.releases == ([True] if mode == "fixed" and not stop_in_first_token else [])


def test_text_stop_does_not_hide_cleanup_failure():
    def source():
        try:
            yield generation.AcceptedTokenEvent(1)
        finally:
            raise RuntimeError("cleanup failed")

    events = speculative.decode_speculative_events(
        source(), SimpleNamespace(decode=lambda *_, **__: "END"), ["END"],
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        list(events)
