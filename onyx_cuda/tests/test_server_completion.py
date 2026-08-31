import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import onyx_cuda.server as server
from onyx_cuda.server import ChatCompletionRequest, create_app, prepare_generation


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, decoded):
        self.decoded = decoded

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        return [1, 2, 3] if tokenize else "prompt"

    def encode(self, text, add_special_tokens=False):
        return {"STOP": [40]}.get(text, [41])

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is True
        return self.decoded[tuple(token_ids)]


def _engine(tokenizer):
    draft = SimpleNamespace(model=object(), tokenizer=tokenizer)
    target = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=100)),
        tokenizer=tokenizer,
    )
    return SimpleNamespace(draft=draft, target=target)


def _result(token_ids, finish_reason, timings=None):
    return SimpleNamespace(
        token_ids=token_ids,
        finish_reason=finish_reason,
        timings=timings,
    )


def _timings(*, speed, acceptance, ttft, iterations):
    return SimpleNamespace(
        decode_tokens_per_second=speed,
        acceptance_rate=acceptance,
        time_to_first_token_seconds=ttft,
        speculative_iteration_count=iterations,
    )


def _fake_generation(monkeypatch, results):
    calls = []

    def generate(arguments):
        calls.append(arguments.copy())
        result = results[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(server, "_generate", generate)
    return calls


def test_non_streaming_chat_completion_returns_usage_reason_and_metrics(monkeypatch):
    tokenizer = FakeTokenizer({(10, 99): "Hello"})
    engine = _engine(tokenizer)
    calls = _fake_generation(
        monkeypatch,
        [
            _result(
                [10, 99],
                "eos",
                _timings(speed=12.5, acceptance=0.5, ttft=0.01, iterations=3),
            )
        ],
    )

    with TestClient(create_app(engine=engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "onyx-speculative"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "eos",
        }
    ]
    assert body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert body["onyx_metrics"] == {
        "tokens_per_second": 12.5,
        "acceptance_rate": 0.5,
        "ttft_ms": 10.0,
        "grammar_constrained": False,
        "speculative_iterations": 3,
    }
    assert len(calls) == 1
    assert calls[0]["measure"] is True
    assert calls[0]["draft_model"] is engine.draft.model
    assert calls[0]["target_model"] is engine.target.model


def test_stop_length_and_multiple_choices_use_final_choice_metrics(monkeypatch):
    tokenizer = FakeTokenizer(
        {
            (20, 21): "first STOP ignored",
            (22,): "second",
        }
    )
    calls = _fake_generation(
        monkeypatch,
        [
            _result(
                [20, 21],
                "stop",
                _timings(speed=5.0, acceptance=1.0, ttft=0.01, iterations=2),
            ),
            _result(
                [22],
                "length",
                _timings(speed=7.0, acceptance=0.25, ttft=0.02, iterations=4),
            ),
        ],
    )

    with TestClient(create_app(engine=_engine(tokenizer))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "n": 2,
                "stop": ["STOP"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "first "},
            "finish_reason": "stop",
        },
        {
            "index": 1,
            "message": {"role": "assistant", "content": "second"},
            "finish_reason": "length",
        },
    ]
    assert body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "total_tokens": 6,
    }
    assert body["onyx_metrics"] == {
        "tokens_per_second": 7.0,
        "acceptance_rate": 0.25,
        "ttft_ms": 20.0,
        "grammar_constrained": False,
        "speculative_iterations": 4,
    }
    assert len(calls) == 2
    assert all(call["stop_sequences"] == [[40]] for call in calls)


def test_regex_response_is_grammar_constrained(monkeypatch):
    tokenizer = FakeTokenizer({(30,): "CUDA Ready"})
    engine = _engine(tokenizer)
    calls = _fake_generation(monkeypatch, [_result([30], "stop")])
    vocabulary = object()
    monkeypatch.setattr(server, "_build_vocabulary", lambda *_args: vocabulary)

    with TestClient(create_app(engine=engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "regex": "CUDA Ready",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "CUDA Ready"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["onyx_metrics"]["grammar_constrained"] is True
    assert body["onyx_metrics"]["tokens_per_second"] is None
    assert calls[0]["regex"] == "CUDA Ready"
    assert calls[0]["token_byte_vocabulary"] is vocabulary


@pytest.mark.parametrize(
    ("text", "compact_json", "expected"),
    [
        (' { "content": "Ready" } ', True, '{"content":"Ready"}'),
        (' { "content": "Ready" } ', False, ' { "content": "Ready" } '),
        ("not json", True, "not json"),
    ],
)
def test_json_response_compacts_only_when_requested(monkeypatch, text, compact_json, expected):
    tokenizer = FakeTokenizer({(31,): text})
    engine = _engine(tokenizer)
    calls = _fake_generation(monkeypatch, [_result([31], "stop")])
    vocabulary = object()
    monkeypatch.setattr(server, "_build_vocabulary", lambda *_args: vocabulary)
    schema = {"type": "object", "required": ["content"]}

    with TestClient(create_app(engine=engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "json_schema": schema,
                "compact_json": compact_json,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == expected
    assert body["onyx_metrics"]["grammar_constrained"] is True
    assert json.loads(calls[0]["json_schema"]) == schema
    assert calls[0]["token_byte_vocabulary"] is vocabulary


def test_streaming_request_is_not_run_as_non_streaming(monkeypatch):
    calls = _fake_generation(monkeypatch, [])
    tokenizer = FakeTokenizer({})

    with TestClient(create_app(engine=_engine(tokenizer))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Streaming is not implemented"}
    assert calls == []


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"top_p": 0},
        {"json_schema": "not an object"},
    ],
)
def test_request_validation_stays_422_and_does_not_generate(monkeypatch, invalid_field):
    tokenizer = FakeTokenizer({(10,): "Ready"})
    calls = _fake_generation(monkeypatch, [_result([10], "stop")])

    with TestClient(create_app(engine=_engine(tokenizer))) as client:
        invalid = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                **invalid_field,
            },
        )
        valid = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert invalid.status_code == 422
    assert valid.status_code == 200
    assert len(calls) == 1


def test_unknown_model_stays_400_and_next_request_succeeds(monkeypatch):
    tokenizer = FakeTokenizer({(10,): "Ready"})
    calls = _fake_generation(monkeypatch, [_result([10], "stop")])

    with TestClient(create_app(engine=_engine(tokenizer))) as client:
        invalid = client.post(
            "/v1/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        valid = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert invalid.status_code == 400
    assert "missing" in invalid.json()["detail"]
    assert valid.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (ValueError("invalid regex"), 400, "invalid regex"),
        (
            RuntimeError("CUDA is unavailable"),
            503,
            "Model or CUDA service unavailable",
        ),
        (
            OSError("model is unavailable"),
            503,
            "Model or CUDA service unavailable",
        ),
        (LookupError("private traceback detail"), 500, "Internal server error"),
    ],
)
def test_generation_errors_are_mapped_and_next_request_succeeds(
    monkeypatch, error, status_code, detail
):
    tokenizer = FakeTokenizer({(10,): "Ready"})
    calls = _fake_generation(
        monkeypatch,
        [error, _result([10], "stop")],
    )

    with TestClient(create_app(engine=_engine(tokenizer)), raise_server_exceptions=False) as client:
        failed = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        valid = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert failed.status_code == status_code
    assert failed.json() == {"detail": detail}
    assert "traceback" not in failed.text.lower()
    assert valid.status_code == 200
    assert len(calls) == 2


def test_real_cuda_request_matches_direct_generation_content_and_counts():
    from onyx_cuda.model import load_model_pair
    from onyx_cuda.speculative import generate_speculative

    pair = load_model_pair()
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with CUDA ready."},
        ],
        "max_tokens": 4,
    }

    with TestClient(create_app(engine=pair)) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    request = ChatCompletionRequest.model_validate(payload)
    arguments = prepare_generation(request, pair)
    arguments["measure"] = True
    direct = generate_speculative(**arguments)
    expected = pair.target.tokenizer.decode(direct.token_ids, skip_special_tokens=True)

    assert body["choices"][0]["message"]["content"] == expected
    assert body["choices"][0]["finish_reason"] == direct.finish_reason
    assert body["usage"] == {
        "prompt_tokens": len(arguments["prompt_token_ids"]),
        "completion_tokens": len(direct.token_ids),
        "total_tokens": len(arguments["prompt_token_ids"]) + len(direct.token_ids),
    }
