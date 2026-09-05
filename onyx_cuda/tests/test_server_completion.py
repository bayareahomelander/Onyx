import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import onyx_cuda.server as server
from onyx_cuda.server import (
    MODEL_ID,
    ChatCompletionRequest,
    create_app,
    prepare_generation,
)


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
    ],
)
def test_json_response_compacts_only_when_requested(monkeypatch, text, compact_json, expected):
    tokenizer = FakeTokenizer({(31,): text})
    engine = _engine(tokenizer)
    calls = _fake_generation(monkeypatch, [_result([31], "stop")])
    vocabulary = object()
    monkeypatch.setattr(server, "_build_vocabulary", lambda *_args: vocabulary)
    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    }

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


def test_json_compaction_preserves_numeric_precision(monkeypatch):
    text = ' { "value": 0.10000000000000000001 } '
    engine = _engine(FakeTokenizer({(31,): text}))
    _fake_generation(monkeypatch, [_result([31], "stop")])
    monkeypatch.setattr(server, "_build_vocabulary", lambda *_args: object())
    with TestClient(create_app(engine=engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "json_schema": {"type": "object", "properties": {"value": {"type": "number"}}},
            },
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"value":0.10000000000000000001}'


@pytest.mark.parametrize("stream", [False, True])
def test_http_schema_rejects_decimal_precision_loss(monkeypatch, stream):
    engine = _engine(FakeTokenizer({}))
    calls = _fake_generation(monkeypatch, [])
    body = (
        '{"messages":[{"role":"user","content":"Hi"}],'
        '"json_schema":{"enum":[0.10000000000000000001]},'
        '"stream":' + json.dumps(stream) + "}"
    )
    with TestClient(create_app(engine=engine)) as client:
        response = client.post(
            "/v1/chat/completions", content=body, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 400
    assert "loses precision" in response.text
    assert calls == []


def _parse_sse(body: str) -> list[str]:
    payloads = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        data = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            payloads.append("\n".join(data))
    return payloads


@pytest.mark.parametrize("text", ["not json", '{"content":1}', "{}"])
@pytest.mark.parametrize("stream", [False, True])
def test_json_response_never_reports_success_for_invalid_decoded_output(monkeypatch, text, stream):
    engine = _engine(FakeTokenizer({(31,): text}))
    _fake_generation(monkeypatch, [_result([31], "stop")])
    _fake_stream(
        monkeypatch, [SimpleNamespace(text=text), SimpleNamespace(result=_result([31], "stop"))]
    )
    monkeypatch.setattr(server, "_build_vocabulary", lambda *_args: object())
    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    }
    with TestClient(create_app(engine=engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "json_schema": schema,
                "stream": stream,
            },
        )
    if not stream:
        assert response.status_code == 400
    else:
        events = _parse_sse(response.text)
        assert events[-1] == "[DONE]"
        chunks = [json.loads(event) for event in events[:-1]]
        assert any("error" in chunk for chunk in chunks)
        assert not any(
            choice["finish_reason"] is not None
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )


def _fake_stream(monkeypatch, events, error=None):
    calls = []

    def fake_iter(arguments, tokenizer, stop):
        calls.append({"arguments": arguments.copy(), "stop": stop})

        def gen():
            yield from events
            if error is not None:
                raise error

        return gen()

    monkeypatch.setattr(server, "_completion_event_iter", fake_iter)
    return calls


def test_streaming_chunks_preserve_id_and_real_finish_reason(monkeypatch):
    engine = _engine(FakeTokenizer({}))
    collected = []

    def generate(_arguments):
        collected.append(1)
        raise AssertionError("streaming must not collect generate_speculative")

    monkeypatch.setattr(server, "_generate", generate)
    calls = _fake_stream(
        monkeypatch,
        [
            SimpleNamespace(text="Hel"),
            SimpleNamespace(text="lo"),
            SimpleNamespace(result=_result([10, 11], "eos")),
        ],
    )

    with TestClient(create_app(engine=engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "stop": ["END"],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = _parse_sse(response.text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(item) for item in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if chunk["choices"][0]["delta"].get("content")
    )
    assert content == "Hello"
    finished = [chunk for chunk in chunks if chunk["choices"][0]["finish_reason"] is not None]
    assert len(finished) == 1
    assert finished[0]["choices"][0]["finish_reason"] == "eos"
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert {chunk["created"] for chunk in chunks} == {chunks[0]["created"]}
    assert {chunk["model"] for chunk in chunks} == {MODEL_ID}
    assert collected == []
    assert calls[0]["arguments"]["measure"] is True
    assert calls[0]["stop"] == ["END"]


def test_streaming_rejects_n_and_unknown_model_before_generation(monkeypatch):
    calls = _fake_stream(monkeypatch, [])

    with TestClient(create_app(engine=_engine(FakeTokenizer({})))) as client:
        many = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": True, "n": 2},
        )
        missing = client.post(
            "/v1/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        invalid = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": True, "top_p": 0},
        )

    assert many.status_code == 400
    assert many.json() == {"detail": "stream=true supports only n=1"}
    assert missing.status_code == 400
    assert "missing" in missing.json()["detail"]
    assert invalid.status_code == 422
    assert calls == []


@pytest.mark.parametrize(
    ("error", "error_type", "message"),
    [
        (ValueError("invalid regex"), "invalid_request", "invalid regex"),
        (
            RuntimeError("CUDA is unavailable"),
            "service_unavailable",
            "Model or CUDA service unavailable",
        ),
        (LookupError("private traceback detail"), "server_error", "Internal server error"),
    ],
)
def test_streaming_mid_stream_errors_emit_event_and_done(monkeypatch, error, error_type, message):
    _fake_stream(monkeypatch, [SimpleNamespace(text="He")], error=error)
    tokenizer = FakeTokenizer({(10,): "Ready"})

    with TestClient(create_app(engine=_engine(tokenizer)), raise_server_exceptions=False) as client:
        streamed = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        monkeypatch.setattr(server, "_generate", lambda _arguments: _result([10], "stop"))
        valid = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    payloads = _parse_sse(streamed.text)
    assert streamed.status_code == 200
    assert payloads[-1] == "[DONE]"
    events = [json.loads(item) for item in payloads[:-1]]
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[1]["choices"][0]["delta"]["content"] == "He"
    assert events[-1] == {"error": {"message": message, "type": error_type}}
    assert "traceback" not in streamed.text.lower()
    assert "private traceback detail" not in streamed.text
    assert valid.status_code == 200


def test_cancelled_stream_releases_lock_and_grammar(monkeypatch):
    tokenizer = FakeTokenizer({(10,): "Ready"})
    started = threading.Event()
    release = threading.Event()
    grammar_states = {"live"}

    def fake_iter(_arguments, _tokenizer, _stop):
        def gen():
            try:
                started.set()
                assert release.wait(timeout=5)
                yield SimpleNamespace(text="Hello")
                yield SimpleNamespace(result=_result([10], "stop"))
            finally:
                grammar_states.clear()

        return gen()

    monkeypatch.setattr(server, "_completion_event_iter", fake_iter)
    monkeypatch.setattr(server, "_generate", lambda _arguments: _result([10], "stop"))

    async def exercise():
        app = create_app(engine=_engine(tokenizer))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                cancelled = asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
                    )
                )
                assert await asyncio.to_thread(started.wait, 5)
                cancelled.cancel()
                await asyncio.sleep(0)
                waiting = asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "Hi"}]},
                    )
                )
                await asyncio.sleep(0.05)
                lock = app.state.engine_locks[MODEL_ID]
                assert lock.locked()
                assert grammar_states == {"live"}
                assert not waiting.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled
                assert (await waiting).status_code == 200
                assert not lock.locked()
                assert grammar_states == set()

    asyncio.run(exercise())


def test_overlapping_requests_serialize_engine_and_leave_routes_responsive(monkeypatch):
    tokenizer = FakeTokenizer({(10,): "Ready"})
    started = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    active = 0
    max_active = 0

    def generate(_arguments):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        started.set()
        try:
            assert release.wait(timeout=5)
            return _result([10], "stop")
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(server, "_generate", generate)
    payload = {"messages": [{"role": "user", "content": "Hi"}]}

    with TestClient(create_app(engine=_engine(tokenizer))) as client:
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(client.post, "/v1/chat/completions", json=payload)
            try:
                assert started.wait(timeout=5)
                second = executor.submit(client.post, "/v1/chat/completions", json=payload)
                root = executor.submit(client.get, "/")
                assert root.result(timeout=2).status_code == 200
                assert not second.done()
            finally:
                release.set()

            assert first.result(timeout=5).status_code == 200
            assert second.result(timeout=5).status_code == 200

    assert max_active == 1


def test_cancelled_request_holds_lock_until_generation_stops(monkeypatch):
    tokenizer = FakeTokenizer({(10,): "Ready"})
    started = threading.Event()
    release = threading.Event()

    def generate(_arguments):
        started.set()
        assert release.wait(timeout=5)
        return _result([10], "stop")

    monkeypatch.setattr(server, "_generate", generate)
    payload = {"messages": [{"role": "user", "content": "Hi"}]}

    async def exercise():
        app = create_app(engine=_engine(tokenizer))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                cancelled = asyncio.create_task(client.post("/v1/chat/completions", json=payload))
                assert await asyncio.to_thread(started.wait, 5)
                cancelled.cancel()
                await asyncio.sleep(0)
                waiting = asyncio.create_task(client.post("/v1/chat/completions", json=payload))
                await asyncio.sleep(0.05)
                lock = app.state.engine_locks[MODEL_ID]
                assert lock.locked()
                assert not waiting.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await cancelled
                assert (await waiting).status_code == 200
                assert not lock.locked()

    asyncio.run(exercise())


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


def test_real_cuda_two_client_requests_match_direct_generation_without_model_copies():
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    lambda _index: client.post("/v1/chat/completions", json=payload),
                    range(2),
                )
            )
        registered = client.app.state.engines[MODEL_ID]

    assert registered is pair
    assert registered.draft.model is pair.draft.model
    assert registered.target.model is pair.target.model
    request = ChatCompletionRequest.model_validate(payload)
    arguments = prepare_generation(request, pair)
    arguments["measure"] = True
    direct = generate_speculative(**arguments)
    expected = pair.target.tokenizer.decode(direct.token_ids, skip_special_tokens=True)

    for response in responses:
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == expected
        assert body["choices"][0]["finish_reason"] == direct.finish_reason
        assert body["usage"] == {
            "prompt_tokens": len(arguments["prompt_token_ids"]),
            "completion_tokens": len(direct.token_ids),
            "total_tokens": len(arguments["prompt_token_ids"]) + len(direct.token_ids),
        }


def test_real_uvicorn_api_phase_gate():
    import socket
    import time

    import torch
    import uvicorn
    from onyx_cuda.model import load_model_pair
    from onyx_cuda.speculative import generate_speculative

    pair = load_model_pair()
    app = create_app(engine=pair)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not server.started:
        assert thread.is_alive()
        assert time.time() < deadline
        time.sleep(0.05)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with CUDA ready."},
        ],
        "max_tokens": 4,
    }
    schema = {
        "type": "object",
        "properties": {"content": {"enum": ["CUDA ready", "Ready"]}},
        "required": ["content"],
    }

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
            root = client.get("/")
            models = client.get("/v1/models")
            assert root.status_code == 200
            assert root.json()["status"] == "ok"
            assert "/v1/chat/completions" in root.json()["endpoints"]
            assert [item["id"] for item in models.json()["data"]] == [MODEL_ID]
            assert (
                client.post(
                    "/v1/chat/completions",
                    json={**payload, "top_p": 0},
                ).status_code
                == 422
            )
            unknown = client.post(
                "/v1/chat/completions",
                json={**payload, "model": "missing"},
            )
            assert unknown.status_code == 400
            assert "missing" in unknown.json()["detail"]

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            warmup = client.post("/v1/chat/completions", json=payload)
            assert warmup.status_code == 200
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            allocated = []
            for _ in range(3):
                repeated = client.post("/v1/chat/completions", json=payload)
                assert repeated.status_code == 200
                torch.cuda.synchronize()
                allocated.append(torch.cuda.memory_allocated())
            peak = torch.cuda.max_memory_allocated()
            assert allocated == [baseline, baseline, baseline]
            assert peak >= baseline

            body = warmup.json()
            request = ChatCompletionRequest.model_validate(payload)
            arguments = prepare_generation(request, pair)
            arguments["measure"] = True
            direct = generate_speculative(**arguments)
            expected = pair.target.tokenizer.decode(direct.token_ids, skip_special_tokens=True)
            assert app.state.engines[MODEL_ID] is pair
            assert body["choices"][0]["message"]["content"] == expected
            assert body["choices"][0]["finish_reason"] == direct.finish_reason

            streamed = client.post("/v1/chat/completions", json={**payload, "stream": True})
            assert streamed.status_code == 200
            assert streamed.headers["content-type"].startswith("text/event-stream")
            payloads = _parse_sse(streamed.text)
            assert payloads[-1] == "[DONE]"
            chunks = [json.loads(item) for item in payloads[:-1]]
            text = "".join(
                chunk["choices"][0]["delta"]["content"]
                for chunk in chunks
                if chunk["choices"][0]["delta"].get("content")
            )
            finished = [
                chunk for chunk in chunks if chunk["choices"][0]["finish_reason"] is not None
            ]
            assert text == expected
            assert len(finished) == 1
            assert finished[0]["choices"][0]["finish_reason"] == direct.finish_reason

            sampled = client.post(
                "/v1/chat/completions",
                json={**payload, "temperature": 0.8, "top_p": 0.9},
            )
            stopped = client.post(
                "/v1/chat/completions",
                json={**payload, "stop": [" Ready"]},
            )
            regex = client.post(
                "/v1/chat/completions",
                json={**payload, "regex": "CUDA Ready", "max_tokens": 8},
            )
            schema_response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "Return compact JSON only."},
                        {
                            "role": "user",
                            "content": "Use no spaces or newlines in the JSON response.",
                        },
                    ],
                    "max_tokens": 32,
                    "json_schema": schema,
                },
            )
            many = client.post("/v1/chat/completions", json={**payload, "n": 2})
            for response in (sampled, stopped, regex, schema_response, many):
                assert response.status_code == 200
                assert response.json()["choices"][0]["finish_reason"] in {
                    "eos",
                    "stop",
                    "length",
                }
            assert regex.json()["choices"][0]["message"]["content"] == "CUDA Ready"
            json.loads(schema_response.json()["choices"][0]["message"]["content"])
            unsupported = client.post(
                "/v1/chat/completions",
                json={
                    **payload,
                    "json_schema": {
                        "type": "object",
                        "properties": {"age": {"type": "integer", "minimum": 18}},
                    },
                },
            )
            assert unsupported.status_code == 400
            assert "unsupported keyword 'minimum'" in unsupported.json()["detail"]

            # Permanent GPU/API regression: raw UTF-8 and surrogate-pair-sized characters.
            unicode_schema = {"type": "string", "enum": ["é🚀"], "minLength": 2, "maxLength": 2}
            unicode_payload = {**payload, "max_tokens": 32, "json_schema": unicode_schema}
            from jsonschema import Draft202012Validator

            for temperature in (0.0, 0.8):
                for stream in (False, True):
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            **unicode_payload,
                            "temperature": temperature,
                            "stream": stream,
                        },
                    )
                    assert response.status_code == 200
                    if stream:
                        events = _parse_sse(response.text)
                        chunks = [json.loads(event) for event in events[:-1]]
                        assert events[-1] == "[DONE]"
                        assert not any("error" in chunk for chunk in chunks)
                        output = "".join(
                            chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks
                        )
                        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
                    else:
                        output = response.json()["choices"][0]["message"]["content"]
                    value = json.loads(output)
                    assert value == "é🚀"
                    Draft202012Validator(unicode_schema).validate(value)
            exhausted = client.post(
                "/v1/chat/completions", json={**unicode_payload, "max_tokens": 1}
            )
            assert exhausted.status_code == 400
            assert "complete valid document" in exhausted.json()["detail"]
            many_body = many.json()
            assert len(many_body["choices"]) == 2
            assert many_body["usage"]["total_tokens"] == (
                many_body["usage"]["prompt_tokens"] + many_body["usage"]["completion_tokens"]
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                concurrent = list(
                    executor.map(
                        lambda _index: client.post("/v1/chat/completions", json=payload),
                        range(2),
                    )
                )
            assert [item.status_code for item in concurrent] == [200, 200]
            assert {item.json()["choices"][0]["message"]["content"] for item in concurrent} == {
                expected
            }

            print(f"uvicorn_gate_baseline_allocated={baseline}")
            print(f"uvicorn_gate_peak_allocated={peak}")
            print(f"uvicorn_gate_post_run_allocated={allocated}")
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert app.state.engines == {}
