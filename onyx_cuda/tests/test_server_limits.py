"""API work limits, overload recovery, and cancellation under backpressure."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import onyx_cuda.server as server


class Tokenizer:
    eos_token_id = 99

    def __init__(self, prompt_length=3):
        self.prompt_length = prompt_length

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1] * self.prompt_length if tokenize else "prompt"

    def decode(self, tokens, skip_special_tokens=False):
        return "OK"


def engine(prompt_length=3, context=32768):
    return SimpleNamespace(
        draft=None,
        target=SimpleNamespace(
            tokenizer=Tokenizer(prompt_length),
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=context)),
        ),
    )


PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1}


def result():
    return SimpleNamespace(token_ids=[10], finish_reason="stop", timings=None)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("extra", [
    {"max_tokens": 1025}, {"max_completion_tokens": 1025}, {"n": 5},
    {"messages": [{"role": "user", "content": "x" * 32769}]},
    {"messages": [{"role": "user", "content": "x"}] * 129},
])
def test_excess_work_rejected_before_generation(monkeypatch, stream, extra):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid request reached model preparation")

    monkeypatch.setattr(server, "_request_prompt_token_ids", unexpected)
    payload = {"messages": PAYLOAD["messages"], "stream": stream, **extra}
    with TestClient(server.create_app(engine=engine())) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422
    assert "text/event-stream" not in response.headers.get("content-type", "")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prompt_length,context", [(2048, 32768), (100, 100), (0, 32768)])
def test_context_rejected_before_stream_or_generation(monkeypatch, stream, prompt_length, context):
    monkeypatch.setattr(server, "_generate", lambda _: pytest.fail("Generation started"))
    with TestClient(server.create_app(engine=engine(prompt_length, context))) as client:
        response = client.post("/v1/chat/completions", json={**PAYLOAD, "stream": stream})
    assert response.status_code == 422
    assert "text/event-stream" not in response.headers.get("content-type", "")


def test_context_boundary_and_max_choices_accepted(monkeypatch):
    monkeypatch.setattr(server, "_generate", lambda _: result())
    with TestClient(server.create_app(engine=engine(1024))) as client:
        response = client.post("/v1/chat/completions", json={
            **PAYLOAD, "max_tokens": 1024, "n": 4,
        })
    assert response.status_code == 200
    assert len(response.json()["choices"]) == 4


def test_overload_rejects_before_work_and_capacity_recovers(monkeypatch):
    monkeypatch.setattr(server, "MAX_ACTIVE_REQUESTS", 2)
    started, release = threading.Event(), threading.Event()

    def generate(_arguments):
        started.set()
        assert release.wait(5)
        return result()

    monkeypatch.setattr(server, "_generate", generate)

    async def exercise():
        app = server.create_app(engine=engine())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/v1/chat/completions", json=PAYLOAD))
                second = None
                try:
                    assert await asyncio.to_thread(started.wait, 3)
                    second = asyncio.create_task(client.post("/v1/chat/completions", json=PAYLOAD))
                    # The second request waits for the model while occupying an
                    # admission slot. Yield until it has entered the middleware.
                    middleware = app.middleware_stack
                    while not isinstance(middleware, server.RequestCapacityMiddleware):
                        middleware = middleware.app
                    async def admitted():
                        while middleware.active != 2:
                            await asyncio.sleep(0.001)
                    await asyncio.wait_for(admitted(), 2)
                    rejected = await client.post("/v1/chat/completions", json={**PAYLOAD, "stream": True})
                    assert rejected.status_code == 429
                    assert rejected.headers["retry-after"] == "1"
                    assert (await client.get("/")).status_code == 200
                    second.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await second
                    assert middleware.active == 1
                finally:
                    release.set()
                    assert (await first).status_code == 200
                    if second is not None and not second.done():
                        await second
                assert middleware.active == 0
                assert (await client.post("/v1/chat/completions", json=PAYLOAD)).status_code == 200
                assert middleware.active == 0

    asyncio.run(exercise())


def test_repeated_cancellation_keeps_model_owned_until_worker_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def generate(_arguments):
        started.set()
        assert release.wait(5)
        return result()

    monkeypatch.setattr(server, "_generate", generate)

    async def exercise():
        app = server.create_app(engine=engine())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                task = asyncio.create_task(client.post("/v1/chat/completions", json=PAYLOAD))
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    for _ in range(3):
                        task.cancel()
                        await asyncio.sleep(0.01)
                        assert not task.done()
                        assert app.state.engine_locks[server.MODEL_ID].locked()
                finally:
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert not app.state.engine_locks[server.MODEL_ID].locked()
                assert (await client.post("/v1/chat/completions", json=PAYLOAD)).status_code == 200

    asyncio.run(exercise())


def test_disconnecting_full_stream_buffer_closes_producer_and_releases_lock(monkeypatch):
    monkeypatch.setattr(server, "STREAM_BUFFER_CHUNKS", 2)
    saturated, closed = threading.Event(), threading.Event()
    produced = []

    def chunks(*args, **kwargs):
        try:
            for index in range(1000):
                produced.append(index)
                if len(produced) == 4:
                    saturated.set()
                yield str(index)
        finally:
            closed.set()

    monkeypatch.setattr(server, "_sse_events", chunks)

    async def exercise():
        app = server.create_app(engine=engine())
        async with app.router.lifespan_context(app):
            stream = server._stream_chat_completion(
                app, server.ChatCompletionRequest(**PAYLOAD, stream=True), app.state.engines[server.MODEL_ID],
            )
            try:
                assert await anext(stream) == "0"
                assert await asyncio.to_thread(saturated.wait, 2)
                assert len(produced) == 4  # consumed + two buffered + one producer-held
                assert app.state.engine_locks[server.MODEL_ID].locked()
            finally:
                await asyncio.wait_for(stream.aclose(), 3)
            assert closed.is_set()
            assert not app.state.engine_locks[server.MODEL_ID].locked()

    asyncio.run(exercise())


@pytest.mark.gpu
@pytest.mark.parametrize("gamma", [0, 2])
def test_repeated_cuda_requests_keep_live_allocations_stable(gamma):
    import gc
    import json
    import re

    import torch
    from onyx_cuda.model import load_model_pair

    pair = load_model_pair(include_draft=gamma > 0)
    schema = {"enum": ["OK"]}
    cases = [
        {"regex": "[0-9]{4}"},
        {"json_schema": schema},
        {"json_schema": schema, "stream": True, "temperature": 0.7, "seed": 42},
        {"regex": "[0-9]{4}", "stream": True},
    ]
    with TestClient(server.create_app(engine=pair, gamma=gamma)) as client:
        # Exercise almost the whole advertised context on real hardware, using
        # a short forced answer so the gate measures prefill rather than luck.
        low, high = 1, server.MAX_CONTEXT_TOKENS
        while low < high:
            middle = (low + high + 1) // 2
            messages = [server.ChatMessage(role="user", content="x " * middle)]
            _, ids = server.format_request_messages(messages, pair.target.tokenizer)
            if len(ids) + 16 <= server.MAX_CONTEXT_TOKENS:
                low = middle
            else:
                high = middle - 1
        boundary = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "x " * low}],
            "max_tokens": 16, "json_schema": schema,
        })
        assert boundary.status_code == 200, boundary.text
        assert boundary.json()["choices"][0]["finish_reason"] == "stop"
        assert json.loads(boundary.json()["choices"][0]["message"]["content"]) == "OK"
        assert server.MAX_CONTEXT_TOKENS - 1 <= boundary.json()["usage"]["prompt_tokens"] + 16 <= server.MAX_CONTEXT_TOKENS
        baseline = None
        for cycle in range(5):  # warm every path, then 16 measured requests
            for options in cases:
                response = client.post("/v1/chat/completions", json={**PAYLOAD, "max_tokens": 16, **options})
                assert response.status_code == 200
                if options.get("stream"):
                    payloads = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
                    assert payloads[-1] == "[DONE]"
                    events = [json.loads(part) for part in payloads[:-1]]
                    assert all("error" not in event for event in events)
                    assert events[-1]["choices"][0]["finish_reason"] == "stop"
                    text = "".join(event["choices"][0]["delta"].get("content") or "" for event in events)
                else:
                    choice = response.json()["choices"][0]
                    assert choice["finish_reason"] == "stop"
                    text = choice["message"]["content"]
                if "json_schema" in options:
                    assert json.loads(text) == "OK"
                else:
                    assert re.fullmatch(r"[0-9]{4}", text)
            assert client.post("/v1/chat/completions", json={**PAYLOAD, "max_tokens": 1025}).status_code == 422
            gc.collect()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            if baseline is None:
                baseline = allocated
            else:
                assert allocated == baseline, "Live CUDA allocations grew across request cycles"
