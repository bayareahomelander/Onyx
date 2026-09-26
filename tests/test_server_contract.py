import asyncio
import importlib
import json
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import anyio
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("ONYX_API_DOCS", raising=False)
    sys.modules.pop("onyx.server", None)
    server = importlib.import_module("onyx.server")
    calls = []
    closed = []
    metrics = dict(prompt_tokens=2, generated_tokens=3, finish_reason="length")

    def stream_generate(**kwargs):
        calls.append(kwargs)
        try:
            yield "abc", None
            yield "", dict(metrics)
        finally:
            closed.append(True)

    def generate(**kwargs):
        calls.append(kwargs)
        return "abc", dict(metrics)

    engine = SimpleNamespace(
        draft_model=object(), tokenizer=SimpleNamespace(
            encode=lambda text: [7, 8], eos_token_id=9),
        generate=generate, stream_generate=stream_generate,
    )
    server._engines = {"onyx-speculative": engine}
    # No lifespan/model loading: exercise the real HTTP and Pydantic layers.
    client = TestClient(server.app)
    yield server, client, engine, calls, closed, metrics
    client.close()
    sys.modules.pop("onyx.server", None)


def payload(**kwargs):
    return {"messages": [{"role": "user", "content": "Hi"}], **kwargs}


@pytest.mark.parametrize("options", [
    {"messages": []}, {"messages": [{"role": "tool", "content": "x"}]},
    {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": None}, {"max_tokens": True},
    {"temperature": -1}, {"temperature": 3}, {"temperature": None},
    {"top_p": 0}, {"top_p": 1.1}, {"top_p": None},
    {"n": 0}, {"n": 2}, {"n": None}, {"n": True},
    {"compact_json": True}, {"compact_json": None},
    {"stop": [""]}, {"stop": "END"}, {"stream": None},
    {"regex": "x", "json_schema": {}}, {"unsupported": True},
])
def test_invalid_options_fail_before_generation(api, options):
    _, client, _, calls, _, _ = api
    response = client.post("/v1/chat/completions", json=payload(**options))
    assert response.status_code == 422
    assert not calls


def chunks(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


@pytest.mark.parametrize("reason", ["length", "stop", "grammar_complete"])
def test_stream_and_collected_contract_match(api, reason):
    _, client, _, calls, closed, metrics = api
    metrics["finish_reason"] = reason
    options = payload(json_schema={}, stop=["END"], temperature=0, top_p=0.5,
                      compact_json=False, n=1, max_tokens=3)
    collected = client.post("/v1/chat/completions", json=options).json()
    streamed = client.post("/v1/chat/completions", json={**options, "stream": True})
    events = chunks(streamed)
    assert collected["choices"][0]["finish_reason"] == reason
    assert events[-1]["choices"][0]["finish_reason"] == reason
    assert "".join(e["choices"][0]["delta"].get("content") or "" for e in events) == "abc"
    assert calls[0] == calls[1]
    assert calls[0]["json_schema"] == "{}"
    assert calls[0]["stop_tokens"] == [[9], [7, 8]]
    assert calls[0]["temperature"] == 0
    assert closed == [True]


def test_split_stop_closes_generation_and_hides_delimiter(api):
    _, client, engine, _, closed, _ = api
    def stream(**kwargs):
        try:
            yield "answer<EN", None
            yield "D>hidden", None
            pytest.fail("Generation should close at the stop delimiter")
        finally:
            closed.append(True)
    engine.stream_generate = stream
    response = client.post("/v1/chat/completions", json=payload(stream=True, stop=["<END>"]))
    events = chunks(response)
    assert "".join(e["choices"][0]["delta"].get("content") or "" for e in events) == "answer"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert closed == [True]


def test_missing_terminal_reason_is_an_error(api):
    _, client, _, _, _, metrics = api
    metrics.pop("finish_reason")
    assert client.post("/v1/chat/completions", json=payload()).status_code == 500
    events = chunks(client.post("/v1/chat/completions", json=payload(stream=True)))
    assert events[-1]["error"]["type"] == "server_error"


def test_unsatisfiable_grammar_is_a_client_error(api):
    _, client, engine, _, closed, _ = api
    message = "Grammar constraint has no valid token continuation"

    def generate(**kwargs):
        raise ValueError(message)

    def stream(**kwargs):
        try:
            yield "a", None
            raise ValueError(message)
        finally:
            closed.append(True)

    engine.generate = generate
    engine.stream_generate = stream
    response = client.post("/v1/chat/completions", json=payload(regex="ab"))
    assert response.status_code == 400
    assert response.json()["detail"] == message
    events = chunks(client.post("/v1/chat/completions", json=payload(regex="ab", stream=True)))
    assert events[-1]["error"] == {"message": message, "type": "invalid_request"}
    assert closed == [True]


def test_documentation_routes_and_compatibility_fields(api, monkeypatch):
    server, client, _, _, _, _ = api
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 404
    schema = client.get("/openapi.json").json()
    properties = schema["components"]["schemas"]["ChatCompletionRequest"]["properties"]
    assert properties["compact_json"]["deprecated"]
    assert properties["n"]["deprecated"]
    monkeypatch.setenv("ONYX_API_DOCS", "0")
    server = importlib.reload(server)
    with_docs_off = TestClient(server.app)
    assert with_docs_off.get("/docs").status_code == 404
    assert with_docs_off.get("/openapi.json").status_code == 200
    with_docs_off.close()



@contextmanager
def serving(server):
    """One event loop for every request, as under uvicorn, without loading models."""
    @asynccontextmanager
    async def no_model_loading(app):
        yield

    server.app.router.lifespan_context = no_model_loading
    with TestClient(server.app) as client:
        yield client


def _post_concurrently(client, bodies):
    threads = [threading.Thread(target=client.post, args=("/v1/chat/completions",),
                                kwargs={"json": body}, daemon=True) for body in bodies]
    for thread in threads:
        thread.start()
        time.sleep(0.02)
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads), "requests did not finish"


def _serialized_engine(engine, metrics, steps=4):
    """Record whether two generations ever run inside the engine at once."""
    active, overlaps, guard = [0], [], threading.Lock()

    def step():
        with guard:
            active[0] += 1
            overlaps.append(active[0] > 1)
        time.sleep(0.05)  # model work on the engine's shared KV caches
        with guard:
            active[0] -= 1

    def stream_generate(**kwargs):
        for _ in range(steps):
            step()
            yield "x", None
        yield "", dict(metrics)

    def generate(**kwargs):
        step()
        return "x", dict(metrics)

    engine.stream_generate, engine.generate = stream_generate, generate
    return overlaps


@pytest.mark.parametrize("streams", [(True, True), (True, False), (False, False)])
def test_requests_never_share_an_engine(api, streams):
    server, _, engine, _, _, metrics = api
    overlaps = _serialized_engine(engine, metrics)
    with serving(server) as client:
        _post_concurrently(client, [payload(stream=stream) for stream in streams])
    assert overlaps and not any(overlaps)


def test_waiting_requests_do_not_starve_the_owning_stream(api):
    server, _, engine, _, _, metrics = api
    overlaps = _serialized_engine(engine, metrics)
    with serving(server) as client:
        # With one worker thread, a request blocking a thread while it waits for
        # the engine would leave the owning stream no thread for its next chunk.
        client.portal.call(lambda: setattr(
            anyio.to_thread.current_default_thread_limiter(), "total_tokens", 1))
        _post_concurrently(client, [payload(stream=True), payload(), payload()])
    assert overlaps and not any(overlaps)


def test_lazy_engine_loads_once_under_concurrent_requests(api):
    server, _, engine, _, _, _ = api
    loads = []

    def load_models():
        loads.append(True)
        time.sleep(0.1)
        engine.draft_model = object()

    engine.draft_model, engine.load_models = None, load_models
    with serving(server) as client:
        _post_concurrently(client, [payload(), payload(stream=True)])
    assert loads == [True]


def test_generation_does_not_block_the_event_loop(api):
    server, _, engine, _, _, metrics = api
    started, release, order = threading.Event(), threading.Event(), []

    def generate(**kwargs):
        started.set()
        release.wait(5)
        order.append("generation")
        return "x", dict(metrics)

    engine.generate = generate
    with serving(server) as client:
        request = threading.Thread(target=client.post, args=("/v1/chat/completions",),
                                   kwargs={"json": payload()})
        request.start()
        assert started.wait(5)
        assert client.get("/").status_code == 200
        order.append("root")
        release.set()
        request.join()
    assert order == ["root", "generation"]


@pytest.mark.parametrize("abandon", ["close", "disconnect"])
def test_abandoned_stream_closes_generation_before_releasing_engine(api, abandon):
    server, _, engine, _, closed, _ = api

    def stream_generate(**kwargs):
        try:
            while True:
                yield "x", None
        finally:
            closed.append(True)

    engine.stream_generate = stream_generate
    request = server.ChatCompletionRequest(**payload(stream=True))

    async def scenario():
        lock = server.engine_lock(request.model)
        stream = server.exclusive_stream(request, engine)
        if abandon == "close":
            await stream.__anext__()  # role
            await stream.__anext__()  # content: generation owns the engine
            assert lock.locked()
            await stream.aclose()
        else:
            # Starlette cancels the response through a task group on disconnect;
            # that cancellation is redelivered at every await until the scope ends.
            produced = anyio.Event()

            async def respond():
                sent = 0
                async for _ in stream:
                    sent += 1
                    if sent == 2:  # role, then content: generation owns the engine
                        produced.set()

            async with anyio.create_task_group() as responses:
                responses.start_soon(respond)
                await produced.wait()
                assert lock.locked()
                responses.cancel_scope.cancel()
        assert closed == [True]
        assert not lock.locked()

    asyncio.run(scenario())
