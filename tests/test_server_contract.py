import importlib
import json
import sys
from types import SimpleNamespace

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
    assert "error" in events[-1]


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
