import asyncio
import importlib
import json
import sys
import types

import pytest


@pytest.fixture
def server_with_fake_deps(monkeypatch):
    fastapi = types.ModuleType("fastapi")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

    class FakeHTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi.FastAPI = FakeFastAPI
    fastapi.HTTPException = FakeHTTPException

    responses = types.ModuleType("fastapi.responses")
    responses.StreamingResponse = type("StreamingResponse", (), {})

    pydantic = types.ModuleType("pydantic")

    class FakeBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump_json(self):
            return "{}"

    pydantic.BaseModel = FakeBaseModel
    pydantic.Field = lambda default=None, **_kwargs: default

    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses)
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)
    monkeypatch.delitem(sys.modules, "onyx.server", raising=False)

    module = importlib.import_module("onyx.server")
    yield module

    sys.modules.pop("onyx.server", None)


def make_messages(server):
    return [
        server.ChatMessage(role="system", content="S"),
        server.ChatMessage(role="user", content="U"),
        server.ChatMessage(role="assistant", content="A"),
        server.ChatMessage(role="tool", content="T"),
    ]


def test_format_messages_for_engine_uses_tokenizer_chat_template(server_with_fake_deps):
    server = server_with_fake_deps

    class TemplateTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            self.calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                }
            )
            return "templated prompt"

    tokenizer = TemplateTokenizer()

    prompt = server.format_messages_for_engine(make_messages(server), tokenizer)

    assert prompt == "templated prompt"
    assert tokenizer.calls == [
        {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A"},
                {"role": "tool", "content": "T"},
            ],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]


def test_format_messages_for_engine_falls_back_without_chat_template(server_with_fake_deps):
    server = server_with_fake_deps

    prompt = server.format_messages_for_engine(make_messages(server), tokenizer=object())

    assert prompt == "System: S\nUser: U\nAssistant: A\nAssistant:"


def test_format_messages_for_engine_falls_back_on_template_type_error(server_with_fake_deps):
    server = server_with_fake_deps

    class TypeErrorTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            raise TypeError("unsupported signature")

    prompt = server.format_messages_for_engine(make_messages(server), TypeErrorTokenizer())

    assert prompt == "System: S\nUser: U\nAssistant: A\nAssistant:"


def test_chat_completion_passes_templated_prompt_to_engine(server_with_fake_deps):
    server = server_with_fake_deps

    class TemplateTokenizer:
        def encode(self, text):
            return []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "api templated prompt"

    class FakeEngine:
        def __init__(self):
            self.draft_model = object()
            self.tokenizer = TemplateTokenizer()
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return (
                "ok",
                {
                    "prompt_tokens": 3,
                    "generated_tokens": 1,
                    "tokens_per_second": 1.0,
                    "acceptance_rate": 100.0,
                    "speculative_iterations": 1,
                    "finish_reason": "length",
                },
            )

    engine = FakeEngine()
    server._engines = {"onyx-speculative": engine}

    request = server.ChatCompletionRequest(
        model="onyx-speculative",
        messages=[server.ChatMessage(role="user", content="Hello")],
        max_tokens=5,
        temperature=0.0,
        stream=False,
        regex="[a-z]+",
        json_schema=None,
        compact_json=True,
        top_p=1.0,
        n=1,
        stop=None,
    )

    response = asyncio.run(server.create_chat_completion(request))

    assert engine.generate_kwargs["prompt"] == "api templated prompt"
    assert engine.generate_kwargs["temperature"] == 0.0
    assert engine.generate_kwargs["top_p"] == 1.0
    assert response.choices[0].finish_reason == "length"


def test_resolve_stop_tokens_preserves_sequences(server_with_fake_deps):
    server = server_with_fake_deps

    class Tokenizer:
        def encode(self, text):
            return {
                "END": [1, 2],
                "STOP": [3],
                "": [],
            }[text]

    assert server.resolve_stop_tokens(["END", "STOP", ""], Tokenizer()) == [[1, 2], [3]]


def test_truncate_and_stream_stop_helpers(server_with_fake_deps):
    server = server_with_fake_deps

    assert server.truncate_at_stop("alpha END beta", ["END"]) == "alpha "
    assert server.truncate_at_stop("alpha beta", ["END"]) == "alpha beta"

    flushed, pending, stopped = server.flush_stream_text("alpha EN", ["END"])
    assert (flushed, pending, stopped) == ("alpha ", "EN", False)

    flushed, pending, stopped = server.flush_stream_text(pending + "D beta", ["END"])
    assert (flushed, pending, stopped) == ("", "", True)

    flushed, pending, stopped = server.flush_stream_text("abc", ["X"])
    assert (flushed, pending, stopped) == ("abc", "", False)


def test_chat_completion_passes_stop_sequences_and_truncates_output(server_with_fake_deps):
    server = server_with_fake_deps

    class Tokenizer:
        def encode(self, text):
            if text == "<END>":
                return [10, 11]
            return []

    class FakeEngine:
        def __init__(self):
            self.draft_model = object()
            self.tokenizer = Tokenizer()
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return (
                "answer<END>hidden",
                {
                    "prompt_tokens": 2,
                    "generated_tokens": 3,
                    "tokens_per_second": 1.0,
                    "acceptance_rate": 100.0,
                    "speculative_iterations": 1,
                    "finish_reason": "stop",
                },
            )

    engine = FakeEngine()
    server._engines = {"onyx-speculative": engine}

    request = server.ChatCompletionRequest(
        model="onyx-speculative",
        messages=[server.ChatMessage(role="user", content="Hello")],
        max_tokens=5,
        temperature=0.0,
        stream=False,
        regex=None,
        json_schema=None,
        compact_json=True,
        top_p=1.0,
        n=1,
        stop=["<END>"],
    )

    response = asyncio.run(server.create_chat_completion(request))

    assert engine.generate_kwargs["stop_tokens"] == [[10, 11]]
    assert response.choices[0].message.content == "answer"


def test_streaming_response_uses_engine_finish_reason(server_with_fake_deps, monkeypatch):
    server = server_with_fake_deps

    class Tokenizer:
        def encode(self, text):
            return []

    class FakeEngine:
        tokenizer = Tokenizer()

        def stream_generate(self, **_kwargs):
            yield "answer", None
            yield "", {"finish_reason": "length"}

    monkeypatch.setattr(
        server.ChatCompletionChunk,
        "model_dump_json",
        lambda self: json.dumps({"finish_reason": self.choices[0].finish_reason}),
    )

    request = server.ChatCompletionRequest(
        model="onyx-speculative",
        messages=[server.ChatMessage(role="user", content="Hello")],
        max_tokens=1,
        temperature=0.0,
        stream=True,
        regex=None,
        json_schema=None,
        compact_json=True,
        top_p=1.0,
        n=1,
        stop=None,
    )

    chunks = list(
        server.create_streaming_response(
            request,
            FakeEngine(),
            temperature=0.0,
            top_p=1.0,
        )
    )

    assert any('"finish_reason": "length"' in chunk for chunk in chunks)


def test_resolve_speculative_sampling_uses_greedy_defaults(server_with_fake_deps):
    server = server_with_fake_deps

    assert server.resolve_speculative_sampling(None, None) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("temperature", "top_p", "stream"),
    [
        (0.1, 1.0, False),
        (0.0, 0.9, True),
    ],
)
def test_chat_completion_rejects_non_greedy_speculative_sampling(
    server_with_fake_deps,
    temperature,
    top_p,
    stream,
):
    server = server_with_fake_deps
    request = server.ChatCompletionRequest(
        model="onyx-speculative",
        messages=[server.ChatMessage(role="user", content="Hello")],
        max_tokens=5,
        temperature=temperature,
        stream=stream,
        regex=None,
        json_schema=None,
        compact_json=True,
        top_p=top_p,
        n=1,
        stop=None,
    )

    with pytest.raises(server.HTTPException) as exc_info:
        asyncio.run(server.create_chat_completion(request))

    assert exc_info.value.status_code == 400
    assert "supports greedy sampling only" in exc_info.value.detail
