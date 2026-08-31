import json
from types import SimpleNamespace

import onyx_cuda.server as server
from onyx_cuda.server import (
    ChatCompletionRequest,
    ChatMessage,
    GAMMA,
    format_request_messages,
    prepare_generation,
    resolve_stop_sequences,
)


def _messages():
    return [
        ChatMessage(role="system", content="S"),
        ChatMessage(role="user", content="U"),
        ChatMessage(role="assistant", content="A"),
        ChatMessage(role="tool", content="T"),
    ]


def _engine(tokenizer, model=None):
    loaded = SimpleNamespace(model=model or object(), tokenizer=tokenizer)
    return SimpleNamespace(draft=loaded, target=loaded)


class TemplateTokenizer:
    eos_token_id = 99

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
        if tokenize:
            return [1, 2, 3]
        return "templated"

    def encode(self, text, add_special_tokens=False):
        return {
            "END": [10, 11],
            "STOP": [12],
            "": [],
            " ": [],
        }.get(text, [7])


def test_format_uses_chat_template_with_role_order_and_generation_prompt():
    tokenizer = TemplateTokenizer()
    text, token_ids = format_request_messages(_messages(), tokenizer)
    assert text == "templated"
    assert token_ids == [1, 2, 3]
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
        },
        {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A"},
                {"role": "tool", "content": "T"},
            ],
            "tokenize": True,
            "add_generation_prompt": True,
        },
    ]


def test_format_falls_back_without_template_and_on_type_error():
    class PlainTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "System: S\nUser: U\nAssistant: A\nAssistant:"
            return [4, 5]

    text, token_ids = format_request_messages(_messages(), PlainTokenizer())
    assert text == "System: S\nUser: U\nAssistant: A\nAssistant:"
    assert token_ids == [4, 5]

    class TypeErrorTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            raise TypeError("unsupported signature")

        def encode(self, text, add_special_tokens=False):
            return [6]

    text, token_ids = format_request_messages(_messages(), TypeErrorTokenizer())
    assert text == "System: S\nUser: U\nAssistant: A\nAssistant:"
    assert token_ids == [6]


def test_stop_sequences_keep_multi_token_and_drop_empty():
    tokenizer = TemplateTokenizer()
    assert resolve_stop_sequences(["END", "STOP", "", " "], tokenizer) == [
        [10, 11],
        [12],
    ]
    assert resolve_stop_sequences(None, tokenizer) is None
    assert resolve_stop_sequences(["", " "], tokenizer) is None


def test_prepare_generation_forwards_exact_arguments_and_json_precedence(
    monkeypatch,
):
    tokenizer = TemplateTokenizer()
    draft_model = object()
    target_model = object()
    engine = SimpleNamespace(
        draft=SimpleNamespace(model=draft_model, tokenizer=tokenizer),
        target=SimpleNamespace(model=target_model, tokenizer=tokenizer),
    )
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        max_tokens=32,
        temperature=0.8,
        top_p=0.9,
        stop=["END"],
    )
    arguments = prepare_generation(request, engine)
    assert arguments == {
        "draft_model": draft_model,
        "target_model": target_model,
        "prompt_token_ids": [1, 2, 3],
        "max_tokens": 32,
        "gamma": GAMMA,
        "eos_token_ids": 99,
        "stop_sequences": [[10, 11]],
        "temperature": 0.8,
        "top_p": 0.9,
        "regex": None,
        "json_schema": None,
    }

    schema = {"type": "object", "properties": {"content": {"type": "string"}}}
    vocab_tokenizer = TemplateTokenizer()
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=4))
    constrained_engine = _engine(vocab_tokenizer, model)
    vocabulary = object()
    captured = []

    def fake_build_vocabulary(tokenizer, logits_vocab_size):
        captured.append((tokenizer, logits_vocab_size))
        return vocabulary

    monkeypatch.setattr(server, "_build_vocabulary", fake_build_vocabulary)
    constrained = prepare_generation(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            regex="CUDA",
            json_schema=schema,
        ),
        constrained_engine,
    )
    assert constrained["regex"] == "CUDA"
    assert constrained["json_schema"] == json.dumps(schema)
    assert json.loads(constrained["json_schema"]) == schema
    assert constrained["token_byte_vocabulary"] is vocabulary
    assert captured == [(vocab_tokenizer, 4)]
