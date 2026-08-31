import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from onyx_cuda.server import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


SRC_PATH = Path(__file__).resolve().parents[1] / "src"


def test_server_import_does_not_load_a_model():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_PATH), existing]))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import onyx_cuda.server as server\n"
            "mods = set(sys.modules)\n"
            "assert 'torch' not in mods\n"
            "assert 'transformers' not in mods\n"
            "assert 'onyx_cuda.model' not in mods\n"
            "assert 'onyx_cuda.generation' not in mods\n"
            "assert 'onyx_cuda.speculative' not in mods\n"
            "assert 'onyx' not in mods\n"
            "assert hasattr(server, 'ChatCompletionRequest')\n"
            "assert hasattr(server, 'create_app')\n"
            "assert not hasattr(server, 'app')\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_valid_request_and_response_round_trip():
    request_payload = {
        "model": "onyx-speculative",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
        "max_tokens": 32,
        "temperature": 0.8,
        "stream": True,
        "regex": "CUDA",
        "json_schema": {"type": "object"},
        "compact_json": False,
        "top_p": 0.9,
        "n": 2,
        "stop": ["END"],
    }
    request = ChatCompletionRequest.model_validate(request_payload)
    assert request.model_dump() == request_payload
    assert ChatCompletionRequest.model_validate(request.model_dump()) == request

    defaults = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert defaults.model == "onyx-speculative"
    assert defaults.max_tokens == 256
    assert defaults.temperature == 0.0
    assert defaults.stream is False
    assert defaults.regex is None
    assert defaults.json_schema is None
    assert defaults.compact_json is True
    assert defaults.top_p == 1.0
    assert defaults.n == 1
    assert defaults.stop is None
    assert ChatCompletionRequest.model_validate(defaults.model_dump()) == defaults

    response_payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "onyx-speculative",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "CUDA"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
        "onyx_metrics": {
            "tokens_per_second": 10.0,
            "acceptance_rate": 0.5,
            "ttft_ms": 12.0,
            "grammar_constrained": True,
            "speculative_iterations": 3,
        },
    }
    response = ChatCompletionResponse.model_validate(response_payload)
    assert response.model_dump() == response_payload
    assert ChatCompletionResponse.model_validate(response.model_dump()) == response

    chunk_payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "onyx-speculative",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "CU"},
                "finish_reason": None,
            }
        ],
    }
    chunk = ChatCompletionChunk.model_validate(chunk_payload)
    assert chunk.model_dump() == chunk_payload
    assert ChatCompletionChunk.model_validate(chunk.model_dump()) == chunk


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"temperature": -0.1},
        {"temperature": float("inf")},
        {"temperature": float("nan")},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"top_p": float("nan")},
        {"n": 0},
        {"messages": "user"},
        {"messages": [{"content": "hi"}]},
        {"json_schema": "object"},
        {"stream": "maybe"},
        {"stop": "END"},
    ],
)
def test_invalid_request_bounds_and_shapes_fail_before_cuda(overrides):
    payload = {"messages": [{"role": "user", "content": "hi"}], **overrides}
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


def test_missing_messages_fail_validation():
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({})
