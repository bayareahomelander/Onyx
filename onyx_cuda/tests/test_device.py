import pytest
import torch

from onyx_cuda.device import require_cuda
from onyx_cuda.generation import generate_tokens
from onyx_cuda.prefill import prefill


def test_require_cuda_returns_first_cuda_device():
    assert require_cuda() == torch.device("cuda:0")


def test_require_cuda_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires an NVIDIA GPU"):
        require_cuda()


def test_prefill_fails_with_cpu_model():
    with pytest.raises(RuntimeError, match="requires a model on CUDA"):
        prefill(torch.nn.Linear(1, 1), [0])


def test_generate_tokens_rejects_invalid_options_before_generation():
    cases = [
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_p": float("nan")}, "top_p"),
        ({"seed": 1.5}, "seed"),
    ]
    for overrides, message in cases:
        options = {"max_tokens": 1, "eos_token_ids": [], **overrides}
        with pytest.raises(ValueError, match=message):
            generate_tokens(torch.nn.Linear(1, 1), [0], **options)

    with pytest.raises(ValueError, match="token_byte_vocabulary"):
        generate_tokens(
            torch.nn.Linear(1, 1),
            [0],
            max_tokens=1,
            eos_token_ids=[],
            regex="CUDA",
        )

    with pytest.raises(ValueError, match="token_byte_vocabulary"):
        generate_tokens(
            torch.nn.Linear(1, 1),
            [0],
            max_tokens=1,
            eos_token_ids=[],
            json_schema='{"type":"object"}',
        )
