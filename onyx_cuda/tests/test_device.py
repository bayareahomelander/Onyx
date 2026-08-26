import pytest
import torch

from onyx_cuda.device import require_cuda
from onyx_cuda.generation import generate_greedy
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


def test_generate_greedy_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="max_tokens must be at least 1"):
        generate_greedy(torch.nn.Linear(1, 1), [0], max_tokens=0)
