"""CUDA device selection."""

import torch


def require_cuda() -> torch.device:
    """Return cuda:0 or fail instead of silently using the CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("Onyx CUDA requires an NVIDIA GPU available to PyTorch")
    return torch.device("cuda:0")
