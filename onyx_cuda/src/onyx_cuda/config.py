"""Runtime settings shared by the API, generators, and validation tools."""

import os


def resolve_greedy_backend(backend: str | None = None) -> str:
    value = backend if backend is not None else os.environ.get("ONYX_GREEDY_BACKEND", "torch")
    if value not in ("torch", "cuda"):
        raise ValueError("ONYX_GREEDY_BACKEND must be 'torch' or 'cuda'")
    return value


def initialize_greedy_backend(backend: str) -> None:
    """Check optional dependencies and compile kernels before accepting requests."""
    if backend == "cuda":
        import torch
        from onyx_cuda.masking import grammar_argmax

        for dtype in (torch.float16, torch.float32):
            logits = torch.tensor([[0.0, 1.0]], device="cuda", dtype=dtype)
            if grammar_argmax(logits, [1], backend=backend).item() != 1:
                raise RuntimeError("CUDA greedy selector startup check failed")
