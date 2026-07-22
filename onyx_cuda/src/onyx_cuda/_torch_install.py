"""Shared guidance for the validated PyTorch CUDA installation."""

from __future__ import annotations

from typing import Any


PYTORCH_CUDA_REQUIREMENT = "torch==2.6.0+cu124"
PYTORCH_CUDA_INSTALL_COMMAND = (
    "python -m pip install torch==2.6.0 "
    "--index-url https://download.pytorch.org/whl/cu124"
)
PYTORCH_CUDA_INSTALL_GUIDANCE = (
    "Install the validated PyTorch CUDA build with: " f"{PYTORCH_CUDA_INSTALL_COMMAND}"
)

_MISSING = object()


def torch_import_error_message(error: BaseException, *, context: str) -> str:
    """Return an import failure with the supported CUDA-wheel command."""

    return f"{context}: {error}. {PYTORCH_CUDA_INSTALL_GUIDANCE}"


def torch_cuda_unavailable_message(torch_module: Any) -> str:
    """Distinguish a CPU-only wheel from a CUDA runtime visibility failure."""

    try:
        torch_version = str(torch_module.__version__)
    except Exception:
        torch_version = "unknown"

    try:
        compiled_cuda = getattr(torch_module.version, "cuda", _MISSING)
    except Exception:
        compiled_cuda = _MISSING

    if compiled_cuda is None:
        return (
            "the installed PyTorch build reports CUDA unavailable: "
            f"PyTorch {torch_version} is CPU-only. {PYTORCH_CUDA_INSTALL_GUIDANCE}"
        )
    if compiled_cuda is _MISSING:
        return (
            "the installed PyTorch build reports CUDA unavailable. Verify that an NVIDIA GPU and "
            "compatible driver are installed and visible to this process."
        )
    return (
        "the installed PyTorch build reports CUDA unavailable "
        f"(PyTorch {torch_version}, compiled CUDA {compiled_cuda}). Verify that an NVIDIA GPU and "
        "compatible driver are installed and visible to this process."
    )
