"""lazy-loaded CUDA masked argmax for grammar-constrained decoding."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

CUDA_EXTENSION_AVAILABLE = False
CUDA_EXTENSION_ERROR: Optional[str] = None

_EXTENSION = None
_EXTENSION_ATTEMPTED = False


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "onyx_cuda requires PyTorch. Install the optional CUDA extra with "
            "`python -m pip install -e .[cuda]` or install a CUDA-enabled torch build."
        ) from exc

    return torch


def _load_extension():
    """build or load the local CUDA extension on first use."""
    global CUDA_EXTENSION_AVAILABLE, CUDA_EXTENSION_ERROR, _EXTENSION, _EXTENSION_ATTEMPTED

    if _EXTENSION is not None:
        return _EXTENSION

    if _EXTENSION_ATTEMPTED and CUDA_EXTENSION_ERROR:
        raise RuntimeError(CUDA_EXTENSION_ERROR)

    _EXTENSION_ATTEMPTED = True
    try:
        torch = _require_torch()
    except RuntimeError as exc:
        CUDA_EXTENSION_AVAILABLE = False
        CUDA_EXTENSION_ERROR = str(exc)
        raise

    if not torch.cuda.is_available():
        CUDA_EXTENSION_AVAILABLE = False
        CUDA_EXTENSION_ERROR = "CUDA is not available to PyTorch in this environment."
        raise RuntimeError(CUDA_EXTENSION_ERROR)

    try:
        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).resolve().parent / "csrc"
        sources = [
            source_dir / "masked_argmax.cpp",
            source_dir / "masked_argmax_kernel.cu",
        ]
        missing_sources = [str(path) for path in sources if not path.exists()]
        if missing_sources:
            raise FileNotFoundError(
                "CUDA extension source files are missing. "
                "This experimental backend currently expects an editable source tree. "
                f"Missing: {', '.join(missing_sources)}"
            )

        cflags = ["/O2"] if os.name == "nt" else ["-O3"]
        _EXTENSION = load(
            name="onyx_cuda_masked_argmax",
            sources=[str(path) for path in sources],
            extra_cflags=cflags,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
        CUDA_EXTENSION_AVAILABLE = False
        CUDA_EXTENSION_ERROR = (
            "Failed to build/load onyx_cuda masked-argmax extension: "
            f"{type(exc).__name__}: {exc}"
        )
        raise RuntimeError(CUDA_EXTENSION_ERROR) from exc

    CUDA_EXTENSION_AVAILABLE = True
    CUDA_EXTENSION_ERROR = None
    return _EXTENSION


def extension_status() -> Tuple[bool, Optional[str]]:
    """return whether the CUDA extension is loaded and the last load error."""
    return CUDA_EXTENSION_AVAILABLE, CUDA_EXTENSION_ERROR


def _as_valid_id_tensor(valid_token_ids: Any, torch, device):
    if torch.is_tensor(valid_token_ids):
        valid_ids = valid_token_ids
        if valid_ids.device != device:
            valid_ids = valid_ids.to(device=device)
        if valid_ids.dtype != torch.long:
            valid_ids = valid_ids.to(dtype=torch.long)
    else:
        if not isinstance(valid_token_ids, Iterable):
            raise TypeError("valid_token_ids must be a 1D tensor or iterable of integers")
        valid_ids = torch.as_tensor(list(valid_token_ids), dtype=torch.long, device=device)

    if valid_ids.ndim != 1:
        raise ValueError("valid_token_ids must be 1D")
    if valid_ids.numel() == 0:
        raise ValueError("valid_token_ids cannot be empty")

    return valid_ids.contiguous()


def _validate_inputs(logits, valid_ids, torch, check_inputs: bool) -> int:
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor")
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if logits.ndim not in (1, 2):
        raise ValueError("logits must have shape [vocab] or [batch, vocab]")
    if logits.dtype not in (torch.float16, torch.float32):
        raise ValueError("logits dtype must be torch.float16 or torch.float32")

    vocab_size = logits.shape[-1]
    if vocab_size <= 0:
        raise ValueError("logits vocabulary dimension cannot be empty")

    if check_inputs:
        below_zero = bool((valid_ids < 0).any().item())
        above_vocab = bool((valid_ids >= vocab_size).any().item())
        if below_zero or above_vocab:
            raise ValueError(
                f"valid_token_ids must be in [0, {vocab_size}); "
                "pass check_inputs=False only for trusted benchmark inputs"
            )

    return vocab_size


def masked_argmax_tensor(logits, valid_token_ids: Any, *, check_inputs: bool = True):
    """return CUDA tensor of selected token IDs after sparse grammar masking.

    args:
        logits: CUDA tensor shaped ``[vocab]`` or ``[batch, vocab]``.
        valid_token_ids: 1D tensor/list of grammar-valid token IDs.
        check_inputs: when true, range-check valid IDs before launching CUDA.

    returns:
        a CUDA int64 tensor shaped ``[1]`` for 1D logits or ``[batch]`` for 2D
        logits. ties are resolved by choosing the smallest token ID.
    """
    torch = _require_torch()
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor")

    valid_ids = _as_valid_id_tensor(valid_token_ids, torch, logits.device)
    _validate_inputs(logits, valid_ids, torch, check_inputs)

    extension = _load_extension()
    return extension.masked_argmax(logits.contiguous(), valid_ids)


def masked_argmax(logits, valid_token_ids: Any, *, check_inputs: bool = True):
    """return selected token ID(s) as Python values.

    this convenience wrapper synchronizes to copy results back to the host. use
    :func:`masked_argmax_tensor` for benchmark loops that should keep results on
    the GPU until an explicit synchronization point.
    """
    result = masked_argmax_tensor(logits, valid_token_ids, check_inputs=check_inputs)
    if result.numel() == 1:
        return int(result.item())
    return [int(item) for item in result.cpu().tolist()]


def torch_reference_masked_argmax(logits, valid_token_ids: Any):
    """reference PyTorch implementation with the same tie-breaking as the CUDA kernel."""
    torch = _require_torch()
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a torch.Tensor")

    valid_ids = _as_valid_id_tensor(valid_token_ids, torch, logits.device)
    _validate_inputs(logits, valid_ids, torch, check_inputs=True)

    if logits.ndim == 1:
        gathered = logits.index_select(0, valid_ids)
        max_value = gathered.max()
        tied_ids = valid_ids[gathered == max_value]
        return tied_ids.min()

    gathered = logits.index_select(1, valid_ids)
    max_values = gathered.max(dim=1, keepdim=True).values
    expanded_ids = valid_ids.view(1, -1).expand(logits.shape[0], -1)
    sentinel = logits.shape[-1]
    tied_ids = torch.where(
        gathered == max_values,
        expanded_ids,
        torch.full_like(expanded_ids, sentinel),
    )
    return tied_ids.min(dim=1).values
