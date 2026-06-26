"""
experimental CUDA helpers for Onyx.
"""

from . import masked_argmax as _masked_argmax
from .grammar_handoff import masked_argmax_from_grammar_state

extension_status = _masked_argmax.extension_status
masked_argmax_tensor = _masked_argmax.masked_argmax_tensor
torch_reference_masked_argmax = _masked_argmax.torch_reference_masked_argmax


def __getattr__(name: str):
    if name in {"CUDA_EXTENSION_AVAILABLE", "CUDA_EXTENSION_ERROR"}:
        return getattr(_masked_argmax, name)
    raise AttributeError(f"module 'onyx_cuda' has no attribute {name!r}")


__all__ = [
    "CUDA_EXTENSION_AVAILABLE",
    "CUDA_EXTENSION_ERROR",
    "extension_status",
    "masked_argmax_from_grammar_state",
    "masked_argmax_tensor",
    "torch_reference_masked_argmax",
]
