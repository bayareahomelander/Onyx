"""
experimental CUDA helpers for Onyx.
"""

from . import masked_argmax as _masked_argmax
from .decode_loop import (
    CudaGrammarDecodeResult,
    CudaGrammarDecodeTimings,
    decode_greedy_from_logits,
)
from .grammar_handoff import (
    CudaValidIdCache,
    masked_argmax_from_cached_grammar_state,
    masked_argmax_from_grammar_state,
)
from .real_logits_handoff import (
    RealLogitsHandoffReport,
    format_real_logits_report,
    run_real_logits_handoff,
)

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
    "CudaGrammarDecodeResult",
    "CudaGrammarDecodeTimings",
    "CudaValidIdCache",
    "RealLogitsHandoffReport",
    "decode_greedy_from_logits",
    "extension_status",
    "format_real_logits_report",
    "masked_argmax_from_cached_grammar_state",
    "masked_argmax_from_grammar_state",
    "masked_argmax_tensor",
    "run_real_logits_handoff",
    "torch_reference_masked_argmax",
]
