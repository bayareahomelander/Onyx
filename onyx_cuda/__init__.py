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
    CudaValidIdLookup,
    CudaValidIdCache,
    masked_argmax_from_cached_grammar_state,
    masked_argmax_from_grammar_state,
)
from .kv_cache_probe import (
    KvCacheLayerSnapshot,
    KvCacheProbeReport,
    KvCacheProbeTimings,
    KvCacheSnapshot,
    format_kv_cache_report,
    inspect_kv_cache,
    run_kv_cache_probe,
)
from .real_logits_handoff import (
    RealLogitsHandoffReport,
    format_real_logits_report,
    run_real_logits_handoff,
)
from .target_generation import (
    CudaSelectionStepDiagnostics,
    CudaTargetGenerationReport,
    CudaTargetGenerationTimings,
    GeneratedToken,
    format_target_generation_report,
    run_target_only_generation,
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
    "CudaSelectionStepDiagnostics",
    "CudaTargetGenerationReport",
    "CudaTargetGenerationTimings",
    "CudaValidIdCache",
    "CudaValidIdLookup",
    "GeneratedToken",
    "KvCacheLayerSnapshot",
    "KvCacheProbeReport",
    "KvCacheProbeTimings",
    "KvCacheSnapshot",
    "RealLogitsHandoffReport",
    "decode_greedy_from_logits",
    "extension_status",
    "format_kv_cache_report",
    "format_real_logits_report",
    "format_target_generation_report",
    "inspect_kv_cache",
    "masked_argmax_from_cached_grammar_state",
    "masked_argmax_from_grammar_state",
    "masked_argmax_tensor",
    "run_kv_cache_probe",
    "run_real_logits_handoff",
    "run_target_only_generation",
    "torch_reference_masked_argmax",
]
