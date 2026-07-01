"""helpers for handing grammar-valid token ids to CUDA selection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .masked_argmax import masked_argmax_tensor


@dataclass(frozen=True)
class CudaValidIdLookup:
    """One instrumented grammar-state lookup and CUDA tensor result."""

    valid_ids: Any
    valid_token_count: int
    tensor_bytes: int
    cache_hit: bool
    grammar_traversal_s: float
    fingerprint_s: float
    cache_lookup_s: float
    cuda_upload_s: float


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CudaValidIdCache requires PyTorch to create CUDA tensors.") from exc

    return torch


class CudaValidIdCache:
    """cache grammar-valid token id tensors on CUDA devices.

    Every lookup re-reads the grammar-valid IDs and compares them with the
    cached fingerprint. This preserves correctness when opaque state handles
    are reused after a grammar reset while still avoiding repeated CUDA uploads
    when the valid-ID set is unchanged.
    """

    def __init__(self, grammar_constraint: Any):
        self.grammar_constraint = grammar_constraint
        self._cache: Dict[Tuple[int, str], Tuple[Tuple[int, ...], Any]] = {}

    def clear(self) -> None:
        """clear all cached CUDA tensors."""
        self._cache.clear()

    def discard(self, state: int) -> None:
        """drop cached tensors for one opaque grammar state on all devices."""
        state = int(state)
        stale_keys = [key for key in self._cache if key[0] == state]
        for key in stale_keys:
            del self._cache[key]

    @property
    def num_entries(self) -> int:
        """return the number of cached state/device tensors."""
        return len(self._cache)

    def get(self, state: int, device):
        """return a CUDA tensor of valid token ids for the grammar state."""
        torch = _require_torch()
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CudaValidIdCache requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        valid_token_ids = tuple(self.grammar_constraint.get_valid_token_ids(state))
        if not valid_token_ids:
            raise ValueError("grammar state produced no valid token ids")

        key = (int(state), str(device))
        cached = self._cache.get(key)
        if cached is not None and cached[0] == valid_token_ids:
            return cached[1]

        valid_ids = torch.as_tensor(valid_token_ids, dtype=torch.long, device=device).contiguous()
        self._cache[key] = (valid_token_ids, valid_ids)
        return valid_ids

    def get_with_diagnostics(self, state: int, device) -> CudaValidIdLookup:
        """return valid IDs with phase-separated host-observed diagnostics.

        CUDA upload timing uses explicit synchronization around cache misses.
        Callers should use :meth:`get` when they do not want this diagnostic
        synchronization to affect the measured path.
        """
        return self._get_with_diagnostics(state, device)

    def _get_with_diagnostics(self, state: int, device) -> CudaValidIdLookup:
        torch = _require_torch()
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CudaValidIdCache requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        stage_start = time.perf_counter()
        raw_valid_token_ids = self.grammar_constraint.get_valid_token_ids(state)
        grammar_traversal_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        valid_token_ids = tuple(raw_valid_token_ids)
        fingerprint_s = time.perf_counter() - stage_start
        if not valid_token_ids:
            raise ValueError("grammar state produced no valid token ids")

        key = (int(state), str(device))
        stage_start = time.perf_counter()
        cached = self._cache.get(key)
        if cached is not None and cached[0] == valid_token_ids:
            cache_lookup_s = time.perf_counter() - stage_start
            valid_ids = cached[1]
            return CudaValidIdLookup(
                valid_ids=valid_ids,
                valid_token_count=len(valid_token_ids),
                tensor_bytes=int(valid_ids.numel()) * int(valid_ids.element_size()),
                cache_hit=True,
                grammar_traversal_s=grammar_traversal_s,
                fingerprint_s=fingerprint_s,
                cache_lookup_s=cache_lookup_s,
                cuda_upload_s=0.0,
            )
        cache_lookup_s = time.perf_counter() - stage_start

        torch.cuda.synchronize(device)
        stage_start = time.perf_counter()
        valid_ids = torch.as_tensor(valid_token_ids, dtype=torch.long, device=device).contiguous()
        torch.cuda.synchronize(device)
        cuda_upload_s = time.perf_counter() - stage_start
        self._cache[key] = (valid_token_ids, valid_ids)
        return CudaValidIdLookup(
            valid_ids=valid_ids,
            valid_token_count=len(valid_token_ids),
            tensor_bytes=int(valid_ids.numel()) * int(valid_ids.element_size()),
            cache_hit=False,
            grammar_traversal_s=grammar_traversal_s,
            fingerprint_s=fingerprint_s,
            cache_lookup_s=cache_lookup_s,
            cuda_upload_s=cuda_upload_s,
        )


def masked_argmax_from_grammar_state(
    logits,
    grammar_constraint: Any,
    state: int,
    *,
    check_inputs: bool = True,
):
    """return CUDA selected token ids for a grammar state."""
    valid_token_ids = grammar_constraint.get_valid_token_ids(state)
    if not valid_token_ids:
        raise ValueError("grammar state produced no valid token ids")

    return masked_argmax_tensor(logits, valid_token_ids, check_inputs=check_inputs)


def masked_argmax_from_cached_grammar_state(
    logits,
    valid_id_cache: CudaValidIdCache,
    state: int,
    *,
    check_inputs: bool = True,
):
    """return CUDA selected token ids using cached grammar-valid id tensors."""
    valid_ids = valid_id_cache.get(state, logits.device)
    return masked_argmax_tensor(logits, valid_ids, check_inputs=check_inputs)
