"""helpers for handing grammar-valid token ids to CUDA selection."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .masked_argmax import masked_argmax_tensor


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
