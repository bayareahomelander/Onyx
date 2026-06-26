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

    cached tensors are valid for one compiled grammar state space. clear or
    recreate the cache after recompiling or otherwise changing the grammar
    constraint.
    """

    def __init__(self, grammar_constraint: Any):
        self.grammar_constraint = grammar_constraint
        self._cache: Dict[Tuple[int, str], Any] = {}

    def clear(self) -> None:
        """clear all cached CUDA tensors."""
        self._cache.clear()

    def get(self, state: int, device):
        """return a CUDA tensor of valid token ids for the grammar state."""
        torch = _require_torch()
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CudaValidIdCache requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        key = (int(state), str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        valid_token_ids = self.grammar_constraint.get_valid_token_ids(state)
        if not valid_token_ids:
            raise ValueError("grammar state produced no valid token ids")

        valid_ids = torch.as_tensor(valid_token_ids, dtype=torch.long, device=device).contiguous()
        self._cache[key] = valid_ids
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
