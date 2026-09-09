"""CUDA grammar logit masking."""

import torch

from onyx_cuda.config import resolve_greedy_backend


def _validate_mask_inputs(
    logits: torch.Tensor, valid_token_ids: list[int]
) -> None:
    if logits.device.type != "cuda":
        raise RuntimeError("Onyx CUDA grammar masking requires logits on CUDA")
    if not torch.is_floating_point(logits):
        raise TypeError("Grammar masking requires floating-point logits")
    if not valid_token_ids:
        raise ValueError("valid_token_ids cannot be empty")
    if logits.ndim == 0:
        raise ValueError("logits must have a vocabulary dimension")

    vocab_size = logits.shape[-1]
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= vocab_size
        for token_id in valid_token_ids
    ):
        raise ValueError(f"valid_token_ids must be integers in [0, {vocab_size})")


def apply_grammar_mask(
    logits: torch.Tensor, valid_token_ids: list[int]
) -> torch.Tensor:
    """Return logits with every invalid token set to negative infinity."""
    _validate_mask_inputs(logits, valid_token_ids)
    token_ids = torch.tensor(
        valid_token_ids, dtype=torch.long, device=logits.device
    )
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits.index_copy_(
        -1, token_ids, logits.index_select(-1, token_ids)
    )
    return masked_logits


def grammar_argmax(
    logits: torch.Tensor, valid_token_ids: list[int], *, backend: str | None = None
) -> torch.Tensor:
    """Select a constrained greedy token; CUDA is an explicit optional backend.

    Unhandled dtypes/ranks use the dense reference, preserving its precision.
    Compilation/import failures for supported CUDA inputs are surfaced to callers.
    """
    backend = resolve_greedy_backend(backend)
    if backend == "torch":
        return apply_grammar_mask(logits, valid_token_ids).argmax(dim=-1)
    _validate_mask_inputs(logits, valid_token_ids)
    if logits.dtype not in (torch.float16, torch.float32) or logits.ndim not in (1, 2):
        return apply_grammar_mask(logits, valid_token_ids).argmax(dim=-1)
    from onyx_cuda._sparse_argmax import sparse_argmax

    token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=logits.device)
    return sparse_argmax(logits, token_ids)
