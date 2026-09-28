"""CUDA grammar logit masking."""

import torch


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


class TokenMask:
    """Selectable tokens from the native engine as a device mask.

    blocked is True where a token is not selectable. count equals the length
    of the equivalent ID list, so emptiness and the numerical guard behave as
    for lists; iterating yields the IDs but synchronizes with the device.
    """

    __slots__ = ("blocked", "count")

    def __init__(self, blocked: torch.Tensor, count: int):
        self.blocked = blocked
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(torch.nonzero(~self.blocked).flatten().tolist())


def _token_id_list(valid):
    return valid if isinstance(valid, list) else list(valid)


def _usable_mask(logits: torch.Tensor, valid) -> bool:
    """Engine-built masks skip per-ID checks; other widths use the ID path."""
    if not isinstance(valid, TokenMask) or valid.blocked.shape[-1] != logits.shape[-1]:
        return False
    if logits.device.type != "cuda":
        raise RuntimeError("Onyx CUDA grammar masking requires logits on CUDA")
    if not torch.is_floating_point(logits):
        raise TypeError("Grammar masking requires floating-point logits")
    if not valid.count:
        raise ValueError("valid_token_ids cannot be empty")
    return True


def apply_grammar_mask(
    logits: torch.Tensor, valid_token_ids: list[int]
) -> torch.Tensor:
    """Return logits with every invalid token set to negative infinity."""
    if _usable_mask(logits, valid_token_ids):
        # Bitwise identical to copying the valid logits into a -inf tensor.
        return logits.masked_fill(valid_token_ids.blocked, -torch.inf)
    valid_token_ids = _token_id_list(valid_token_ids)
    _validate_mask_inputs(logits, valid_token_ids)
    token_ids = torch.tensor(
        valid_token_ids, dtype=torch.long, device=logits.device
    )
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits.index_copy_(
        -1, token_ids, logits.index_select(-1, token_ids)
    )
    return masked_logits


def grammar_argmax(logits: torch.Tensor, valid_token_ids: list[int]) -> torch.Tensor:
    """Select the greedy token among the grammar's valid tokens."""
    return apply_grammar_mask(logits, valid_token_ids).argmax(dim=-1)
