"""CUDA grammar logit masking."""

import torch


def apply_grammar_mask(
    logits: torch.Tensor, valid_token_ids: list[int]
) -> torch.Tensor:
    """Return logits with every invalid token set to negative infinity."""
    if logits.device.type != "cuda":
        raise RuntimeError("Onyx CUDA grammar masking requires logits on CUDA")
    if not torch.is_floating_point(logits):
        raise TypeError("Grammar masking requires floating-point logits")
    if not valid_token_ids:
        raise ValueError("valid_token_ids cannot be empty")

    vocab_size = logits.shape[-1]
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= vocab_size
        for token_id in valid_token_ids
    ):
        raise ValueError(f"valid_token_ids must be integers in [0, {vocab_size})")

    token_ids = torch.tensor(
        valid_token_ids, dtype=torch.long, device=logits.device
    )
    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits.index_copy_(
        -1, token_ids, logits.index_select(-1, token_ids)
    )
    return masked_logits
