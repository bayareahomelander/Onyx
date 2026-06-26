"""helpers for handing grammar-valid token ids to CUDA selection."""

from __future__ import annotations

from typing import Any

from .masked_argmax import masked_argmax_tensor


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
