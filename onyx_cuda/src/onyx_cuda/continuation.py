"""Framework-neutral continuation selection over completed acceptance evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

from .acceptance import MatchReplaceAcceptanceResult


class PostIterationContinuationError(RuntimeError):
    """Base error raised by the framework-neutral continuation decision."""


class PostIterationContinuationInvariantError(PostIterationContinuationError):
    """Raised when proposal, row, or acceptance relationships are inconsistent."""


LogitsT = TypeVar("LogitsT")


@dataclass(frozen=True, slots=True)
class PostIterationContinuationResult:
    """Immutable emitted-token output and its uncached continuation token."""

    output_token_ids: tuple[int, ...]
    uncached_next_token_id: int

    def __post_init__(self) -> None:
        _validate_output_token_ids(self.output_token_ids)
        _validate_nonnegative_token_id(
            self.uncached_next_token_id,
            label="uncached_next_token_id",
        )
        if self.uncached_next_token_id != self.output_token_ids[-1]:
            raise PostIterationContinuationInvariantError(
                "uncached_next_token_id must equal the final output token"
            )


def decide_post_iteration_continuation(
    proposal_token_ids: tuple[int, ...],
    target_logit_rows: tuple[LogitsT, ...],
    acceptance_result: MatchReplaceAcceptanceResult,
    *,
    vocab_size: int,
    select_token: Callable[[LogitsT], int],
) -> PostIterationContinuationResult:
    """Derive one uncached continuation from exact D30 and D33 evidence.

    A mismatch reuses D33's replacement without selecting another row. Full acceptance passes
    only the final post-proposal row to the caller-owned selector exactly once.
    """

    accepted_count, replacement_token_id = _validate_decision_inputs(
        proposal_token_ids,
        target_logit_rows,
        acceptance_result,
        vocab_size=vocab_size,
        select_token=select_token,
    )

    if accepted_count < len(proposal_token_ids):
        replacement = cast(int, replacement_token_id)
        output_token_ids = proposal_token_ids[:accepted_count] + (replacement,)
        return PostIterationContinuationResult(
            output_token_ids=output_token_ids,
            uncached_next_token_id=replacement,
        )

    selected_bonus_token_id = select_token(target_logit_rows[-1])
    _validate_token_in_vocabulary(
        selected_bonus_token_id,
        vocab_size,
        label="selected bonus token",
    )
    bonus = cast(int, selected_bonus_token_id)
    return PostIterationContinuationResult(
        output_token_ids=proposal_token_ids + (bonus,),
        uncached_next_token_id=bonus,
    )


def _validate_decision_inputs(
    proposal_token_ids: object,
    target_logit_rows: object,
    acceptance_result: object,
    *,
    vocab_size: object,
    select_token: object,
) -> tuple[int, int | None]:
    _validate_proposal_token_ids(proposal_token_ids)
    validated_vocab_size = _validate_vocab_size(vocab_size)
    proposal = cast(tuple[int, ...], proposal_token_ids)
    for position, token_id in enumerate(proposal):
        _validate_token_in_vocabulary(
            token_id,
            validated_vocab_size,
            label=f"proposal token at position {position}",
        )

    if type(target_logit_rows) is not tuple:
        raise TypeError("target_logit_rows must be a tuple")
    rows = cast(tuple[object, ...], target_logit_rows)
    expected_row_count = len(proposal) + 1
    if len(rows) != expected_row_count:
        raise PostIterationContinuationInvariantError(
            f"target_logit_rows contains {len(rows)} rows; expected {expected_row_count} "
            f"for proposal length {len(proposal)}"
        )

    if not isinstance(acceptance_result, MatchReplaceAcceptanceResult):
        raise TypeError("acceptance_result must be a MatchReplaceAcceptanceResult")
    try:
        evidence_proposal = acceptance_result.proposal_token_ids
        accepted_count = acceptance_result.accepted_count
        replacement_token_id = acceptance_result.replacement_token_id
    except AttributeError as exc:
        raise PostIterationContinuationInvariantError(
            "acceptance_result fields are unavailable"
        ) from exc

    if type(evidence_proposal) is not tuple:
        raise PostIterationContinuationInvariantError(
            "acceptance_result.proposal_token_ids must be a tuple"
        )
    for position, token_id in enumerate(evidence_proposal):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise PostIterationContinuationInvariantError(
                "acceptance_result proposal token at position "
                f"{position} must be an integer"
            )
        if token_id < 0 or token_id >= validated_vocab_size:
            raise PostIterationContinuationInvariantError(
                "acceptance_result proposal token at position "
                f"{position} must be within [0, {validated_vocab_size})"
            )
    if evidence_proposal != proposal:
        raise PostIterationContinuationInvariantError(
            "acceptance_result proposal does not match proposal_token_ids"
        )

    if isinstance(accepted_count, bool) or not isinstance(accepted_count, int):
        raise PostIterationContinuationInvariantError(
            "acceptance_result.accepted_count must be an integer"
        )
    if accepted_count < 0 or accepted_count > len(proposal):
        raise PostIterationContinuationInvariantError(
            f"acceptance_result.accepted_count must be within [0, {len(proposal)}]"
        )

    if accepted_count == len(proposal):
        if replacement_token_id is not None:
            raise PostIterationContinuationInvariantError(
                "fully accepted evidence cannot contain a replacement token"
            )
    else:
        if replacement_token_id is None:
            raise PostIterationContinuationInvariantError(
                "mismatch evidence must contain a replacement token"
            )
        if isinstance(replacement_token_id, bool) or not isinstance(replacement_token_id, int):
            raise PostIterationContinuationInvariantError(
                "acceptance_result.replacement_token_id must be an integer"
            )
        if replacement_token_id < 0 or replacement_token_id >= validated_vocab_size:
            raise PostIterationContinuationInvariantError(
                "acceptance_result.replacement_token_id must be within "
                f"[0, {validated_vocab_size})"
            )
        if replacement_token_id == proposal[accepted_count]:
            raise PostIterationContinuationInvariantError(
                "acceptance_result.replacement_token_id must differ from the rejected "
                "proposal token"
            )

    if not callable(select_token):
        raise TypeError("select_token must be callable")
    return accepted_count, replacement_token_id


def _validate_output_token_ids(output_token_ids: object) -> None:
    if type(output_token_ids) is not tuple:
        raise TypeError("output_token_ids must be a tuple")
    output = cast(tuple[object, ...], output_token_ids)
    if not output:
        raise ValueError("output_token_ids cannot be empty")
    for position, token_id in enumerate(output):
        _validate_nonnegative_token_id(
            token_id,
            label=f"output token at position {position}",
        )


def _validate_proposal_token_ids(proposal_token_ids: object) -> None:
    if type(proposal_token_ids) is not tuple:
        raise TypeError("proposal_token_ids must be a tuple")
    proposal = cast(tuple[object, ...], proposal_token_ids)
    if not proposal:
        raise ValueError("proposal_token_ids cannot be empty")
    for position, token_id in enumerate(proposal):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"proposal token at position {position} must be an integer")


def _validate_vocab_size(vocab_size: object) -> int:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")
    return vocab_size


def _validate_token_in_vocabulary(token_id: object, vocab_size: int, *, label: str) -> None:
    _validate_nonnegative_token_id(token_id, label=label)
    if token_id >= vocab_size:
        raise ValueError(f"{label} must be within [0, {vocab_size})")


def _validate_nonnegative_token_id(token_id: object, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0:
        raise ValueError(f"{label} cannot be negative")


__all__ = [
    "PostIterationContinuationError",
    "PostIterationContinuationInvariantError",
    "PostIterationContinuationResult",
    "decide_post_iteration_continuation",
]
