"""Framework-neutral match/replace acceptance over already-produced evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast


class MatchReplaceAcceptanceError(RuntimeError):
    """Base error raised by framework-neutral match/replace acceptance."""


class MatchReplaceAcceptanceInvariantError(MatchReplaceAcceptanceError):
    """Raised when proposal, row, or outcome relationships violate the D33 contract."""


LogitsT = TypeVar("LogitsT")


@dataclass(frozen=True, slots=True)
class MatchReplaceAcceptanceResult:
    """Immutable accepted-prefix and optional first-mismatch replacement outcome."""

    proposal_token_ids: tuple[int, ...]
    accepted_count: int
    replacement_token_id: int | None

    def __post_init__(self) -> None:
        _validate_proposal_token_ids(self.proposal_token_ids)

        if isinstance(self.accepted_count, bool) or not isinstance(self.accepted_count, int):
            raise TypeError("accepted_count must be an integer")
        proposal_length = len(self.proposal_token_ids)
        if self.accepted_count < 0 or self.accepted_count > proposal_length:
            raise MatchReplaceAcceptanceInvariantError(
                f"accepted_count must be within [0, {proposal_length}]"
            )

        if self.accepted_count == proposal_length:
            if self.replacement_token_id is not None:
                raise MatchReplaceAcceptanceInvariantError(
                    "fully accepted result cannot contain a replacement token"
                )
            return

        if self.replacement_token_id is None:
            raise MatchReplaceAcceptanceInvariantError(
                "partially accepted result must contain a replacement token"
            )
        _validate_nonnegative_token_id(
            self.replacement_token_id,
            label="replacement_token_id",
        )
        if self.replacement_token_id == self.proposal_token_ids[self.accepted_count]:
            raise MatchReplaceAcceptanceInvariantError(
                "replacement_token_id must differ from the rejected proposal token"
            )

    @property
    def fully_accepted(self) -> bool:
        """Whether every proposal token matched its target decision row."""

        return self.accepted_count == len(self.proposal_token_ids)

    @property
    def accepted_token_ids(self) -> tuple[int, ...]:
        """Return the exact proposal prefix accepted before a mismatch."""

        return self.proposal_token_ids[: self.accepted_count]

    @property
    def rejected_proposal_token_id(self) -> int | None:
        """Return the first rejected proposal token, if one exists."""

        if self.fully_accepted:
            return None
        return self.proposal_token_ids[self.accepted_count]

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Return only tokens newly emitted by this acceptance decision."""

        if self.fully_accepted:
            return self.accepted_token_ids
        replacement_token_id = cast(int, self.replacement_token_id)
        return self.accepted_token_ids + (replacement_token_id,)

    @property
    def next_current_token_id(self) -> int:
        """Return the final emitted token for later loop composition."""

        return self.output_token_ids[-1]


def decide_match_replace_acceptance(
    proposal_token_ids: tuple[int, ...],
    target_logit_rows: tuple[LogitsT, ...],
    *,
    select_token: Callable[[LogitsT], int],
) -> MatchReplaceAcceptanceResult:
    """Accept matching proposal tokens and replace the first target mismatch.

    ``target_logit_rows`` must contain the complete D30/D31 ``n + 1`` row tuple. Only rows
    ``0`` through ``n - 1`` are passed to the borrowed selector; the final post-proposal row is
    required structurally but deliberately remains unused.
    """

    _validate_decision_inputs(
        proposal_token_ids,
        target_logit_rows,
        select_token=select_token,
    )

    for position, proposal_token_id in enumerate(proposal_token_ids):
        selected_token_id = select_token(target_logit_rows[position])
        _validate_nonnegative_token_id(
            selected_token_id,
            label=f"selected token at proposal position {position}",
        )
        if selected_token_id != proposal_token_id:
            return MatchReplaceAcceptanceResult(
                proposal_token_ids=proposal_token_ids,
                accepted_count=position,
                replacement_token_id=selected_token_id,
            )

    return MatchReplaceAcceptanceResult(
        proposal_token_ids=proposal_token_ids,
        accepted_count=len(proposal_token_ids),
        replacement_token_id=None,
    )


def _validate_decision_inputs(
    proposal_token_ids: object,
    target_logit_rows: object,
    *,
    select_token: object,
) -> None:
    _validate_proposal_token_ids(proposal_token_ids)
    if type(target_logit_rows) is not tuple:
        raise TypeError("target_logit_rows must be a tuple")

    proposal_length = len(cast(tuple[object, ...], proposal_token_ids))
    actual_row_count = len(target_logit_rows)
    expected_row_count = proposal_length + 1
    if actual_row_count != expected_row_count:
        raise MatchReplaceAcceptanceInvariantError(
            f"target_logit_rows contains {actual_row_count} rows; expected {expected_row_count} "
            f"for proposal length {proposal_length}"
        )
    if not callable(select_token):
        raise TypeError("select_token must be callable")


def _validate_proposal_token_ids(proposal_token_ids: object) -> None:
    if type(proposal_token_ids) is not tuple:
        raise TypeError("proposal_token_ids must be a tuple")
    proposal = cast(tuple[object, ...], proposal_token_ids)
    if not proposal:
        raise ValueError("proposal_token_ids cannot be empty")
    for position, token_id in enumerate(proposal):
        _validate_nonnegative_token_id(
            token_id,
            label=f"proposal token at position {position}",
        )


def _validate_nonnegative_token_id(token_id: object, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0:
        raise ValueError(f"{label} cannot be negative")


__all__ = [
    "MatchReplaceAcceptanceError",
    "MatchReplaceAcceptanceInvariantError",
    "MatchReplaceAcceptanceResult",
    "decide_match_replace_acceptance",
]
