"""Pure classification of one completed grammar-masked speculative transaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar, cast

from .grammar_selection import GrammarMaskedSelectionResult
from .grammar_speculative_iteration import (
    GrammarMaskedSpeculativeIterationError,
    GrammarMaskedSpeculativeIterationResult,
)

__all__ = [
    "GrammarMaskedSpeculativeOutcomeError",
    "GrammarMaskedSpeculativeOutcomeInvariantError",
    "GrammarMaskedSpeculativeOutcomeResult",
    "classify_grammar_masked_speculative_outcome",
]


class GrammarMaskedSpeculativeOutcomeError(GrammarMaskedSpeculativeIterationError):
    """Base error raised while classifying a completed D47 transaction."""


class GrammarMaskedSpeculativeOutcomeInvariantError(
    GrammarMaskedSpeculativeOutcomeError
):
    """Raised when stored transaction or composed outcome evidence is inconsistent."""


@dataclass(frozen=True, slots=True)
class GrammarMaskedSpeculativeOutcomeResult:
    """One explicit semantic classification of a completed D47 transaction."""

    kind: Literal[
        "handoff_available",
        "grammar_complete",
        "grammar_no_continuation",
    ]

    def __post_init__(self) -> None:
        if type(self.kind) is not str:
            raise TypeError("kind must be an exact string")
        if self.kind not in {
            "handoff_available",
            "grammar_complete",
            "grammar_no_continuation",
        }:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                f"unsupported grammar-masked speculative outcome kind: {self.kind!r}"
            )


_OUTCOME_RESULT_TYPE = GrammarMaskedSpeculativeOutcomeResult
StateT = TypeVar("StateT")


def classify_grammar_masked_speculative_outcome(
    iteration_result: GrammarMaskedSpeculativeIterationResult[StateT],
) -> GrammarMaskedSpeculativeOutcomeResult:
    """Validate and classify one borrowed completed D47 result without side effects."""

    try:
        is_iteration_result = isinstance(
            iteration_result,
            GrammarMaskedSpeculativeIterationResult,
        )
    except Exception as exc:
        raise TypeError(
            "iteration_result type could not be determined"
        ) from exc
    if not is_iteration_result:
        raise TypeError(
            "iteration_result must be a GrammarMaskedSpeculativeIterationResult"
        )

    proposal_token_ids = _read_attribute(iteration_result, "proposal_token_ids")
    accepted_count = _read_attribute(iteration_result, "accepted_count")
    replacement_token_id = _read_attribute(iteration_result, "replacement_token_id")
    initial_cache_length = _read_attribute(iteration_result, "initial_cache_length")
    final_cache_length = _read_attribute(iteration_result, "final_cache_length")
    uncached_next_token_id = _read_attribute(
        iteration_result,
        "uncached_next_token_id",
    )
    shortening_selection = _read_attribute(iteration_result, "shortening_selection")
    acceptance_no_decision_selection = _read_attribute(
        iteration_result,
        "acceptance_no_decision_selection",
    )
    final_row_no_decision_selection = _read_attribute(
        iteration_result,
        "final_row_no_decision_selection",
    )
    _committed_state = _read_attribute(iteration_result, "committed_state")
    committed_state_is_match = _read_attribute(
        iteration_result,
        "committed_state_is_match",
    )

    terminal_is_match = _validate_iteration_evidence(
        proposal_token_ids=proposal_token_ids,
        accepted_count=accepted_count,
        replacement_token_id=replacement_token_id,
        initial_cache_length=initial_cache_length,
        final_cache_length=final_cache_length,
        uncached_next_token_id=uncached_next_token_id,
        shortening_selection=shortening_selection,
        acceptance_no_decision_selection=acceptance_no_decision_selection,
        final_row_no_decision_selection=final_row_no_decision_selection,
        committed_state_is_match=committed_state_is_match,
    )

    if uncached_next_token_id is not None:
        kind = "handoff_available"
    elif terminal_is_match is True:
        kind = "grammar_complete"
    else:
        kind = "grammar_no_continuation"
    return _construct_result(kind)


def _validate_iteration_evidence(
    *,
    proposal_token_ids: object,
    accepted_count: object,
    replacement_token_id: object,
    initial_cache_length: object,
    final_cache_length: object,
    uncached_next_token_id: object,
    shortening_selection: object,
    acceptance_no_decision_selection: object,
    final_row_no_decision_selection: object,
    committed_state_is_match: object,
) -> bool | None:
    proposal = _validate_proposal(proposal_token_ids)
    proposal_length = len(proposal)
    accepted = _validate_accepted_count(accepted_count, proposal_length)
    initial_length = _validate_cache_length(
        initial_cache_length,
        label="initial_cache_length",
    )
    if initial_length == 0:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "initial_cache_length must be greater than zero"
        )
    final_length = _validate_cache_length(
        final_cache_length,
        label="final_cache_length",
    )
    replacement = _validate_optional_token(
        replacement_token_id,
        label="replacement_token_id",
    )
    handoff = _validate_optional_token(
        uncached_next_token_id,
        label="uncached_next_token_id",
    )
    if type(committed_state_is_match) is not bool:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "committed_state_is_match must be a boolean"
        )

    shortening_is_match = _validate_optional_selection(
        shortening_selection,
        label="shortening_selection",
    )
    acceptance_is_match = _validate_optional_selection(
        acceptance_no_decision_selection,
        label="acceptance_no_decision_selection",
    )
    final_row_is_match = _validate_optional_selection(
        final_row_no_decision_selection,
        label="final_row_no_decision_selection",
    )
    if (
        acceptance_no_decision_selection is not None
        and final_row_no_decision_selection is not None
    ):
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "acceptance and final-row no-decision selections are mutually exclusive"
        )

    if proposal_length == 0:
        if accepted != 0:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "an empty proposal requires accepted_count zero"
            )
        if shortening_selection is None:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "an empty proposal requires shortening_selection"
            )
        if (
            replacement is not None
            or handoff is not None
            or acceptance_no_decision_selection is not None
            or final_row_no_decision_selection is not None
        ):
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "an empty proposal cannot contain target decision evidence"
            )
        _require_final_cache_length(
            final_length,
            expected=initial_length + 1,
        )
        if committed_state_is_match is not shortening_is_match:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "empty-proposal match fact must equal shortening_selection"
            )
        return cast(bool, shortening_is_match)

    if acceptance_no_decision_selection is not None:
        if accepted >= proposal_length:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "acceptance no-decision must stop before full acceptance"
            )
        if replacement is not None or handoff is not None:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "acceptance no-decision cannot contain replacement or handoff evidence"
            )
        _require_final_cache_length(
            final_length,
            expected=initial_length + 1 + accepted,
        )
        if committed_state_is_match is not acceptance_is_match:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "no-decision match fact must equal its terminal selection"
            )
        return cast(bool, acceptance_is_match)

    if accepted < proposal_length:
        if replacement is None:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "a decided mismatch requires a replacement token"
            )
        if replacement == proposal[accepted]:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "replacement_token_id must differ from the rejected proposal token"
            )
        if handoff != replacement:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "mismatch uncached_next_token_id must equal replacement_token_id"
            )
        if final_row_no_decision_selection is not None:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "a mismatch cannot contain final-row no-decision evidence"
            )
        _require_final_cache_length(
            final_length,
            expected=initial_length + 1 + accepted,
        )
        return None

    if replacement is not None:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "fully accepted result cannot contain a replacement token"
        )
    _require_final_cache_length(
        final_length,
        expected=initial_length + 1 + proposal_length,
    )
    if final_row_no_decision_selection is None:
        if handoff is None:
            raise GrammarMaskedSpeculativeOutcomeInvariantError(
                "decided full acceptance requires an uncached bonus token"
            )
        return None
    if handoff is not None:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "final-row no-decision cannot contain an uncached token"
        )
    if committed_state_is_match is not final_row_is_match:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "final-row match fact must equal its terminal selection"
        )
    return cast(bool, final_row_is_match)


def _validate_proposal(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "proposal_token_ids must be an exact tuple"
        )
    for position, token_id in enumerate(value):
        _validate_token(token_id, label=f"proposal token at position {position}")
    return cast(tuple[int, ...], value)


def _validate_accepted_count(value: object, proposal_length: int) -> int:
    if type(value) is not int:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "accepted_count must be an integer"
        )
    if value < 0 or value > proposal_length:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"accepted_count must be within [0, {proposal_length}]"
        )
    return value


def _validate_cache_length(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} must be an integer"
        )
    if value < 0:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} cannot be negative"
        )
    return value


def _validate_optional_token(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    _validate_token(value, label=label)
    return cast(int, value)


def _validate_token(value: object, *, label: str) -> None:
    if type(value) is not int:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} must be an integer"
        )
    if value < 0:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} cannot be negative"
        )


def _validate_optional_selection(selection: object, *, label: str) -> bool | None:
    if selection is None:
        return None
    try:
        is_selection = isinstance(selection, GrammarMaskedSelectionResult)
    except Exception as exc:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} type could not be determined"
        ) from exc
    if not is_selection:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} must be a GrammarMaskedSelectionResult or None"
        )
    valid_token_ids = _read_attribute(selection, "valid_token_ids", label=label)
    is_match = _read_attribute(selection, "is_match", label=label)
    selected_token_id = _read_attribute(selection, "selected_token_id", label=label)
    if type(valid_token_ids) is not tuple or valid_token_ids:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} must contain an exact empty support tuple"
        )
    if type(is_match) is not bool:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} is_match must be a boolean"
        )
    if selected_token_id is not None:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{label} must not contain a selected token"
        )
    return is_match


def _require_final_cache_length(value: int, *, expected: int) -> None:
    if value != expected:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"final_cache_length is {value}; expected {expected}"
        )


def _read_attribute(value: object, name: str, *, label: str | None = None) -> object:
    owner = label or "iteration_result"
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            f"{owner} {name} could not be read"
        ) from exc


def _construct_result(
    kind: Literal[
        "handoff_available",
        "grammar_complete",
        "grammar_no_continuation",
    ],
) -> GrammarMaskedSpeculativeOutcomeResult:
    result = GrammarMaskedSpeculativeOutcomeResult(kind=kind)
    if type(result) is not _OUTCOME_RESULT_TYPE:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "outcome construction must return a GrammarMaskedSpeculativeOutcomeResult"
        )
    result_kind = _read_attribute(result, "kind", label="outcome result")
    if type(result_kind) is not str or result_kind != kind:
        raise GrammarMaskedSpeculativeOutcomeInvariantError(
            "outcome result must retain the exact requested kind"
        )
    return result
