"""Grammar-masked continuation for one decided target-acceptance outcome."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .constrained_generation import GrammarLogitMask
from .continuation import PostIterationContinuationError
from .grammar import GrammarConstraint
from .grammar_acceptance import GrammarMaskedTargetAcceptanceResult
from .grammar_selection import GrammarMaskedSelectionResult
from .grammar_transition import (
    GrammarMaskedTransitionResult,
    select_and_advance_grammar_state,
)


class GrammarMaskedPostAcceptanceContinuationError(PostIterationContinuationError):
    """Base error raised by grammar-masked post-acceptance continuation."""


class GrammarMaskedPostAcceptanceContinuationInvariantError(
    GrammarMaskedPostAcceptanceContinuationError
):
    """Raised when acceptance or transition evidence violates the D46 contract."""


class GrammarMaskedPostAcceptanceContinuationCleanupError(
    GrammarMaskedPostAcceptanceContinuationError
):
    """Raised when failed continuation cannot release every owned grammar state."""

    def __init__(
        self,
        original_failure: BaseException,
        cleanup_failures: Sequence[tuple[str, Exception]],
    ) -> None:
        failures = tuple(cleanup_failures)
        if not failures:
            raise ValueError("cleanup_failures cannot be empty")
        self.original_failure = original_failure
        self.cleanup_failures = failures
        self.__cause__ = original_failure
        details = "; ".join(
            f"{operation} also failed: {failure}"
            for operation, failure in failures
        )
        super().__init__(
            "grammar-masked post-acceptance continuation failed: "
            f"{original_failure}; {details}"
        )


StateT = TypeVar("StateT")
LogitsT = TypeVar("LogitsT")


@dataclass(frozen=True, slots=True)
class GrammarMaskedPostAcceptanceContinuationResult(Generic[StateT]):
    """One emitted continuation outcome and its transferred grammar state."""

    output_token_ids: tuple[int, ...]
    uncached_next_token_id: int | None
    final_row_no_decision_selection: GrammarMaskedSelectionResult | None
    committed_state: StateT
    committed_state_is_match: bool

    def __post_init__(self) -> None:
        _validate_output_token_ids(self.output_token_ids)
        if type(self.committed_state_is_match) is not bool:
            raise TypeError("committed_state_is_match must be a boolean")

        if self.final_row_no_decision_selection is None:
            _validate_nonnegative_token_id(
                self.uncached_next_token_id,
                label="uncached_next_token_id",
            )
            if self.uncached_next_token_id != self.output_token_ids[-1]:
                raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                    "uncached_next_token_id must equal the final output token"
                )
            return

        selection_is_match = _validate_no_decision_selection_for_result(
            self.final_row_no_decision_selection
        )
        if self.uncached_next_token_id is not None:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "a final-row no-decision result cannot contain an uncached token"
            )
        if self.committed_state_is_match is not selection_is_match:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "the committed-state match fact must equal the final-row parent fact"
            )


_NO_STATE = object()


def decide_grammar_masked_post_acceptance_continuation(
    proposal_token_ids: tuple[int, ...],
    target_logit_rows: tuple[LogitsT, ...],
    acceptance_result: GrammarMaskedTargetAcceptanceResult[StateT],
    constraint: GrammarConstraint[StateT],
    logit_mask: GrammarLogitMask[LogitsT],
    *,
    vocab_size: int,
    select_token: Callable[[LogitsT], int],
) -> GrammarMaskedPostAcceptanceContinuationResult[StateT]:
    """Continue one decided D45 outcome without touching either model cache."""

    accepted_count, replacement_token_id, parent_state, parent_is_match = (
        _validate_continuation_inputs(
            proposal_token_ids,
            target_logit_rows,
            acceptance_result,
            vocab_size=vocab_size,
            select_token=select_token,
        )
    )

    owned_parent: object = parent_state
    owned_child: object = _NO_STATE
    try:
        if accepted_count < len(proposal_token_ids):
            replacement = cast(int, replacement_token_id)
            output_token_ids = proposal_token_ids[:accepted_count] + (replacement,)
            result = GrammarMaskedPostAcceptanceContinuationResult(
                output_token_ids=output_token_ids,
                uncached_next_token_id=replacement,
                final_row_no_decision_selection=None,
                committed_state=parent_state,
                committed_state_is_match=parent_is_match,
            )
            _validate_composed_result(
                result,
                output_token_ids=output_token_ids,
                uncached_next_token_id=replacement,
                final_row_no_decision_selection=None,
                committed_state=parent_state,
                committed_state_is_match=parent_is_match,
            )
            owned_parent = _NO_STATE
            return result

        transition = select_and_advance_grammar_state(
            constraint,
            parent_state,
            target_logit_rows[-1],
            logit_mask,
            vocab_size=vocab_size,
            select_token=select_token,
        )
        child_state = _read_transition_child(transition)
        if child_state is not None:
            if child_state is parent_state:
                raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                    "grammar bonus child aliases the committed parent"
                )
            owned_child = child_state
        selection = _read_transition_selection(transition)
        selected_token_id = _read_selection_token_for_ownership(selection)
        if child_state is None and selected_token_id is not None:
            if child_state is parent_state:
                raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                    "grammar bonus child aliases the committed parent"
                )
            owned_child = child_state
        child_is_match = _read_attribute(
            transition,
            "child_is_match",
            label="transition result",
        )
        if (
            child_state is None
            and selected_token_id is None
            and child_is_match is not None
        ):
            if child_state is parent_state:
                raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                    "grammar bonus child aliases the committed parent"
                )
            owned_child = child_state
        transitioned = _read_transitioned(transition)

        valid_token_ids, selection_is_match, selected_token_id = (
            _validate_selection_evidence(selection, vocab_size=vocab_size)
        )
        if selection_is_match is not parent_is_match:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "final-row selection parent-match evidence changed from D45"
            )

        if selected_token_id is None:
            _validate_no_transition_evidence(
                transition,
                selection=selection,
                valid_token_ids=valid_token_ids,
                child_state=child_state,
                child_is_match=child_is_match,
                transitioned=transitioned,
            )
            if owned_child is not _NO_STATE:
                raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                    "a no-transition result cannot transfer a bonus child"
                )
            result = GrammarMaskedPostAcceptanceContinuationResult(
                output_token_ids=proposal_token_ids,
                uncached_next_token_id=None,
                final_row_no_decision_selection=cast(
                    GrammarMaskedSelectionResult, selection
                ),
                committed_state=parent_state,
                committed_state_is_match=parent_is_match,
            )
            _validate_composed_result(
                result,
                output_token_ids=proposal_token_ids,
                uncached_next_token_id=None,
                final_row_no_decision_selection=cast(
                    GrammarMaskedSelectionResult, selection
                ),
                committed_state=parent_state,
                committed_state_is_match=parent_is_match,
            )
            owned_parent = _NO_STATE
            return result

        _validate_transitioned_evidence(
            transition,
            selection=selection,
            valid_token_ids=valid_token_ids,
            child_state=child_state,
            child_is_match=child_is_match,
            transitioned=transitioned,
        )
        bonus = selected_token_id
        output_token_ids = proposal_token_ids + (bonus,)
        result = GrammarMaskedPostAcceptanceContinuationResult(
            output_token_ids=output_token_ids,
            uncached_next_token_id=bonus,
            final_row_no_decision_selection=None,
            committed_state=cast(StateT, child_state),
            committed_state_is_match=cast(bool, child_is_match),
        )
        _validate_composed_result(
            result,
            output_token_ids=output_token_ids,
            uncached_next_token_id=bonus,
            final_row_no_decision_selection=None,
            committed_state=child_state,
            committed_state_is_match=cast(bool, child_is_match),
        )

        constraint.release_state(parent_state)
        owned_parent = _NO_STATE
        _validate_retained_child(
            constraint,
            cast(StateT, child_state),
            expected_is_match=cast(bool, child_is_match),
        )
        if owned_child is _NO_STATE or owned_child is not child_state:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "bonus-child ownership is inconsistent at transfer"
            )
        owned_child = _NO_STATE
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_continuation(
            constraint,
            owned_parent=owned_parent,
            owned_child=owned_child,
        )
        if cleanup_failures:
            raise GrammarMaskedPostAcceptanceContinuationCleanupError(
                failure,
                cleanup_failures,
            ) from failure
        raise


def _validate_continuation_inputs(
    proposal_token_ids: object,
    target_logit_rows: object,
    acceptance_result: object,
    *,
    vocab_size: object,
    select_token: object,
) -> tuple[int, int | None, StateT, bool]:
    if type(vocab_size) is not int:
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")

    if type(proposal_token_ids) is not tuple:
        raise TypeError("proposal_token_ids must be a tuple")
    proposal = cast(tuple[object, ...], proposal_token_ids)
    if not proposal:
        raise ValueError("proposal_token_ids cannot be empty")
    for position, token_id in enumerate(proposal):
        _validate_token_in_vocabulary(
            token_id,
            vocab_size,
            label=f"proposal token at position {position}",
        )

    if type(target_logit_rows) is not tuple:
        raise TypeError("target_logit_rows must be a tuple")
    expected_row_count = len(proposal) + 1
    actual_row_count = len(target_logit_rows)
    if actual_row_count != expected_row_count:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            f"target_logit_rows contains {actual_row_count} rows; expected "
            f"{expected_row_count} for proposal length {len(proposal)}"
        )

    if not isinstance(acceptance_result, GrammarMaskedTargetAcceptanceResult):
        raise TypeError(
            "acceptance_result must be a GrammarMaskedTargetAcceptanceResult"
        )
    evidence_proposal = _read_attribute(
        acceptance_result,
        "proposal_token_ids",
        label="acceptance result",
    )
    accepted_count = _read_attribute(
        acceptance_result,
        "accepted_count",
        label="acceptance result",
    )
    replacement_token_id = _read_attribute(
        acceptance_result,
        "replacement_token_id",
        label="acceptance result",
    )
    no_decision_selection = _read_attribute(
        acceptance_result,
        "no_decision_selection",
        label="acceptance result",
    )
    committed_state = _read_attribute(
        acceptance_result,
        "committed_state",
        label="acceptance result",
    )
    committed_state_is_match = _read_attribute(
        acceptance_result,
        "committed_state_is_match",
        label="acceptance result",
    )

    if type(evidence_proposal) is not tuple:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "acceptance result proposal_token_ids must be an exact tuple"
        )
    for position, token_id in enumerate(evidence_proposal):
        _validate_acceptance_token_evidence(
            token_id,
            vocab_size,
            label=f"acceptance result proposal token at position {position}",
        )
    if evidence_proposal != proposal:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "acceptance result proposal does not match proposal_token_ids"
        )

    if type(accepted_count) is not int:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "acceptance result accepted_count must be an integer"
        )
    if accepted_count < 0 or accepted_count > len(proposal):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            f"acceptance result accepted_count must be within [0, {len(proposal)}]"
        )

    if accepted_count == len(proposal):
        if replacement_token_id is not None:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "fully accepted evidence cannot contain a replacement token"
            )
    else:
        if replacement_token_id is None:
            if no_decision_selection is None:
                raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                    "decided mismatch evidence must contain a replacement token"
                )
        else:
            _validate_acceptance_token_evidence(
                replacement_token_id,
                vocab_size,
                label="acceptance result replacement_token_id",
            )
            if replacement_token_id == proposal[accepted_count]:
                raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                    "acceptance result replacement_token_id must differ from the "
                    "rejected proposal token"
                )

    if no_decision_selection is not None:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "D45 no-decision outcomes are outside D46"
        )
    if type(committed_state_is_match) is not bool:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "a decided acceptance result requires a boolean committed-state match fact"
        )
    if not callable(select_token):
        raise TypeError("select_token must be callable")

    return (
        accepted_count,
        cast(int | None, replacement_token_id),
        cast(StateT, committed_state),
        committed_state_is_match,
    )


def _require_transition_result(transition: object) -> None:
    if not isinstance(transition, GrammarMaskedTransitionResult):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "D43 must return a GrammarMaskedTransitionResult"
        )


def _read_transition_child(transition: object) -> object:
    _require_transition_result(transition)
    return _read_attribute(
        transition,
        "child_state",
        label="transition result",
    )


def _read_transition_selection(transition: object) -> object:
    _require_transition_result(transition)
    return _read_attribute(transition, "selection", label="transition result")


def _read_transitioned(transition: object) -> bool:
    transitioned = _read_attribute(transition, "transitioned", label="transition result")
    if type(transitioned) is not bool:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition result transitioned must be a boolean"
        )
    return transitioned


def _read_selection_token_for_ownership(selection: object) -> object:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition selection must be a GrammarMaskedSelectionResult"
        )
    return _read_attribute(
        selection,
        "selected_token_id",
        label="transition selection",
    )


def _validate_selection_evidence(
    selection: object,
    *,
    vocab_size: int,
) -> tuple[tuple[int, ...], bool, int | None]:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition selection must be a GrammarMaskedSelectionResult"
        )
    valid_token_ids = _read_attribute(
        selection,
        "valid_token_ids",
        label="transition selection",
    )
    is_match = _read_attribute(
        selection,
        "is_match",
        label="transition selection",
    )
    selected_token_id = _read_attribute(
        selection,
        "selected_token_id",
        label="transition selection",
    )
    if type(valid_token_ids) is not tuple:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition selection valid_token_ids must be an exact tuple"
        )
    if type(is_match) is not bool:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition selection is_match must be a boolean"
        )

    previous = -1
    selected_found = False
    for token_id in valid_token_ids:
        if type(token_id) is not int:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "transition selection valid_token_ids must contain Python integers"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                f"transition selection token ID {token_id} is outside vocabulary "
                f"range [0, {vocab_size})"
            )
        if token_id <= previous:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "transition selection valid_token_ids must be strictly increasing and unique"
            )
        if selected_token_id is not None and token_id == selected_token_id:
            selected_found = True
        previous = token_id

    if not valid_token_ids:
        if selected_token_id is not None:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "empty transition support requires no selected token"
            )
    else:
        if type(selected_token_id) is not int:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "nonempty transition support requires an integer selected token"
            )
        if selected_token_id < 0 or selected_token_id >= vocab_size:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                f"selected bonus token ID {selected_token_id} is outside vocabulary "
                f"range [0, {vocab_size})"
            )
        if not selected_found:
            raise GrammarMaskedPostAcceptanceContinuationInvariantError(
                "selected bonus token must belong to valid_token_ids"
            )
    return valid_token_ids, is_match, cast(int | None, selected_token_id)


def _validate_no_transition_evidence(
    transition: object,
    *,
    selection: object,
    valid_token_ids: tuple[int, ...],
    child_state: object,
    child_is_match: object,
    transitioned: bool,
) -> None:
    if valid_token_ids:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "a no-transition result requires empty grammar support"
        )
    if child_state is not None or child_is_match is not None:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "a no-transition result requires both child fields to be None"
        )
    _validate_transition_identity(
        transition,
        selection=selection,
        child_state=None,
        child_is_match=None,
        transitioned=transitioned,
        expected_transitioned=False,
    )


def _validate_transitioned_evidence(
    transition: object,
    *,
    selection: object,
    valid_token_ids: tuple[int, ...],
    child_state: object,
    child_is_match: object,
    transitioned: bool,
) -> None:
    if not valid_token_ids:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "a transitioned result requires nonempty grammar support"
        )
    if type(child_is_match) is not bool:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "a transitioned result requires a boolean child_is_match"
        )
    _validate_transition_identity(
        transition,
        selection=selection,
        child_state=child_state,
        child_is_match=child_is_match,
        transitioned=transitioned,
        expected_transitioned=True,
    )


def _validate_transition_identity(
    transition: object,
    *,
    selection: object,
    child_state: object,
    child_is_match: object,
    transitioned: bool,
    expected_transitioned: bool,
) -> None:
    if _read_attribute(transition, "selection", label="transition result") is not selection:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition result must retain the exact selection"
        )
    if (
        _read_attribute(transition, "child_state", label="transition result")
        is not child_state
    ):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition result must retain the exact child state"
        )
    if (
        _read_attribute(transition, "child_is_match", label="transition result")
        is not child_is_match
    ):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition result must retain the exact child match fact"
        )
    if transitioned is not expected_transitioned:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition result reports inconsistent transition evidence"
        )
    if (
        _read_attribute(transition, "transitioned", label="transition result")
        is not expected_transitioned
    ):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "transition result changed its transitioned evidence"
        )


def _validate_retained_child(
    constraint: GrammarConstraint[StateT],
    child_state: StateT,
    *,
    expected_is_match: bool,
) -> None:
    is_dead = _require_state_boolean(
        constraint.is_dead_state(child_state),
        operation="is_dead_state",
    )
    if is_dead:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "bonus child must remain live"
        )
    is_match = _require_state_boolean(
        constraint.is_match_state(child_state),
        operation="is_match_state",
    )
    if is_match is not expected_is_match:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "bonus-child match status changed before transfer"
        )


def _validate_composed_result(
    result: object,
    *,
    output_token_ids: tuple[int, ...],
    uncached_next_token_id: int | None,
    final_row_no_decision_selection: GrammarMaskedSelectionResult | None,
    committed_state: object,
    committed_state_is_match: bool,
) -> None:
    if (
        _read_attribute(result, "output_token_ids", label="continuation result")
        is not output_token_ids
    ):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "continuation result must retain the exact output-token tuple"
        )
    result_uncached = _read_attribute(
        result,
        "uncached_next_token_id",
        label="continuation result",
    )
    if (
        result_uncached is not None and type(result_uncached) is not int
    ) or result_uncached != uncached_next_token_id:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "continuation result must retain the exact uncached token"
        )
    if (
        _read_attribute(
            result,
            "final_row_no_decision_selection",
            label="continuation result",
        )
        is not final_row_no_decision_selection
    ):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "continuation result must retain the exact final-row selection"
        )
    if (
        _read_attribute(result, "committed_state", label="continuation result")
        is not committed_state
    ):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "continuation result must retain the exact committed state"
        )
    if (
        _read_attribute(
            result,
            "committed_state_is_match",
            label="continuation result",
        )
        is not committed_state_is_match
    ):
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            "continuation result must retain the exact committed-state match fact"
        )


def _validate_no_decision_selection_for_result(selection: object) -> bool:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise TypeError(
            "final_row_no_decision_selection must be a "
            "GrammarMaskedSelectionResult or None"
        )
    try:
        valid_token_ids = selection.valid_token_ids
        is_match = selection.is_match
        selected_token_id = selection.selected_token_id
    except Exception as exc:
        raise TypeError(
            "final_row_no_decision_selection fields must be readable"
        ) from exc
    if type(valid_token_ids) is not tuple:
        raise TypeError(
            "final_row_no_decision_selection valid_token_ids must be an exact tuple"
        )
    if valid_token_ids:
        raise ValueError(
            "final_row_no_decision_selection must contain empty valid_token_ids"
        )
    if type(is_match) is not bool:
        raise TypeError("final_row_no_decision_selection is_match must be a boolean")
    if selected_token_id is not None:
        raise ValueError(
            "final_row_no_decision_selection must not contain a selected token"
        )
    return is_match


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


def _validate_token_in_vocabulary(
    token_id: object,
    vocab_size: int,
    *,
    label: str,
) -> None:
    if type(token_id) is not int:
        raise TypeError(f"{label} must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _validate_acceptance_token_evidence(
    token_id: object,
    vocab_size: int,
    *,
    label: str,
) -> None:
    if type(token_id) is not int:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            f"{label} must be an integer"
        )
    if token_id < 0 or token_id >= vocab_size:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _validate_nonnegative_token_id(token_id: object, *, label: str) -> None:
    if type(token_id) is not int:
        raise TypeError(f"{label} must be an integer")
    if token_id < 0:
        raise ValueError(f"{label} cannot be negative")


def _require_state_boolean(value: object, *, operation: str) -> bool:
    if type(value) is not bool:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedPostAcceptanceContinuationInvariantError(
            f"{label} {name} could not be read"
        ) from exc


def _cleanup_failed_continuation(
    constraint: GrammarConstraint[StateT],
    *,
    owned_parent: object,
    owned_child: object,
) -> tuple[tuple[str, Exception], ...]:
    cleanup_failures: list[tuple[str, Exception]] = []
    attempted_states: list[object] = []
    for label, state in (
        ("committed parent state release", owned_parent),
        ("bonus child state release", owned_child),
    ):
        if state is _NO_STATE or any(state is attempted for attempted in attempted_states):
            continue
        attempted_states.append(state)
        try:
            constraint.release_state(cast(StateT, state))
        except Exception as cleanup_failure:
            cleanup_failures.append((label, cleanup_failure))
    return tuple(cleanup_failures)


__all__ = [
    "GrammarMaskedPostAcceptanceContinuationCleanupError",
    "GrammarMaskedPostAcceptanceContinuationError",
    "GrammarMaskedPostAcceptanceContinuationInvariantError",
    "GrammarMaskedPostAcceptanceContinuationResult",
    "decide_grammar_masked_post_acceptance_continuation",
]
