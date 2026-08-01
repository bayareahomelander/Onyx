"""Framework-neutral grammar-masked target match/replace acceptance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .acceptance import MatchReplaceAcceptanceError
from .constrained_generation import GrammarLogitMask
from .grammar import GrammarConstraint
from .grammar_selection import GrammarMaskedSelectionResult
from .grammar_transition import (
    GrammarMaskedTransitionResult,
    select_and_advance_grammar_state,
)


class GrammarMaskedTargetAcceptanceError(MatchReplaceAcceptanceError):
    """Base error raised by grammar-masked target acceptance."""


class GrammarMaskedTargetAcceptanceInvariantError(GrammarMaskedTargetAcceptanceError):
    """Raised when target-selection or grammar-state evidence violates the D45 contract."""


class GrammarMaskedTargetAcceptanceCleanupError(GrammarMaskedTargetAcceptanceError):
    """Raised when failed target acceptance cannot release all owned grammar children."""

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
            f"grammar-masked target acceptance failed: {original_failure}; {details}"
        )


StateT = TypeVar("StateT")
LogitsT = TypeVar("LogitsT")


@dataclass(frozen=True, slots=True)
class GrammarMaskedTargetAcceptanceResult(Generic[StateT]):
    """One grammar-masked accepted prefix, replacement, or no-decision outcome."""

    proposal_token_ids: tuple[int, ...]
    accepted_count: int
    replacement_token_id: int | None
    no_decision_selection: GrammarMaskedSelectionResult | None
    committed_state: StateT | None
    committed_state_is_match: bool | None

    def __post_init__(self) -> None:
        _validate_result_proposal(self.proposal_token_ids)

        if type(self.accepted_count) is not int:
            raise TypeError("accepted_count must be an integer")
        proposal_length = len(self.proposal_token_ids)
        if self.accepted_count < 0 or self.accepted_count > proposal_length:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"accepted_count must be within [0, {proposal_length}]"
            )

        no_decision_is_match: bool | None = None
        if self.no_decision_selection is not None:
            no_decision_is_match = _validate_no_decision_selection_for_result(
                self.no_decision_selection
            )
            if self.accepted_count == proposal_length:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "a no-decision result must stop before full acceptance"
                )
            if self.replacement_token_id is not None:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "a no-decision result cannot contain a replacement token"
                )
        elif self.accepted_count == proposal_length:
            if self.replacement_token_id is not None:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "fully accepted result cannot contain a replacement token"
                )
        else:
            if self.replacement_token_id is None:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "a decided partial result must contain a replacement token"
                )
            _validate_nonnegative_token_id(
                self.replacement_token_id,
                label="replacement_token_id",
            )
            if self.replacement_token_id == self.proposal_token_ids[self.accepted_count]:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "replacement_token_id must differ from the rejected proposal token"
                )

        if self.committed_state_is_match is not None and type(
            self.committed_state_is_match
        ) is not bool:
            raise TypeError("committed_state_is_match must be a boolean or None")

        if self.decision_made or self.accepted_count > 0:
            if type(self.committed_state_is_match) is not bool:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "a transferred committed state requires a boolean match fact"
                )
            if (
                not self.decision_made
                and self.committed_state_is_match is not no_decision_is_match
            ):
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "a no-decision committed-state match fact must equal its parent fact"
                )
        elif self.committed_state is not None or self.committed_state_is_match is not None:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                "position-zero no-decision cannot transfer or retain a grammar state"
            )

    @property
    def decision_made(self) -> bool:
        """Whether a mismatch or full-acceptance decision was made."""

        return self.no_decision_selection is None

    @property
    def fully_accepted(self) -> bool:
        """Whether the decided outcome accepted every proposal token."""

        return self.decision_made and self.accepted_count == len(self.proposal_token_ids)

    @property
    def accepted_token_ids(self) -> tuple[int, ...]:
        """Return the exact accepted proposal prefix."""

        return self.proposal_token_ids[: self.accepted_count]

    @property
    def committed_token_ids(self) -> tuple[int, ...]:
        """Return the tokens represented by the committed target branch."""

        if self.replacement_token_id is None:
            return self.accepted_token_ids
        return self.accepted_token_ids + (self.replacement_token_id,)

    @property
    def committed_state_transferred(self) -> bool:
        """Whether ownership of one committed child transferred to the caller."""

        return self.decision_made or self.accepted_count > 0


_OwnedState = tuple[int, StateT]


def decide_grammar_masked_target_acceptance(
    proposal_token_ids: tuple[int, ...],
    target_logit_rows: tuple[LogitsT, ...],
    constraint: GrammarConstraint[StateT],
    starting_state: StateT,
    logit_mask: GrammarLogitMask[LogitsT],
    *,
    vocab_size: int,
    select_token: Callable[[LogitsT], int],
) -> GrammarMaskedTargetAcceptanceResult[StateT]:
    """Grammar-mask proposal-aligned target rows and transfer the committed child."""

    _validate_acceptance_inputs(
        proposal_token_ids,
        target_logit_rows,
        vocab_size=vocab_size,
        select_token=select_token,
    )

    current_parent = starting_state
    current_parent_is_owned = False
    current_parent_is_match: bool | None = None
    first_parent_is_match: bool | None = None
    accepted_count = 0
    replacement_token_id: int | None = None
    no_decision_selection: GrammarMaskedSelectionResult | None = None
    owned_states: list[_OwnedState[StateT]] = []
    state_identity_history: list[StateT] = []

    try:
        for position, proposal_token_id in enumerate(proposal_token_ids):
            transition = select_and_advance_grammar_state(
                constraint,
                current_parent,
                target_logit_rows[position],
                logit_mask,
                vocab_size=vocab_size,
                select_token=select_token,
            )
            selection, transitioned = _read_transition_disposition(transition)

            if transitioned:
                child_state = _read_attribute(
                    transition,
                    "child_state",
                    label="transition result",
                )
                if child_state is starting_state:
                    raise GrammarMaskedTargetAcceptanceInvariantError(
                        f"grammar child at position {position} aliases starting_state"
                    )
                if child_state is current_parent:
                    raise GrammarMaskedTargetAcceptanceInvariantError(
                        f"grammar child at position {position} aliases its parent"
                    )
                if _contains_identity(state_identity_history, child_state):
                    raise GrammarMaskedTargetAcceptanceInvariantError(
                        f"grammar child at position {position} aliases an earlier child"
                    )

                owned_states.append((position, child_state))
                state_identity_history.append(child_state)
                child_is_match = _read_attribute(
                    transition,
                    "child_is_match",
                    label="transition result",
                )
                selected_token_id, selection_is_match = _validate_transitioned_evidence(
                    transition,
                    selection=selection,
                    child_state=child_state,
                    child_is_match=child_is_match,
                    vocab_size=vocab_size,
                )
                if first_parent_is_match is None:
                    first_parent_is_match = selection_is_match

                if current_parent_is_owned:
                    _release_superseded_parent(
                        constraint,
                        owned_states,
                        current_parent=current_parent,
                        new_child=child_state,
                    )

                current_parent = cast(StateT, child_state)
                current_parent_is_owned = True
                current_parent_is_match = cast(bool, child_is_match)
                if selected_token_id == proposal_token_id:
                    accepted_count += 1
                    continue

                replacement_token_id = selected_token_id
                break

            child_state = _read_attribute(
                transition,
                "child_state",
                label="transition result",
            )
            child_is_match = _read_attribute(
                transition,
                "child_is_match",
                label="transition result",
            )
            selection_is_match = _validate_no_transition_evidence(
                transition,
                selection=selection,
                child_state=child_state,
                child_is_match=child_is_match,
                vocab_size=vocab_size,
            )
            if first_parent_is_match is None:
                first_parent_is_match = selection_is_match
            no_decision_selection = cast(GrammarMaskedSelectionResult, selection)
            if current_parent_is_owned:
                current_parent_is_match = selection_is_match
            break

        if first_parent_is_match is None:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                "the first inspected target row did not provide match evidence"
            )

        transfer_required = no_decision_selection is None or accepted_count > 0
        if transfer_required and not current_parent_is_owned:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                "the decided target branch did not retain a committed child"
            )
        if not transfer_required and current_parent_is_owned:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                "position-zero no-decision unexpectedly owns a child"
            )

        _validate_retained_state(
            constraint,
            starting_state,
            expected_is_match=first_parent_is_match,
            label="starting_state",
        )
        if transfer_required:
            if type(current_parent_is_match) is not bool:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "the committed child is missing an exact match fact"
                )
            _validate_retained_state(
                constraint,
                current_parent,
                expected_is_match=current_parent_is_match,
                label="committed_state",
            )

        result = GrammarMaskedTargetAcceptanceResult(
            proposal_token_ids=proposal_token_ids,
            accepted_count=accepted_count,
            replacement_token_id=replacement_token_id,
            no_decision_selection=no_decision_selection,
            committed_state=current_parent if transfer_required else None,
            committed_state_is_match=(
                current_parent_is_match if transfer_required else None
            ),
        )
        _validate_composed_result(
            result,
            proposal_token_ids=proposal_token_ids,
            accepted_count=accepted_count,
            replacement_token_id=replacement_token_id,
            no_decision_selection=no_decision_selection,
            committed_state=current_parent if transfer_required else None,
            committed_state_is_match=(
                current_parent_is_match if transfer_required else None
            ),
            transfer_required=transfer_required,
        )

        _validate_retained_state(
            constraint,
            starting_state,
            expected_is_match=first_parent_is_match,
            label="starting_state",
        )
        if transfer_required:
            _validate_retained_state(
                constraint,
                current_parent,
                expected_is_match=cast(bool, current_parent_is_match),
                label="committed_state",
            )

        if transfer_required:
            if len(owned_states) != 1 or owned_states[0][1] is not current_parent:
                raise GrammarMaskedTargetAcceptanceInvariantError(
                    "target grammar-state ownership is inconsistent at transfer"
                )
            owned_states.pop()
        elif owned_states:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                "position-zero no-decision retained an owned child"
            )
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_acceptance(
            constraint,
            starting_state=starting_state,
            owned_states=owned_states,
        )
        if cleanup_failures:
            raise GrammarMaskedTargetAcceptanceCleanupError(
                failure,
                cleanup_failures,
            ) from failure
        raise


def _validate_acceptance_inputs(
    proposal_token_ids: object,
    target_logit_rows: object,
    *,
    vocab_size: object,
    select_token: object,
) -> None:
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
        _validate_token_id(
            token_id,
            vocab_size,
            label=f"proposal token at position {position}",
        )

    if type(target_logit_rows) is not tuple:
        raise TypeError("target_logit_rows must be a tuple")
    actual_row_count = len(target_logit_rows)
    expected_row_count = len(proposal) + 1
    if actual_row_count != expected_row_count:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"target_logit_rows contains {actual_row_count} rows; expected "
            f"{expected_row_count} for proposal length {len(proposal)}"
        )
    if not callable(select_token):
        raise TypeError("select_token must be callable")


def _read_transition_disposition(transition: object) -> tuple[object, bool]:
    if not isinstance(transition, GrammarMaskedTransitionResult):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "D43 must return a GrammarMaskedTransitionResult"
        )
    selection = _read_attribute(transition, "selection", label="transition result")
    transitioned = _read_attribute(
        transition,
        "transitioned",
        label="transition result",
    )
    if type(transitioned) is not bool:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "transition result transitioned must be a boolean"
        )
    return selection, transitioned


def _validate_transitioned_evidence(
    transition: object,
    *,
    selection: object,
    child_state: object,
    child_is_match: object,
    vocab_size: int,
) -> tuple[int, bool]:
    valid_token_ids, selection_is_match, selected_token_id = _validate_selection_evidence(
        selection,
        vocab_size=vocab_size,
        label="transition selection",
    )
    if not valid_token_ids:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "a transitioned result requires nonempty grammar support"
        )
    if selected_token_id is None:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "a transitioned result requires a selected token"
        )
    if type(child_is_match) is not bool:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "a transitioned result requires a boolean child_is_match"
        )
    _validate_transition_identity(
        transition,
        selection=selection,
        child_state=child_state,
        child_is_match=child_is_match,
        expected_transitioned=True,
    )
    return selected_token_id, selection_is_match


def _validate_no_transition_evidence(
    transition: object,
    *,
    selection: object,
    child_state: object,
    child_is_match: object,
    vocab_size: int,
) -> bool:
    valid_token_ids, selection_is_match, selected_token_id = _validate_selection_evidence(
        selection,
        vocab_size=vocab_size,
        label="transition selection",
    )
    if valid_token_ids:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "a no-transition result requires empty grammar support"
        )
    if selected_token_id is not None:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "a no-transition result cannot contain a selected token"
        )
    if child_state is not None or child_is_match is not None:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "a no-transition result requires both child fields to be None"
        )
    _validate_transition_identity(
        transition,
        selection=selection,
        child_state=None,
        child_is_match=None,
        expected_transitioned=False,
    )
    return selection_is_match


def _validate_selection_evidence(
    selection: object,
    *,
    vocab_size: int,
    label: str,
) -> tuple[tuple[int, ...], bool, int | None]:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"{label} must be a GrammarMaskedSelectionResult"
        )
    valid_token_ids = _read_attribute(selection, "valid_token_ids", label=label)
    is_match = _read_attribute(selection, "is_match", label=label)
    selected_token_id = _read_attribute(selection, "selected_token_id", label=label)

    if type(valid_token_ids) is not tuple:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"{label} valid_token_ids must be an exact tuple"
        )
    if type(is_match) is not bool:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"{label} is_match must be a boolean"
        )

    previous = -1
    selected_found = False
    for token_id in valid_token_ids:
        if type(token_id) is not int:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"{label} valid_token_ids must contain Python integers"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"{label} token ID {token_id} is outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if token_id <= previous:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"{label} valid_token_ids must be strictly increasing and unique"
            )
        if selected_token_id is not None and token_id == selected_token_id:
            selected_found = True
        previous = token_id

    if not valid_token_ids:
        if selected_token_id is not None:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"{label} empty support requires no selected token"
            )
    else:
        if type(selected_token_id) is not int:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"{label} nonempty support requires an integer selected token"
            )
        if selected_token_id < 0 or selected_token_id >= vocab_size:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"{label} selected token ID {selected_token_id} is outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if not selected_found:
            raise GrammarMaskedTargetAcceptanceInvariantError(
                f"{label} selected token must belong to valid_token_ids"
            )
    return valid_token_ids, is_match, selected_token_id


def _validate_transition_identity(
    transition: object,
    *,
    selection: object,
    child_state: object,
    child_is_match: object,
    expected_transitioned: bool,
) -> None:
    if _read_attribute(transition, "selection", label="transition result") is not selection:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "transition result must retain the exact selection"
        )
    if _read_attribute(transition, "child_state", label="transition result") is not child_state:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "transition result must retain the exact child state"
        )
    if (
        _read_attribute(transition, "child_is_match", label="transition result")
        is not child_is_match
    ):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "transition result must retain the exact child match flag"
        )
    if (
        _read_attribute(transition, "transitioned", label="transition result")
        is not expected_transitioned
    ):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "transition result reports inconsistent transition evidence"
        )


def _release_superseded_parent(
    constraint: GrammarConstraint[StateT],
    owned_states: list[_OwnedState[StateT]],
    *,
    current_parent: StateT,
    new_child: object,
) -> None:
    if (
        len(owned_states) != 2
        or owned_states[0][1] is not current_parent
        or owned_states[1][1] is not new_child
    ):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "target grammar-state ownership is inconsistent during ancestor settlement"
        )
    constraint.release_state(current_parent)
    owned_states.pop(0)


def _validate_retained_state(
    constraint: GrammarConstraint[StateT],
    state: StateT,
    *,
    expected_is_match: bool,
    label: str,
) -> None:
    is_dead = _require_state_boolean(
        constraint.is_dead_state(state),
        operation="is_dead_state",
    )
    if is_dead:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"{label} must remain live"
        )
    is_match = _require_state_boolean(
        constraint.is_match_state(state),
        operation="is_match_state",
    )
    if is_match is not expected_is_match:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"{label} match status changed during target acceptance"
        )


def _validate_composed_result(
    result: object,
    *,
    proposal_token_ids: tuple[int, ...],
    accepted_count: int,
    replacement_token_id: int | None,
    no_decision_selection: GrammarMaskedSelectionResult | None,
    committed_state: object,
    committed_state_is_match: bool | None,
    transfer_required: bool,
) -> None:
    if (
        _read_attribute(result, "proposal_token_ids", label="acceptance result")
        is not proposal_token_ids
    ):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result must retain the exact proposal-token tuple"
        )
    result_accepted_count = _read_attribute(
        result,
        "accepted_count",
        label="acceptance result",
    )
    if type(result_accepted_count) is not int or result_accepted_count != accepted_count:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result must retain the exact accepted count"
        )
    result_replacement_token_id = _read_attribute(
        result,
        "replacement_token_id",
        label="acceptance result",
    )
    if (
        result_replacement_token_id is not None
        and type(result_replacement_token_id) is not int
    ) or result_replacement_token_id != replacement_token_id:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result must retain the exact replacement token"
        )
    if (
        _read_attribute(
            result,
            "no_decision_selection",
            label="acceptance result",
        )
        is not no_decision_selection
    ):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result must retain the exact no-decision selection"
        )
    if (
        _read_attribute(result, "committed_state", label="acceptance result")
        is not committed_state
    ):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result must retain the exact committed state"
        )
    if (
        _read_attribute(
            result,
            "committed_state_is_match",
            label="acceptance result",
        )
        is not committed_state_is_match
    ):
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result must retain the exact committed-state match fact"
        )

    decision_made = _read_attribute(result, "decision_made", label="acceptance result")
    fully_accepted = _read_attribute(result, "fully_accepted", label="acceptance result")
    accepted_token_ids = _read_attribute(
        result,
        "accepted_token_ids",
        label="acceptance result",
    )
    committed_token_ids = _read_attribute(
        result,
        "committed_token_ids",
        label="acceptance result",
    )
    committed_state_transferred = _read_attribute(
        result,
        "committed_state_transferred",
        label="acceptance result",
    )
    expected_decision_made = no_decision_selection is None
    expected_fully_accepted = (
        expected_decision_made and accepted_count == len(proposal_token_ids)
    )
    expected_accepted_tokens = proposal_token_ids[:accepted_count]
    expected_committed_tokens = expected_accepted_tokens
    if replacement_token_id is not None:
        expected_committed_tokens += (replacement_token_id,)

    if decision_made is not expected_decision_made:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result reports an inconsistent decision_made value"
        )
    if fully_accepted is not expected_fully_accepted:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result reports an inconsistent fully_accepted value"
        )
    if type(accepted_token_ids) is not tuple or accepted_token_ids != expected_accepted_tokens:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result reports inconsistent accepted_token_ids"
        )
    if type(committed_token_ids) is not tuple or committed_token_ids != expected_committed_tokens:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result reports inconsistent committed_token_ids"
        )
    if committed_state_transferred is not transfer_required:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            "acceptance result reports an inconsistent state-transfer value"
        )


def _validate_no_decision_selection_for_result(selection: object) -> bool:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise TypeError(
            "no_decision_selection must be a GrammarMaskedSelectionResult or None"
        )
    try:
        valid_token_ids = selection.valid_token_ids
        is_match = selection.is_match
        selected_token_id = selection.selected_token_id
    except Exception as exc:
        raise TypeError("no_decision_selection fields must be readable") from exc
    if type(valid_token_ids) is not tuple:
        raise TypeError("no_decision_selection valid_token_ids must be an exact tuple")
    if valid_token_ids:
        raise ValueError("no_decision_selection must contain empty valid_token_ids")
    if type(is_match) is not bool:
        raise TypeError("no_decision_selection is_match must be a boolean")
    if selected_token_id is not None:
        raise ValueError("no_decision_selection must not contain a selected token")
    return is_match


def _validate_result_proposal(proposal_token_ids: object) -> None:
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


def _validate_token_id(token_id: object, vocab_size: int, *, label: str) -> None:
    if type(token_id) is not int:
        raise TypeError(f"{label} must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _validate_nonnegative_token_id(token_id: object, *, label: str) -> None:
    if type(token_id) is not int:
        raise TypeError(f"{label} must be an integer")
    if token_id < 0:
        raise ValueError(f"{label} cannot be negative")


def _require_state_boolean(value: object, *, operation: str) -> bool:
    if type(value) is not bool:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedTargetAcceptanceInvariantError(
            f"{label} {name} could not be read"
        ) from exc


def _contains_identity(values: Sequence[StateT], candidate: StateT) -> bool:
    return any(value is candidate for value in values)


def _cleanup_failed_acceptance(
    constraint: GrammarConstraint[StateT],
    *,
    starting_state: StateT,
    owned_states: Sequence[_OwnedState[StateT]],
) -> tuple[tuple[str, Exception], ...]:
    cleanup_failures: list[tuple[str, Exception]] = []
    attempted_states: list[StateT] = []
    for position, state in owned_states:
        if state is starting_state or _contains_identity(attempted_states, state):
            continue
        attempted_states.append(state)
        try:
            constraint.release_state(state)
        except Exception as cleanup_failure:
            cleanup_failures.append(
                (f"target state release at position {position}", cleanup_failure)
            )
    return tuple(cleanup_failures)


__all__ = [
    "GrammarMaskedTargetAcceptanceCleanupError",
    "GrammarMaskedTargetAcceptanceError",
    "GrammarMaskedTargetAcceptanceInvariantError",
    "GrammarMaskedTargetAcceptanceResult",
    "decide_grammar_masked_target_acceptance",
]
