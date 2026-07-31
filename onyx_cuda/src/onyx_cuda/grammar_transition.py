"""One grammar-masked selection and optional child-state transfer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .constrained_generation import GrammarLogitMask
from .grammar import GrammarConstraint, GrammarError
from .grammar_selection import (
    GrammarMaskedSelectionResult,
    select_grammar_masked_token,
)

__all__ = [
    "GrammarMaskedTransitionCleanupError",
    "GrammarMaskedTransitionError",
    "GrammarMaskedTransitionInvariantError",
    "GrammarMaskedTransitionResult",
    "select_and_advance_grammar_state",
]


class GrammarMaskedTransitionError(GrammarError):
    """Base error raised by one grammar-masked state transition."""


class GrammarMaskedTransitionInvariantError(GrammarMaskedTransitionError):
    """Raised when selection or state evidence violates the D43 contract."""


class GrammarMaskedTransitionCleanupError(GrammarMaskedTransitionError):
    """Raised when a failed transition also cannot release its owned child."""

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
            f"{operation} also failed: {failure}" for operation, failure in failures
        )
        super().__init__(
            f"grammar-masked transition failed: {original_failure}; {details}"
        )


StateT = TypeVar("StateT")
LogitsT = TypeVar("LogitsT")


@dataclass(frozen=True, slots=True)
class GrammarMaskedTransitionResult(Generic[StateT]):
    """One D42 selection and its optional transferred grammar child."""

    selection: GrammarMaskedSelectionResult
    child_state: StateT | None
    child_is_match: bool | None

    def __post_init__(self) -> None:
        valid_token_ids, _is_match, selected_token_id = (
            _validate_nested_selection_for_result(self.selection)
        )
        if selected_token_id is None:
            if valid_token_ids:
                raise ValueError("no selection requires empty valid_token_ids")
            if self.child_state is not None or self.child_is_match is not None:
                raise ValueError("no selection requires both child fields to be None")
            return
        if not valid_token_ids:
            raise ValueError("a selected token requires nonempty valid_token_ids")
        if type(self.child_is_match) is not bool:
            raise TypeError("a selected transition requires child_is_match to be a boolean")

    @property
    def transitioned(self) -> bool:
        """Whether the selection produced and transferred a child state."""

        return self.selection.selected_token_id is not None


_NO_CHILD = object()


def select_and_advance_grammar_state(
    constraint: GrammarConstraint[StateT],
    state: StateT,
    logits: LogitsT,
    logit_mask: GrammarLogitMask[LogitsT],
    *,
    vocab_size: int,
    select_token: Callable[[LogitsT], int],
) -> GrammarMaskedTransitionResult[StateT]:
    """Select once and transfer at most one validated independent child state."""

    selection = select_grammar_masked_token(
        constraint,
        state,
        logits,
        logit_mask,
        vocab_size=vocab_size,
        select_token=select_token,
    )
    valid_token_ids, selection_is_match, selected_token_id = (
        _validate_selection_evidence(selection, vocab_size=vocab_size)
    )
    _validate_parent_state(
        constraint,
        state,
        expected_is_match=selection_is_match,
    )

    if selected_token_id is None:
        result = GrammarMaskedTransitionResult(
            selection=selection,
            child_state=None,
            child_is_match=None,
        )
        _validate_composed_result(
            result,
            selection=selection,
            child_state=None,
            child_is_match=None,
            expected_transitioned=False,
        )
        return result

    if not valid_token_ids:
        raise GrammarMaskedTransitionInvariantError(
            "a selected token requires nonempty valid_token_ids"
        )

    owned_child: object = _NO_CHILD
    try:
        child_state = constraint.advance_state(state, selected_token_id)
        owned_child = child_state
        if child_state is state:
            owned_child = _NO_CHILD
            raise GrammarMaskedTransitionInvariantError(
                "grammar advancement must return an independent child state"
            )

        child_is_dead = _require_state_boolean(
            constraint.is_dead_state(child_state),
            operation="is_dead_state",
        )
        if child_is_dead:
            raise GrammarMaskedTransitionInvariantError(
                f"grammar-valid token ID {selected_token_id} advanced to a dead child state"
            )
        child_is_match = _require_state_boolean(
            constraint.is_match_state(child_state),
            operation="is_match_state",
        )
        _validate_parent_state(
            constraint,
            state,
            expected_is_match=selection_is_match,
        )

        result = GrammarMaskedTransitionResult(
            selection=selection,
            child_state=child_state,
            child_is_match=child_is_match,
        )
        _validate_composed_result(
            result,
            selection=selection,
            child_state=child_state,
            child_is_match=child_is_match,
            expected_transitioned=True,
        )
        owned_child = _NO_CHILD
        return result
    except BaseException as failure:
        if owned_child is _NO_CHILD:
            raise
        child_to_release = cast(StateT, owned_child)
        try:
            constraint.release_state(child_to_release)
        except Exception as cleanup_failure:
            raise GrammarMaskedTransitionCleanupError(
                failure,
                (("child state release", cleanup_failure),),
            ) from failure
        raise


def _validate_nested_selection_for_result(
    selection: object,
) -> tuple[tuple[int, ...], bool, int | None]:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise TypeError("selection must be a GrammarMaskedSelectionResult")
    try:
        valid_token_ids = selection.valid_token_ids
        is_match = selection.is_match
        selected_token_id = selection.selected_token_id
    except Exception as exc:
        raise TypeError("selection fields must be readable") from exc

    if type(valid_token_ids) is not tuple:
        raise TypeError("selection valid_token_ids must be an exact tuple")
    previous = -1
    selected_found = False
    for token_id in valid_token_ids:
        if type(token_id) is not int:
            raise TypeError("selection valid_token_ids must contain Python integers")
        if token_id < 0:
            raise ValueError("selection valid_token_ids cannot contain negative token IDs")
        if token_id <= previous:
            raise ValueError(
                "selection valid_token_ids must be strictly increasing and unique"
            )
        if selected_token_id is not None and token_id == selected_token_id:
            selected_found = True
        previous = token_id
    if type(is_match) is not bool:
        raise TypeError("selection is_match must be a boolean")
    if not valid_token_ids:
        if selected_token_id is not None:
            raise ValueError("empty selection support requires no selected token")
    else:
        if type(selected_token_id) is not int:
            raise TypeError(
                "nonempty selection support requires an integer selected token"
            )
        if selected_token_id < 0:
            raise ValueError("selection selected_token_id cannot be negative")
        if not selected_found:
            raise ValueError("selection selected_token_id must belong to valid_token_ids")
    return valid_token_ids, is_match, selected_token_id


def _validate_selection_evidence(
    selection: object,
    *,
    vocab_size: object,
) -> tuple[tuple[int, ...], bool, int | None]:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise GrammarMaskedTransitionInvariantError(
            "D42 must return a GrammarMaskedSelectionResult"
        )
    valid_token_ids = _read_attribute(
        selection,
        "valid_token_ids",
        label="selection",
    )
    is_match = _read_attribute(selection, "is_match", label="selection")
    selected_token_id = _read_attribute(
        selection,
        "selected_token_id",
        label="selection",
    )

    if type(vocab_size) is not int or vocab_size <= 0:
        raise GrammarMaskedTransitionInvariantError(
            "vocab_size evidence must be a positive integer"
        )
    if type(valid_token_ids) is not tuple:
        raise GrammarMaskedTransitionInvariantError(
            "selection valid_token_ids must be an exact tuple"
        )
    if type(is_match) is not bool:
        raise GrammarMaskedTransitionInvariantError(
            "selection is_match must be a boolean"
        )

    previous = -1
    selected_found = False
    for token_id in valid_token_ids:
        if type(token_id) is not int:
            raise GrammarMaskedTransitionInvariantError(
                "selection valid_token_ids must contain Python integers"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise GrammarMaskedTransitionInvariantError(
                f"selection token ID {token_id} is outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if token_id <= previous:
            raise GrammarMaskedTransitionInvariantError(
                "selection valid_token_ids must be strictly increasing and unique"
            )
        if selected_token_id is not None and token_id == selected_token_id:
            selected_found = True
        previous = token_id

    if not valid_token_ids:
        if selected_token_id is not None:
            raise GrammarMaskedTransitionInvariantError(
                "empty selection support requires no selected token"
            )
    else:
        if type(selected_token_id) is not int:
            raise GrammarMaskedTransitionInvariantError(
                "nonempty selection support requires an integer selected token"
            )
        if selected_token_id < 0 or selected_token_id >= vocab_size:
            raise GrammarMaskedTransitionInvariantError(
                f"selected token ID {selected_token_id} is outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if not selected_found:
            raise GrammarMaskedTransitionInvariantError(
                "selected token ID must belong to selection valid_token_ids"
            )
    return valid_token_ids, is_match, selected_token_id


def _validate_parent_state(
    constraint: GrammarConstraint[StateT],
    state: StateT,
    *,
    expected_is_match: bool,
) -> None:
    is_dead = _require_state_boolean(
        constraint.is_dead_state(state),
        operation="is_dead_state",
    )
    if is_dead:
        raise GrammarMaskedTransitionInvariantError(
            "borrowed parent state must remain live"
        )
    is_match = _require_state_boolean(
        constraint.is_match_state(state),
        operation="is_match_state",
    )
    if is_match is not expected_is_match:
        raise GrammarMaskedTransitionInvariantError(
            "borrowed parent match status changed after selection"
        )


def _require_state_boolean(value: object, *, operation: str) -> bool:
    if type(value) is not bool:
        raise GrammarMaskedTransitionInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedTransitionInvariantError(
            f"{label} {name} could not be read"
        ) from exc


def _validate_composed_result(
    result: object,
    *,
    selection: GrammarMaskedSelectionResult,
    child_state: StateT | None,
    child_is_match: bool | None,
    expected_transitioned: bool,
) -> None:
    result_selection = _read_attribute(result, "selection", label="transition result")
    result_child_state = _read_attribute(
        result,
        "child_state",
        label="transition result",
    )
    result_child_is_match = _read_attribute(
        result,
        "child_is_match",
        label="transition result",
    )
    transitioned = _read_attribute(
        result,
        "transitioned",
        label="transition result",
    )
    if result_selection is not selection:
        raise GrammarMaskedTransitionInvariantError(
            "transition result must retain the exact D42 selection"
        )
    if result_child_state is not child_state:
        raise GrammarMaskedTransitionInvariantError(
            "transition result must retain the exact child state"
        )
    if result_child_is_match is not child_is_match:
        raise GrammarMaskedTransitionInvariantError(
            "transition result must retain the exact child match flag"
        )
    if transitioned is not expected_transitioned:
        raise GrammarMaskedTransitionInvariantError(
            "transition result reports an inconsistent transitioned value"
        )
