"""Framework-neutral grammar-state reconciliation for one speculative iteration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .grammar import GrammarConstraint, GrammarError
from .speculative_iteration import ContinuationAwareSpeculativeIterationResult


class SpeculativeGrammarReconciliationError(GrammarError):
    """Base error raised by one speculative grammar-state reconciliation."""


class SpeculativeGrammarReconciliationInvariantError(
    SpeculativeGrammarReconciliationError
):
    """Raised when grammar or iteration evidence violates the D41 contract."""


class SpeculativeGrammarReconciliationCleanupError(
    SpeculativeGrammarReconciliationError
):
    """Raised when reconciliation fails and owned states cannot all be released."""

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
            f"speculative grammar reconciliation failed: {original_failure}; {details}"
        )


StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class SpeculativeGrammarReconciliationResult(Generic[StateT]):
    """The final committed grammar state transferred to the caller."""

    committed_state: StateT
    is_match: bool

    def __post_init__(self) -> None:
        if type(self.is_match) is not bool:
            raise TypeError("is_match must be a boolean")


_OwnedState = tuple[int, StateT]


def reconcile_speculative_grammar_state(
    constraint: GrammarConstraint[StateT],
    starting_state: StateT,
    iteration_result: ContinuationAwareSpeculativeIterationResult,
    *,
    vocab_size: int,
) -> SpeculativeGrammarReconciliationResult[StateT]:
    """Replay draft and committed branches and transfer the final committed state."""

    validated_vocab_size = _validate_vocab_size(vocab_size)
    _validate_constraint(constraint, vocab_size=validated_vocab_size)
    proposal_token_ids, output_token_ids = _validate_iteration_result(
        iteration_result,
        vocab_size=validated_vocab_size,
    )
    starting_is_dead = _require_state_boolean(
        constraint.is_dead_state(starting_state),
        operation="is_dead_state",
    )
    if starting_is_dead:
        raise SpeculativeGrammarReconciliationInvariantError(
            "starting_state must not be dead"
        )
    starting_is_match = _require_state_boolean(
        constraint.is_match_state(starting_state),
        operation="is_match_state",
    )

    owned_draft_states: list[_OwnedState[StateT]] = []
    owned_committed_states: list[_OwnedState[StateT]] = []

    try:
        draft_parent = starting_state
        for position, token_id in enumerate(proposal_token_ids):
            draft_parent = _acquire_child(
                constraint,
                parent=draft_parent,
                token_id=token_id,
                position=position,
                branch="draft",
                starting_state=starting_state,
                owned_draft_states=owned_draft_states,
                owned_committed_states=owned_committed_states,
            )

        committed_parent = starting_state
        for position, token_id in enumerate(output_token_ids):
            committed_parent = _acquire_child(
                constraint,
                parent=committed_parent,
                token_id=token_id,
                position=position,
                branch="committed",
                starting_state=starting_state,
                owned_draft_states=owned_draft_states,
                owned_committed_states=owned_committed_states,
            )

        final_committed_state = owned_committed_states[-1][1]
        _validate_retained_state(
            constraint,
            starting_state,
            expected_is_dead=starting_is_dead,
            expected_is_match=starting_is_match,
            label="starting_state",
        )
        final_is_dead = _require_state_boolean(
            constraint.is_dead_state(final_committed_state),
            operation="is_dead_state",
        )
        if final_is_dead:
            raise SpeculativeGrammarReconciliationInvariantError(
                "final committed state must not be dead"
            )
        final_is_match = _require_state_boolean(
            constraint.is_match_state(final_committed_state),
            operation="is_match_state",
        )
        _validate_branch_independence(
            owned_draft_states,
            owned_committed_states,
        )

        _release_owned_prefix(
            constraint,
            owned_draft_states,
            count=len(owned_draft_states),
        )
        _release_owned_prefix(
            constraint,
            owned_committed_states,
            count=len(owned_committed_states) - 1,
        )

        _validate_retained_state(
            constraint,
            starting_state,
            expected_is_dead=starting_is_dead,
            expected_is_match=starting_is_match,
            label="starting_state",
        )
        _validate_retained_state(
            constraint,
            final_committed_state,
            expected_is_dead=False,
            expected_is_match=final_is_match,
            label="final committed state",
        )

        result = SpeculativeGrammarReconciliationResult(
            committed_state=final_committed_state,
            is_match=final_is_match,
        )
        try:
            result_state = result.committed_state
            result_is_match = result.is_match
        except Exception as exc:
            raise SpeculativeGrammarReconciliationInvariantError(
                "reconciliation result fields must be readable"
            ) from exc
        if result_state is not final_committed_state:
            raise SpeculativeGrammarReconciliationInvariantError(
                "reconciliation result must retain the exact final committed state"
            )
        if type(result_is_match) is not bool or result_is_match is not final_is_match:
            raise SpeculativeGrammarReconciliationInvariantError(
                "reconciliation result must retain the exact final match flag"
            )

        if (
            owned_draft_states
            or len(owned_committed_states) != 1
            or owned_committed_states[0][0] != len(output_token_ids) - 1
            or owned_committed_states[0][1] is not final_committed_state
        ):
            raise SpeculativeGrammarReconciliationInvariantError(
                "reconciliation ownership transfer is inconsistent"
            )
        owned_committed_states.pop()
        return result
    except BaseException as failure:
        if not owned_draft_states and not owned_committed_states:
            raise
        cleanup_failures = _cleanup_owned_states(
            constraint,
            starting_state=starting_state,
            owned_draft_states=owned_draft_states,
            owned_committed_states=owned_committed_states,
        )
        if cleanup_failures:
            raise SpeculativeGrammarReconciliationCleanupError(
                failure,
                cleanup_failures,
            ) from failure
        raise


def _validate_vocab_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("vocab_size must be an integer")
    if value <= 0:
        raise ValueError("vocab_size must be greater than zero")
    return value


def _validate_constraint(
    constraint: object,
    *,
    vocab_size: int,
) -> None:
    try:
        conforms = isinstance(constraint, GrammarConstraint)
    except Exception as exc:
        raise SpeculativeGrammarReconciliationInvariantError(
            "constraint runtime conformance could not be determined"
        ) from exc
    if not conforms:
        raise TypeError("constraint must satisfy GrammarConstraint")

    constraint_vocab_size = _read_attribute(
        constraint,
        "vocab_size",
        label="constraint",
    )
    if (
        isinstance(constraint_vocab_size, bool)
        or not isinstance(constraint_vocab_size, int)
        or constraint_vocab_size <= 0
    ):
        raise SpeculativeGrammarReconciliationInvariantError(
            "constraint vocab_size must be a positive integer"
        )
    if constraint_vocab_size != vocab_size:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"constraint vocab_size is {constraint_vocab_size}; expected {vocab_size}"
        )

    grammar_type = _read_attribute(
        constraint,
        "grammar_type",
        label="constraint",
    )
    if type(grammar_type) is not str or grammar_type not in {
        "regex",
        "json_schema",
    }:
        raise SpeculativeGrammarReconciliationInvariantError(
            "constraint grammar_type must be 'regex' or 'json_schema'"
        )


def _validate_iteration_result(
    result: object,
    *,
    vocab_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(result, ContinuationAwareSpeculativeIterationResult):
        raise SpeculativeGrammarReconciliationInvariantError(
            "iteration_result must be a ContinuationAwareSpeculativeIterationResult"
        )

    proposal_token_ids = _read_attribute(
        result,
        "proposal_token_ids",
        label="iteration_result",
    )
    accepted_count = _read_attribute(
        result,
        "accepted_count",
        label="iteration_result",
    )
    replacement_token_id = _read_attribute(
        result,
        "replacement_token_id",
        label="iteration_result",
    )
    initial_cache_length = _read_attribute(
        result,
        "initial_cache_length",
        label="iteration_result",
    )
    final_cache_length = _read_attribute(
        result,
        "final_cache_length",
        label="iteration_result",
    )
    uncached_next_token_id = _read_attribute(
        result,
        "uncached_next_token_id",
        label="iteration_result",
    )
    output_token_ids = _read_attribute(
        result,
        "output_token_ids",
        label="iteration_result",
    )

    if type(proposal_token_ids) is not tuple or not proposal_token_ids:
        raise SpeculativeGrammarReconciliationInvariantError(
            "iteration_result proposal_token_ids must be an exact nonempty tuple"
        )
    for position, token_id in enumerate(proposal_token_ids):
        _validate_result_token(
            token_id,
            vocab_size=vocab_size,
            label=f"proposal token at position {position}",
        )

    if isinstance(accepted_count, bool) or not isinstance(accepted_count, int):
        raise SpeculativeGrammarReconciliationInvariantError(
            "iteration_result accepted_count must be an integer"
        )
    proposal_length = len(proposal_token_ids)
    if accepted_count < 0 or accepted_count > proposal_length:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"iteration_result accepted_count must be within [0, {proposal_length}]"
        )

    if accepted_count == proposal_length:
        if replacement_token_id is not None:
            raise SpeculativeGrammarReconciliationInvariantError(
                "fully accepted iteration_result cannot contain a replacement token"
            )
    else:
        _validate_result_token(
            replacement_token_id,
            vocab_size=vocab_size,
            label="replacement_token_id",
        )
        if replacement_token_id == proposal_token_ids[accepted_count]:
            raise SpeculativeGrammarReconciliationInvariantError(
                "replacement_token_id must differ from the rejected proposal token"
            )

    initial_length = _validate_result_length(
        initial_cache_length,
        label="initial_cache_length",
    )
    final_length = _validate_result_length(
        final_cache_length,
        label="final_cache_length",
    )
    expected_final_length = initial_length + accepted_count + 1
    if final_length != expected_final_length:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"iteration_result final_cache_length is {final_length}; "
            f"expected {expected_final_length}"
        )

    _validate_result_token(
        uncached_next_token_id,
        vocab_size=vocab_size,
        label="uncached_next_token_id",
    )
    if (
        accepted_count < proposal_length
        and uncached_next_token_id != replacement_token_id
    ):
        raise SpeculativeGrammarReconciliationInvariantError(
            "mismatch uncached_next_token_id must equal replacement_token_id"
        )

    if type(output_token_ids) is not tuple or not output_token_ids:
        raise SpeculativeGrammarReconciliationInvariantError(
            "iteration_result output_token_ids must be an exact nonempty tuple"
        )
    for position, token_id in enumerate(output_token_ids):
        _validate_result_token(
            token_id,
            vocab_size=vocab_size,
            label=f"output token at position {position}",
        )
    expected_output = proposal_token_ids[:accepted_count] + (
        cast(int, uncached_next_token_id),
    )
    if output_token_ids != expected_output:
        raise SpeculativeGrammarReconciliationInvariantError(
            "iteration_result output does not match its accepted prefix and uncached token"
        )
    if output_token_ids[-1] != uncached_next_token_id:
        raise SpeculativeGrammarReconciliationInvariantError(
            "iteration_result uncached token must equal the final output token"
        )

    return proposal_token_ids, output_token_ids


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{label} {name} could not be read"
        ) from exc


def _validate_result_token(
    token_id: object,
    *,
    vocab_size: int,
    label: str,
) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{label} must be an integer"
        )
    if token_id < 0 or token_id >= vocab_size:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _validate_result_length(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeculativeGrammarReconciliationInvariantError(
            f"iteration_result {label} must be an integer"
        )
    if value <= 0:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"iteration_result {label} must be greater than zero"
        )
    return value


def _acquire_child(
    constraint: GrammarConstraint[StateT],
    *,
    parent: StateT,
    token_id: int,
    position: int,
    branch: str,
    starting_state: StateT,
    owned_draft_states: list[_OwnedState[StateT]],
    owned_committed_states: list[_OwnedState[StateT]],
) -> StateT:
    child = constraint.advance_state(parent, token_id)

    if child is starting_state:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{branch} child at position {position} aliases starting_state"
        )
    if child is parent:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{branch} child at position {position} aliases its parent"
        )
    if _contains_state_identity(owned_draft_states, child):
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{branch} child at position {position} aliases an owned draft state"
        )
    if _contains_state_identity(owned_committed_states, child):
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{branch} child at position {position} aliases an owned committed state"
        )

    owned_states = (
        owned_draft_states if branch == "draft" else owned_committed_states
    )
    owned_states.append((position, child))
    is_dead = _require_state_boolean(
        constraint.is_dead_state(child),
        operation="is_dead_state",
    )
    if branch == "committed" and is_dead:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"committed state at position {position} must not be dead"
        )
    return child


def _contains_state_identity(
    owned_states: Sequence[_OwnedState[StateT]],
    candidate: StateT,
) -> bool:
    return any(state is candidate for _position, state in owned_states)


def _require_state_boolean(value: object, *, operation: str) -> bool:
    if type(value) is not bool:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _validate_retained_state(
    constraint: GrammarConstraint[StateT],
    state: StateT,
    *,
    expected_is_dead: bool,
    expected_is_match: bool,
    label: str,
) -> None:
    is_dead = _require_state_boolean(
        constraint.is_dead_state(state),
        operation="is_dead_state",
    )
    if is_dead is not expected_is_dead:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{label} dead status changed during reconciliation"
        )
    is_match = _require_state_boolean(
        constraint.is_match_state(state),
        operation="is_match_state",
    )
    if is_match is not expected_is_match:
        raise SpeculativeGrammarReconciliationInvariantError(
            f"{label} match status changed during reconciliation"
        )


def _validate_branch_independence(
    owned_draft_states: Sequence[_OwnedState[StateT]],
    owned_committed_states: Sequence[_OwnedState[StateT]],
) -> None:
    for _draft_position, draft_state in owned_draft_states:
        if _contains_state_identity(owned_committed_states, draft_state):
            raise SpeculativeGrammarReconciliationInvariantError(
                "draft and committed branches must contain independent states"
            )


def _release_owned_prefix(
    constraint: GrammarConstraint[StateT],
    owned_states: list[_OwnedState[StateT]],
    *,
    count: int,
) -> None:
    for _ in range(count):
        _position, state = owned_states[0]
        constraint.release_state(state)
        owned_states.pop(0)


def _cleanup_owned_states(
    constraint: GrammarConstraint[StateT],
    *,
    starting_state: StateT,
    owned_draft_states: Sequence[_OwnedState[StateT]],
    owned_committed_states: Sequence[_OwnedState[StateT]],
) -> tuple[tuple[str, Exception], ...]:
    cleanup_failures: list[tuple[str, Exception]] = []
    attempted_states: list[StateT] = []

    for branch, owned_states in (
        ("draft", owned_draft_states),
        ("committed", owned_committed_states),
    ):
        for position, state in owned_states:
            if state is starting_state or any(
                state is attempted_state for attempted_state in attempted_states
            ):
                continue
            attempted_states.append(state)
            try:
                constraint.release_state(state)
            except Exception as cleanup_failure:
                cleanup_failures.append(
                    (
                        f"{branch} state release at position {position}",
                        cleanup_failure,
                    )
                )

    return tuple(cleanup_failures)


__all__ = [
    "SpeculativeGrammarReconciliationCleanupError",
    "SpeculativeGrammarReconciliationError",
    "SpeculativeGrammarReconciliationInvariantError",
    "SpeculativeGrammarReconciliationResult",
    "reconcile_speculative_grammar_state",
]
