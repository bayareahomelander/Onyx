"""Framework-neutral grammar-masked bounded draft-proposal orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .backend import BackendStateError, ModelStep
from .cache import CacheCheckpoint, CheckpointableAutoregressiveBackend
from .constrained_generation import GrammarLogitMask
from .draft import DraftProposalError
from .grammar import GrammarConstraint
from .grammar_selection import GrammarMaskedSelectionResult
from .grammar_transition import (
    GrammarMaskedTransitionResult,
    select_and_advance_grammar_state,
)


class GrammarMaskedDraftProposalError(DraftProposalError):
    """Base error raised by grammar-masked draft-proposal orchestration."""


class GrammarMaskedDraftProposalInvariantError(GrammarMaskedDraftProposalError):
    """Raised when draft, checkpoint, or grammar evidence violates the D44 contract."""


class GrammarMaskedDraftProposalCleanupError(GrammarMaskedDraftProposalError):
    """Raised when a failed grammar-masked proposal cannot settle all owned resources."""

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
            f"grammar-masked draft proposal failed: {original_failure}; {details}"
        )


LogitsT = TypeVar("LogitsT")
StateT = TypeVar("StateT")
CheckpointT = TypeVar("CheckpointT", bound=CacheCheckpoint)


@dataclass(frozen=True, slots=True)
class GrammarMaskedDraftProposalResult(Generic[CheckpointT]):
    """A bounded proposal and explicit evidence when grammar support shortened it."""

    proposal_token_ids: tuple[int, ...]
    rollback_checkpoints: tuple[CheckpointT, ...]
    initial_cache_length: int
    final_cache_length: int
    shortening_selection: GrammarMaskedSelectionResult | None

    def __post_init__(self) -> None:
        if type(self.proposal_token_ids) is not tuple:
            raise TypeError("proposal_token_ids must be a tuple")
        for position, token_id in enumerate(self.proposal_token_ids):
            _validate_nonnegative_token_id(
                token_id,
                label=f"proposal token at position {position}",
            )

        if type(self.rollback_checkpoints) is not tuple:
            raise TypeError("rollback_checkpoints must be a tuple")
        if len(self.rollback_checkpoints) != len(self.proposal_token_ids):
            raise ValueError(
                "rollback_checkpoints must contain exactly one checkpoint per proposal token"
            )

        initial_cache_length = _validate_cache_length_metadata(
            self.initial_cache_length,
            label="initial_cache_length",
        )
        if initial_cache_length == 0:
            raise GrammarMaskedDraftProposalInvariantError(
                "initial_cache_length must be greater than zero"
            )
        final_cache_length = _validate_cache_length_metadata(
            self.final_cache_length,
            label="final_cache_length",
        )
        expected_final_length = initial_cache_length + len(self.proposal_token_ids) + 1
        if final_cache_length != expected_final_length:
            raise GrammarMaskedDraftProposalInvariantError(
                f"final_cache_length is {final_cache_length}; "
                f"expected {expected_final_length}"
            )

        for position, checkpoint in enumerate(self.rollback_checkpoints):
            _validate_checkpoint(
                checkpoint,
                expected_cache_length=initial_cache_length + 1 + position,
                label=f"rollback checkpoint at position {position}",
            )

        if self.shortening_selection is None:
            if not self.proposal_token_ids:
                raise ValueError(
                    "an empty proposal requires shortening_selection"
                )
        else:
            valid_token_ids, _is_match, selected_token_id = (
                _validate_nested_selection_for_result(self.shortening_selection)
            )
            if valid_token_ids:
                raise ValueError(
                    "shortening_selection must contain empty valid_token_ids"
                )
            if selected_token_id is not None:
                raise ValueError(
                    "shortening_selection must not contain a selected token"
                )

    @property
    def shortened(self) -> bool:
        """Whether exact empty-support evidence shortened the requested proposal."""

        return self.shortening_selection is not None


@dataclass(slots=True)
class _OwnedCheckpoint:
    position: int
    checkpoint: object
    shortening_only: bool = False
    owned: bool = True


_OwnedState = tuple[int, StateT]


def generate_grammar_masked_draft_proposal(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    current_token_id: int,
    constraint: GrammarConstraint[StateT],
    starting_state: StateT,
    logit_mask: GrammarLogitMask[LogitsT],
    *,
    proposal_bound: int,
    select_token: Callable[[LogitsT], int],
) -> GrammarMaskedDraftProposalResult[CheckpointT]:
    """Generate up to ``proposal_bound`` grammar-masked draft tokens.

    ``starting_state`` is borrowed and must already represent the grammar state after the
    uncached current token. Returned checkpoints are transferred to the caller; every grammar
    child created for the draft branch is settled before this function returns.
    """

    initial_cache_length, vocab_size = _validate_proposal_inputs(
        backend,
        current_token_id,
        proposal_bound=proposal_bound,
        select_token=select_token,
    )

    start_checkpoint = backend.create_cache_checkpoint()
    start_checkpoint_usable = False
    start_checkpoint_owned = True
    inspected_checkpoints: list[_OwnedCheckpoint] = []
    owned_states: list[_OwnedState[StateT]] = []
    state_identity_history: list[StateT] = []

    try:
        _validate_checkpoint(
            start_checkpoint,
            expected_cache_length=initial_cache_length,
            label="start checkpoint",
        )
        start_checkpoint_usable = True

        expected_cache_length = initial_cache_length + 1
        step = backend.decode(current_token_id)
        logits = _validate_backend_step(backend, step, expected_cache_length)

        proposal_token_ids: list[int] = []
        current_parent = starting_state
        current_parent_is_owned = False
        first_parent_is_match: bool | None = None
        shortening_selection: GrammarMaskedSelectionResult | None = None

        for position in range(proposal_bound):
            checkpoint = backend.create_cache_checkpoint()
            checkpoint_record = _OwnedCheckpoint(position, checkpoint)
            inspected_checkpoints.append(checkpoint_record)
            _validate_checkpoint(
                checkpoint,
                expected_cache_length=expected_cache_length,
                label=f"rollback checkpoint at position {position}",
            )

            transition = select_and_advance_grammar_state(
                constraint,
                current_parent,
                logits,
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
                    raise GrammarMaskedDraftProposalInvariantError(
                        f"grammar child at position {position} aliases starting_state"
                    )
                if child_state is current_parent:
                    raise GrammarMaskedDraftProposalInvariantError(
                        f"grammar child at position {position} aliases its parent"
                    )
                if _contains_identity(state_identity_history, child_state):
                    raise GrammarMaskedDraftProposalInvariantError(
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

                proposal_token_ids.append(selected_token_id)
                expected_cache_length += 1
                step = backend.decode(selected_token_id)
                logits = _validate_backend_step(
                    backend,
                    step,
                    expected_cache_length,
                )

                if current_parent_is_owned:
                    _release_current_parent(constraint, owned_states)
                current_parent = child_state
                current_parent_is_owned = True
                continue

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
            checkpoint_record.shortening_only = True
            if first_parent_is_match is None:
                first_parent_is_match = selection_is_match
            shortening_selection = selection
            break

        produced_count = len(proposal_token_ids)
        final_cache_length = initial_cache_length + 1 + produced_count
        _validate_backend_cache_length(backend, final_cache_length)
        if shortening_selection is None:
            if produced_count != proposal_bound:
                raise GrammarMaskedDraftProposalInvariantError(
                    "a full-bound result must contain proposal_bound tokens"
                )
        elif produced_count >= proposal_bound:
            raise GrammarMaskedDraftProposalInvariantError(
                "a shortened result must contain fewer than proposal_bound tokens"
            )

        if first_parent_is_match is None:
            raise GrammarMaskedDraftProposalInvariantError(
                "the first inspected grammar row did not provide match evidence"
            )
        _validate_borrowed_start(
            constraint,
            starting_state,
            expected_is_match=first_parent_is_match,
        )

        proposal_tuple = tuple(proposal_token_ids)
        rollback_tuple = tuple(
            cast(CheckpointT, record.checkpoint)
            for record in inspected_checkpoints
            if not record.shortening_only
        )
        result = GrammarMaskedDraftProposalResult(
            proposal_token_ids=proposal_tuple,
            rollback_checkpoints=rollback_tuple,
            initial_cache_length=initial_cache_length,
            final_cache_length=final_cache_length,
            shortening_selection=shortening_selection,
        )
        _validate_composed_result(
            result,
            proposal_token_ids=proposal_tuple,
            rollback_checkpoints=rollback_tuple,
            initial_cache_length=initial_cache_length,
            final_cache_length=final_cache_length,
            shortening_selection=shortening_selection,
        )

        if current_parent_is_owned:
            _release_current_parent(constraint, owned_states)
            current_parent_is_owned = False
        _validate_borrowed_start(
            constraint,
            starting_state,
            expected_is_match=first_parent_is_match,
        )

        for record in inspected_checkpoints:
            if record.shortening_only and record.owned:
                backend.release_cache_checkpoint(
                    cast(CheckpointT, record.checkpoint)
                )
                record.owned = False

        backend.release_cache_checkpoint(start_checkpoint)
        start_checkpoint_owned = False

        for record in inspected_checkpoints:
            if not record.shortening_only:
                record.owned = False
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_proposal(
            backend,
            constraint,
            starting_state=starting_state,
            start_checkpoint=start_checkpoint,
            start_checkpoint_usable=start_checkpoint_usable,
            start_checkpoint_owned=start_checkpoint_owned,
            inspected_checkpoints=inspected_checkpoints,
            owned_states=owned_states,
        )
        if cleanup_failures:
            raise GrammarMaskedDraftProposalCleanupError(
                failure,
                cleanup_failures,
            ) from failure
        raise


def _validate_proposal_inputs(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    current_token_id: int,
    *,
    proposal_bound: int,
    select_token: Callable[[LogitsT], int],
) -> tuple[int, int]:
    if isinstance(proposal_bound, bool) or not isinstance(proposal_bound, int):
        raise TypeError("proposal_bound must be an integer")
    if proposal_bound <= 0:
        raise ValueError("proposal_bound must be greater than zero")
    if not callable(select_token):
        raise TypeError("select_token must be callable")
    try:
        backend_conforms = isinstance(backend, CheckpointableAutoregressiveBackend)
    except Exception as exc:
        raise GrammarMaskedDraftProposalInvariantError(
            "backend checkpoint capability could not be determined"
        ) from exc
    if not backend_conforms:
        raise TypeError("backend must satisfy CheckpointableAutoregressiveBackend")

    vocab_size = _read_attribute(backend, "vocab_size", label="backend")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise GrammarMaskedDraftProposalInvariantError(
            "backend vocab_size must be an integer"
        )
    if vocab_size <= 0:
        raise GrammarMaskedDraftProposalInvariantError(
            "backend vocab_size must be greater than zero"
        )

    _validate_token_id(current_token_id, vocab_size, label="current_token_id")

    initial_cache_length = _read_backend_cache_length(backend)
    if initial_cache_length == 0:
        raise BackendStateError(
            "prefill must establish an active cache before grammar-masked draft proposal"
        )
    return initial_cache_length, vocab_size


def _read_transition_disposition(
    transition: object,
) -> tuple[object, bool]:
    if not isinstance(transition, GrammarMaskedTransitionResult):
        raise GrammarMaskedDraftProposalInvariantError(
            "D43 must return a GrammarMaskedTransitionResult"
        )
    selection = _read_attribute(transition, "selection", label="transition result")
    transitioned = _read_attribute(
        transition,
        "transitioned",
        label="transition result",
    )
    if type(transitioned) is not bool:
        raise GrammarMaskedDraftProposalInvariantError(
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
        raise GrammarMaskedDraftProposalInvariantError(
            "a transitioned result requires nonempty grammar support"
        )
    if selected_token_id is None:
        raise GrammarMaskedDraftProposalInvariantError(
            "a transitioned result requires a selected token"
        )
    if type(child_is_match) is not bool:
        raise GrammarMaskedDraftProposalInvariantError(
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
        raise GrammarMaskedDraftProposalInvariantError(
            "a no-transition result requires empty grammar support"
        )
    if selected_token_id is not None:
        raise GrammarMaskedDraftProposalInvariantError(
            "a no-transition result cannot contain a selected token"
        )
    if child_state is not None or child_is_match is not None:
        raise GrammarMaskedDraftProposalInvariantError(
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
    vocab_size: int | None,
    label: str,
) -> tuple[tuple[int, ...], bool, int | None]:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} must be a GrammarMaskedSelectionResult"
        )
    valid_token_ids = _read_attribute(
        selection,
        "valid_token_ids",
        label=label,
    )
    is_match = _read_attribute(selection, "is_match", label=label)
    selected_token_id = _read_attribute(
        selection,
        "selected_token_id",
        label=label,
    )

    if type(valid_token_ids) is not tuple:
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} valid_token_ids must be an exact tuple"
        )
    if type(is_match) is not bool:
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} is_match must be a boolean"
        )

    previous = -1
    selected_found = False
    for token_id in valid_token_ids:
        if type(token_id) is not int:
            raise GrammarMaskedDraftProposalInvariantError(
                f"{label} valid_token_ids must contain Python integers"
            )
        if token_id < 0 or (vocab_size is not None and token_id >= vocab_size):
            if vocab_size is None:
                raise GrammarMaskedDraftProposalInvariantError(
                    f"{label} valid_token_ids cannot contain negative token IDs"
                )
            raise GrammarMaskedDraftProposalInvariantError(
                f"{label} token ID {token_id} is outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if token_id <= previous:
            raise GrammarMaskedDraftProposalInvariantError(
                f"{label} valid_token_ids must be strictly increasing and unique"
            )
        if selected_token_id is not None and token_id == selected_token_id:
            selected_found = True
        previous = token_id

    if not valid_token_ids:
        if selected_token_id is not None:
            raise GrammarMaskedDraftProposalInvariantError(
                f"{label} empty support requires no selected token"
            )
    else:
        if type(selected_token_id) is not int:
            raise GrammarMaskedDraftProposalInvariantError(
                f"{label} nonempty support requires an integer selected token"
            )
        if selected_token_id < 0 or (
            vocab_size is not None and selected_token_id >= vocab_size
        ):
            if vocab_size is None:
                raise GrammarMaskedDraftProposalInvariantError(
                    f"{label} selected_token_id cannot be negative"
                )
            raise GrammarMaskedDraftProposalInvariantError(
                f"{label} selected token ID {selected_token_id} is outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if not selected_found:
            raise GrammarMaskedDraftProposalInvariantError(
                f"{label} selected token must belong to valid_token_ids"
            )
    return valid_token_ids, is_match, selected_token_id


def _validate_nested_selection_for_result(
    selection: object,
) -> tuple[tuple[int, ...], bool, int | None]:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise TypeError(
            "shortening_selection must be a GrammarMaskedSelectionResult"
        )
    try:
        valid_token_ids = selection.valid_token_ids
        is_match = selection.is_match
        selected_token_id = selection.selected_token_id
    except Exception as exc:
        raise TypeError("shortening_selection fields must be readable") from exc

    if type(valid_token_ids) is not tuple:
        raise TypeError(
            "shortening_selection valid_token_ids must be an exact tuple"
        )
    previous = -1
    selected_found = False
    for token_id in valid_token_ids:
        if type(token_id) is not int:
            raise TypeError(
                "shortening_selection valid_token_ids must contain Python integers"
            )
        if token_id < 0:
            raise ValueError(
                "shortening_selection valid_token_ids cannot contain negative token IDs"
            )
        if token_id <= previous:
            raise ValueError(
                "shortening_selection valid_token_ids must be strictly increasing and unique"
            )
        if selected_token_id is not None and token_id == selected_token_id:
            selected_found = True
        previous = token_id
    if type(is_match) is not bool:
        raise TypeError("shortening_selection is_match must be a boolean")
    if not valid_token_ids:
        if selected_token_id is not None:
            raise ValueError(
                "shortening_selection empty support requires no selected token"
            )
    else:
        if type(selected_token_id) is not int:
            raise TypeError(
                "shortening_selection nonempty support requires an integer selected token"
            )
        if selected_token_id < 0:
            raise ValueError(
                "shortening_selection selected_token_id cannot be negative"
            )
        if not selected_found:
            raise ValueError(
                "shortening_selection selected token must belong to valid_token_ids"
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
        raise GrammarMaskedDraftProposalInvariantError(
            "transition result must retain the exact selection"
        )
    if _read_attribute(transition, "child_state", label="transition result") is not child_state:
        raise GrammarMaskedDraftProposalInvariantError(
            "transition result must retain the exact child state"
        )
    if (
        _read_attribute(transition, "child_is_match", label="transition result")
        is not child_is_match
    ):
        raise GrammarMaskedDraftProposalInvariantError(
            "transition result must retain the exact child match flag"
        )
    if (
        _read_attribute(transition, "transitioned", label="transition result")
        is not expected_transitioned
    ):
        raise GrammarMaskedDraftProposalInvariantError(
            "transition result reports inconsistent transition evidence"
        )


def _release_current_parent(
    constraint: GrammarConstraint[StateT],
    owned_states: list[_OwnedState[StateT]],
) -> None:
    if not owned_states:
        raise GrammarMaskedDraftProposalInvariantError(
            "grammar-state ownership is inconsistent"
        )
    _position, state = owned_states[0]
    constraint.release_state(state)
    owned_states.pop(0)


def _validate_borrowed_start(
    constraint: GrammarConstraint[StateT],
    starting_state: StateT,
    *,
    expected_is_match: bool,
) -> None:
    is_dead = _require_state_boolean(
        constraint.is_dead_state(starting_state),
        operation="is_dead_state",
    )
    if is_dead:
        raise GrammarMaskedDraftProposalInvariantError(
            "starting_state must remain live"
        )
    is_match = _require_state_boolean(
        constraint.is_match_state(starting_state),
        operation="is_match_state",
    )
    if is_match is not expected_is_match:
        raise GrammarMaskedDraftProposalInvariantError(
            "starting_state match status changed during draft proposal"
        )


def _validate_composed_result(
    result: object,
    *,
    proposal_token_ids: tuple[int, ...],
    rollback_checkpoints: tuple[CheckpointT, ...],
    initial_cache_length: int,
    final_cache_length: int,
    shortening_selection: GrammarMaskedSelectionResult | None,
) -> None:
    if (
        _read_attribute(result, "proposal_token_ids", label="proposal result")
        is not proposal_token_ids
    ):
        raise GrammarMaskedDraftProposalInvariantError(
            "proposal result must retain the exact proposal-token tuple"
        )
    if (
        _read_attribute(result, "rollback_checkpoints", label="proposal result")
        is not rollback_checkpoints
    ):
        raise GrammarMaskedDraftProposalInvariantError(
            "proposal result must retain the exact checkpoint tuple"
        )
    if (
        _read_attribute(result, "initial_cache_length", label="proposal result")
        is not initial_cache_length
    ):
        raise GrammarMaskedDraftProposalInvariantError(
            "proposal result must retain the exact initial cache length"
        )
    if (
        _read_attribute(result, "final_cache_length", label="proposal result")
        is not final_cache_length
    ):
        raise GrammarMaskedDraftProposalInvariantError(
            "proposal result must retain the exact final cache length"
        )
    if (
        _read_attribute(result, "shortening_selection", label="proposal result")
        is not shortening_selection
    ):
        raise GrammarMaskedDraftProposalInvariantError(
            "proposal result must retain the exact shortening selection"
        )


def _validate_backend_step(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    step: ModelStep[LogitsT],
    expected_cache_length: int,
) -> LogitsT:
    if not isinstance(step, ModelStep):
        raise GrammarMaskedDraftProposalInvariantError(
            "backend decode must return a ModelStep"
        )
    reported_cache_length = _validate_cache_length_metadata(
        _read_attribute(step, "cache_length", label="backend step"),
        label="backend step cache_length",
    )
    if reported_cache_length != expected_cache_length:
        raise GrammarMaskedDraftProposalInvariantError(
            f"backend step reported cache length {reported_cache_length}; "
            f"expected {expected_cache_length}"
        )
    _validate_backend_cache_length(backend, expected_cache_length)
    return cast(LogitsT, _read_attribute(step, "logits", label="backend step"))


def _validate_backend_cache_length(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    expected_cache_length: int,
) -> int:
    cache_length = _read_backend_cache_length(backend)
    if cache_length != expected_cache_length:
        raise GrammarMaskedDraftProposalInvariantError(
            f"backend state reported cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )
    return cache_length


def _read_backend_cache_length(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
) -> int:
    return _validate_cache_length_metadata(
        _read_attribute(backend, "cache_length", label="backend"),
        label="backend cache_length",
    )


def _validate_cache_length_metadata(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} must be an integer"
        )
    if value < 0:
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} cannot be negative"
        )
    return value


def _validate_checkpoint(
    checkpoint: object,
    *,
    expected_cache_length: int,
    label: str,
) -> None:
    try:
        is_checkpoint = isinstance(checkpoint, CacheCheckpoint)
    except Exception as exc:
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        ) from exc
    if not is_checkpoint:
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        )

    cache_length = _read_attribute(checkpoint, "cache_length", label=label)
    cache_length = _validate_cache_length_metadata(
        cache_length,
        label=f"{label} cache_length",
    )
    if cache_length != expected_cache_length:
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} reports cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )


def _validate_token_id(token_id: object, vocab_size: int, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _validate_nonnegative_token_id(token_id: object, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0:
        raise ValueError(f"{label} cannot be negative")


def _require_state_boolean(value: object, *, operation: str) -> bool:
    if type(value) is not bool:
        raise GrammarMaskedDraftProposalInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedDraftProposalInvariantError(
            f"{label} {name} could not be read"
        ) from exc


def _contains_identity(values: Sequence[StateT], candidate: StateT) -> bool:
    return any(value is candidate for value in values)


def _cleanup_failed_proposal(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    constraint: GrammarConstraint[StateT],
    *,
    starting_state: StateT,
    start_checkpoint: object,
    start_checkpoint_usable: bool,
    start_checkpoint_owned: bool,
    inspected_checkpoints: Sequence[_OwnedCheckpoint],
    owned_states: Sequence[_OwnedState[StateT]],
) -> tuple[tuple[str, Exception], ...]:
    cleanup_failures: list[tuple[str, Exception]] = []

    if start_checkpoint_usable:
        try:
            backend.rollback_cache(cast(CheckpointT, start_checkpoint))
        except Exception as cleanup_failure:
            cleanup_failures.append(("start checkpoint rollback", cleanup_failure))

    released_checkpoint_objects: list[object] = []
    for record in inspected_checkpoints:
        if not record.owned or _contains_identity(
            released_checkpoint_objects,
            record.checkpoint,
        ):
            continue
        released_checkpoint_objects.append(record.checkpoint)
        operation = (
            f"shortening checkpoint {record.position} release"
            if record.shortening_only
            else f"rollback checkpoint {record.position} release"
        )
        try:
            backend.release_cache_checkpoint(cast(CheckpointT, record.checkpoint))
        except Exception as cleanup_failure:
            cleanup_failures.append((operation, cleanup_failure))

    if start_checkpoint_owned and not _contains_identity(
        released_checkpoint_objects,
        start_checkpoint,
    ):
        try:
            backend.release_cache_checkpoint(cast(CheckpointT, start_checkpoint))
        except Exception as cleanup_failure:
            cleanup_failures.append(("start checkpoint release", cleanup_failure))

    attempted_states: list[StateT] = []
    for position, state in owned_states:
        if state is starting_state or _contains_identity(attempted_states, state):
            continue
        attempted_states.append(state)
        try:
            constraint.release_state(state)
        except Exception as cleanup_failure:
            cleanup_failures.append(
                (f"grammar state release at position {position}", cleanup_failure)
            )

    return tuple(cleanup_failures)


__all__ = [
    "GrammarMaskedDraftProposalCleanupError",
    "GrammarMaskedDraftProposalError",
    "GrammarMaskedDraftProposalInvariantError",
    "GrammarMaskedDraftProposalResult",
    "generate_grammar_masked_draft_proposal",
]
