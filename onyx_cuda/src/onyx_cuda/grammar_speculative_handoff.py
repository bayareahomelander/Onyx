"""Framework-neutral bounded routing across grammar-masked transactions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .cache import CacheCheckpoint, CheckpointableAutoregressiveBackend
from .constrained_generation import GrammarLogitMask
from .grammar import GrammarConstraint
from .grammar_speculative_iteration import (
    GrammarMaskedSpeculativeIterationResult,
    coordinate_grammar_masked_speculative_iteration,
)
from .grammar_speculative_outcome import (
    GrammarMaskedSpeculativeOutcomeError,
    GrammarMaskedSpeculativeOutcomeResult,
    classify_grammar_masked_speculative_outcome,
)


class GrammarMaskedSpeculativeHandoffError(GrammarMaskedSpeculativeOutcomeError):
    """Base error raised by one bounded grammar-masked handoff."""


class GrammarMaskedSpeculativeHandoffInvariantError(
    GrammarMaskedSpeculativeHandoffError
):
    """Raised when completed transaction or handoff evidence is inconsistent."""


class GrammarMaskedSpeculativeHandoffCleanupError(
    GrammarMaskedSpeculativeHandoffError
):
    """Raised when a failed handoff cannot settle every owned resource."""

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
            f"grammar-masked speculative handoff failed: {original_failure}; {details}"
        )


DraftLogitsT = TypeVar("DraftLogitsT")
DraftCheckpointT = TypeVar("DraftCheckpointT", bound=CacheCheckpoint)
TargetLogitsT = TypeVar("TargetLogitsT")
TargetCheckpointT = TypeVar("TargetCheckpointT", bound=CacheCheckpoint)
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class GrammarMaskedSpeculativeHandoffResult(Generic[StateT]):
    """Exact bounded output and final D47/D48 evidence."""

    output_token_ids: tuple[int, ...]
    final_iteration: GrammarMaskedSpeculativeIterationResult[StateT]
    final_outcome: GrammarMaskedSpeculativeOutcomeResult

    def __post_init__(self) -> None:
        output_token_ids = _validate_result_output(self.output_token_ids)
        final_iteration = _require_iteration_result(
            self.final_iteration,
            label="final_iteration",
            error_type=TypeError,
        )
        final_outcome = _require_outcome_result(
            self.final_outcome,
            label="final_outcome",
            error_type=TypeError,
        )
        final_output = _read_result_output(
            final_iteration,
            label="final_iteration",
            vocab_size=None,
        )
        if len(final_output) > len(output_token_ids) or (
            final_output
            and output_token_ids[-len(final_output) :] != final_output
        ):
            raise GrammarMaskedSpeculativeHandoffInvariantError(
                "final_iteration output must be an exact suffix of output_token_ids"
            )
        _validate_outcome_relationship(
            final_iteration,
            final_outcome,
            label="final",
        )


def coordinate_grammar_masked_speculative_handoff(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT, DraftCheckpointT
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT, TargetCheckpointT
    ],
    current_token_id: int,
    constraint: GrammarConstraint[StateT],
    starting_state: StateT,
    draft_logit_mask: GrammarLogitMask[DraftLogitsT],
    target_logit_mask: GrammarLogitMask[TargetLogitsT],
    *,
    proposal_bound: int,
    draft_select_token: Callable[[DraftLogitsT], int],
    target_select_token: Callable[[TargetLogitsT], int],
    draft_root_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
) -> GrammarMaskedSpeculativeHandoffResult[StateT]:
    """Run one D47 transaction and at most one classified handoff transaction."""

    initial_cache_length = _validate_initial_root_metadata(
        draft_root_checkpoint,
        target_root_checkpoint,
    )

    first_iteration = coordinate_grammar_masked_speculative_iteration(
        draft_backend,
        target_backend,
        current_token_id,
        constraint,
        starting_state,
        draft_logit_mask,
        target_logit_mask,
        proposal_bound=proposal_bound,
        draft_select_token=draft_select_token,
        target_select_token=target_select_token,
        draft_root_checkpoint=draft_root_checkpoint,
        target_root_checkpoint=target_root_checkpoint,
    )

    draft_intermediate: object = None
    target_intermediate: object = None
    draft_intermediate_owned = False
    target_intermediate_owned = False
    owned_state: object = None
    owned_state_acquired = False
    owned_state_is_match = False

    try:
        first_iteration = _require_iteration_result(
            first_iteration,
            label="first D47 result",
            error_type=GrammarMaskedSpeculativeHandoffInvariantError,
        )
        owned_state = _read_attribute(
            first_iteration,
            "committed_state",
            label="first D47 result",
        )
        owned_state_acquired = True
        owned_state_is_match = _read_state_match_fact(
            first_iteration,
            label="first D47 result",
        )

        first_outcome = classify_grammar_masked_speculative_outcome(first_iteration)
        first_outcome = _require_outcome_result(
            first_outcome,
            label="first D48 result",
            error_type=GrammarMaskedSpeculativeHandoffInvariantError,
        )
        _validate_outcome_relationship(
            first_iteration,
            first_outcome,
            label="first",
        )

        vocab_size = _read_common_vocab_size(
            draft_backend,
            target_backend,
            constraint,
        )
        first_output, first_final_length = _validate_completed_iteration(
            draft_backend,
            target_backend,
            constraint,
            first_iteration,
            expected_initial_cache_length=initial_cache_length,
            vocab_size=vocab_size,
            acquired_state=owned_state,
            acquired_state_is_match=owned_state_is_match,
            label="first D47 result",
        )

        first_kind = _read_outcome_kind(first_outcome, label="first D48 result")
        if first_kind != "handoff_available":
            result = GrammarMaskedSpeculativeHandoffResult(
                output_token_ids=first_output,
                final_iteration=first_iteration,
                final_outcome=first_outcome,
            )
            _validate_composed_result(
                result,
                output_token_ids=first_output,
                final_iteration=first_iteration,
                final_outcome=first_outcome,
            )
            _validate_backend_pair_cache_length(
                draft_backend,
                target_backend,
                first_final_length,
            )
            _validate_live_state(
                constraint,
                cast(StateT, owned_state),
                expected_is_match=owned_state_is_match,
                label="final committed_state",
            )
            owned_state_acquired = False
            return result

        handoff_token_id = _read_optional_token(
            first_iteration,
            "uncached_next_token_id",
            vocab_size=vocab_size,
            label="first D47 result",
        )
        if handoff_token_id is None:
            raise GrammarMaskedSpeculativeHandoffInvariantError(
                "first handoff outcome requires one uncached token"
            )
        if not first_output or first_output[-1] != handoff_token_id:
            raise GrammarMaskedSpeculativeHandoffInvariantError(
                "first handoff token must be the final first output token"
            )

        draft_intermediate = draft_backend.create_cache_checkpoint()
        draft_intermediate_owned = True
        _validate_intermediate_checkpoint(
            draft_intermediate,
            expected_cache_length=first_final_length,
            label="draft intermediate checkpoint",
        )
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            first_final_length,
        )

        target_intermediate = target_backend.create_cache_checkpoint()
        target_intermediate_owned = True
        _validate_intermediate_checkpoint(
            target_intermediate,
            expected_cache_length=first_final_length,
            label="target intermediate checkpoint",
        )
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            first_final_length,
        )

        second_iteration = coordinate_grammar_masked_speculative_iteration(
            draft_backend,
            target_backend,
            handoff_token_id,
            constraint,
            cast(StateT, owned_state),
            draft_logit_mask,
            target_logit_mask,
            proposal_bound=proposal_bound,
            draft_select_token=draft_select_token,
            target_select_token=target_select_token,
            draft_root_checkpoint=cast(DraftCheckpointT, draft_intermediate),
            target_root_checkpoint=cast(TargetCheckpointT, target_intermediate),
        )

        second_iteration = _require_iteration_result(
            second_iteration,
            label="second D47 result",
            error_type=GrammarMaskedSpeculativeHandoffInvariantError,
        )
        owned_state = _read_attribute(
            second_iteration,
            "committed_state",
            label="second D47 result",
        )
        owned_state_acquired = True
        owned_state_is_match = _read_state_match_fact(
            second_iteration,
            label="second D47 result",
        )

        second_outcome = classify_grammar_masked_speculative_outcome(second_iteration)
        second_outcome = _require_outcome_result(
            second_outcome,
            label="second D48 result",
            error_type=GrammarMaskedSpeculativeHandoffInvariantError,
        )
        _validate_outcome_relationship(
            second_iteration,
            second_outcome,
            label="second",
        )

        second_output, second_final_length = _validate_completed_iteration(
            draft_backend,
            target_backend,
            constraint,
            second_iteration,
            expected_initial_cache_length=first_final_length,
            vocab_size=vocab_size,
            acquired_state=owned_state,
            acquired_state_is_match=owned_state_is_match,
            label="second D47 result",
        )
        combined_output = first_output + second_output
        result = GrammarMaskedSpeculativeHandoffResult(
            output_token_ids=combined_output,
            final_iteration=second_iteration,
            final_outcome=second_outcome,
        )
        _validate_composed_result(
            result,
            output_token_ids=combined_output,
            final_iteration=second_iteration,
            final_outcome=second_outcome,
        )

        draft_backend.release_cache_checkpoint(
            cast(DraftCheckpointT, draft_intermediate)
        )
        draft_intermediate_owned = False
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            second_final_length,
        )
        target_backend.release_cache_checkpoint(
            cast(TargetCheckpointT, target_intermediate)
        )
        target_intermediate_owned = False
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            second_final_length,
        )

        _validate_live_state(
            constraint,
            cast(StateT, owned_state),
            expected_is_match=owned_state_is_match,
            label="final committed_state",
        )
        owned_state_acquired = False
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_handoff(
            draft_backend,
            target_backend,
            constraint,
            draft_root_checkpoint=draft_root_checkpoint,
            target_root_checkpoint=target_root_checkpoint,
            draft_intermediate=draft_intermediate,
            target_intermediate=target_intermediate,
            draft_intermediate_owned=draft_intermediate_owned,
            target_intermediate_owned=target_intermediate_owned,
            initial_cache_length=initial_cache_length,
            owned_state=owned_state,
            owned_state_acquired=owned_state_acquired,
        )
        if cleanup_failures:
            raise GrammarMaskedSpeculativeHandoffCleanupError(
                failure,
                cleanup_failures,
            ) from failure
        raise


def _validate_result_output(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("output_token_ids must be an exact tuple")
    for position, token_id in enumerate(value):
        if type(token_id) is not int:
            raise TypeError(f"output token at position {position} must be an integer")
        if token_id < 0:
            raise ValueError(f"output token at position {position} cannot be negative")
    return cast(tuple[int, ...], value)


def _validate_initial_root_metadata(
    draft_root_checkpoint: object,
    target_root_checkpoint: object,
) -> int:
    draft_length = _read_initial_root_length(
        draft_root_checkpoint,
        label="draft_root_checkpoint",
    )
    target_length = _read_initial_root_length(
        target_root_checkpoint,
        label="target_root_checkpoint",
    )
    if draft_length != target_length:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"initial root lengths differ: draft reports {draft_length}, "
            f"target reports {target_length}"
        )
    return draft_length


def _read_initial_root_length(checkpoint: object, *, label: str) -> int:
    try:
        is_checkpoint = isinstance(checkpoint, CacheCheckpoint)
    except Exception as exc:
        raise TypeError(f"{label} must satisfy CacheCheckpoint") from exc
    if not is_checkpoint:
        raise TypeError(f"{label} must satisfy CacheCheckpoint")
    try:
        cache_length = checkpoint.cache_length
    except Exception as exc:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} cache_length could not be read"
        ) from exc
    if type(cache_length) is not int:
        raise TypeError(f"{label} cache_length must be an integer")
    if cache_length <= 0:
        raise ValueError(f"{label} cache_length must be greater than zero")
    return cache_length


def _require_iteration_result(
    value: object,
    *,
    label: str,
    error_type: type[Exception],
) -> GrammarMaskedSpeculativeIterationResult[object]:
    try:
        is_result = isinstance(value, GrammarMaskedSpeculativeIterationResult)
    except Exception as exc:
        raise error_type(f"{label} type could not be determined") from exc
    if not is_result:
        raise error_type(
            f"{label} must be a GrammarMaskedSpeculativeIterationResult"
        )
    return cast(GrammarMaskedSpeculativeIterationResult[object], value)


def _require_outcome_result(
    value: object,
    *,
    label: str,
    error_type: type[Exception],
) -> GrammarMaskedSpeculativeOutcomeResult:
    try:
        is_result = isinstance(value, GrammarMaskedSpeculativeOutcomeResult)
    except Exception as exc:
        raise error_type(f"{label} type could not be determined") from exc
    if not is_result:
        raise error_type(f"{label} must be a GrammarMaskedSpeculativeOutcomeResult")
    _read_outcome_kind(cast(GrammarMaskedSpeculativeOutcomeResult, value), label=label)
    return cast(GrammarMaskedSpeculativeOutcomeResult, value)


def _read_state_match_fact(
    result: GrammarMaskedSpeculativeIterationResult[object],
    *,
    label: str,
) -> bool:
    is_match = _read_attribute(result, "committed_state_is_match", label=label)
    if type(is_match) is not bool:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} committed_state_is_match must be a boolean"
        )
    return is_match


def _validate_completed_iteration(
    draft_backend: object,
    target_backend: object,
    constraint: GrammarConstraint[StateT],
    result: GrammarMaskedSpeculativeIterationResult[object],
    *,
    expected_initial_cache_length: int,
    vocab_size: int,
    acquired_state: object,
    acquired_state_is_match: bool,
    label: str,
) -> tuple[tuple[int, ...], int]:
    current_vocab_size = _read_common_vocab_size(
        draft_backend,
        target_backend,
        constraint,
    )
    if current_vocab_size != vocab_size:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"common vocab_size changed to {current_vocab_size}; expected {vocab_size}"
        )
    proposal = _read_attribute(result, "proposal_token_ids", label=label)
    if type(proposal) is not tuple:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} proposal_token_ids must be an exact tuple"
        )
    for position, token_id in enumerate(proposal):
        _validate_operation_token(
            token_id,
            vocab_size=vocab_size,
            label=f"{label} proposal token at position {position}",
        )

    accepted_count = _read_attribute(result, "accepted_count", label=label)
    if type(accepted_count) is not int:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} accepted_count must be an integer"
        )
    if accepted_count < 0 or accepted_count > len(proposal):
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} accepted_count must be within [0, {len(proposal)}]"
        )

    initial_cache_length = _read_cache_length(
        _read_attribute(result, "initial_cache_length", label=label),
        label=f"{label} initial_cache_length",
    )
    if initial_cache_length != expected_initial_cache_length:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} initial cache length is {initial_cache_length}; "
            f"expected {expected_initial_cache_length}"
        )
    final_cache_length = _read_cache_length(
        _read_attribute(result, "final_cache_length", label=label),
        label=f"{label} final_cache_length",
    )
    expected_final_cache_length = initial_cache_length + 1 + accepted_count
    if final_cache_length != expected_final_cache_length:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} final cache length is {final_cache_length}; "
            f"expected {expected_final_cache_length}"
        )

    output = _read_result_output(result, label=label, vocab_size=vocab_size)
    handoff = _read_optional_token(
        result,
        "uncached_next_token_id",
        vocab_size=vocab_size,
        label=label,
    )
    expected_output = cast(tuple[int, ...], proposal)[:accepted_count]
    if handoff is not None:
        expected_output += (handoff,)
    if output != expected_output:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} output does not match its accepted prefix and uncached token"
        )

    if _read_attribute(result, "committed_state", label=label) is not acquired_state:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} must retain the exact acquired committed_state"
        )
    stored_match = _read_attribute(result, "committed_state_is_match", label=label)
    if type(stored_match) is not bool or stored_match is not acquired_state_is_match:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} must retain the exact acquired committed_state_is_match"
        )
    _validate_backend_pair_cache_length(
        draft_backend,
        target_backend,
        final_cache_length,
    )
    _validate_live_state(
        constraint,
        cast(StateT, acquired_state),
        expected_is_match=acquired_state_is_match,
        label=f"{label} committed_state",
    )
    return output, final_cache_length


def _read_result_output(
    result: GrammarMaskedSpeculativeIterationResult[object],
    *,
    label: str,
    vocab_size: int | None,
) -> tuple[int, ...]:
    output = _read_attribute(result, "output_token_ids", label=label)
    if type(output) is not tuple:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} output_token_ids must be an exact tuple"
        )
    for position, token_id in enumerate(output):
        _validate_operation_token(
            token_id,
            vocab_size=vocab_size,
            label=f"{label} output token at position {position}",
        )
    return cast(tuple[int, ...], output)


def _read_optional_token(
    result: GrammarMaskedSpeculativeIterationResult[object],
    field_name: str,
    *,
    vocab_size: int,
    label: str,
) -> int | None:
    token_id = _read_attribute(result, field_name, label=label)
    if token_id is None:
        return None
    _validate_operation_token(
        token_id,
        vocab_size=vocab_size,
        label=f"{label} {field_name}",
    )
    return cast(int, token_id)


def _read_outcome_kind(
    result: GrammarMaskedSpeculativeOutcomeResult,
    *,
    label: str,
) -> str:
    kind = _read_attribute(result, "kind", label=label)
    if type(kind) is not str or kind not in {
        "handoff_available",
        "grammar_complete",
        "grammar_no_continuation",
    }:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} contains an unsupported kind"
        )
    return kind


def _validate_outcome_relationship(
    iteration: GrammarMaskedSpeculativeIterationResult[object],
    outcome: GrammarMaskedSpeculativeOutcomeResult,
    *,
    label: str,
) -> None:
    handoff = _read_attribute(
        iteration,
        "uncached_next_token_id",
        label=f"{label} iteration",
    )
    kind = _read_outcome_kind(outcome, label=f"{label} outcome")
    if (kind == "handoff_available") is (handoff is None):
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} outcome handoff kind disagrees with its iteration token"
        )


def _read_common_vocab_size(
    draft_backend: object,
    target_backend: object,
    constraint: object,
) -> int:
    values = []
    for component, label in (
        (draft_backend, "draft backend"),
        (target_backend, "target backend"),
        (constraint, "constraint"),
    ):
        vocab_size = _read_attribute(component, "vocab_size", label=label)
        if type(vocab_size) is not int:
            raise GrammarMaskedSpeculativeHandoffInvariantError(
                f"{label} vocab_size must be an integer"
            )
        if vocab_size <= 0:
            raise GrammarMaskedSpeculativeHandoffInvariantError(
                f"{label} vocab_size must be greater than zero"
            )
        values.append(vocab_size)
    if values[0] != values[1] or values[0] != values[2]:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "draft backend, target backend, and constraint vocabulary sizes "
            "must match exactly"
        )
    return values[0]


def _validate_intermediate_checkpoint(
    checkpoint: object,
    *,
    expected_cache_length: int,
    label: str,
) -> None:
    try:
        is_checkpoint = isinstance(checkpoint, CacheCheckpoint)
    except Exception as exc:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        ) from exc
    if not is_checkpoint:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        )
    cache_length = _read_cache_length(
        _read_attribute(checkpoint, "cache_length", label=label),
        label=f"{label} cache_length",
    )
    if cache_length != expected_cache_length:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} reports cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )


def _validate_backend_pair_cache_length(
    draft_backend: object,
    target_backend: object,
    expected_cache_length: int,
) -> None:
    _validate_backend_cache_length(
        draft_backend,
        expected_cache_length,
        role="draft",
    )
    _validate_backend_cache_length(
        target_backend,
        expected_cache_length,
        role="target",
    )


def _validate_backend_cache_length(
    backend: object,
    expected_cache_length: int,
    *,
    role: str,
) -> None:
    cache_length = _read_cache_length(
        _read_attribute(backend, "cache_length", label=f"{role} backend"),
        label=f"{role} backend cache_length",
    )
    if cache_length != expected_cache_length:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{role} backend reported cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )


def _read_cache_length(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} must be an integer"
        )
    if value < 0:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} cannot be negative"
        )
    return value


def _validate_live_state(
    constraint: GrammarConstraint[StateT],
    state: StateT,
    *,
    expected_is_match: bool,
    label: str,
) -> None:
    try:
        is_dead = constraint.is_dead_state(state)
    except Exception as exc:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} dead-state status could not be read"
        ) from exc
    if type(is_dead) is not bool:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "constraint is_dead_state() must return a boolean"
        )
    if is_dead:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} must remain live"
        )
    try:
        is_match = constraint.is_match_state(state)
    except Exception as exc:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} match status could not be read"
        ) from exc
    if type(is_match) is not bool:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "constraint is_match_state() must return a boolean"
        )
    if is_match is not expected_is_match:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} match status changed during the handoff"
        )


def _validate_composed_result(
    result: object,
    *,
    output_token_ids: tuple[int, ...],
    final_iteration: GrammarMaskedSpeculativeIterationResult[object],
    final_outcome: GrammarMaskedSpeculativeOutcomeResult,
) -> None:
    try:
        is_result = isinstance(result, GrammarMaskedSpeculativeHandoffResult)
    except Exception as exc:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "D49 result type could not be determined"
        ) from exc
    if not is_result:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "D49 must return a GrammarMaskedSpeculativeHandoffResult"
        )
    if _read_attribute(result, "output_token_ids", label="D49 result") is not output_token_ids:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "D49 result must retain the exact output_token_ids tuple"
        )
    if _read_attribute(result, "final_iteration", label="D49 result") is not final_iteration:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "D49 result must retain the exact final D47 result"
        )
    if _read_attribute(result, "final_outcome", label="D49 result") is not final_outcome:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            "D49 result must retain the exact final D48 result"
        )


def _cleanup_failed_handoff(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT, DraftCheckpointT
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT, TargetCheckpointT
    ],
    constraint: GrammarConstraint[StateT],
    *,
    draft_root_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
    draft_intermediate: object,
    target_intermediate: object,
    draft_intermediate_owned: bool,
    target_intermediate_owned: bool,
    initial_cache_length: int,
    owned_state: object,
    owned_state_acquired: bool,
) -> tuple[tuple[str, Exception], ...]:
    cleanup_failures: list[tuple[str, Exception]] = []

    try:
        draft_backend.rollback_cache(draft_root_checkpoint)
        _validate_backend_cache_length(
            draft_backend,
            initial_cache_length,
            role="draft",
        )
    except Exception as cleanup_failure:
        cleanup_failures.append(("draft initial root rollback", cleanup_failure))

    try:
        target_backend.rollback_cache(target_root_checkpoint)
        _validate_backend_cache_length(
            target_backend,
            initial_cache_length,
            role="target",
        )
    except Exception as cleanup_failure:
        cleanup_failures.append(("target initial root rollback", cleanup_failure))

    if draft_intermediate_owned:
        try:
            draft_backend.release_cache_checkpoint(
                cast(DraftCheckpointT, draft_intermediate)
            )
        except Exception as cleanup_failure:
            cleanup_failures.append(
                ("draft intermediate root release", cleanup_failure)
            )

    if target_intermediate_owned:
        try:
            target_backend.release_cache_checkpoint(
                cast(TargetCheckpointT, target_intermediate)
            )
        except Exception as cleanup_failure:
            cleanup_failures.append(
                ("target intermediate root release", cleanup_failure)
            )

    if owned_state_acquired:
        try:
            constraint.release_state(cast(StateT, owned_state))
        except Exception as cleanup_failure:
            cleanup_failures.append(("committed state release", cleanup_failure))

    return tuple(cleanup_failures)


def _validate_operation_token(
    token_id: object,
    *,
    vocab_size: int | None,
    label: str,
) -> None:
    if type(token_id) is not int:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} must be an integer"
        )
    if token_id < 0:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} cannot be negative"
        )
    if vocab_size is not None and token_id >= vocab_size:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedSpeculativeHandoffInvariantError(
            f"{label} {name} could not be read"
        ) from exc


__all__ = [
    "GrammarMaskedSpeculativeHandoffCleanupError",
    "GrammarMaskedSpeculativeHandoffError",
    "GrammarMaskedSpeculativeHandoffInvariantError",
    "GrammarMaskedSpeculativeHandoffResult",
    "coordinate_grammar_masked_speculative_handoff",
]
