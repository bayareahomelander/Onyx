"""Framework-neutral bounded handoffs across speculative transactions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from .backend import BackendError
from .cache import CacheCheckpoint, CheckpointableAutoregressiveBackend
from .speculative_iteration import (
    ContinuationAwareSpeculativeIterationResult,
    coordinate_continuation_aware_speculative_iteration,
)


class SpeculativeHandoffError(BackendError):
    """Base error raised by a bounded speculative handoff."""


class SpeculativeHandoffInvariantError(SpeculativeHandoffError):
    """Raised when completed transaction evidence violates a handoff contract."""


class SpeculativeHandoffCleanupError(SpeculativeHandoffError):
    """Raised when a failed handoff also cannot restore or settle borrowed state."""

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
        super().__init__(f"speculative handoff failed: {original_failure}; {details}")


DraftLogitsT = TypeVar("DraftLogitsT")
DraftCheckpointT = TypeVar("DraftCheckpointT", bound=CacheCheckpoint)
TargetLogitsT = TypeVar("TargetLogitsT")
TargetCheckpointT = TypeVar("TargetCheckpointT", bound=CacheCheckpoint)


@dataclass(frozen=True, slots=True)
class TwoIterationSpeculativeHandoffResult:
    """Immutable results and derived output for exactly two D38 transactions."""

    first_iteration: ContinuationAwareSpeculativeIterationResult
    second_iteration: ContinuationAwareSpeculativeIterationResult

    def __post_init__(self) -> None:
        first_proposal_length = _validate_result_structure(
            self.first_iteration,
            label="first_iteration",
        )
        second_proposal_length = _validate_result_structure(
            self.second_iteration,
            label="second_iteration",
        )
        if first_proposal_length != second_proposal_length:
            raise SpeculativeHandoffInvariantError(
                "both iterations must use the same positive proposal length"
            )
        if (
            self.first_iteration.final_cache_length
            != self.second_iteration.initial_cache_length
        ):
            raise SpeculativeHandoffInvariantError(
                "first final cache length must equal second initial cache length"
            )

        output_token_ids = self.output_token_ids
        if type(output_token_ids) is not tuple or not output_token_ids:
            raise SpeculativeHandoffInvariantError(
                "combined output_token_ids must be an exact nonempty tuple"
            )
        if output_token_ids[-1] != self.uncached_next_token_id:
            raise SpeculativeHandoffInvariantError(
                "combined output must end in the final uncached token"
            )

    @property
    def handoff_token_id(self) -> int:
        """Return the first result's token consumed by the second transaction."""

        return self.first_iteration.uncached_next_token_id

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Return the exact concatenation of both transaction outputs."""

        return (
            self.first_iteration.output_token_ids
            + self.second_iteration.output_token_ids
        )

    @property
    def uncached_next_token_id(self) -> int:
        """Return the only final token that remains outside both caches."""

        return self.second_iteration.uncached_next_token_id

    @property
    def initial_cache_length(self) -> int:
        """Return the caller-owned initial-root length."""

        return self.first_iteration.initial_cache_length

    @property
    def intermediate_cache_length(self) -> int:
        """Return the D39-owned intermediate-root length."""

        return self.first_iteration.final_cache_length

    @property
    def final_cache_length(self) -> int:
        """Return the common final cache length after the second transaction."""

        return self.second_iteration.final_cache_length


@dataclass(frozen=True, slots=True)
class MultiIterationSpeculativeHandoffResult:
    """Immutable transaction sequence and derived bounded-handoff outcome."""

    iterations: tuple[ContinuationAwareSpeculativeIterationResult, ...]

    def __post_init__(self) -> None:
        if type(self.iterations) is not tuple:
            raise TypeError("iterations must be an exact tuple")
        if not self.iterations:
            raise ValueError("iterations cannot be empty")

        proposal_length: int | None = None
        previous_final_cache_length: int | None = None
        for position, iteration in enumerate(self.iterations):
            current_proposal_length = _validate_result_structure(
                iteration,
                label=f"iterations[{position}]",
            )
            if proposal_length is None:
                proposal_length = current_proposal_length
            elif current_proposal_length != proposal_length:
                raise SpeculativeHandoffInvariantError(
                    "all iterations must use the same positive proposal length"
                )
            if (
                previous_final_cache_length is not None
                and iteration.initial_cache_length != previous_final_cache_length
            ):
                raise SpeculativeHandoffInvariantError(
                    "adjacent iteration cache lengths must be continuous"
                )
            previous_final_cache_length = iteration.final_cache_length

        output_token_ids = self.output_token_ids
        if type(output_token_ids) is not tuple or not output_token_ids:
            raise SpeculativeHandoffInvariantError(
                "combined output_token_ids must be an exact nonempty tuple"
            )
        if output_token_ids[-1] != self.uncached_next_token_id:
            raise SpeculativeHandoffInvariantError(
                "combined output must end in the final uncached token"
            )

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Return the exact ordered concatenation of all transaction outputs."""

        return tuple(
            token_id
            for iteration in self.iterations
            for token_id in iteration.output_token_ids
        )

    @property
    def uncached_next_token_id(self) -> int:
        """Return the only final token that remains outside both caches."""

        return self.iterations[-1].uncached_next_token_id

    @property
    def initial_cache_length(self) -> int:
        """Return the caller-owned initial-root length."""

        return self.iterations[0].initial_cache_length

    @property
    def final_cache_length(self) -> int:
        """Return the common final cache length after all transactions."""

        return self.iterations[-1].final_cache_length


def coordinate_two_iteration_speculative_handoff(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT,
        DraftCheckpointT,
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT,
        TargetCheckpointT,
    ],
    current_token_id: int,
    *,
    proposal_length: int,
    draft_select_token: Callable[[DraftLogitsT], int],
    target_select_token: Callable[[TargetLogitsT], int],
    draft_root_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
) -> TwoIterationSpeculativeHandoffResult:
    """Coordinate one exact handoff across two completed D38 transactions."""

    initial_cache_length = _validate_initial_root_metadata(
        draft_root_checkpoint,
        target_root_checkpoint,
    )

    first_result = coordinate_continuation_aware_speculative_iteration(
        draft_backend,
        target_backend,
        current_token_id,
        proposal_length=proposal_length,
        draft_select_token=draft_select_token,
        target_select_token=target_select_token,
        draft_root_checkpoint=draft_root_checkpoint,
        target_root_checkpoint=target_root_checkpoint,
    )

    draft_intermediate: object = None
    target_intermediate: object = None
    draft_intermediate_acquired = False
    target_intermediate_acquired = False

    try:
        vocab_size = _read_common_vocab_size(draft_backend, target_backend)
        first_final_length = _validate_completed_iteration_result(
            first_result,
            expected_initial_cache_length=initial_cache_length,
            expected_proposal_length=proposal_length,
            vocab_size=vocab_size,
            label="first iteration",
        )
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            first_final_length,
        )

        draft_intermediate = draft_backend.create_cache_checkpoint()
        draft_intermediate_acquired = True
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
        target_intermediate_acquired = True
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

        second_result = coordinate_continuation_aware_speculative_iteration(
            draft_backend,
            target_backend,
            first_result.uncached_next_token_id,
            proposal_length=proposal_length,
            draft_select_token=draft_select_token,
            target_select_token=target_select_token,
            draft_root_checkpoint=cast(DraftCheckpointT, draft_intermediate),
            target_root_checkpoint=cast(TargetCheckpointT, target_intermediate),
        )

        second_final_length = _validate_completed_iteration_result(
            second_result,
            expected_initial_cache_length=first_final_length,
            expected_proposal_length=proposal_length,
            vocab_size=vocab_size,
            label="second iteration",
        )
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            second_final_length,
        )

        result = TwoIterationSpeculativeHandoffResult(
            first_iteration=first_result,
            second_iteration=second_result,
        )
        _validate_composed_result(
            result,
            first_result=first_result,
            second_result=second_result,
            intermediate_cache_length=first_final_length,
            final_cache_length=second_final_length,
        )

        draft_backend.release_cache_checkpoint(
            cast(DraftCheckpointT, draft_intermediate)
        )
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            second_final_length,
        )
        target_backend.release_cache_checkpoint(
            cast(TargetCheckpointT, target_intermediate)
        )
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            second_final_length,
        )
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_handoff(
            draft_backend,
            target_backend,
            draft_root_checkpoint=draft_root_checkpoint,
            target_root_checkpoint=target_root_checkpoint,
            draft_intermediate=draft_intermediate,
            target_intermediate=target_intermediate,
            draft_intermediate_acquired=draft_intermediate_acquired,
            target_intermediate_acquired=target_intermediate_acquired,
            initial_cache_length=initial_cache_length,
        )
        if cleanup_failures:
            raise SpeculativeHandoffCleanupError(failure, cleanup_failures) from failure
        raise


def coordinate_multi_iteration_speculative_handoff(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT,
        DraftCheckpointT,
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT,
        TargetCheckpointT,
    ],
    current_token_id: int,
    *,
    iteration_count: int,
    proposal_length: int,
    draft_select_token: Callable[[DraftLogitsT], int],
    target_select_token: Callable[[TargetLogitsT], int],
    draft_root_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
) -> MultiIterationSpeculativeHandoffResult:
    """Coordinate a positive caller-bounded sequence of completed D38 transactions."""

    _validate_iteration_count(iteration_count)
    initial_cache_length = _validate_initial_root_metadata(
        draft_root_checkpoint,
        target_root_checkpoint,
    )

    first_result = coordinate_continuation_aware_speculative_iteration(
        draft_backend,
        target_backend,
        current_token_id,
        proposal_length=proposal_length,
        draft_select_token=draft_select_token,
        target_select_token=target_select_token,
        draft_root_checkpoint=draft_root_checkpoint,
        target_root_checkpoint=target_root_checkpoint,
    )

    current_draft: object = None
    current_target: object = None
    current_draft_acquired = False
    current_target_acquired = False
    next_draft: object = None
    next_target: object = None
    next_draft_acquired = False
    next_target_acquired = False

    try:
        vocab_size = _read_common_vocab_size(draft_backend, target_backend)
        first_final_length = _validate_completed_iteration_result(
            first_result,
            expected_initial_cache_length=initial_cache_length,
            expected_proposal_length=proposal_length,
            vocab_size=vocab_size,
            label="iteration 1",
        )
        _validate_backend_pair_cache_length(
            draft_backend,
            target_backend,
            first_final_length,
        )

        completed_results = [first_result]
        latest_final_length = first_final_length

        while len(completed_results) < iteration_count:
            next_draft = draft_backend.create_cache_checkpoint()
            next_draft_acquired = True
            _validate_intermediate_checkpoint(
                next_draft,
                expected_cache_length=latest_final_length,
                label="draft next intermediate checkpoint",
            )
            _validate_backend_pair_cache_length(
                draft_backend,
                target_backend,
                latest_final_length,
            )

            next_target = target_backend.create_cache_checkpoint()
            next_target_acquired = True
            _validate_intermediate_checkpoint(
                next_target,
                expected_cache_length=latest_final_length,
                label="target next intermediate checkpoint",
            )
            _validate_backend_pair_cache_length(
                draft_backend,
                target_backend,
                latest_final_length,
            )

            if current_draft_acquired:
                draft_backend.release_cache_checkpoint(
                    cast(DraftCheckpointT, current_draft)
                )
                _validate_backend_pair_cache_length(
                    draft_backend,
                    target_backend,
                    latest_final_length,
                )
                target_backend.release_cache_checkpoint(
                    cast(TargetCheckpointT, current_target)
                )
                _validate_backend_pair_cache_length(
                    draft_backend,
                    target_backend,
                    latest_final_length,
                )
                current_draft = None
                current_target = None
                current_draft_acquired = False
                current_target_acquired = False

            current_draft = next_draft
            current_target = next_target
            current_draft_acquired = next_draft_acquired
            current_target_acquired = next_target_acquired
            next_draft = None
            next_target = None
            next_draft_acquired = False
            next_target_acquired = False

            previous_result = completed_results[-1]
            candidate_result = coordinate_continuation_aware_speculative_iteration(
                draft_backend,
                target_backend,
                previous_result.uncached_next_token_id,
                proposal_length=proposal_length,
                draft_select_token=draft_select_token,
                target_select_token=target_select_token,
                draft_root_checkpoint=cast(DraftCheckpointT, current_draft),
                target_root_checkpoint=cast(TargetCheckpointT, current_target),
            )
            latest_final_length = _validate_completed_iteration_result(
                candidate_result,
                expected_initial_cache_length=latest_final_length,
                expected_proposal_length=proposal_length,
                vocab_size=vocab_size,
                label=f"iteration {len(completed_results) + 1}",
            )
            _validate_backend_pair_cache_length(
                draft_backend,
                target_backend,
                latest_final_length,
            )
            completed_results.append(candidate_result)

        result = MultiIterationSpeculativeHandoffResult(
            iterations=tuple(completed_results),
        )
        _validate_multi_iteration_composed_result(
            result,
            completed_results=completed_results,
            iteration_count=iteration_count,
            proposal_length=proposal_length,
            initial_cache_length=initial_cache_length,
            final_cache_length=latest_final_length,
        )

        if current_draft_acquired:
            draft_backend.release_cache_checkpoint(
                cast(DraftCheckpointT, current_draft)
            )
            _validate_backend_pair_cache_length(
                draft_backend,
                target_backend,
                latest_final_length,
            )
            target_backend.release_cache_checkpoint(
                cast(TargetCheckpointT, current_target)
            )
            _validate_backend_pair_cache_length(
                draft_backend,
                target_backend,
                latest_final_length,
            )
            current_draft = None
            current_target = None
            current_draft_acquired = False
            current_target_acquired = False
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_multi_iteration_handoff(
            draft_backend,
            target_backend,
            draft_root_checkpoint=draft_root_checkpoint,
            target_root_checkpoint=target_root_checkpoint,
            current_draft=current_draft,
            current_target=current_target,
            current_draft_acquired=current_draft_acquired,
            current_target_acquired=current_target_acquired,
            next_draft=next_draft,
            next_target=next_target,
            next_draft_acquired=next_draft_acquired,
            next_target_acquired=next_target_acquired,
            initial_cache_length=initial_cache_length,
        )
        if cleanup_failures:
            raise SpeculativeHandoffCleanupError(failure, cleanup_failures) from failure
        raise


def _validate_iteration_count(iteration_count: object) -> int:
    if isinstance(iteration_count, bool) or not isinstance(iteration_count, int):
        raise TypeError("iteration_count must be an integer")
    if iteration_count <= 0:
        raise ValueError("iteration_count must be greater than zero")
    return iteration_count


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
        raise SpeculativeHandoffInvariantError(
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
        raise SpeculativeHandoffInvariantError(
            f"{label} cache_length could not be read"
        ) from exc
    if isinstance(cache_length, bool) or not isinstance(cache_length, int):
        raise TypeError(f"{label} cache_length must be an integer")
    if cache_length <= 0:
        raise ValueError(f"{label} cache_length must be greater than zero")
    return cache_length


def _validate_completed_iteration_result(
    result: object,
    *,
    expected_initial_cache_length: int,
    expected_proposal_length: int,
    vocab_size: int,
    label: str,
) -> int:
    proposal_length = _validate_result_structure(
        result,
        vocab_size=vocab_size,
        label=label,
    )
    typed_result = cast(ContinuationAwareSpeculativeIterationResult, result)
    if proposal_length != expected_proposal_length:
        raise SpeculativeHandoffInvariantError(
            f"{label} proposal length is {proposal_length}; "
            f"expected {expected_proposal_length}"
        )
    if typed_result.initial_cache_length != expected_initial_cache_length:
        raise SpeculativeHandoffInvariantError(
            f"{label} initial cache length is {typed_result.initial_cache_length}; "
            f"expected {expected_initial_cache_length}"
        )
    return typed_result.final_cache_length


def _validate_result_structure(
    result: object,
    *,
    vocab_size: int | None = None,
    label: str,
) -> int:
    if not isinstance(result, ContinuationAwareSpeculativeIterationResult):
        raise SpeculativeHandoffInvariantError(
            f"{label} must be a ContinuationAwareSpeculativeIterationResult"
        )
    try:
        proposal_token_ids = result.proposal_token_ids
        accepted_count = result.accepted_count
        replacement_token_id = result.replacement_token_id
        initial_cache_length = result.initial_cache_length
        final_cache_length = result.final_cache_length
        uncached_next_token_id = result.uncached_next_token_id
        output_token_ids = result.output_token_ids
    except Exception as exc:
        raise SpeculativeHandoffInvariantError(
            f"{label} fields could not be read"
        ) from exc

    if type(proposal_token_ids) is not tuple or not proposal_token_ids:
        raise SpeculativeHandoffInvariantError(
            f"{label} proposal_token_ids must be an exact nonempty tuple"
        )
    for position, token_id in enumerate(proposal_token_ids):
        _validate_result_token(
            token_id,
            vocab_size=vocab_size,
            label=f"{label} proposal token at position {position}",
        )
    proposal_length = len(proposal_token_ids)

    if isinstance(accepted_count, bool) or not isinstance(accepted_count, int):
        raise SpeculativeHandoffInvariantError(
            f"{label} accepted_count must be an integer"
        )
    if accepted_count < 0 or accepted_count > proposal_length:
        raise SpeculativeHandoffInvariantError(
            f"{label} accepted_count must be within [0, {proposal_length}]"
        )

    _validate_positive_result_length(
        initial_cache_length,
        label=f"{label} initial_cache_length",
    )
    _validate_positive_result_length(
        final_cache_length,
        label=f"{label} final_cache_length",
    )
    expected_final_length = initial_cache_length + accepted_count + 1
    if final_cache_length != expected_final_length:
        raise SpeculativeHandoffInvariantError(
            f"{label} final cache length is {final_cache_length}; "
            f"expected {expected_final_length}"
        )

    if accepted_count == proposal_length:
        if replacement_token_id is not None:
            raise SpeculativeHandoffInvariantError(
                f"{label} fully accepted result cannot contain a replacement token"
            )
    else:
        _validate_result_token(
            replacement_token_id,
            vocab_size=vocab_size,
            label=f"{label} replacement_token_id",
        )
        if replacement_token_id == proposal_token_ids[accepted_count]:
            raise SpeculativeHandoffInvariantError(
                f"{label} replacement token must differ from the rejected proposal token"
            )

    _validate_result_token(
        uncached_next_token_id,
        vocab_size=vocab_size,
        label=f"{label} uncached_next_token_id",
    )
    if accepted_count < proposal_length and uncached_next_token_id != replacement_token_id:
        raise SpeculativeHandoffInvariantError(
            f"{label} mismatch uncached token must equal the replacement token"
        )

    if type(output_token_ids) is not tuple or not output_token_ids:
        raise SpeculativeHandoffInvariantError(
            f"{label} output_token_ids must be an exact nonempty tuple"
        )
    for position, token_id in enumerate(output_token_ids):
        _validate_result_token(
            token_id,
            vocab_size=vocab_size,
            label=f"{label} output token at position {position}",
        )
    expected_output = proposal_token_ids[:accepted_count] + (uncached_next_token_id,)
    if output_token_ids != expected_output:
        raise SpeculativeHandoffInvariantError(
            f"{label} output does not match its accepted prefix and uncached token"
        )
    if output_token_ids[-1] != uncached_next_token_id:
        raise SpeculativeHandoffInvariantError(
            f"{label} uncached token must equal the final output token"
        )
    return proposal_length


def _validate_result_token(
    token_id: object,
    *,
    vocab_size: int | None,
    label: str,
) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise SpeculativeHandoffInvariantError(f"{label} must be an integer")
    if token_id < 0:
        raise SpeculativeHandoffInvariantError(f"{label} cannot be negative")
    if vocab_size is not None and token_id >= vocab_size:
        raise SpeculativeHandoffInvariantError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _validate_positive_result_length(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeculativeHandoffInvariantError(f"{label} must be an integer")
    if value <= 0:
        raise SpeculativeHandoffInvariantError(f"{label} must be greater than zero")
    return value


def _read_common_vocab_size(
    draft_backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
    target_backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
) -> int:
    draft_vocab_size = _validate_vocab_size(draft_backend.vocab_size, role="draft")
    target_vocab_size = _validate_vocab_size(target_backend.vocab_size, role="target")
    if draft_vocab_size != target_vocab_size:
        raise SpeculativeHandoffInvariantError(
            f"backend vocabulary sizes differ: draft reports {draft_vocab_size}, "
            f"target reports {target_vocab_size}"
        )
    return draft_vocab_size


def _validate_vocab_size(value: object, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeculativeHandoffInvariantError(
            f"{role} backend vocab_size must be an integer"
        )
    if value <= 0:
        raise SpeculativeHandoffInvariantError(
            f"{role} backend vocab_size must be greater than zero"
        )
    return value


def _validate_intermediate_checkpoint(
    checkpoint: object,
    *,
    expected_cache_length: int,
    label: str,
) -> None:
    try:
        is_checkpoint = isinstance(checkpoint, CacheCheckpoint)
    except Exception as exc:
        raise SpeculativeHandoffInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        ) from exc
    if not is_checkpoint:
        raise SpeculativeHandoffInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        )
    try:
        cache_length = checkpoint.cache_length
    except Exception as exc:
        raise SpeculativeHandoffInvariantError(
            f"{label} cache_length could not be read"
        ) from exc
    if isinstance(cache_length, bool) or not isinstance(cache_length, int):
        raise SpeculativeHandoffInvariantError(
            f"{label} cache_length must be an integer"
        )
    if cache_length < 0:
        raise SpeculativeHandoffInvariantError(
            f"{label} cache_length cannot be negative"
        )
    if cache_length != expected_cache_length:
        raise SpeculativeHandoffInvariantError(
            f"{label} reports cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )


def _validate_backend_pair_cache_length(
    draft_backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
    target_backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
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
    backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
    expected_cache_length: int,
    *,
    role: str,
) -> None:
    cache_length = backend.cache_length
    if isinstance(cache_length, bool) or not isinstance(cache_length, int):
        raise SpeculativeHandoffInvariantError(
            f"{role} backend cache_length must be an integer"
        )
    if cache_length < 0:
        raise SpeculativeHandoffInvariantError(
            f"{role} backend cache_length cannot be negative"
        )
    if cache_length != expected_cache_length:
        raise SpeculativeHandoffInvariantError(
            f"{role} backend reported cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )


def _validate_composed_result(
    result: object,
    *,
    first_result: ContinuationAwareSpeculativeIterationResult,
    second_result: ContinuationAwareSpeculativeIterationResult,
    intermediate_cache_length: int,
    final_cache_length: int,
) -> None:
    if not isinstance(result, TwoIterationSpeculativeHandoffResult):
        raise SpeculativeHandoffInvariantError(
            "handoff result must be a TwoIterationSpeculativeHandoffResult"
        )
    if result.first_iteration is not first_result or result.second_iteration is not second_result:
        raise SpeculativeHandoffInvariantError(
            "handoff result must retain the exact completed iteration results"
        )
    if result.intermediate_cache_length != intermediate_cache_length:
        raise SpeculativeHandoffInvariantError(
            "handoff result reports the wrong intermediate cache length"
        )
    if result.final_cache_length != final_cache_length:
        raise SpeculativeHandoffInvariantError(
            "handoff result reports the wrong final cache length"
        )
    if result.handoff_token_id != first_result.uncached_next_token_id:
        raise SpeculativeHandoffInvariantError(
            "handoff result reports the wrong handoff token"
        )
    expected_output = first_result.output_token_ids + second_result.output_token_ids
    if result.output_token_ids != expected_output:
        raise SpeculativeHandoffInvariantError(
            "handoff output must exactly concatenate both iteration outputs"
        )
    if result.uncached_next_token_id != second_result.uncached_next_token_id:
        raise SpeculativeHandoffInvariantError(
            "handoff result reports the wrong final uncached token"
        )
    if not result.output_token_ids or (
        result.output_token_ids[-1] != result.uncached_next_token_id
    ):
        raise SpeculativeHandoffInvariantError(
            "handoff output must end in the final uncached token"
        )


def _validate_multi_iteration_composed_result(
    result: object,
    *,
    completed_results: list[ContinuationAwareSpeculativeIterationResult],
    iteration_count: int,
    proposal_length: int,
    initial_cache_length: int,
    final_cache_length: int,
) -> None:
    if not isinstance(result, MultiIterationSpeculativeHandoffResult):
        raise SpeculativeHandoffInvariantError(
            "handoff result must be a MultiIterationSpeculativeHandoffResult"
        )
    if len(result.iterations) != iteration_count:
        raise SpeculativeHandoffInvariantError(
            "handoff result contains the wrong number of iterations"
        )
    if any(
        stored is not completed
        for stored, completed in zip(result.iterations, completed_results)
    ):
        raise SpeculativeHandoffInvariantError(
            "handoff result must retain the exact completed iteration results"
        )
    if any(
        len(iteration.proposal_token_ids) != proposal_length
        for iteration in result.iterations
    ):
        raise SpeculativeHandoffInvariantError(
            "handoff result contains the wrong proposal length"
        )
    if any(
        first.final_cache_length != second.initial_cache_length
        for first, second in zip(result.iterations, result.iterations[1:])
    ):
        raise SpeculativeHandoffInvariantError(
            "handoff result cache lengths must be continuous"
        )
    if result.initial_cache_length != initial_cache_length:
        raise SpeculativeHandoffInvariantError(
            "handoff result reports the wrong initial cache length"
        )
    if result.final_cache_length != final_cache_length:
        raise SpeculativeHandoffInvariantError(
            "handoff result reports the wrong final cache length"
        )
    expected_output = tuple(
        token_id
        for iteration in completed_results
        for token_id in iteration.output_token_ids
    )
    if result.output_token_ids != expected_output:
        raise SpeculativeHandoffInvariantError(
            "handoff output must exactly concatenate every iteration output"
        )
    if result.uncached_next_token_id != completed_results[-1].uncached_next_token_id:
        raise SpeculativeHandoffInvariantError(
            "handoff result reports the wrong final uncached token"
        )
    if not result.output_token_ids or (
        result.output_token_ids[-1] != result.uncached_next_token_id
    ):
        raise SpeculativeHandoffInvariantError(
            "handoff output must end in the final uncached token"
        )


def _cleanup_failed_handoff(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT,
        DraftCheckpointT,
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT,
        TargetCheckpointT,
    ],
    *,
    draft_root_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
    draft_intermediate: object,
    target_intermediate: object,
    draft_intermediate_acquired: bool,
    target_intermediate_acquired: bool,
    initial_cache_length: int,
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

    if draft_intermediate_acquired:
        try:
            draft_backend.release_cache_checkpoint(
                cast(DraftCheckpointT, draft_intermediate)
            )
        except Exception as cleanup_failure:
            cleanup_failures.append(
                ("draft intermediate root release", cleanup_failure)
            )

    if target_intermediate_acquired:
        try:
            target_backend.release_cache_checkpoint(
                cast(TargetCheckpointT, target_intermediate)
            )
        except Exception as cleanup_failure:
            cleanup_failures.append(
                ("target intermediate root release", cleanup_failure)
            )

    return tuple(cleanup_failures)


def _cleanup_failed_multi_iteration_handoff(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT,
        DraftCheckpointT,
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT,
        TargetCheckpointT,
    ],
    *,
    draft_root_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
    current_draft: object,
    current_target: object,
    current_draft_acquired: bool,
    current_target_acquired: bool,
    next_draft: object,
    next_target: object,
    next_draft_acquired: bool,
    next_target_acquired: bool,
    initial_cache_length: int,
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

    has_current_pair = current_draft_acquired or current_target_acquired
    if has_current_pair:
        _settle_multi_iteration_checkpoint(
            draft_backend,
            current_draft,
            acquired=current_draft_acquired,
            label="draft intermediate root release",
            cleanup_failures=cleanup_failures,
        )
        _settle_multi_iteration_checkpoint(
            target_backend,
            current_target,
            acquired=current_target_acquired,
            label="target intermediate root release",
            cleanup_failures=cleanup_failures,
        )
        _settle_multi_iteration_checkpoint(
            draft_backend,
            next_draft,
            acquired=next_draft_acquired,
            label="draft next intermediate root release",
            cleanup_failures=cleanup_failures,
        )
        _settle_multi_iteration_checkpoint(
            target_backend,
            next_target,
            acquired=next_target_acquired,
            label="target next intermediate root release",
            cleanup_failures=cleanup_failures,
        )
    else:
        _settle_multi_iteration_checkpoint(
            draft_backend,
            next_draft,
            acquired=next_draft_acquired,
            label="draft intermediate root release",
            cleanup_failures=cleanup_failures,
        )
        _settle_multi_iteration_checkpoint(
            target_backend,
            next_target,
            acquired=next_target_acquired,
            label="target intermediate root release",
            cleanup_failures=cleanup_failures,
        )

    return tuple(cleanup_failures)


def _settle_multi_iteration_checkpoint(
    backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
    checkpoint: object,
    *,
    acquired: bool,
    label: str,
    cleanup_failures: list[tuple[str, Exception]],
) -> None:
    if not acquired:
        return
    try:
        backend.release_cache_checkpoint(cast(CacheCheckpoint, checkpoint))
    except Exception as cleanup_failure:
        cleanup_failures.append((label, cleanup_failure))


__all__ = [
    "SpeculativeHandoffCleanupError",
    "SpeculativeHandoffError",
    "SpeculativeHandoffInvariantError",
    "TwoIterationSpeculativeHandoffResult",
    "coordinate_two_iteration_speculative_handoff",
    "MultiIterationSpeculativeHandoffResult",
    "coordinate_multi_iteration_speculative_handoff",
]
