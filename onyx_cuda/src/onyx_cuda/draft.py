"""Framework-neutral draft-proposal orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .backend import BackendError, BackendStateError, ModelStep
from .cache import CacheCheckpoint, CheckpointableAutoregressiveBackend


class DraftProposalError(BackendError):
    """Base error raised by framework-neutral draft-proposal orchestration."""


class DraftProposalInvariantError(DraftProposalError):
    """Raised when a backend/checkpoint result violates the proposal contract."""


class DraftProposalCleanupError(DraftProposalError):
    """Raised when a failed proposal also cannot release or restore owned state."""

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
        details = "; ".join(
            f"{operation} also failed: {failure}" for operation, failure in failures
        )
        super().__init__(f"draft proposal failed: {original_failure}; {details}")


LogitsT = TypeVar("LogitsT")
CheckpointT = TypeVar("CheckpointT", bound=CacheCheckpoint)


@dataclass(frozen=True, slots=True)
class DraftProposalResult(Generic[CheckpointT]):
    """Ordered proposal tokens and caller-owned rejection rollback checkpoints."""

    proposal_token_ids: tuple[int, ...]
    rollback_checkpoints: tuple[CheckpointT, ...]
    initial_cache_length: int
    final_cache_length: int

    def __post_init__(self) -> None:
        if type(self.proposal_token_ids) is not tuple:
            raise TypeError("proposal_token_ids must be a tuple")
        if not self.proposal_token_ids:
            raise ValueError("proposal_token_ids cannot be empty")
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
            raise DraftProposalInvariantError(
                "initial_cache_length must be greater than zero"
            )
        final_cache_length = _validate_cache_length_metadata(
            self.final_cache_length,
            label="final_cache_length",
        )
        expected_final_length = initial_cache_length + len(self.proposal_token_ids) + 1
        if final_cache_length != expected_final_length:
            raise DraftProposalInvariantError(
                f"final_cache_length is {final_cache_length}; expected {expected_final_length}"
            )

        for position, checkpoint in enumerate(self.rollback_checkpoints):
            expected_length = initial_cache_length + 1 + position
            _validate_checkpoint(
                checkpoint,
                expected_cache_length=expected_length,
                label=f"rollback checkpoint at position {position}",
            )


def generate_draft_proposal(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    current_token_id: int,
    *,
    proposal_length: int,
    select_token: Callable[[LogitsT], int],
) -> DraftProposalResult[CheckpointT]:
    """Generate one exact proposal from an already-prefilled checkpointable backend.

    The returned checkpoints are owned by the caller. Checkpoint ``k`` restores the cache after
    the current token and the first ``k`` proposal tokens have been consumed.
    """

    initial_cache_length, vocab_size = _validate_proposal_inputs(
        backend,
        current_token_id,
        proposal_length=proposal_length,
        select_token=select_token,
    )

    start_checkpoint = backend.create_cache_checkpoint()
    rollback_checkpoints: list[CheckpointT] = []
    start_checkpoint_usable = False

    try:
        _validate_checkpoint(
            start_checkpoint,
            expected_cache_length=initial_cache_length,
            label="start checkpoint",
        )
        start_checkpoint_usable = True

        expected_cache_length = initial_cache_length + 1
        step = backend.decode(current_token_id)
        _validate_backend_step(backend, step, expected_cache_length)

        proposal_token_ids: list[int] = []
        for position in range(proposal_length):
            rollback_checkpoint = backend.create_cache_checkpoint()
            rollback_checkpoints.append(rollback_checkpoint)
            _validate_checkpoint(
                rollback_checkpoint,
                expected_cache_length=expected_cache_length,
                label=f"rollback checkpoint at position {position}",
            )

            selected_token_id = select_token(step.logits)
            _validate_selected_token_id(
                selected_token_id,
                vocab_size,
                position=position,
            )
            proposal_token_ids.append(selected_token_id)

            expected_cache_length += 1
            step = backend.decode(selected_token_id)
            _validate_backend_step(backend, step, expected_cache_length)

        final_cache_length = initial_cache_length + proposal_length + 1
        _validate_backend_cache_length(backend, final_cache_length)
        result = DraftProposalResult(
            proposal_token_ids=tuple(proposal_token_ids),
            rollback_checkpoints=tuple(rollback_checkpoints),
            initial_cache_length=initial_cache_length,
            final_cache_length=final_cache_length,
        )

        backend.release_cache_checkpoint(start_checkpoint)
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_proposal(
            backend,
            start_checkpoint=start_checkpoint,
            start_checkpoint_usable=start_checkpoint_usable,
            rollback_checkpoints=rollback_checkpoints,
        )
        if cleanup_failures:
            raise DraftProposalCleanupError(failure, cleanup_failures) from failure
        raise


def _validate_proposal_inputs(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    current_token_id: int,
    *,
    proposal_length: int,
    select_token: Callable[[LogitsT], int],
) -> tuple[int, int]:
    if isinstance(proposal_length, bool) or not isinstance(proposal_length, int):
        raise TypeError("proposal_length must be an integer")
    if proposal_length <= 0:
        raise ValueError("proposal_length must be greater than zero")
    if not callable(select_token):
        raise TypeError("select_token must be callable")
    if not isinstance(backend, CheckpointableAutoregressiveBackend):
        raise TypeError("backend must satisfy CheckpointableAutoregressiveBackend")

    vocab_size = backend.vocab_size
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise DraftProposalInvariantError("backend vocab_size must be an integer")
    if vocab_size <= 0:
        raise DraftProposalInvariantError("backend vocab_size must be greater than zero")

    _validate_token_id(current_token_id, vocab_size, label="current_token_id")

    initial_cache_length = _read_backend_cache_length(backend)
    if initial_cache_length == 0:
        raise BackendStateError("prefill must establish an active cache before draft proposal")
    return initial_cache_length, vocab_size


def _validate_backend_step(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    step: ModelStep[LogitsT],
    expected_cache_length: int,
) -> None:
    if not isinstance(step, ModelStep):
        raise DraftProposalInvariantError("backend decode must return a ModelStep")
    reported_cache_length = _validate_cache_length_metadata(
        step.cache_length,
        label="backend step cache_length",
    )
    if reported_cache_length != expected_cache_length:
        raise DraftProposalInvariantError(
            f"backend step reported cache length {reported_cache_length}; "
            f"expected {expected_cache_length}"
        )
    _validate_backend_cache_length(backend, expected_cache_length)


def _validate_backend_cache_length(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    expected_cache_length: int,
) -> int:
    cache_length = _read_backend_cache_length(backend)
    if cache_length != expected_cache_length:
        raise DraftProposalInvariantError(
            f"backend state reported cache length {cache_length}; expected {expected_cache_length}"
        )
    return cache_length


def _read_backend_cache_length(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
) -> int:
    return _validate_cache_length_metadata(
        backend.cache_length,
        label="backend cache_length",
    )


def _validate_cache_length_metadata(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DraftProposalInvariantError(f"{label} must be an integer")
    if value < 0:
        raise DraftProposalInvariantError(f"{label} cannot be negative")
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
        raise DraftProposalInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        ) from exc
    if not is_checkpoint:
        raise DraftProposalInvariantError(f"{label} must satisfy CacheCheckpoint")

    try:
        cache_length = checkpoint.cache_length
    except Exception as exc:
        raise DraftProposalInvariantError(
            f"{label} cache_length could not be read"
        ) from exc
    cache_length = _validate_cache_length_metadata(
        cache_length,
        label=f"{label} cache_length",
    )
    if cache_length != expected_cache_length:
        raise DraftProposalInvariantError(
            f"{label} reports cache length {cache_length}; expected {expected_cache_length}"
        )


def _validate_selected_token_id(token_id: object, vocab_size: int, *, position: int) -> None:
    _validate_token_id(
        token_id,
        vocab_size,
        label=f"selected token at proposal position {position}",
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


def _cleanup_failed_proposal(
    backend: CheckpointableAutoregressiveBackend[LogitsT, CheckpointT],
    *,
    start_checkpoint: object,
    start_checkpoint_usable: bool,
    rollback_checkpoints: Sequence[object],
) -> tuple[tuple[str, Exception], ...]:
    cleanup_failures: list[tuple[str, Exception]] = []

    if start_checkpoint_usable:
        try:
            backend.rollback_cache(cast(CheckpointT, start_checkpoint))
        except Exception as cleanup_failure:
            cleanup_failures.append(("start checkpoint rollback", cleanup_failure))

    for position, checkpoint in enumerate(rollback_checkpoints):
        try:
            backend.release_cache_checkpoint(cast(CheckpointT, checkpoint))
        except Exception as cleanup_failure:
            cleanup_failures.append(
                (f"rollback checkpoint {position} release", cleanup_failure)
            )

    try:
        backend.release_cache_checkpoint(cast(CheckpointT, start_checkpoint))
    except Exception as cleanup_failure:
        cleanup_failures.append(("start checkpoint release", cleanup_failure))

    return tuple(cleanup_failures)


__all__ = [
    "DraftProposalCleanupError",
    "DraftProposalError",
    "DraftProposalInvariantError",
    "DraftProposalResult",
    "generate_draft_proposal",
]
