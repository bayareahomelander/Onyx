"""Framework-neutral coordination for one speculative decoding iteration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from .acceptance import (
    MatchReplaceAcceptanceResult,
    decide_match_replace_acceptance,
)
from .backend import BackendError, ModelStep
from .cache import CacheCheckpoint, CheckpointableAutoregressiveBackend
from .draft import DraftProposalResult, generate_draft_proposal
from .verification import (
    BatchedTargetVerificationBackend,
    BatchedTargetVerificationResult,
)


class SpeculativeIterationError(BackendError):
    """Base error raised by one framework-neutral speculative transaction."""


class SpeculativeIterationInvariantError(SpeculativeIterationError):
    """Raised when composed backend evidence violates the D35 contract."""


class SpeculativeIterationCleanupError(SpeculativeIterationError):
    """Raised when a failed iteration also cannot release or restore owned state."""

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
        super().__init__(f"speculative iteration failed: {original_failure}; {details}")


DraftLogitsT = TypeVar("DraftLogitsT")
DraftCheckpointT = TypeVar("DraftCheckpointT", bound=CacheCheckpoint)
TargetLogitsT = TypeVar("TargetLogitsT")
TargetCheckpointT = TypeVar("TargetCheckpointT", bound=CacheCheckpoint)


@dataclass(frozen=True, slots=True)
class SpeculativeIterationResult:
    """Immutable token outcome and reconciled cache lengths for one transaction."""

    proposal_token_ids: tuple[int, ...]
    accepted_count: int
    replacement_token_id: int | None
    initial_cache_length: int
    final_cache_length: int

    def __post_init__(self) -> None:
        _validate_proposal_token_ids(self.proposal_token_ids)

        if isinstance(self.accepted_count, bool) or not isinstance(self.accepted_count, int):
            raise TypeError("accepted_count must be an integer")
        proposal_length = len(self.proposal_token_ids)
        if self.accepted_count < 0 or self.accepted_count > proposal_length:
            raise SpeculativeIterationInvariantError(
                f"accepted_count must be within [0, {proposal_length}]"
            )

        if self.accepted_count == proposal_length:
            if self.replacement_token_id is not None:
                raise SpeculativeIterationInvariantError(
                    "fully accepted result cannot contain a replacement token"
                )
        else:
            if self.replacement_token_id is None:
                raise SpeculativeIterationInvariantError(
                    "partially accepted result must contain a replacement token"
                )
            _validate_nonnegative_token_id(
                self.replacement_token_id,
                label="replacement_token_id",
            )
            if self.replacement_token_id == self.proposal_token_ids[self.accepted_count]:
                raise SpeculativeIterationInvariantError(
                    "replacement_token_id must differ from the rejected proposal token"
                )

        initial_cache_length = _validate_cache_length_metadata(
            self.initial_cache_length,
            label="initial_cache_length",
        )
        if initial_cache_length == 0:
            raise SpeculativeIterationInvariantError(
                "initial_cache_length must be greater than zero"
            )
        final_cache_length = _validate_cache_length_metadata(
            self.final_cache_length,
            label="final_cache_length",
        )
        expected_final_length = (
            initial_cache_length + proposal_length + 1
            if self.fully_accepted
            else initial_cache_length + self.accepted_count + 1
        )
        if final_cache_length != expected_final_length:
            raise SpeculativeIterationInvariantError(
                f"final_cache_length is {final_cache_length}; expected {expected_final_length}"
            )

    @property
    def fully_accepted(self) -> bool:
        """Whether every proposal token matched its target decision row."""

        return self.accepted_count == len(self.proposal_token_ids)

    @property
    def accepted_token_ids(self) -> tuple[int, ...]:
        """Return the exact accepted proposal prefix."""

        return self.proposal_token_ids[: self.accepted_count]

    @property
    def rejected_proposal_token_id(self) -> int | None:
        """Return the first rejected proposal token, if one exists."""

        if self.fully_accepted:
            return None
        return self.proposal_token_ids[self.accepted_count]

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Return only tokens newly emitted by this transaction."""

        if self.fully_accepted:
            return self.accepted_token_ids
        return self.accepted_token_ids + (cast(int, self.replacement_token_id),)

    @property
    def uncached_next_token_id(self) -> int | None:
        """Return the uncached mismatch replacement, if the transaction produced one."""

        return self.replacement_token_id


def coordinate_speculative_iteration(
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
) -> SpeculativeIterationResult:
    """Coordinate exactly one proposal, verification, acceptance, and cache outcome."""

    initial_cache_length, vocab_size = _validate_iteration_inputs(
        draft_backend,
        target_backend,
        current_token_id,
        proposal_length=proposal_length,
        draft_select_token=draft_select_token,
        target_select_token=target_select_token,
        draft_root_checkpoint=draft_root_checkpoint,
        target_root_checkpoint=target_root_checkpoint,
    )
    target_verifier = cast(
        BatchedTargetVerificationBackend[TargetLogitsT],
        target_backend,
    )
    owned_checkpoints: tuple[DraftCheckpointT, ...] = ()

    try:
        proposal = generate_draft_proposal(
            draft_backend,
            current_token_id,
            proposal_length=proposal_length,
            select_token=draft_select_token,
        )
        if not isinstance(proposal, DraftProposalResult):
            raise SpeculativeIterationInvariantError(
                "draft proposal operation must return a DraftProposalResult"
            )

        raw_owned_checkpoints = proposal.rollback_checkpoints
        if type(raw_owned_checkpoints) is tuple:
            owned_checkpoints = raw_owned_checkpoints
        else:
            try:
                owned_checkpoints = tuple(raw_owned_checkpoints)
            except (TypeError, ValueError) as exc:
                raise SpeculativeIterationInvariantError(
                    "draft proposal rollback_checkpoints could not be acquired"
                ) from exc

        proposal_token_ids = _validate_draft_proposal(
            draft_backend,
            proposal,
            proposal_length=proposal_length,
            initial_cache_length=initial_cache_length,
            vocab_size=vocab_size,
        )

        verification = target_verifier.verify_proposal(
            current_token_id,
            proposal_token_ids,
        )
        target_logit_rows = _validate_target_verification(
            target_backend,
            verification,
            proposal_length=proposal_length,
            initial_cache_length=initial_cache_length,
        )

        decision = decide_match_replace_acceptance(
            proposal_token_ids,
            target_logit_rows,
            select_token=target_select_token,
        )
        accepted_count, replacement_token_id = _validate_acceptance_decision(
            decision,
            proposal_token_ids=proposal_token_ids,
            vocab_size=vocab_size,
        )

        if accepted_count == proposal_length:
            final_cache_length = initial_cache_length + proposal_length + 1
            _validate_backend_cache_length(
                draft_backend,
                final_cache_length,
                role="draft",
            )
            _validate_backend_cache_length(
                target_backend,
                final_cache_length,
                role="target",
            )
        else:
            final_cache_length = _reconcile_mismatch(
                draft_backend,
                target_backend,
                current_token_id,
                proposal_token_ids=proposal_token_ids,
                accepted_count=accepted_count,
                initial_cache_length=initial_cache_length,
                draft_rollback_checkpoint=owned_checkpoints[accepted_count],
                target_root_checkpoint=target_root_checkpoint,
            )

        result = SpeculativeIterationResult(
            proposal_token_ids=proposal_token_ids,
            accepted_count=accepted_count,
            replacement_token_id=replacement_token_id,
            initial_cache_length=initial_cache_length,
            final_cache_length=final_cache_length,
        )

        for checkpoint in owned_checkpoints:
            draft_backend.release_cache_checkpoint(checkpoint)

        _validate_backend_cache_length(
            draft_backend,
            final_cache_length,
            role="draft",
        )
        _validate_backend_cache_length(
            target_backend,
            final_cache_length,
            role="target",
        )
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_iteration(
            draft_backend,
            target_backend,
            draft_root_checkpoint=draft_root_checkpoint,
            target_root_checkpoint=target_root_checkpoint,
            owned_checkpoints=owned_checkpoints,
            initial_cache_length=initial_cache_length,
        )
        if cleanup_failures:
            raise SpeculativeIterationCleanupError(failure, cleanup_failures) from failure
        raise


def _validate_iteration_inputs(
    draft_backend: object,
    target_backend: object,
    current_token_id: object,
    *,
    proposal_length: object,
    draft_select_token: object,
    target_select_token: object,
    draft_root_checkpoint: object,
    target_root_checkpoint: object,
) -> tuple[int, int]:
    if draft_backend is target_backend:
        raise ValueError("draft_backend and target_backend must be distinct objects")
    if not isinstance(draft_backend, CheckpointableAutoregressiveBackend):
        raise TypeError("draft_backend must satisfy CheckpointableAutoregressiveBackend")
    if not isinstance(target_backend, CheckpointableAutoregressiveBackend):
        raise TypeError("target_backend must satisfy CheckpointableAutoregressiveBackend")
    if not isinstance(target_backend, BatchedTargetVerificationBackend):
        raise TypeError("target_backend must satisfy BatchedTargetVerificationBackend")
    if isinstance(proposal_length, bool) or not isinstance(proposal_length, int):
        raise TypeError("proposal_length must be an integer")
    if proposal_length <= 0:
        raise ValueError("proposal_length must be greater than zero")
    if not callable(draft_select_token):
        raise TypeError("draft_select_token must be callable")
    if not callable(target_select_token):
        raise TypeError("target_select_token must be callable")

    draft = cast(CheckpointableAutoregressiveBackend[object, CacheCheckpoint], draft_backend)
    target = cast(CheckpointableAutoregressiveBackend[object, CacheCheckpoint], target_backend)
    draft_vocab_size = _validate_vocab_size(draft.vocab_size, role="draft")
    target_vocab_size = _validate_vocab_size(target.vocab_size, role="target")
    if draft_vocab_size != target_vocab_size:
        raise SpeculativeIterationInvariantError(
            f"backend vocabulary sizes differ: draft reports {draft_vocab_size}, "
            f"target reports {target_vocab_size}"
        )
    _validate_token_id(current_token_id, draft_vocab_size, label="current_token_id")

    draft_cache_length = _read_backend_cache_length(draft, role="draft")
    target_cache_length = _read_backend_cache_length(target, role="target")
    if draft_cache_length == 0 or target_cache_length == 0:
        raise SpeculativeIterationInvariantError(
            "both backends must have an active nonempty prefilled cache"
        )
    if draft_cache_length != target_cache_length:
        raise SpeculativeIterationInvariantError(
            f"backend cache lengths differ: draft reports {draft_cache_length}, "
            f"target reports {target_cache_length}"
        )
    initial_cache_length = draft_cache_length

    _validate_root_checkpoint(
        draft_root_checkpoint,
        expected_cache_length=initial_cache_length,
        label="draft_root_checkpoint",
    )
    _validate_root_checkpoint(
        target_root_checkpoint,
        expected_cache_length=initial_cache_length,
        label="target_root_checkpoint",
    )

    draft.rollback_cache(cast(CacheCheckpoint, draft_root_checkpoint))
    _validate_backend_cache_length(draft, initial_cache_length, role="draft")
    target.rollback_cache(cast(CacheCheckpoint, target_root_checkpoint))
    _validate_backend_cache_length(target, initial_cache_length, role="target")
    _validate_backend_cache_length(draft, initial_cache_length, role="draft")
    _validate_backend_cache_length(target, initial_cache_length, role="target")
    return initial_cache_length, draft_vocab_size


def _validate_draft_proposal(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT,
        DraftCheckpointT,
    ],
    proposal: DraftProposalResult[DraftCheckpointT],
    *,
    proposal_length: int,
    initial_cache_length: int,
    vocab_size: int,
) -> tuple[int, ...]:
    proposal_token_ids = proposal.proposal_token_ids
    if type(proposal_token_ids) is not tuple:
        raise SpeculativeIterationInvariantError("draft proposal_token_ids must be an exact tuple")
    if len(proposal_token_ids) != proposal_length:
        raise SpeculativeIterationInvariantError(
            f"draft proposal contains {len(proposal_token_ids)} tokens; expected {proposal_length}"
        )
    for position, token_id in enumerate(proposal_token_ids):
        try:
            _validate_token_id(
                token_id,
                vocab_size,
                label=f"draft proposal token at position {position}",
            )
        except (TypeError, ValueError) as exc:
            raise SpeculativeIterationInvariantError(str(exc)) from exc

    reported_initial_cache_length = _validate_cache_length_metadata(
        proposal.initial_cache_length,
        label="draft proposal initial_cache_length",
    )
    if reported_initial_cache_length != initial_cache_length:
        raise SpeculativeIterationInvariantError(
            f"draft proposal initial cache length is {reported_initial_cache_length}; "
            f"expected {initial_cache_length}"
        )
    expected_final_length = initial_cache_length + proposal_length + 1
    reported_final_cache_length = _validate_cache_length_metadata(
        proposal.final_cache_length,
        label="draft proposal final_cache_length",
    )
    if reported_final_cache_length != expected_final_length:
        raise SpeculativeIterationInvariantError(
            f"draft proposal final cache length is {reported_final_cache_length}; "
            f"expected {expected_final_length}"
        )

    checkpoints = proposal.rollback_checkpoints
    if type(checkpoints) is not tuple:
        raise SpeculativeIterationInvariantError(
            "draft proposal rollback_checkpoints must be an exact tuple"
        )
    if len(checkpoints) != proposal_length:
        raise SpeculativeIterationInvariantError(
            f"draft proposal contains {len(checkpoints)} rollback checkpoints; "
            f"expected {proposal_length}"
        )
    for position, checkpoint in enumerate(checkpoints):
        _validate_checkpoint(
            checkpoint,
            expected_cache_length=initial_cache_length + 1 + position,
            label=f"draft proposal checkpoint at position {position}",
        )

    _validate_backend_cache_length(
        draft_backend,
        expected_final_length,
        role="draft",
    )
    return proposal_token_ids


def _validate_target_verification(
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT,
        TargetCheckpointT,
    ],
    verification: object,
    *,
    proposal_length: int,
    initial_cache_length: int,
) -> tuple[TargetLogitsT, ...]:
    if not isinstance(verification, BatchedTargetVerificationResult):
        raise SpeculativeIterationInvariantError(
            "target verification must return a BatchedTargetVerificationResult"
        )
    logit_rows = verification.logit_rows
    if type(logit_rows) is not tuple:
        raise SpeculativeIterationInvariantError(
            "target verification logit_rows must be an exact tuple"
        )
    expected_row_count = proposal_length + 1
    if len(logit_rows) != expected_row_count:
        raise SpeculativeIterationInvariantError(
            f"target verification returned {len(logit_rows)} rows; expected {expected_row_count}"
        )
    expected_cache_length = initial_cache_length + expected_row_count
    reported_cache_length = _validate_cache_length_metadata(
        verification.cache_length,
        label="target verification cache_length",
    )
    if reported_cache_length != expected_cache_length:
        raise SpeculativeIterationInvariantError(
            f"target verification cache length is {reported_cache_length}; "
            f"expected {expected_cache_length}"
        )
    _validate_backend_cache_length(
        target_backend,
        expected_cache_length,
        role="target",
    )
    return logit_rows


def _validate_acceptance_decision(
    decision: object,
    *,
    proposal_token_ids: tuple[int, ...],
    vocab_size: int,
) -> tuple[int, int | None]:
    if not isinstance(decision, MatchReplaceAcceptanceResult):
        raise SpeculativeIterationInvariantError(
            "acceptance operation must return a MatchReplaceAcceptanceResult"
        )
    if type(decision.proposal_token_ids) is not tuple:
        raise SpeculativeIterationInvariantError(
            "acceptance result proposal_token_ids must be an exact tuple"
        )
    if decision.proposal_token_ids != proposal_token_ids:
        raise SpeculativeIterationInvariantError(
            "acceptance result proposal_token_ids differ from the draft proposal"
        )

    accepted_count = decision.accepted_count
    if isinstance(accepted_count, bool) or not isinstance(accepted_count, int):
        raise SpeculativeIterationInvariantError(
            "acceptance result accepted_count must be an integer"
        )
    proposal_length = len(proposal_token_ids)
    if accepted_count < 0 or accepted_count > proposal_length:
        raise SpeculativeIterationInvariantError(
            f"acceptance result accepted_count must be within [0, {proposal_length}]"
        )

    replacement_token_id = decision.replacement_token_id
    if accepted_count == proposal_length:
        if replacement_token_id is not None:
            raise SpeculativeIterationInvariantError(
                "fully accepted decision cannot contain a replacement token"
            )
        return accepted_count, None
    if replacement_token_id is None:
        raise SpeculativeIterationInvariantError(
            "mismatch decision must contain a replacement token"
        )
    try:
        _validate_token_id(
            replacement_token_id,
            vocab_size,
            label="target-selected replacement_token_id",
        )
    except (TypeError, ValueError) as exc:
        raise SpeculativeIterationInvariantError(str(exc)) from exc
    if replacement_token_id == proposal_token_ids[accepted_count]:
        raise SpeculativeIterationInvariantError(
            "replacement token must differ from the rejected proposal token"
        )
    return accepted_count, replacement_token_id


def _reconcile_mismatch(
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
    proposal_token_ids: tuple[int, ...],
    accepted_count: int,
    initial_cache_length: int,
    draft_rollback_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
) -> int:
    final_cache_length = initial_cache_length + accepted_count + 1
    draft_backend.rollback_cache(draft_rollback_checkpoint)
    _validate_backend_cache_length(
        draft_backend,
        final_cache_length,
        role="draft",
    )

    target_backend.rollback_cache(target_root_checkpoint)
    _validate_backend_cache_length(
        target_backend,
        initial_cache_length,
        role="target",
    )
    replay_token_ids = (current_token_id, *proposal_token_ids[:accepted_count])
    for position, token_id in enumerate(replay_token_ids):
        expected_cache_length = initial_cache_length + position + 1
        step = target_backend.decode(token_id)
        _validate_replay_step(
            target_backend,
            step,
            expected_cache_length=expected_cache_length,
        )
        del step

    _validate_backend_cache_length(
        draft_backend,
        final_cache_length,
        role="draft",
    )
    _validate_backend_cache_length(
        target_backend,
        final_cache_length,
        role="target",
    )
    return final_cache_length


def _validate_replay_step(
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT,
        TargetCheckpointT,
    ],
    step: object,
    *,
    expected_cache_length: int,
) -> None:
    if not isinstance(step, ModelStep):
        raise SpeculativeIterationInvariantError("target replay decode must return a ModelStep")
    reported_cache_length = _validate_cache_length_metadata(
        step.cache_length,
        label="target replay step cache_length",
    )
    if reported_cache_length != expected_cache_length:
        raise SpeculativeIterationInvariantError(
            f"target replay step reported cache length {reported_cache_length}; "
            f"expected {expected_cache_length}"
        )
    _validate_backend_cache_length(
        target_backend,
        expected_cache_length,
        role="target",
    )


def _cleanup_failed_iteration(
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
    owned_checkpoints: Sequence[DraftCheckpointT],
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
        cleanup_failures.append(("draft root rollback", cleanup_failure))

    try:
        target_backend.rollback_cache(target_root_checkpoint)
        _validate_backend_cache_length(
            target_backend,
            initial_cache_length,
            role="target",
        )
    except Exception as cleanup_failure:
        cleanup_failures.append(("target root rollback", cleanup_failure))

    for position, checkpoint in enumerate(owned_checkpoints):
        try:
            draft_backend.release_cache_checkpoint(checkpoint)
        except Exception as cleanup_failure:
            cleanup_failures.append(
                (f"draft proposal checkpoint {position} release", cleanup_failure)
            )

    return tuple(cleanup_failures)


def _validate_vocab_size(value: object, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeculativeIterationInvariantError(f"{role} backend vocab_size must be an integer")
    if value <= 0:
        raise SpeculativeIterationInvariantError(
            f"{role} backend vocab_size must be greater than zero"
        )
    return value


def _read_backend_cache_length(
    backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
    *,
    role: str,
) -> int:
    return _validate_cache_length_metadata(
        backend.cache_length,
        label=f"{role} backend cache_length",
    )


def _validate_backend_cache_length(
    backend: CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
    expected_cache_length: int,
    *,
    role: str,
) -> int:
    cache_length = _read_backend_cache_length(backend, role=role)
    if cache_length != expected_cache_length:
        raise SpeculativeIterationInvariantError(
            f"{role} backend reported cache length {cache_length}; expected {expected_cache_length}"
        )
    return cache_length


def _validate_cache_length_metadata(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeculativeIterationInvariantError(f"{label} must be an integer")
    if value < 0:
        raise SpeculativeIterationInvariantError(f"{label} cannot be negative")
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
        raise SpeculativeIterationInvariantError(f"{label} must satisfy CacheCheckpoint") from exc
    if not is_checkpoint:
        raise SpeculativeIterationInvariantError(f"{label} must satisfy CacheCheckpoint")
    try:
        cache_length = checkpoint.cache_length
    except Exception as exc:
        raise SpeculativeIterationInvariantError(f"{label} cache_length could not be read") from exc
    cache_length = _validate_cache_length_metadata(
        cache_length,
        label=f"{label} cache_length",
    )
    if cache_length != expected_cache_length:
        raise SpeculativeIterationInvariantError(
            f"{label} reports cache length {cache_length}; expected {expected_cache_length}"
        )


def _validate_root_checkpoint(
    checkpoint: object,
    *,
    expected_cache_length: int,
    label: str,
) -> None:
    try:
        is_checkpoint = isinstance(checkpoint, CacheCheckpoint)
    except Exception as exc:
        raise TypeError(f"{label} must satisfy CacheCheckpoint") from exc
    if not is_checkpoint:
        raise TypeError(f"{label} must satisfy CacheCheckpoint")
    try:
        cache_length = checkpoint.cache_length
    except Exception as exc:
        raise SpeculativeIterationInvariantError(f"{label} cache_length could not be read") from exc
    cache_length = _validate_cache_length_metadata(
        cache_length,
        label=f"{label} cache_length",
    )
    if cache_length != expected_cache_length:
        raise SpeculativeIterationInvariantError(
            f"{label} reports cache length {cache_length}; expected {expected_cache_length}"
        )


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


def _validate_token_id(token_id: object, vocab_size: int, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{label} {token_id} is outside vocabulary range [0, {vocab_size})")


def _validate_nonnegative_token_id(token_id: object, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0:
        raise ValueError(f"{label} cannot be negative")


__all__ = [
    "SpeculativeIterationCleanupError",
    "SpeculativeIterationError",
    "SpeculativeIterationInvariantError",
    "SpeculativeIterationResult",
    "coordinate_speculative_iteration",
]
