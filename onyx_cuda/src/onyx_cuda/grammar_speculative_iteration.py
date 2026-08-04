"""One framework-neutral grammar-masked speculative decoding transaction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from .backend import ModelStep
from .cache import CacheCheckpoint, CheckpointableAutoregressiveBackend
from .constrained_generation import GrammarLogitMask
from .grammar import GrammarConstraint
from .grammar_acceptance import (
    GrammarMaskedTargetAcceptanceResult,
    decide_grammar_masked_target_acceptance,
)
from .grammar_continuation import (
    GrammarMaskedPostAcceptanceContinuationResult,
    decide_grammar_masked_post_acceptance_continuation,
)
from .grammar_draft import (
    GrammarMaskedDraftProposalResult,
    generate_grammar_masked_draft_proposal,
)
from .grammar_selection import GrammarMaskedSelectionResult
from .speculative_iteration import SpeculativeIterationError
from .verification import (
    BatchedTargetVerificationBackend,
    BatchedTargetVerificationResult,
)


class GrammarMaskedSpeculativeIterationError(SpeculativeIterationError):
    """Base error raised by one grammar-masked speculative transaction."""


class GrammarMaskedSpeculativeIterationInvariantError(
    GrammarMaskedSpeculativeIterationError
):
    """Raised when composed cache or grammar evidence violates the D47 contract."""


class GrammarMaskedSpeculativeIterationCleanupError(
    GrammarMaskedSpeculativeIterationError
):
    """Raised when a failed transaction cannot settle every owned resource."""

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
            f"grammar-masked speculative iteration failed: {original_failure}; {details}"
        )


DraftLogitsT = TypeVar("DraftLogitsT")
DraftCheckpointT = TypeVar("DraftCheckpointT", bound=CacheCheckpoint)
TargetLogitsT = TypeVar("TargetLogitsT")
TargetCheckpointT = TypeVar("TargetCheckpointT", bound=CacheCheckpoint)
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class GrammarMaskedSpeculativeIterationResult(Generic[StateT]):
    """One emitted outcome, aligned cache pair, and transferred grammar state."""

    proposal_token_ids: tuple[int, ...]
    accepted_count: int
    replacement_token_id: int | None
    initial_cache_length: int
    final_cache_length: int
    uncached_next_token_id: int | None
    shortening_selection: GrammarMaskedSelectionResult | None
    acceptance_no_decision_selection: GrammarMaskedSelectionResult | None
    final_row_no_decision_selection: GrammarMaskedSelectionResult | None
    committed_state: StateT
    committed_state_is_match: bool

    def __post_init__(self) -> None:
        _validate_result_proposal(self.proposal_token_ids)
        proposal_length = len(self.proposal_token_ids)

        if type(self.accepted_count) is not int:
            raise TypeError("accepted_count must be an integer")
        if self.accepted_count < 0 or self.accepted_count > proposal_length:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"accepted_count must be within [0, {proposal_length}]"
            )
        if type(self.committed_state_is_match) is not bool:
            raise TypeError("committed_state_is_match must be a boolean")

        initial_cache_length = _validate_result_cache_length(
            self.initial_cache_length,
            label="initial_cache_length",
        )
        if initial_cache_length == 0:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "initial_cache_length must be greater than zero"
            )
        final_cache_length = _validate_result_cache_length(
            self.final_cache_length,
            label="final_cache_length",
        )

        shortening_is_match = _validate_optional_empty_selection_for_result(
            self.shortening_selection,
            label="shortening_selection",
        )
        acceptance_no_decision_is_match = (
            _validate_optional_empty_selection_for_result(
                self.acceptance_no_decision_selection,
                label="acceptance_no_decision_selection",
            )
        )
        final_row_no_decision_is_match = (
            _validate_optional_empty_selection_for_result(
                self.final_row_no_decision_selection,
                label="final_row_no_decision_selection",
            )
        )
        if (
            self.acceptance_no_decision_selection is not None
            and self.final_row_no_decision_selection is not None
        ):
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "acceptance and final-row no-decision selections are mutually exclusive"
            )

        if proposal_length == 0:
            if self.accepted_count != 0:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "an empty proposal requires accepted_count zero"
                )
            if self.shortening_selection is None:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "an empty proposal requires shortening_selection"
                )
            if (
                self.replacement_token_id is not None
                or self.uncached_next_token_id is not None
                or self.acceptance_no_decision_selection is not None
                or self.final_row_no_decision_selection is not None
            ):
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "an empty proposal cannot contain target decision evidence"
                )
            if final_cache_length != initial_cache_length + 1:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    f"final_cache_length is {final_cache_length}; "
                    f"expected {initial_cache_length + 1}"
                )
            if self.committed_state_is_match is not shortening_is_match:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "empty-proposal match fact must equal shortening_selection"
                )
            return

        if self.acceptance_no_decision_selection is not None:
            if self.accepted_count >= proposal_length:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "acceptance no-decision must stop before full acceptance"
                )
            if self.replacement_token_id is not None:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "acceptance no-decision cannot contain a replacement token"
                )
            if self.uncached_next_token_id is not None:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "acceptance no-decision cannot contain an uncached token"
                )
            if self.committed_state_is_match is not acceptance_no_decision_is_match:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "no-decision match fact must equal its terminal selection"
                )
            expected_final_length = initial_cache_length + 1 + self.accepted_count
        elif self.accepted_count < proposal_length:
            if self.replacement_token_id is None:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "a decided mismatch requires a replacement token"
                )
            _validate_result_nonnegative_token(
                self.replacement_token_id,
                label="replacement_token_id",
            )
            if self.replacement_token_id == self.proposal_token_ids[self.accepted_count]:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "replacement_token_id must differ from the rejected proposal token"
                )
            _validate_result_nonnegative_token(
                self.uncached_next_token_id,
                label="uncached_next_token_id",
            )
            if self.uncached_next_token_id != self.replacement_token_id:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "mismatch uncached_next_token_id must equal replacement_token_id"
                )
            if self.final_row_no_decision_selection is not None:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "a mismatch cannot contain final-row no-decision evidence"
                )
            expected_final_length = initial_cache_length + 1 + self.accepted_count
        else:
            if self.replacement_token_id is not None:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "fully accepted result cannot contain a replacement token"
                )
            if self.final_row_no_decision_selection is None:
                if self.uncached_next_token_id is None:
                    raise GrammarMaskedSpeculativeIterationInvariantError(
                        "decided full acceptance requires an uncached bonus token"
                    )
                _validate_result_nonnegative_token(
                    self.uncached_next_token_id,
                    label="uncached_next_token_id",
                )
            else:
                if self.uncached_next_token_id is not None:
                    raise GrammarMaskedSpeculativeIterationInvariantError(
                        "final-row no-decision cannot contain an uncached token"
                    )
                if self.committed_state_is_match is not final_row_no_decision_is_match:
                    raise GrammarMaskedSpeculativeIterationInvariantError(
                        "final-row match fact must equal its terminal selection"
                    )
            expected_final_length = initial_cache_length + 1 + proposal_length

        if final_cache_length != expected_final_length:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"final_cache_length is {final_cache_length}; "
                f"expected {expected_final_length}"
            )

    @property
    def shortened(self) -> bool:
        """Whether draft grammar support ended before the caller's bound."""

        return self.shortening_selection is not None

    @property
    def acceptance_decision_made(self) -> bool:
        """Whether nonempty target acceptance reached mismatch or full acceptance."""

        return bool(self.proposal_token_ids) and self.acceptance_no_decision_selection is None

    @property
    def fully_accepted(self) -> bool:
        """Whether a decided target outcome accepted every nonempty proposal token."""

        return (
            self.acceptance_decision_made
            and self.accepted_count == len(self.proposal_token_ids)
        )

    @property
    def accepted_token_ids(self) -> tuple[int, ...]:
        """Return the exact accepted proposal prefix."""

        return self.proposal_token_ids[: self.accepted_count]

    @property
    def rejected_proposal_token_id(self) -> int | None:
        """Return the rejected proposal token only for an actual mismatch."""

        if self.replacement_token_id is None:
            return None
        return self.proposal_token_ids[self.accepted_count]

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Return all tokens emitted by this transaction."""

        if self.uncached_next_token_id is None:
            return self.accepted_token_ids
        return self.accepted_token_ids + (self.uncached_next_token_id,)


@dataclass(slots=True)
class _OwnedCheckpoint:
    position: int
    checkpoint: object
    owned: bool = True


def coordinate_grammar_masked_speculative_iteration(
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
) -> GrammarMaskedSpeculativeIterationResult[StateT]:
    """Coordinate exactly one grammar-masked speculative transaction.

    Successful preflight consumes ``starting_state``. A successful result transfers exactly
    one live ``committed_state``; callers must not release the input state separately afterward.
    """

    initial_cache_length, vocab_size, initial_state_is_match = _validate_iteration_inputs(
        draft_backend,
        target_backend,
        current_token_id,
        constraint,
        starting_state,
        proposal_bound=proposal_bound,
        draft_select_token=draft_select_token,
        target_select_token=target_select_token,
        draft_root_checkpoint=draft_root_checkpoint,
        target_root_checkpoint=target_root_checkpoint,
    )
    target_verifier = cast(
        BatchedTargetVerificationBackend[TargetLogitsT],
        target_backend,
    )

    owned_checkpoints: list[_OwnedCheckpoint] = []
    starting_state_owned = True
    downstream_state: object = None
    downstream_state_owned = False

    try:
        proposal_result = generate_grammar_masked_draft_proposal(
            draft_backend,
            current_token_id,
            constraint,
            starting_state,
            draft_logit_mask,
            proposal_bound=proposal_bound,
            select_token=draft_select_token,
        )
        if not isinstance(proposal_result, GrammarMaskedDraftProposalResult):
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D44 must return a GrammarMaskedDraftProposalResult"
            )

        raw_checkpoints = _read_attribute(
            proposal_result,
            "rollback_checkpoints",
            label="D44 result",
        )
        if type(raw_checkpoints) is tuple:
            acquired_checkpoints = raw_checkpoints
        else:
            try:
                acquired_checkpoints = tuple(raw_checkpoints)
            except (TypeError, ValueError) as exc:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "D44 rollback_checkpoints could not be acquired"
                ) from exc
        owned_checkpoints = _register_acquired_checkpoints(
            acquired_checkpoints,
            draft_root_checkpoint=draft_root_checkpoint,
            target_root_checkpoint=target_root_checkpoint,
        )
        if any(
            record.checkpoint is draft_root_checkpoint
            or record.checkpoint is target_root_checkpoint
            for record in owned_checkpoints
        ):
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D44 rollback checkpoints cannot alias either caller root"
            )

        proposal_token_ids, shortening_selection = _validate_draft_result(
            draft_backend,
            target_backend,
            constraint,
            proposal_result,
            acquired_checkpoints=acquired_checkpoints,
            proposal_bound=proposal_bound,
            initial_cache_length=initial_cache_length,
            vocab_size=vocab_size,
        )
        proposal_length = len(proposal_token_ids)

        if proposal_length == 0:
            step = target_backend.decode(current_token_id)
            _validate_target_decode_step(
                target_backend,
                step,
                expected_cache_length=initial_cache_length + 1,
                label="zero-proposal target decode",
            )
            final_cache_length = initial_cache_length + 1
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
            shortening_match = _validate_empty_selection_evidence(
                shortening_selection,
                label="D44 shortening_selection",
            )
            final_state = starting_state
            final_state_is_match = shortening_match
            accepted_count = 0
            replacement_token_id = None
            acceptance_no_decision_selection = None
            final_row_no_decision_selection = None
            uncached_next_token_id = None
        else:
            verification = target_verifier.verify_proposal(
                current_token_id,
                proposal_token_ids,
            )
            target_logit_rows = _validate_target_verification(
                draft_backend,
                target_backend,
                constraint,
                verification,
                proposal_length=proposal_length,
                initial_cache_length=initial_cache_length,
                vocab_size=vocab_size,
            )

            acceptance_result = decide_grammar_masked_target_acceptance(
                proposal_token_ids,
                target_logit_rows,
                constraint,
                starting_state,
                target_logit_mask,
                vocab_size=vocab_size,
                select_token=target_select_token,
            )
            (
                acquired_acceptance_state_transferred,
                acquired_acceptance_state,
            ) = _acquire_acceptance_state(acceptance_result)
            if acquired_acceptance_state_transferred:
                downstream_state = acquired_acceptance_state
                downstream_state_owned = True
            (
                accepted_count,
                replacement_token_id,
                acceptance_no_decision_selection,
                acceptance_state,
                acceptance_state_is_match,
            ) = _validate_acceptance_result(
                constraint,
                acceptance_result,
                proposal_token_ids=proposal_token_ids,
                starting_state=starting_state,
                initial_state_is_match=initial_state_is_match,
                vocab_size=vocab_size,
                acquired_state_transferred=acquired_acceptance_state_transferred,
                acquired_state=acquired_acceptance_state,
            )

            if accepted_count < proposal_length:
                final_cache_length = _reconcile_partial_outcome(
                    draft_backend,
                    target_backend,
                    current_token_id,
                    proposal_token_ids=proposal_token_ids,
                    accepted_count=accepted_count,
                    initial_cache_length=initial_cache_length,
                    draft_rollback_checkpoint=cast(
                        DraftCheckpointT,
                        owned_checkpoints[accepted_count].checkpoint,
                    ),
                    target_root_checkpoint=target_root_checkpoint,
                )
            else:
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

            if acceptance_no_decision_selection is not None:
                if accepted_count == 0:
                    final_state = starting_state
                    final_state_is_match = _validate_empty_selection_evidence(
                        acceptance_no_decision_selection,
                        label="D45 no_decision_selection",
                    )
                else:
                    final_state = cast(StateT, acceptance_state)
                    final_state_is_match = cast(bool, acceptance_state_is_match)
                final_row_no_decision_selection = None
                uncached_next_token_id = None
            else:
                continuation_result = (
                    decide_grammar_masked_post_acceptance_continuation(
                        proposal_token_ids,
                        target_logit_rows,
                        cast(
                            GrammarMaskedTargetAcceptanceResult[StateT],
                            acceptance_result,
                        ),
                        constraint,
                        target_logit_mask,
                        vocab_size=vocab_size,
                        select_token=target_select_token,
                    )
                )
                acquired_continuation_state = _acquire_continuation_state(
                    continuation_result
                )
                if acquired_continuation_state is not acceptance_state:
                    downstream_state = acquired_continuation_state
                (
                    uncached_next_token_id,
                    final_row_no_decision_selection,
                    continuation_state,
                    continuation_state_is_match,
                ) = _validate_continuation_result(
                    constraint,
                    continuation_result,
                    proposal_token_ids=proposal_token_ids,
                    accepted_count=accepted_count,
                    replacement_token_id=replacement_token_id,
                    acceptance_state=acceptance_state,
                    acceptance_state_is_match=cast(bool, acceptance_state_is_match),
                    starting_state=starting_state,
                    vocab_size=vocab_size,
                    acquired_state=acquired_continuation_state,
                )
                final_state = cast(StateT, continuation_state)
                final_state_is_match = continuation_state_is_match

        result = GrammarMaskedSpeculativeIterationResult(
            proposal_token_ids=proposal_token_ids,
            accepted_count=accepted_count,
            replacement_token_id=replacement_token_id,
            initial_cache_length=initial_cache_length,
            final_cache_length=final_cache_length,
            uncached_next_token_id=uncached_next_token_id,
            shortening_selection=shortening_selection,
            acceptance_no_decision_selection=acceptance_no_decision_selection,
            final_row_no_decision_selection=final_row_no_decision_selection,
            committed_state=final_state,
            committed_state_is_match=final_state_is_match,
        )
        _validate_composed_result(
            result,
            proposal_token_ids=proposal_token_ids,
            accepted_count=accepted_count,
            replacement_token_id=replacement_token_id,
            initial_cache_length=initial_cache_length,
            final_cache_length=final_cache_length,
            uncached_next_token_id=uncached_next_token_id,
            shortening_selection=shortening_selection,
            acceptance_no_decision_selection=acceptance_no_decision_selection,
            final_row_no_decision_selection=final_row_no_decision_selection,
            committed_state=final_state,
            committed_state_is_match=final_state_is_match,
        )

        for record in owned_checkpoints:
            if not record.owned:
                continue
            draft_backend.release_cache_checkpoint(
                cast(DraftCheckpointT, record.checkpoint)
            )
            record.owned = False

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

        if final_state is starting_state:
            if not starting_state_owned or downstream_state_owned:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "grammar-state ownership is inconsistent before transfer"
                )
            _validate_live_state(
                constraint,
                starting_state,
                expected_is_match=initial_state_is_match,
                label="starting_state",
            )
        else:
            if not downstream_state_owned or downstream_state is not final_state:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "downstream grammar-state ownership is inconsistent before transfer"
                )

        _validate_live_state(
            constraint,
            final_state,
            expected_is_match=final_state_is_match,
            label="committed_state",
        )

        if final_state is not starting_state:
            constraint.release_state(starting_state)
            starting_state_owned = False
            _validate_live_state(
                constraint,
                final_state,
                expected_is_match=final_state_is_match,
                label="committed_state",
            )

        if final_state is starting_state:
            starting_state_owned = False
        else:
            downstream_state_owned = False
        return result
    except BaseException as failure:
        cleanup_failures = _cleanup_failed_iteration(
            draft_backend,
            target_backend,
            constraint,
            draft_root_checkpoint=draft_root_checkpoint,
            target_root_checkpoint=target_root_checkpoint,
            owned_checkpoints=owned_checkpoints,
            initial_cache_length=initial_cache_length,
            starting_state=starting_state,
            starting_state_owned=starting_state_owned,
            downstream_state=downstream_state,
            downstream_state_owned=downstream_state_owned,
        )
        if cleanup_failures:
            raise GrammarMaskedSpeculativeIterationCleanupError(
                failure,
                cleanup_failures,
            ) from failure
        raise


def _validate_iteration_inputs(
    draft_backend: object,
    target_backend: object,
    current_token_id: object,
    constraint: object,
    starting_state: StateT,
    *,
    proposal_bound: object,
    draft_select_token: object,
    target_select_token: object,
    draft_root_checkpoint: object,
    target_root_checkpoint: object,
) -> tuple[int, int, bool]:
    if draft_backend is target_backend:
        raise ValueError("draft_backend and target_backend must be distinct objects")
    _require_protocol_conformance(
        draft_backend,
        CheckpointableAutoregressiveBackend,
        label="draft_backend",
    )
    _require_protocol_conformance(
        target_backend,
        CheckpointableAutoregressiveBackend,
        label="target_backend",
    )
    _require_protocol_conformance(
        target_backend,
        BatchedTargetVerificationBackend,
        label="target_backend",
    )
    if isinstance(proposal_bound, bool) or not isinstance(proposal_bound, int):
        raise TypeError("proposal_bound must be an integer")
    if proposal_bound <= 0:
        raise ValueError("proposal_bound must be greater than zero")
    if not callable(draft_select_token):
        raise TypeError("draft_select_token must be callable")
    if not callable(target_select_token):
        raise TypeError("target_select_token must be callable")

    draft = cast(
        CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
        draft_backend,
    )
    target = cast(
        CheckpointableAutoregressiveBackend[object, CacheCheckpoint],
        target_backend,
    )
    draft_vocab_size = _read_positive_vocab_size(draft, label="draft backend")
    target_vocab_size = _read_positive_vocab_size(target, label="target backend")
    if draft_vocab_size != target_vocab_size:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"backend vocabulary sizes differ: draft reports {draft_vocab_size}, "
            f"target reports {target_vocab_size}"
        )
    _validate_token_id(current_token_id, draft_vocab_size, label="current_token_id")

    draft_cache_length = _read_backend_cache_length(draft, role="draft")
    target_cache_length = _read_backend_cache_length(target, role="target")
    if draft_cache_length == 0 or target_cache_length == 0:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "both backends must have an active nonempty prefilled cache"
        )
    if draft_cache_length != target_cache_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
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

    _require_protocol_conformance(constraint, GrammarConstraint, label="constraint")
    constraint_vocab_size = _read_positive_vocab_size(constraint, label="constraint")
    if constraint_vocab_size != draft_vocab_size:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "backend and constraint vocabulary sizes must match exactly: "
            f"backend={draft_vocab_size}, constraint={constraint_vocab_size}"
        )
    grammar_type = _read_attribute(constraint, "grammar_type", label="constraint")
    if type(grammar_type) is not str or grammar_type not in {"regex", "json_schema"}:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "constraint grammar_type must be 'regex' or 'json_schema'"
        )

    typed_constraint = cast(GrammarConstraint[StateT], constraint)
    initial_state_is_match = _validate_live_state(
        typed_constraint,
        starting_state,
        expected_is_match=None,
        label="starting_state",
    )

    draft.rollback_cache(cast(CacheCheckpoint, draft_root_checkpoint))
    _validate_backend_cache_length(draft, initial_cache_length, role="draft")
    _validate_backend_cache_length(target, initial_cache_length, role="target")
    target.rollback_cache(cast(CacheCheckpoint, target_root_checkpoint))
    _validate_backend_cache_length(draft, initial_cache_length, role="draft")
    _validate_backend_cache_length(target, initial_cache_length, role="target")

    _validate_live_state(
        typed_constraint,
        starting_state,
        expected_is_match=initial_state_is_match,
        label="starting_state",
    )
    return initial_cache_length, draft_vocab_size, initial_state_is_match


def _register_acquired_checkpoints(
    checkpoints: tuple[object, ...],
    *,
    draft_root_checkpoint: object,
    target_root_checkpoint: object,
) -> list[_OwnedCheckpoint]:
    records: list[_OwnedCheckpoint] = []
    seen: list[object] = []
    for position, checkpoint in enumerate(checkpoints):
        owned = not (
            checkpoint is draft_root_checkpoint
            or checkpoint is target_root_checkpoint
            or _contains_identity(seen, checkpoint)
        )
        records.append(
            _OwnedCheckpoint(
                position=position,
                checkpoint=checkpoint,
                owned=owned,
            )
        )
        seen.append(checkpoint)
    return records


def _validate_draft_result(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT, DraftCheckpointT
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT, TargetCheckpointT
    ],
    constraint: GrammarConstraint[StateT],
    result: GrammarMaskedDraftProposalResult[DraftCheckpointT],
    *,
    acquired_checkpoints: tuple[object, ...],
    proposal_bound: int,
    initial_cache_length: int,
    vocab_size: int,
) -> tuple[tuple[int, ...], GrammarMaskedSelectionResult | None]:
    proposal_token_ids = _read_attribute(
        result,
        "proposal_token_ids",
        label="D44 result",
    )
    if type(proposal_token_ids) is not tuple:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D44 proposal_token_ids must be an exact tuple"
        )
    if len(proposal_token_ids) > proposal_bound:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"D44 proposal contains {len(proposal_token_ids)} tokens; "
            f"bound is {proposal_bound}"
        )
    for position, token_id in enumerate(proposal_token_ids):
        _validate_operation_token(
            token_id,
            vocab_size,
            label=f"D44 proposal token at position {position}",
        )

    reported_initial = _read_cache_length_metadata(
        _read_attribute(result, "initial_cache_length", label="D44 result"),
        label="D44 initial_cache_length",
    )
    if reported_initial != initial_cache_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"D44 initial cache length is {reported_initial}; "
            f"expected {initial_cache_length}"
        )
    expected_final = initial_cache_length + 1 + len(proposal_token_ids)
    reported_final = _read_cache_length_metadata(
        _read_attribute(result, "final_cache_length", label="D44 result"),
        label="D44 final_cache_length",
    )
    if reported_final != expected_final:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"D44 final cache length is {reported_final}; expected {expected_final}"
        )

    result_checkpoints = _read_attribute(
        result,
        "rollback_checkpoints",
        label="D44 result",
    )
    if type(result_checkpoints) is not tuple:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D44 rollback_checkpoints must be an exact tuple"
        )
    if result_checkpoints is not acquired_checkpoints:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D44 result changed its rollback checkpoint tuple after acquisition"
        )
    if len(result_checkpoints) != len(proposal_token_ids):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D44 must return exactly one rollback checkpoint per proposal token"
        )
    seen_checkpoints: list[object] = []
    for position, checkpoint in enumerate(result_checkpoints):
        if _contains_identity(seen_checkpoints, checkpoint):
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"D44 checkpoint at position {position} aliases an earlier checkpoint"
            )
        seen_checkpoints.append(checkpoint)
        _validate_checkpoint(
            checkpoint,
            expected_cache_length=initial_cache_length + 1 + position,
            label=f"D44 checkpoint at position {position}",
        )

    shortening_selection = _read_attribute(
        result,
        "shortening_selection",
        label="D44 result",
    )
    if shortening_selection is not None:
        _validate_empty_selection_evidence(
            shortening_selection,
            label="D44 shortening_selection",
        )
        if len(proposal_token_ids) >= proposal_bound:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "a shortened D44 result must contain fewer than proposal_bound tokens"
            )
    elif len(proposal_token_ids) != proposal_bound:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "a full-bound D44 result must contain proposal_bound tokens"
        )
    shortened = _read_attribute(result, "shortened", label="D44 result")
    if type(shortened) is not bool or shortened is not (shortening_selection is not None):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D44 result reports inconsistent shortening evidence"
        )

    _validate_backend_cache_length(draft_backend, expected_final, role="draft")
    _validate_backend_cache_length(
        target_backend,
        initial_cache_length,
        role="target",
    )
    _validate_common_vocab_metadata(
        draft_backend,
        target_backend,
        constraint,
        expected_vocab_size=vocab_size,
    )
    return cast(tuple[int, ...], proposal_token_ids), cast(
        GrammarMaskedSelectionResult | None,
        shortening_selection,
    )


def _validate_target_verification(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT, DraftCheckpointT
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT, TargetCheckpointT
    ],
    constraint: GrammarConstraint[StateT],
    verification: object,
    *,
    proposal_length: int,
    initial_cache_length: int,
    vocab_size: int,
) -> tuple[TargetLogitsT, ...]:
    if not isinstance(verification, BatchedTargetVerificationResult):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "target verification must return a BatchedTargetVerificationResult"
        )
    logit_rows = _read_attribute(
        verification,
        "logit_rows",
        label="target verification result",
    )
    if type(logit_rows) is not tuple:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "target verification logit_rows must be an exact tuple"
        )
    expected_row_count = proposal_length + 1
    if len(logit_rows) != expected_row_count:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"target verification returned {len(logit_rows)} rows; "
            f"expected {expected_row_count}"
        )
    expected_cache_length = initial_cache_length + expected_row_count
    reported_cache_length = _read_cache_length_metadata(
        _read_attribute(
            verification,
            "cache_length",
            label="target verification result",
        ),
        label="target verification cache_length",
    )
    if reported_cache_length != expected_cache_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"target verification cache length is {reported_cache_length}; "
            f"expected {expected_cache_length}"
        )
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
    _validate_common_vocab_metadata(
        draft_backend,
        target_backend,
        constraint,
        expected_vocab_size=vocab_size,
    )
    return cast(tuple[TargetLogitsT, ...], logit_rows)


def _validate_acceptance_result(
    constraint: GrammarConstraint[StateT],
    result: object,
    *,
    proposal_token_ids: tuple[int, ...],
    starting_state: StateT,
    initial_state_is_match: bool,
    vocab_size: int,
    acquired_state_transferred: bool,
    acquired_state: object,
) -> tuple[
    int,
    int | None,
    GrammarMaskedSelectionResult | None,
    object,
    bool | None,
]:
    if not isinstance(result, GrammarMaskedTargetAcceptanceResult):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 must return a GrammarMaskedTargetAcceptanceResult"
        )
    result_proposal = _read_attribute(
        result,
        "proposal_token_ids",
        label="D45 result",
    )
    if type(result_proposal) is not tuple or result_proposal is not proposal_token_ids:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 result must retain the exact proposal-token tuple"
        )

    accepted_count = _read_attribute(result, "accepted_count", label="D45 result")
    if type(accepted_count) is not int:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 accepted_count must be an integer"
        )
    proposal_length = len(proposal_token_ids)
    if accepted_count < 0 or accepted_count > proposal_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"D45 accepted_count must be within [0, {proposal_length}]"
        )

    replacement_token_id = _read_attribute(
        result,
        "replacement_token_id",
        label="D45 result",
    )
    no_decision_selection = _read_attribute(
        result,
        "no_decision_selection",
        label="D45 result",
    )
    if no_decision_selection is not None:
        _validate_empty_selection_evidence(
            no_decision_selection,
            label="D45 no_decision_selection",
        )
        if accepted_count >= proposal_length:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D45 no-decision must stop before full acceptance"
            )
        if replacement_token_id is not None:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D45 no-decision cannot contain a replacement token"
            )
    elif accepted_count == proposal_length:
        if replacement_token_id is not None:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D45 full acceptance cannot contain a replacement token"
            )
    else:
        _validate_operation_token(
            replacement_token_id,
            vocab_size,
            label="D45 replacement_token_id",
        )
        if replacement_token_id == proposal_token_ids[accepted_count]:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D45 replacement token must differ from the rejected proposal token"
            )

    expected_transfer = no_decision_selection is None or accepted_count > 0
    transfer_flag = _read_attribute(
        result,
        "committed_state_transferred",
        label="D45 result",
    )
    if (
        type(transfer_flag) is not bool
        or transfer_flag is not acquired_state_transferred
        or transfer_flag is not expected_transfer
    ):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 result reports inconsistent state-transfer evidence"
        )
    decision_made = _read_attribute(result, "decision_made", label="D45 result")
    if type(decision_made) is not bool or decision_made is not (
        no_decision_selection is None
    ):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 result reports inconsistent decision evidence"
        )
    fully_accepted = _read_attribute(result, "fully_accepted", label="D45 result")
    expected_full_acceptance = (
        no_decision_selection is None and accepted_count == proposal_length
    )
    if type(fully_accepted) is not bool or fully_accepted is not expected_full_acceptance:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 result reports inconsistent full-acceptance evidence"
        )

    committed_state = _read_attribute(
        result,
        "committed_state",
        label="D45 result",
    )
    if committed_state is not acquired_state:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 result changed its committed state after ownership acquisition"
        )
    committed_state_is_match = _read_attribute(
        result,
        "committed_state_is_match",
        label="D45 result",
    )
    if expected_transfer:
        if committed_state is starting_state:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D45 committed state must be identity-distinct from starting_state"
            )
        if type(committed_state_is_match) is not bool:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D45 transferred state requires a boolean match fact"
            )
        if no_decision_selection is not None:
            no_decision_match = _validate_empty_selection_evidence(
                no_decision_selection,
                label="D45 no_decision_selection",
            )
            if committed_state_is_match is not no_decision_match:
                raise GrammarMaskedSpeculativeIterationInvariantError(
                    "D45 no-decision state match fact changed from terminal selection"
                )
        _validate_live_state(
            constraint,
            cast(StateT, committed_state),
            expected_is_match=committed_state_is_match,
            label="D45 committed_state",
        )
    elif committed_state is not None or committed_state_is_match is not None:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "position-zero D45 no-decision cannot transfer state evidence"
        )

    _validate_live_state(
        constraint,
        starting_state,
        expected_is_match=initial_state_is_match,
        label="starting_state",
    )
    return (
        accepted_count,
        cast(int | None, replacement_token_id),
        cast(GrammarMaskedSelectionResult | None, no_decision_selection),
        committed_state,
        cast(bool | None, committed_state_is_match),
    )


def _acquire_acceptance_state(result: object) -> tuple[bool, object]:
    if not isinstance(result, GrammarMaskedTargetAcceptanceResult):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 must return a GrammarMaskedTargetAcceptanceResult"
        )
    transferred = _read_attribute(
        result,
        "committed_state_transferred",
        label="D45 result",
    )
    if type(transferred) is not bool:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D45 committed_state_transferred must be a boolean"
        )
    state = _read_attribute(result, "committed_state", label="D45 result")
    return transferred, state


def _reconcile_partial_outcome(
    draft_backend: CheckpointableAutoregressiveBackend[
        DraftLogitsT, DraftCheckpointT
    ],
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT, TargetCheckpointT
    ],
    current_token_id: int,
    *,
    proposal_token_ids: tuple[int, ...],
    accepted_count: int,
    initial_cache_length: int,
    draft_rollback_checkpoint: DraftCheckpointT,
    target_root_checkpoint: TargetCheckpointT,
) -> int:
    final_cache_length = initial_cache_length + 1 + accepted_count
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
    for position, token_id in enumerate(
        (current_token_id, *proposal_token_ids[:accepted_count])
    ):
        expected_cache_length = initial_cache_length + position + 1
        step = target_backend.decode(token_id)
        _validate_target_decode_step(
            target_backend,
            step,
            expected_cache_length=expected_cache_length,
            label=f"target replay decode at position {position}",
        )

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


def _validate_continuation_result(
    constraint: GrammarConstraint[StateT],
    result: object,
    *,
    proposal_token_ids: tuple[int, ...],
    accepted_count: int,
    replacement_token_id: int | None,
    acceptance_state: object,
    acceptance_state_is_match: bool,
    starting_state: StateT,
    vocab_size: int,
    acquired_state: object,
) -> tuple[int | None, GrammarMaskedSelectionResult | None, object, bool]:
    if not isinstance(result, GrammarMaskedPostAcceptanceContinuationResult):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D46 must return a GrammarMaskedPostAcceptanceContinuationResult"
        )
    output_token_ids = _read_attribute(
        result,
        "output_token_ids",
        label="D46 result",
    )
    if type(output_token_ids) is not tuple:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D46 output_token_ids must be an exact tuple"
        )
    for position, token_id in enumerate(output_token_ids):
        _validate_operation_token(
            token_id,
            vocab_size,
            label=f"D46 output token at position {position}",
        )

    uncached_next_token_id = _read_attribute(
        result,
        "uncached_next_token_id",
        label="D46 result",
    )
    final_row_selection = _read_attribute(
        result,
        "final_row_no_decision_selection",
        label="D46 result",
    )
    committed_state = _read_attribute(
        result,
        "committed_state",
        label="D46 result",
    )
    if committed_state is not acquired_state:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D46 result changed its committed state after ownership acquisition"
        )
    committed_state_is_match = _read_attribute(
        result,
        "committed_state_is_match",
        label="D46 result",
    )
    if type(committed_state_is_match) is not bool:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D46 committed state requires a boolean match fact"
        )

    if accepted_count < len(proposal_token_ids):
        expected_output = proposal_token_ids[:accepted_count] + (
            cast(int, replacement_token_id),
        )
        if output_token_ids != expected_output:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 mismatch output disagrees with D45"
            )
        if uncached_next_token_id != replacement_token_id:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 mismatch uncached token disagrees with D45"
            )
        if final_row_selection is not None:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 mismatch cannot contain final-row selection evidence"
            )
        if committed_state is not acceptance_state:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 mismatch must transfer the exact D45 committed state"
            )
        if committed_state_is_match is not acceptance_state_is_match:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 mismatch committed-state match fact changed from D45"
            )
    elif final_row_selection is not None:
        final_row_match = _validate_empty_selection_evidence(
            final_row_selection,
            label="D46 final_row_no_decision_selection",
        )
        if output_token_ids != proposal_token_ids:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 final-row no-decision output must equal the proposal"
            )
        if uncached_next_token_id is not None:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 final-row no-decision cannot contain an uncached token"
            )
        if committed_state is not acceptance_state:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 final-row no-decision must retain the exact D45 state"
            )
        if (
            committed_state_is_match is not acceptance_state_is_match
            or committed_state_is_match is not final_row_match
        ):
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 final-row match evidence changed from D45"
            )
    else:
        if len(output_token_ids) != len(proposal_token_ids) + 1:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 bonus output must contain the proposal and one bonus token"
            )
        if output_token_ids[:-1] != proposal_token_ids:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 bonus output must begin with the exact proposal"
            )
        _validate_operation_token(
            uncached_next_token_id,
            vocab_size,
            label="D46 uncached_next_token_id",
        )
        if uncached_next_token_id != output_token_ids[-1]:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 uncached token must equal its final output token"
            )
        if committed_state is acceptance_state:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 bonus child must be identity-distinct from the D45 state"
            )
        if committed_state is starting_state:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                "D46 bonus child must be identity-distinct from starting_state"
            )

    _validate_live_state(
        constraint,
        cast(StateT, committed_state),
        expected_is_match=committed_state_is_match,
        label="D46 committed_state",
    )
    return (
        cast(int | None, uncached_next_token_id),
        cast(GrammarMaskedSelectionResult | None, final_row_selection),
        committed_state,
        committed_state_is_match,
    )


def _acquire_continuation_state(result: object) -> object:
    if not isinstance(result, GrammarMaskedPostAcceptanceContinuationResult):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            "D46 must return a GrammarMaskedPostAcceptanceContinuationResult"
        )
    return _read_attribute(result, "committed_state", label="D46 result")


def _validate_composed_result(
    result: object,
    *,
    proposal_token_ids: tuple[int, ...],
    accepted_count: int,
    replacement_token_id: int | None,
    initial_cache_length: int,
    final_cache_length: int,
    uncached_next_token_id: int | None,
    shortening_selection: GrammarMaskedSelectionResult | None,
    acceptance_no_decision_selection: GrammarMaskedSelectionResult | None,
    final_row_no_decision_selection: GrammarMaskedSelectionResult | None,
    committed_state: object,
    committed_state_is_match: bool,
) -> None:
    identity_fields = (
        ("proposal_token_ids", proposal_token_ids),
        ("shortening_selection", shortening_selection),
        ("acceptance_no_decision_selection", acceptance_no_decision_selection),
        ("final_row_no_decision_selection", final_row_no_decision_selection),
        ("committed_state", committed_state),
        ("committed_state_is_match", committed_state_is_match),
    )
    for field_name, expected in identity_fields:
        if _read_attribute(result, field_name, label="D47 result") is not expected:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"D47 result must retain the exact {field_name}"
            )
    value_fields = (
        ("accepted_count", accepted_count),
        ("replacement_token_id", replacement_token_id),
        ("initial_cache_length", initial_cache_length),
        ("final_cache_length", final_cache_length),
        ("uncached_next_token_id", uncached_next_token_id),
    )
    for field_name, expected in value_fields:
        value = _read_attribute(result, field_name, label="D47 result")
        if value != expected or (
            value is not None and type(value) is not type(expected)
        ):
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"D47 result must retain the exact {field_name}"
            )

    expected_shortened = shortening_selection is not None
    expected_decision = bool(proposal_token_ids) and (
        acceptance_no_decision_selection is None
    )
    expected_full = expected_decision and accepted_count == len(proposal_token_ids)
    expected_accepted = proposal_token_ids[:accepted_count]
    expected_rejected = (
        proposal_token_ids[accepted_count]
        if replacement_token_id is not None
        else None
    )
    expected_output = expected_accepted
    if uncached_next_token_id is not None:
        expected_output += (uncached_next_token_id,)

    derived_fields = (
        ("shortened", expected_shortened, bool),
        ("acceptance_decision_made", expected_decision, bool),
        ("fully_accepted", expected_full, bool),
        ("accepted_token_ids", expected_accepted, tuple),
        ("rejected_proposal_token_id", expected_rejected, None),
        ("output_token_ids", expected_output, tuple),
    )
    for field_name, expected, expected_type in derived_fields:
        value = _read_attribute(result, field_name, label="D47 result")
        if value != expected:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"D47 result reports inconsistent {field_name}"
            )
        if expected_type is not None and type(value) is not expected_type:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"D47 result reports malformed {field_name}"
            )


def _cleanup_failed_iteration(
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
    owned_checkpoints: Sequence[_OwnedCheckpoint],
    initial_cache_length: int,
    starting_state: StateT,
    starting_state_owned: bool,
    downstream_state: object,
    downstream_state_owned: bool,
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

    attempted_checkpoints: list[object] = []
    for record in owned_checkpoints:
        if not record.owned or _contains_identity(
            attempted_checkpoints,
            record.checkpoint,
        ):
            continue
        attempted_checkpoints.append(record.checkpoint)
        try:
            draft_backend.release_cache_checkpoint(
                cast(DraftCheckpointT, record.checkpoint)
            )
            record.owned = False
        except Exception as cleanup_failure:
            cleanup_failures.append(
                (
                    f"draft proposal checkpoint {record.position} release",
                    cleanup_failure,
                )
            )

    attempted_states: list[object] = []
    if starting_state_owned:
        attempted_states.append(starting_state)
        try:
            constraint.release_state(starting_state)
        except Exception as cleanup_failure:
            cleanup_failures.append(("starting state release", cleanup_failure))

    if downstream_state_owned and not _contains_identity(
        attempted_states,
        downstream_state,
    ):
        try:
            constraint.release_state(cast(StateT, downstream_state))
        except Exception as cleanup_failure:
            cleanup_failures.append(
                ("downstream committed state release", cleanup_failure)
            )
    return tuple(cleanup_failures)


def _validate_target_decode_step(
    target_backend: CheckpointableAutoregressiveBackend[
        TargetLogitsT, TargetCheckpointT
    ],
    step: object,
    *,
    expected_cache_length: int,
    label: str,
) -> None:
    if not isinstance(step, ModelStep):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} must return a ModelStep"
        )
    reported_cache_length = _read_cache_length_metadata(
        _read_attribute(step, "cache_length", label=label),
        label=f"{label} cache_length",
    )
    if reported_cache_length != expected_cache_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} reported cache length {reported_cache_length}; "
            f"expected {expected_cache_length}"
        )
    _validate_backend_cache_length(
        target_backend,
        expected_cache_length,
        role="target",
    )


def _validate_common_vocab_metadata(
    draft_backend: object,
    target_backend: object,
    constraint: object,
    *,
    expected_vocab_size: int,
) -> None:
    for component, label in (
        (draft_backend, "draft backend"),
        (target_backend, "target backend"),
        (constraint, "constraint"),
    ):
        actual = _read_positive_vocab_size(component, label=label)
        if actual != expected_vocab_size:
            raise GrammarMaskedSpeculativeIterationInvariantError(
                f"{label} vocab_size changed to {actual}; expected {expected_vocab_size}"
            )


def _validate_live_state(
    constraint: GrammarConstraint[StateT],
    state: StateT,
    *,
    expected_is_match: bool | None,
    label: str,
) -> bool:
    is_dead = _require_state_boolean(
        constraint.is_dead_state(state),
        operation="is_dead_state",
    )
    if is_dead:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} must remain live"
        )
    is_match = _require_state_boolean(
        constraint.is_match_state(state),
        operation="is_match_state",
    )
    if expected_is_match is not None and is_match is not expected_is_match:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} match status changed during the transaction"
        )
    return is_match


def _validate_empty_selection_evidence(selection: object, *, label: str) -> bool:
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} must be a GrammarMaskedSelectionResult"
        )
    valid_token_ids = _read_attribute(selection, "valid_token_ids", label=label)
    is_match = _read_attribute(selection, "is_match", label=label)
    selected_token_id = _read_attribute(selection, "selected_token_id", label=label)
    if type(valid_token_ids) is not tuple or valid_token_ids:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} must contain an exact empty support tuple"
        )
    if type(is_match) is not bool:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} is_match must be a boolean"
        )
    if selected_token_id is not None:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} must not contain a selected token"
        )
    return is_match


def _validate_optional_empty_selection_for_result(
    selection: object,
    *,
    label: str,
) -> bool | None:
    if selection is None:
        return None
    if not isinstance(selection, GrammarMaskedSelectionResult):
        raise TypeError(f"{label} must be a GrammarMaskedSelectionResult or None")
    try:
        valid_token_ids = selection.valid_token_ids
        is_match = selection.is_match
        selected_token_id = selection.selected_token_id
    except Exception as exc:
        raise TypeError(f"{label} fields must be readable") from exc
    if type(valid_token_ids) is not tuple:
        raise TypeError(f"{label} valid_token_ids must be an exact tuple")
    if valid_token_ids:
        raise ValueError(f"{label} must contain empty valid_token_ids")
    if type(is_match) is not bool:
        raise TypeError(f"{label} is_match must be a boolean")
    if selected_token_id is not None:
        raise ValueError(f"{label} must not contain a selected token")
    return is_match


def _require_protocol_conformance(value: object, protocol: object, *, label: str) -> None:
    try:
        conforms = isinstance(value, protocol)
    except Exception as exc:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} capability could not be determined"
        ) from exc
    if not conforms:
        raise TypeError(f"{label} must satisfy {cast(type, protocol).__name__}")


def _read_positive_vocab_size(component: object, *, label: str) -> int:
    value = _read_attribute(component, "vocab_size", label=label)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} vocab_size must be an integer"
        )
    if value <= 0:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} vocab_size must be greater than zero"
        )
    return value


def _read_backend_cache_length(backend: object, *, role: str) -> int:
    return _read_cache_length_metadata(
        _read_attribute(backend, "cache_length", label=f"{role} backend"),
        label=f"{role} backend cache_length",
    )


def _validate_backend_cache_length(
    backend: object,
    expected_cache_length: int,
    *,
    role: str,
) -> int:
    cache_length = _read_backend_cache_length(backend, role=role)
    if cache_length != expected_cache_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{role} backend reported cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )
    return cache_length


def _read_cache_length_metadata(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} must be an integer"
        )
    if value < 0:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} cannot be negative"
        )
    return value


def _validate_root_checkpoint(
    checkpoint: object,
    *,
    expected_cache_length: int,
    label: str,
) -> None:
    try:
        conforms = isinstance(checkpoint, CacheCheckpoint)
    except Exception as exc:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} capability could not be determined"
        ) from exc
    if not conforms:
        raise TypeError(f"{label} must satisfy CacheCheckpoint")
    cache_length = _read_cache_length_metadata(
        _read_attribute(checkpoint, "cache_length", label=label),
        label=f"{label} cache_length",
    )
    if cache_length != expected_cache_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} reports cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )


def _validate_checkpoint(
    checkpoint: object,
    *,
    expected_cache_length: int,
    label: str,
) -> None:
    try:
        conforms = isinstance(checkpoint, CacheCheckpoint)
    except Exception as exc:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} capability could not be determined"
        ) from exc
    if not conforms:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} must satisfy CacheCheckpoint"
        )
    cache_length = _read_cache_length_metadata(
        _read_attribute(checkpoint, "cache_length", label=label),
        label=f"{label} cache_length",
    )
    if cache_length != expected_cache_length:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} reports cache length {cache_length}; "
            f"expected {expected_cache_length}"
        )


def _validate_result_proposal(proposal_token_ids: object) -> None:
    if type(proposal_token_ids) is not tuple:
        raise TypeError("proposal_token_ids must be a tuple")
    for position, token_id in enumerate(cast(tuple[object, ...], proposal_token_ids)):
        _validate_result_nonnegative_token(
            token_id,
            label=f"proposal token at position {position}",
        )


def _validate_result_nonnegative_token(token_id: object, *, label: str) -> None:
    if type(token_id) is not int:
        raise TypeError(f"{label} must be an integer")
    if token_id < 0:
        raise ValueError(f"{label} cannot be negative")


def _validate_result_cache_length(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return value


def _validate_token_id(token_id: object, vocab_size: int, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )


def _validate_operation_token(token_id: object, vocab_size: int, *, label: str) -> None:
    try:
        _validate_token_id(token_id, vocab_size, label=label)
    except (TypeError, ValueError) as exc:
        raise GrammarMaskedSpeculativeIterationInvariantError(str(exc)) from exc


def _require_state_boolean(value: object, *, operation: str) -> bool:
    if type(value) is not bool:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _read_attribute(value: object, name: str, *, label: str) -> object:
    try:
        return getattr(value, name)
    except Exception as exc:
        raise GrammarMaskedSpeculativeIterationInvariantError(
            f"{label} {name} could not be read"
        ) from exc


def _contains_identity(values: Sequence[object], candidate: object) -> bool:
    return any(value is candidate for value in values)


__all__ = [
    "GrammarMaskedSpeculativeIterationCleanupError",
    "GrammarMaskedSpeculativeIterationError",
    "GrammarMaskedSpeculativeIterationInvariantError",
    "GrammarMaskedSpeculativeIterationResult",
    "coordinate_grammar_masked_speculative_iteration",
]
