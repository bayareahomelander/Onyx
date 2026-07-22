import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path

import pytest

import onyx_cuda.draft as draft_module
from onyx_cuda import (
    AutoregressiveBackend,
    BackendError,
    BackendStateError,
    CacheCheckpointStateError,
    CheckpointableAutoregressiveBackend,
    DraftProposalCleanupError,
    DraftProposalError,
    DraftProposalInvariantError,
    DraftProposalResult,
    ModelStep,
    TemperatureTopPSelection,
    create_reference_sampler,
    generate_draft_proposal,
    select_highest_logit,
)
from onyx_cuda.testing import FakeAutoregressiveBackend, ScriptExhaustedError


VOCAB_SIZE = 5
SCRIPT = tuple(
    tuple(
        20.0 if token_id == row_number % VOCAB_SIZE else float(-token_id)
        for token_id in range(VOCAB_SIZE)
    )
    for row_number in range(64)
)
_UNSET = object()


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    cache_length: int


class TupleSubclass(tuple):
    pass


class MinimumBackend:
    model_id = "minimum"
    vocab_size = VOCAB_SIZE
    cache_length = 1

    def __init__(self):
        self.decode_calls = 0

    def prefill(self, prompt_token_ids, /):
        return ModelStep(logits=SCRIPT[0], cache_length=len(prompt_token_ids))

    def decode(self, token_id, /):
        self.decode_calls += 1
        return ModelStep(logits=SCRIPT[1], cache_length=self.cache_length + 1)

    def reset(self):
        return None


class RecordingBackend(FakeAutoregressiveBackend):
    def __init__(self, scripted_logits=SCRIPT):
        self.decode_token_ids = []
        self.create_attempts = 0
        self.created_checkpoints = []
        self.rollback_calls = []
        self.release_calls = []
        self.reset_calls = 0
        super().__init__(scripted_logits)

    def decode(self, token_id, /):
        self.decode_token_ids.append(token_id)
        return super().decode(token_id)

    def create_cache_checkpoint(self):
        self.create_attempts += 1
        checkpoint = super().create_cache_checkpoint()
        self.created_checkpoints.append(checkpoint)
        return checkpoint

    def rollback_cache(self, checkpoint, /):
        self.rollback_calls.append(checkpoint)
        return super().rollback_cache(checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        return super().release_cache_checkpoint(checkpoint)

    def reset(self):
        self.reset_calls += 1
        return super().reset()


class MetadataOverrideBackend(RecordingBackend):
    def __init__(self, scripted_logits=SCRIPT):
        self.vocab_override = _UNSET
        self.cache_override = _UNSET
        super().__init__(scripted_logits)

    @property
    def vocab_size(self):
        if self.vocab_override is not _UNSET:
            return self.vocab_override
        return super().vocab_size

    @property
    def cache_length(self):
        if self.cache_override is not _UNSET:
            return self.cache_override
        return super().cache_length


class CreateFailureBackend(RecordingBackend):
    def __init__(self, scripted_logits=SCRIPT):
        self.fail_on_attempt = None
        super().__init__(scripted_logits)

    def create_cache_checkpoint(self):
        next_attempt = self.create_attempts + 1
        if next_attempt == self.fail_on_attempt:
            self.create_attempts = next_attempt
            raise RuntimeError("injected checkpoint creation failure")
        return super().create_cache_checkpoint()


class CheckpointResultFaultBackend(RecordingBackend):
    def __init__(self, scripted_logits=SCRIPT):
        self.fault_on_attempt = None
        self.fault_kind = "non-capability"
        self._checkpoint_aliases = {}
        super().__init__(scripted_logits)

    def create_cache_checkpoint(self):
        checkpoint = super().create_cache_checkpoint()
        if self.create_attempts != self.fault_on_attempt:
            return checkpoint
        if self.fault_kind == "non-capability":
            exposed = object()
        else:
            exposed = CheckpointRecord(checkpoint.cache_length + 1)
        self._checkpoint_aliases[id(exposed)] = checkpoint
        return exposed

    def rollback_cache(self, checkpoint, /):
        canonical = self._checkpoint_aliases.get(id(checkpoint), checkpoint)
        return super().rollback_cache(canonical)

    def release_cache_checkpoint(self, checkpoint, /):
        canonical = self._checkpoint_aliases.get(id(checkpoint), checkpoint)
        return super().release_cache_checkpoint(canonical)


class StepFaultBackend(RecordingBackend):
    def __init__(self, *, fault_call, fault_kind, scripted_logits=SCRIPT):
        self.fault_call = fault_call
        self.fault_kind = fault_kind
        super().__init__(scripted_logits)

    def decode(self, token_id, /):
        step = super().decode(token_id)
        if len(self.decode_token_ids) != self.fault_call:
            return step
        if self.fault_kind == "non-step":
            return object()
        if self.fault_kind == "non-integer":
            malformed = object.__new__(ModelStep)
            object.__setattr__(malformed, "logits", step.logits)
            object.__setattr__(malformed, "cache_length", "invalid")
            return malformed
        return ModelStep(logits=step.logits, cache_length=step.cache_length + 1)


class ReportedCacheFaultBackend(RecordingBackend):
    def __init__(self, *, fault_call, scripted_logits=SCRIPT):
        self.fault_call = fault_call
        super().__init__(scripted_logits)

    @property
    def cache_length(self):
        actual = super().cache_length
        if len(self.decode_token_ids) == self.fault_call:
            return actual + 1
        return actual


class CleanupFaultBackend(RecordingBackend):
    def __init__(self, scripted_logits=SCRIPT):
        self.fail_rollback = False
        self.fail_rollback_release = False
        self.start_release_failures = 0
        self.cleanup_root_length = None
        super().__init__(scripted_logits)

    def rollback_cache(self, checkpoint, /):
        self.rollback_calls.append(checkpoint)
        if self.fail_rollback:
            raise RuntimeError("injected rollback failure")
        return FakeAutoregressiveBackend.rollback_cache(self, checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        if (
            self.cleanup_root_length is not None
            and checkpoint.cache_length == self.cleanup_root_length
            and self.start_release_failures > 0
        ):
            self.start_release_failures -= 1
            raise RuntimeError("injected start release failure")
        if (
            self.cleanup_root_length is not None
            and checkpoint.cache_length > self.cleanup_root_length
            and self.fail_rollback_release
        ):
            raise RuntimeError("injected rollback checkpoint release failure")
        return FakeAutoregressiveBackend.release_cache_checkpoint(self, checkpoint)


def _sequence_state(backend):
    return (
        backend._cache_length,
        backend.cached_token_ids,
        backend._next_row,
        backend._epoch,
        tuple(backend._cache_checkpoints.items()),
    )


def _complete_state(backend):
    return (*_sequence_state(backend), backend._next_checkpoint_id)


def _release_result(backend, result):
    for checkpoint in result.rollback_checkpoints:
        backend.release_cache_checkpoint(checkpoint)


def _prefilled_backend(backend_type=RecordingBackend, script=SCRIPT):
    backend = backend_type(script)
    backend.prefill((4, 0))
    return backend


def test_result_is_frozen_slotted_generic_minimal_and_retains_checkpoint_identity():
    first = CheckpointRecord(3)
    second = CheckpointRecord(4)
    result = DraftProposalResult[CheckpointRecord](
        proposal_token_ids=(1, 2),
        rollback_checkpoints=(first, second),
        initial_cache_length=2,
        final_cache_length=5,
    )

    assert result.rollback_checkpoints[0] is first
    assert result.rollback_checkpoints[1] is second
    assert [field.name for field in fields(result)] == [
        "proposal_token_ids",
        "rollback_checkpoints",
        "initial_cache_length",
        "final_cache_length",
    ]
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.final_cache_length = 6


@pytest.mark.parametrize(
    ("proposal", "checkpoints", "error", "message"),
    [
        ([], (CheckpointRecord(3),), TypeError, "proposal_token_ids must be a tuple"),
        (
            TupleSubclass((1,)),
            (CheckpointRecord(3),),
            TypeError,
            "proposal_token_ids must be a tuple",
        ),
        ((), (), ValueError, "proposal_token_ids cannot be empty"),
        ((1,), [CheckpointRecord(3)], TypeError, "rollback_checkpoints must be a tuple"),
        (
            (1,),
            TupleSubclass((CheckpointRecord(3),)),
            TypeError,
            "rollback_checkpoints must be a tuple",
        ),
        ((1, 2), (CheckpointRecord(3),), ValueError, "one checkpoint per proposal token"),
    ],
)
def test_result_requires_exact_nonempty_tuple_collections(
    proposal,
    checkpoints,
    error,
    message,
):
    with pytest.raises(error, match=message):
        DraftProposalResult(
            proposal_token_ids=proposal,
            rollback_checkpoints=checkpoints,
            initial_cache_length=2,
            final_cache_length=2 + len(proposal) + 1,
        )


@pytest.mark.parametrize(
    ("token_id", "error", "message"),
    [
        (True, TypeError, "proposal token at position 1 must be an integer"),
        (1.0, TypeError, "proposal token at position 1 must be an integer"),
        ("1", TypeError, "proposal token at position 1 must be an integer"),
        (-1, ValueError, "proposal token at position 1 cannot be negative"),
    ],
)
def test_result_rejects_invalid_proposal_token_ids(token_id, error, message):
    with pytest.raises(error, match=message):
        DraftProposalResult(
            proposal_token_ids=(1, token_id),
            rollback_checkpoints=(CheckpointRecord(3), CheckpointRecord(4)),
            initial_cache_length=2,
            final_cache_length=5,
        )


@pytest.mark.parametrize(
    ("checkpoints", "message"),
    [
        ((object(),), "must satisfy CacheCheckpoint"),
        ((CheckpointRecord(2),), "reports cache length 2; expected 3"),
        ((CheckpointRecord(True),), "cache_length must be an integer"),
    ],
)
def test_result_rejects_invalid_checkpoint_capabilities_and_positions(checkpoints, message):
    with pytest.raises(DraftProposalInvariantError, match=message):
        DraftProposalResult(
            proposal_token_ids=(1,),
            rollback_checkpoints=checkpoints,
            initial_cache_length=2,
            final_cache_length=4,
        )


@pytest.mark.parametrize(
    ("initial_length", "final_length", "message"),
    [
        (True, 3, "initial_cache_length must be an integer"),
        (1.0, 3, "initial_cache_length must be an integer"),
        (-1, 1, "initial_cache_length cannot be negative"),
        (0, 2, "initial_cache_length must be greater than zero"),
        (2, True, "final_cache_length must be an integer"),
        (2, -1, "final_cache_length cannot be negative"),
        (2, 5, "final_cache_length is 5; expected 4"),
    ],
)
def test_result_rejects_invalid_cache_length_metadata_and_relation(
    initial_length,
    final_length,
    message,
):
    with pytest.raises(DraftProposalInvariantError, match=message):
        DraftProposalResult(
            proposal_token_ids=(1,),
            rollback_checkpoints=(CheckpointRecord(initial_length + 1),),
            initial_cache_length=initial_length,
            final_cache_length=final_length,
        )


def test_cleanup_error_retains_original_and_immutable_ordered_failures():
    original = RuntimeError("selection failed")
    rollback = RuntimeError("rollback failed")
    release = RuntimeError("release failed")
    error = DraftProposalCleanupError(
        original,
        [("start checkpoint rollback", rollback), ("start checkpoint release", release)],
    )

    assert error.original_failure is original
    assert error.cleanup_failures == (
        ("start checkpoint rollback", rollback),
        ("start checkpoint release", release),
    )
    assert "selection failed" in str(error)
    assert "rollback failed" in str(error)
    assert "release failed" in str(error)
    with pytest.raises(ValueError, match="cannot be empty"):
        DraftProposalCleanupError(original, [])


def test_public_contract_module_ownership_exports_and_error_hierarchy():
    import onyx_cuda

    for symbol in (
        DraftProposalError,
        DraftProposalInvariantError,
        DraftProposalCleanupError,
        DraftProposalResult,
        generate_draft_proposal,
    ):
        assert symbol.__module__ == "onyx_cuda.draft"
        assert getattr(onyx_cuda, symbol.__name__) is symbol
        assert symbol.__name__ in onyx_cuda.__all__
    assert issubclass(DraftProposalError, BackendError)
    assert issubclass(DraftProposalInvariantError, DraftProposalError)
    assert issubclass(DraftProposalCleanupError, DraftProposalError)


def test_one_token_proposal_consumes_current_and_final_token_with_one_rollback_handle():
    backend = _prefilled_backend()
    initial_reset_calls = backend.reset_calls
    seen_rows = []

    def selector(row):
        seen_rows.append(row)
        return select_highest_logit(row)

    result = generate_draft_proposal(
        backend,
        4,
        proposal_length=1,
        select_token=selector,
    )

    assert result == DraftProposalResult(
        proposal_token_ids=(1,),
        rollback_checkpoints=result.rollback_checkpoints,
        initial_cache_length=2,
        final_cache_length=4,
    )
    assert backend.decode_token_ids == [4, 1]
    assert seen_rows == [SCRIPT[1]]
    assert seen_rows[0] is backend._scripted_logits[1]
    assert backend._next_row == 3
    assert backend.cached_token_ids == (4, 0, 4, 1)
    assert result.rollback_checkpoints[0].cache_length == 3
    assert backend.active_checkpoint_count == 1
    assert backend.reset_calls == initial_reset_calls
    _release_result(backend, result)
    assert backend.active_checkpoint_count == 0


def test_multi_token_proposal_has_exact_row_token_cache_and_checkpoint_alignment():
    backend = _prefilled_backend()
    seen_rows = []

    def selector(row):
        seen_rows.append(row)
        return select_highest_logit(row)

    result = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=selector,
    )

    assert result.proposal_token_ids == (1, 2, 3)
    assert backend.decode_token_ids == [4, 1, 2, 3]
    assert seen_rows == list(SCRIPT[1:4])
    assert all(
        actual is scripted
        for actual, scripted in zip(seen_rows, backend._scripted_logits[1:4])
    )
    assert all(row is not backend._scripted_logits[4] for row in seen_rows)
    assert backend._next_row == 5
    assert backend.cache_length == result.final_cache_length == 6
    assert backend.cached_token_ids == (4, 0, 4, 1, 2, 3)
    assert result.initial_cache_length == 2
    assert tuple(cp.cache_length for cp in result.rollback_checkpoints) == (3, 4, 5)
    assert backend.create_attempts == 4
    assert backend.release_calls[0].cache_length == 2
    assert backend.active_checkpoint_count == 3
    _release_result(backend, result)


def test_proposal_length_has_no_hidden_operation_limit():
    backend = _prefilled_backend()

    result = generate_draft_proposal(
        backend,
        4,
        proposal_length=12,
        select_token=select_highest_logit,
    )

    assert len(result.proposal_token_ids) == 12
    assert len(result.rollback_checkpoints) == 12
    assert len(backend.decode_token_ids) == 13
    assert backend.cache_length == 15
    _release_result(backend, result)


def test_stateful_selector_session_continues_across_proposals_without_recreation():
    backend = _prefilled_backend()
    selected = iter((0, 1, 2, 3))
    calls = []

    def selector(row):
        calls.append(row)
        return next(selected)

    first = generate_draft_proposal(
        backend,
        4,
        proposal_length=2,
        select_token=selector,
    )
    second = generate_draft_proposal(
        backend,
        4,
        proposal_length=2,
        select_token=selector,
    )

    assert first.proposal_token_ids == (0, 1)
    assert second.proposal_token_ids == (2, 3)
    assert calls == list(SCRIPT[1:3]) + list(SCRIPT[4:6])
    _release_result(backend, first)
    _release_result(backend, second)


def test_fresh_same_seed_sessions_replay_after_caller_owned_root_rollback():
    backend = _prefilled_backend()
    root = backend.create_cache_checkpoint()
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=9182)

    first = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=create_reference_sampler(policy),
    )
    first_allocations = tuple(cp.allocation_id for cp in first.rollback_checkpoints)
    backend.rollback_cache(root)
    _release_result(backend, first)

    replay = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=create_reference_sampler(policy),
    )

    assert replay.proposal_token_ids == first.proposal_token_ids
    assert replay.initial_cache_length == first.initial_cache_length
    assert replay.final_cache_length == first.final_cache_length
    assert tuple(cp.cache_length for cp in replay.rollback_checkpoints) == tuple(
        cp.cache_length for cp in first.rollback_checkpoints
    )
    assert tuple(cp.allocation_id for cp in replay.rollback_checkpoints) != first_allocations
    assert backend.cached_token_ids == (4, 0, 4, *replay.proposal_token_ids)
    _release_result(backend, replay)
    backend.release_cache_checkpoint(root)


def test_alternative_selector_after_root_rollback_produces_an_exact_alternative_suffix():
    backend = _prefilled_backend()
    root = backend.create_cache_checkpoint()
    greedy = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=select_highest_logit,
    )

    backend.rollback_cache(root)
    _release_result(backend, greedy)
    alternative = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=lambda row: 0,
    )

    assert greedy.proposal_token_ids == (1, 2, 3)
    assert alternative.proposal_token_ids == (0, 0, 0)
    assert backend.cached_token_ids == (4, 0, 4, 0, 0, 0)
    _release_result(backend, alternative)
    backend.release_cache_checkpoint(root)


@pytest.mark.parametrize("proposal_length", [True, 1.0, "1", 0, -1])
def test_invalid_proposal_lengths_are_rejected_before_mutation(proposal_length):
    backend = _prefilled_backend()
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _complete_state(backend)
    counts = (len(backend.decode_token_ids), backend.create_attempts)
    selector_calls = []
    error = TypeError if proposal_length in (True, 1.0, "1") else ValueError

    with pytest.raises(error):
        generate_draft_proposal(
            backend,
            1,
            proposal_length=proposal_length,
            select_token=lambda row: selector_calls.append(row),
        )

    assert _complete_state(backend) == before
    assert (len(backend.decode_token_ids), backend.create_attempts) == counts
    assert selector_calls == []
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize("selector", [None, 1, "selector"])
def test_noncallable_selector_is_rejected_before_mutation(selector):
    backend = _prefilled_backend()
    before = _complete_state(backend)

    with pytest.raises(TypeError, match="select_token must be callable"):
        generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=selector,
        )

    assert _complete_state(backend) == before
    assert backend.create_attempts == 0
    assert backend.decode_token_ids == []


def test_minimum_backend_is_rejected_without_mutation_or_selection():
    backend = MinimumBackend()
    selector_calls = []

    assert isinstance(backend, AutoregressiveBackend)
    assert not isinstance(backend, CheckpointableAutoregressiveBackend)
    with pytest.raises(TypeError, match="CheckpointableAutoregressiveBackend"):
        generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=lambda row: selector_calls.append(row),
        )

    assert backend.decode_calls == 0
    assert selector_calls == []


@pytest.mark.parametrize("vocab_size", [True, 3.0, "3", 0, -1])
def test_invalid_backend_vocabulary_metadata_is_rejected_before_mutation(vocab_size):
    backend = _prefilled_backend(MetadataOverrideBackend)
    caller_checkpoint = backend.create_cache_checkpoint()
    backend.vocab_override = vocab_size
    before = _complete_state(backend)
    counts = (len(backend.decode_token_ids), backend.create_attempts)

    with pytest.raises(DraftProposalInvariantError, match="vocab_size"):
        generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=select_highest_logit,
        )

    assert _complete_state(backend) == before
    assert (len(backend.decode_token_ids), backend.create_attempts) == counts
    backend.vocab_override = _UNSET
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize(
    ("current_token", "error", "message"),
    [
        (True, TypeError, "current_token_id must be an integer"),
        (1.0, TypeError, "current_token_id must be an integer"),
        ("1", TypeError, "current_token_id must be an integer"),
        (-1, ValueError, "current_token_id -1"),
        (VOCAB_SIZE, ValueError, f"current_token_id {VOCAB_SIZE}"),
    ],
)
def test_invalid_current_token_is_rejected_before_mutation(current_token, error, message):
    backend = _prefilled_backend()
    before = _complete_state(backend)

    with pytest.raises(error, match=message):
        generate_draft_proposal(
            backend,
            current_token,
            proposal_length=1,
            select_token=select_highest_logit,
        )

    assert _complete_state(backend) == before
    assert backend.create_attempts == 0
    assert backend.decode_token_ids == []


@pytest.mark.parametrize("cache_length", [True, 1.0, "1", -1])
def test_invalid_starting_cache_metadata_is_rejected_before_mutation(cache_length):
    backend = _prefilled_backend(MetadataOverrideBackend)
    caller_checkpoint = backend.create_cache_checkpoint()
    backend.cache_override = cache_length
    before = _complete_state(backend)
    counts = (len(backend.decode_token_ids), backend.create_attempts)

    with pytest.raises(DraftProposalInvariantError, match="cache_length"):
        generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=select_highest_logit,
        )

    assert _complete_state(backend) == before
    assert (len(backend.decode_token_ids), backend.create_attempts) == counts
    backend.cache_override = _UNSET
    backend.rollback_cache(caller_checkpoint)


def test_proposal_requires_prefill_and_does_not_create_a_checkpoint():
    backend = RecordingBackend()
    before = _complete_state(backend)

    with pytest.raises(BackendStateError, match="prefill"):
        generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=select_highest_logit,
        )

    assert _complete_state(backend) == before
    assert backend.create_attempts == 0
    assert backend.decode_token_ids == []


@pytest.mark.parametrize("fault_call", [1, 2, 4])
@pytest.mark.parametrize("fault_kind", ["wrong-length", "non-integer"])
def test_malformed_backend_step_restores_the_exact_starting_sequence(fault_call, fault_kind):
    backend = StepFaultBackend(fault_call=fault_call, fault_kind=fault_kind)
    backend.prefill((4, 0))
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)
    next_allocation = backend._next_checkpoint_id
    reset_calls = backend.reset_calls

    with pytest.raises(DraftProposalInvariantError, match="cache length|must be an integer"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=3,
            select_token=select_highest_logit,
        )

    assert _sequence_state(backend) == before
    assert backend._next_checkpoint_id > next_allocation
    assert backend.reset_calls == reset_calls
    backend.rollback_cache(caller_checkpoint)


def test_non_model_step_result_restores_the_exact_starting_sequence():
    backend = StepFaultBackend(fault_call=1, fault_kind="non-step")
    backend.prefill((4, 0))
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)

    with pytest.raises(DraftProposalInvariantError, match="must return a ModelStep"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=1,
            select_token=select_highest_logit,
        )

    assert _sequence_state(backend) == before
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize("fault_call", [1, 2, 4])
def test_backend_reported_cache_disagreement_at_each_stage_is_atomic(fault_call):
    backend = ReportedCacheFaultBackend(fault_call=fault_call)
    backend.prefill((4, 0))
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)

    with pytest.raises(DraftProposalInvariantError, match="backend state reported cache length"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=3,
            select_token=select_highest_logit,
        )

    assert _sequence_state(backend) == before
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize(
    ("relative_attempt", "fault_kind"),
    [
        (1, "non-capability"),
        (1, "wrong-length"),
        (2, "non-capability"),
        (2, "wrong-length"),
    ],
)
def test_malformed_start_or_rollback_checkpoint_is_released_without_leak(
    relative_attempt,
    fault_kind,
):
    backend = _prefilled_backend(CheckpointResultFaultBackend)
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)
    backend.fault_on_attempt = backend.create_attempts + relative_attempt
    backend.fault_kind = fault_kind

    with pytest.raises(DraftProposalInvariantError, match="CacheCheckpoint|reports cache length"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=2,
            select_token=select_highest_logit,
        )

    assert _sequence_state(backend) == before
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize("relative_attempt", [1, 2, 3])
def test_checkpoint_creation_failure_before_or_after_mutation_is_atomic(relative_attempt):
    backend = _prefilled_backend(CreateFailureBackend)
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)
    backend.fail_on_attempt = backend.create_attempts + relative_attempt

    with pytest.raises(RuntimeError, match="checkpoint creation failure"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=3,
            select_token=select_highest_logit,
        )

    assert _sequence_state(backend) == before
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize(
    ("selected_token", "error", "message"),
    [
        (True, TypeError, "proposal position 0 must be an integer"),
        (1.0, TypeError, "proposal position 0 must be an integer"),
        ("1", TypeError, "proposal position 0 must be an integer"),
        (-1, ValueError, "proposal position 0 -1"),
        (VOCAB_SIZE, ValueError, f"proposal position 0 {VOCAB_SIZE}"),
    ],
)
def test_invalid_selected_token_restores_backend_and_preserves_original_error(
    selected_token,
    error,
    message,
):
    backend = _prefilled_backend()
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)

    with pytest.raises(error, match=message):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=1,
            select_token=lambda row: selected_token,
        )

    assert _sequence_state(backend) == before
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_selector_failure_at_every_position_restores_without_an_extra_call(failure_call):
    backend = _prefilled_backend()
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)
    calls = []

    def selector(row):
        calls.append(row)
        if len(calls) == failure_call:
            raise LookupError(f"selector failed at {failure_call}")
        return select_highest_logit(row)

    with pytest.raises(LookupError, match=f"selector failed at {failure_call}"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=3,
            select_token=selector,
        )

    assert len(calls) == failure_call
    assert _sequence_state(backend) == before
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize("available_decode_rows", [0, 1, 2, 3])
def test_script_exhaustion_during_any_decode_restores_the_starting_sequence(
    available_decode_rows,
):
    backend = RecordingBackend(SCRIPT[: available_decode_rows + 1])
    backend.prefill((4, 0))
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)

    with pytest.raises(ScriptExhaustedError, match="scripted logits"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=3,
            select_token=select_highest_logit,
        )

    assert _sequence_state(backend) == before
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(caller_checkpoint)


def test_result_construction_failure_after_full_suffix_consumption_is_atomic(monkeypatch):
    backend = _prefilled_backend()
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)

    def fail_result_construction(*args, **kwargs):
        raise RuntimeError("injected result construction failure")

    monkeypatch.setattr(draft_module, "DraftProposalResult", fail_result_construction)
    with pytest.raises(RuntimeError, match="result construction failure"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=3,
            select_token=select_highest_logit,
        )

    assert backend.decode_token_ids == [4, 1, 2, 3]
    assert _sequence_state(backend) == before
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(caller_checkpoint)


@pytest.mark.parametrize("rejection_index", [0, 1, 2])
def test_each_rejection_checkpoint_restores_the_exact_accepted_prefix(rejection_index):
    backend = _prefilled_backend()
    root = backend.create_cache_checkpoint()
    result = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=select_highest_logit,
    )

    target = result.rollback_checkpoints[rejection_index]
    backend.rollback_cache(target)

    assert backend.cached_token_ids == (4, 0, 4, *result.proposal_token_ids[:rejection_index])
    assert backend.cache_length == 3 + rejection_index
    assert backend._next_row == 2 + rejection_index
    backend.rollback_cache(target)
    for checkpoint in result.rollback_checkpoints[: rejection_index + 1]:
        backend.rollback_cache(checkpoint)
        break
    for checkpoint in result.rollback_checkpoints[rejection_index + 1 :]:
        with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
            backend.rollback_cache(checkpoint)

    cache_state = (backend._cache_length, backend.cached_token_ids, backend._next_row)
    _release_result(backend, result)
    _release_result(backend, result)
    assert (backend._cache_length, backend.cached_token_ids, backend._next_row) == cache_state
    assert backend.active_checkpoint_count == 1
    backend.release_cache_checkpoint(root)


def test_full_acceptance_needs_no_final_checkpoint_and_release_is_cache_neutral():
    backend = _prefilled_backend()
    result = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=select_highest_logit,
    )
    final_state = (backend.cache_length, backend.cached_token_ids, backend._next_row)

    assert result.final_cache_length == 6
    assert all(cp.cache_length < result.final_cache_length for cp in result.rollback_checkpoints)
    _release_result(backend, result)
    _release_result(backend, result)

    assert (backend.cache_length, backend.cached_token_ids, backend._next_row) == final_state
    assert backend.active_checkpoint_count == 0


def test_earlier_caller_rollback_invalidates_every_draft_suffix_handle():
    backend = _prefilled_backend()
    root = backend.create_cache_checkpoint()
    result = generate_draft_proposal(
        backend,
        4,
        proposal_length=3,
        select_token=select_highest_logit,
    )

    backend.rollback_cache(root)

    assert backend.cached_token_ids == (4, 0)
    for checkpoint in result.rollback_checkpoints:
        with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
            backend.rollback_cache(checkpoint)
        backend.release_cache_checkpoint(checkpoint)
    assert backend.active_checkpoint_count == 1
    backend.release_cache_checkpoint(root)


@pytest.mark.parametrize("replacement", [None, (2,)])
def test_reset_or_replacement_prefill_invalidates_draft_checkpoints(replacement):
    backend = _prefilled_backend()
    result = generate_draft_proposal(
        backend,
        4,
        proposal_length=2,
        select_token=select_highest_logit,
    )

    if replacement is None:
        backend.reset()
    else:
        backend.prefill(replacement)

    for checkpoint in result.rollback_checkpoints:
        with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
            backend.rollback_cache(checkpoint)
        backend.release_cache_checkpoint(checkpoint)
    assert backend.active_checkpoint_count == 0


def test_successful_failure_cleanup_consumes_but_never_recycles_allocation_identity():
    backend = _prefilled_backend()
    caller_checkpoint = backend.create_cache_checkpoint()
    next_allocation = backend._next_checkpoint_id

    with pytest.raises(LookupError, match="selection failed"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=1,
            select_token=lambda row: (_ for _ in ()).throw(LookupError("selection failed")),
        )

    stale = backend.created_checkpoints[-1]
    assert stale.allocation_id >= next_allocation
    assert backend._next_checkpoint_id > stale.allocation_id
    assert backend.active_checkpoint_count == 1
    later = backend.create_cache_checkpoint()
    assert later.allocation_id > stale.allocation_id
    with pytest.raises(CacheCheckpointStateError, match="unknown, released"):
        backend.rollback_cache(stale)
    backend.release_cache_checkpoint(later)
    backend.release_cache_checkpoint(caller_checkpoint)


def test_rollback_failure_is_reported_with_original_failure_and_cleanup_continues():
    backend = _prefilled_backend(CleanupFaultBackend)
    caller_checkpoint = backend.create_cache_checkpoint()
    backend.cleanup_root_length = backend.cache_length
    backend.fail_rollback = True

    with pytest.raises(DraftProposalCleanupError) as raised:
        generate_draft_proposal(
            backend,
            4,
            proposal_length=1,
            select_token=lambda row: (_ for _ in ()).throw(LookupError("selection failed")),
        )

    error = raised.value
    assert isinstance(error.original_failure, LookupError)
    assert [operation for operation, _ in error.cleanup_failures] == [
        "start checkpoint rollback"
    ]
    assert backend.active_checkpoint_count == 1
    assert backend.reset_calls == 2
    backend.fail_rollback = False
    backend.rollback_cache(caller_checkpoint)
    backend.release_cache_checkpoint(caller_checkpoint)


def test_combined_cleanup_failures_are_retained_in_deterministic_operation_order():
    backend = _prefilled_backend(CleanupFaultBackend)
    backend.cleanup_root_length = backend.cache_length
    backend.fail_rollback = True
    backend.fail_rollback_release = True
    backend.start_release_failures = 1

    with pytest.raises(DraftProposalCleanupError) as raised:
        generate_draft_proposal(
            backend,
            4,
            proposal_length=1,
            select_token=lambda row: (_ for _ in ()).throw(ValueError("bad selection")),
        )

    error = raised.value
    assert isinstance(error.original_failure, ValueError)
    assert [operation for operation, _ in error.cleanup_failures] == [
        "start checkpoint rollback",
        "rollback checkpoint 0 release",
        "start checkpoint release",
    ]
    assert "bad selection" in str(error)
    assert "rollback failure" in str(error)
    assert "rollback checkpoint release failure" in str(error)
    assert "start release failure" in str(error)


def test_private_start_release_failure_prevents_result_and_restores_when_retry_succeeds():
    backend = _prefilled_backend(CleanupFaultBackend)
    caller_checkpoint = backend.create_cache_checkpoint()
    before = _sequence_state(backend)
    backend.cleanup_root_length = backend.cache_length
    backend.start_release_failures = 1

    with pytest.raises(RuntimeError, match="start release failure"):
        generate_draft_proposal(
            backend,
            4,
            proposal_length=2,
            select_token=select_highest_logit,
        )

    assert _sequence_state(backend) == before
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(caller_checkpoint)


def test_private_start_release_and_cleanup_release_failures_are_aggregated():
    backend = _prefilled_backend(CleanupFaultBackend)
    backend.cleanup_root_length = backend.cache_length
    backend.start_release_failures = 2

    with pytest.raises(DraftProposalCleanupError) as raised:
        generate_draft_proposal(
            backend,
            4,
            proposal_length=1,
            select_token=select_highest_logit,
        )

    assert isinstance(raised.value.original_failure, RuntimeError)
    assert [operation for operation, _ in raised.value.cleanup_failures] == [
        "start checkpoint release"
    ]
    assert backend.cached_token_ids == (4, 0)


def test_bounded_propose_rollback_release_reuse_has_no_registry_cursor_or_epoch_drift():
    backend = RecordingBackend(SCRIPT[:3])
    backend.prefill((0,))
    root = backend.create_cache_checkpoint()
    base_state = (backend.cache_length, backend.cached_token_ids, backend._next_row, backend._epoch)

    for _ in range(1_000):
        result = generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=select_highest_logit,
        )
        assert result.proposal_token_ids == (1,)
        backend.rollback_cache(root)
        _release_result(backend, result)
        assert (
            backend.cache_length,
            backend.cached_token_ids,
            backend._next_row,
            backend._epoch,
        ) == base_state
        assert backend.active_checkpoint_count == 1

    backend.release_cache_checkpoint(root)
    assert backend.active_checkpoint_count == 0


def test_repeated_prefill_epochs_cannot_alias_old_draft_checkpoint_allocations():
    backend = RecordingBackend(SCRIPT[:3])
    previous = None

    for epoch_number in range(100):
        backend.prefill((epoch_number % VOCAB_SIZE,))
        result = generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=select_highest_logit,
        )
        current = result.rollback_checkpoints[0]
        if previous is not None:
            assert current.allocation_id == previous.allocation_id == 2
            assert current.epoch != previous.epoch
            with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
                backend.rollback_cache(previous)
            backend.release_cache_checkpoint(previous)
        previous = current

    assert backend.active_checkpoint_count == 1
    backend.release_cache_checkpoint(previous)


def test_isolated_source_import_and_complete_fake_lifecycle_load_no_optional_runtime():
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        """
        import sys

        sys.path.insert(0, sys.argv[1])
        import onyx_cuda
        from onyx_cuda.testing import FakeAutoregressiveBackend

        rows = ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
        backend = FakeAutoregressiveBackend(rows)
        backend.prefill((0,))
        root = backend.create_cache_checkpoint()
        result = onyx_cuda.generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=onyx_cuda.select_highest_logit,
        )
        assert result.proposal_token_ids == (1,)
        backend.rollback_cache(result.rollback_checkpoints[0])
        for checkpoint in result.rollback_checkpoints:
            backend.release_cache_checkpoint(checkpoint)
        backend.release_cache_checkpoint(root)
        assert backend.active_checkpoint_count == 0

        forbidden = (
            "onyx", "mlx", "torch", "transformers", "bitsandbytes", "accelerate",
            "huggingface_hub", "tokenizers", "psutil", "onnxruntime",
        )
        loaded = tuple(sys.modules)
        assert "onyx_cuda._grammar_native" not in loaded
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in loaded
            for prefix in forbidden
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
