import gc
import importlib
import inspect
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path

import pytest

import onyx_cuda.speculative_handoff as handoff_module
from onyx_cuda import (
    BackendError,
    CacheCheckpointStateError,
    ContinuationAwareSpeculativeIterationResult,
    SpeculativeHandoffCleanupError,
    SpeculativeHandoffError,
    SpeculativeHandoffInvariantError,
    SpeculativeIterationCleanupError,
    TwoIterationSpeculativeHandoffResult,
    coordinate_continuation_aware_speculative_iteration,
    coordinate_two_iteration_speculative_handoff,
)
from onyx_cuda.testing import FakeAutoregressiveBackend


VOCAB_SIZE = 8
PROMPT = (6, 7)
CURRENT_TOKEN = 0
FIRST_PROPOSAL = (1, 2, 3)
SECOND_PROPOSAL = (4, 5, 6)


def _row(selected_token_id=0):
    return tuple(
        20.0 if token_id == selected_token_id else float(-token_id)
        for token_id in range(VOCAB_SIZE)
    )


def _script(row_count=96):
    return tuple(_row() for _ in range(row_count))


class ScriptedSelector:
    __slots__ = ("outcomes", "calls", "__weakref__")

    def __init__(self, outcomes):
        self.outcomes = tuple(outcomes)
        self.calls = []

    def __call__(self, row):
        self.calls.append(row)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingBackend(FakeAutoregressiveBackend):
    def __init__(self, role, events):
        self.role = role
        self.events = events
        self.create_calls = []
        self.rollback_calls = []
        self.release_calls = []
        self.decode_calls = []
        self.verify_calls = []
        self.reset_calls = 0
        super().__init__(_script(), model_id=f"{role}-fake")

    def decode(self, token_id, /):
        self.decode_calls.append(token_id)
        return super().decode(token_id)

    def verify_proposal(self, current_token_id, proposal_token_ids, /):
        self.verify_calls.append((current_token_id, tuple(proposal_token_ids)))
        return super().verify_proposal(current_token_id, proposal_token_ids)

    def create_cache_checkpoint(self):
        checkpoint = super().create_cache_checkpoint()
        self.create_calls.append(checkpoint)
        self.events.append((self.role, "create", checkpoint))
        return checkpoint

    def rollback_cache(self, checkpoint, /):
        self.rollback_calls.append(checkpoint)
        self.events.append((self.role, "rollback", checkpoint))
        return super().rollback_cache(checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        self.events.append((self.role, "release", checkpoint))
        return super().release_cache_checkpoint(checkpoint)

    def reset(self):
        self.reset_calls += 1
        return super().reset()


class FaultBackend(RecordingBackend):
    def __init__(self, role, events):
        self.create_attempts = 0
        self.create_failures = {}
        self.rollback_attempts = 0
        self.rollback_failures = {}
        self.release_failures_by_id = {}
        super().__init__(role, events)

    def create_cache_checkpoint(self):
        self.create_attempts += 1
        failure = self.create_failures.get(self.create_attempts)
        if failure is not None:
            self.events.append((self.role, "create failure", failure))
            raise failure
        return super().create_cache_checkpoint()

    def rollback_cache(self, checkpoint, /):
        self.rollback_attempts += 1
        self.rollback_calls.append(checkpoint)
        self.events.append((self.role, "rollback", checkpoint))
        failure = self.rollback_failures.get(self.rollback_attempts)
        if failure is not None:
            raise failure
        return FakeAutoregressiveBackend.rollback_cache(self, checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        self.events.append((self.role, "release", checkpoint))
        allocation_id = getattr(checkpoint, "allocation_id", None)
        failures = self.release_failures_by_id.get(allocation_id)
        if failures:
            raise failures.pop(0)
        return FakeAutoregressiveBackend.release_cache_checkpoint(self, checkpoint)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    cache_length: int


class SettledMalformedTarget(FaultBackend):
    def __init__(self, role, events):
        self.malformed_checkpoint = None
        super().__init__(role, events)

    def create_cache_checkpoint(self):
        if self.create_attempts == 1:
            self.create_attempts += 1
            self.malformed_checkpoint = CheckpointRecord(self.cache_length + 1)
            self.create_calls.append(self.malformed_checkpoint)
            self.events.append((self.role, "create", self.malformed_checkpoint))
            return self.malformed_checkpoint
        return super().create_cache_checkpoint()

    def release_cache_checkpoint(self, checkpoint, /):
        if checkpoint is self.malformed_checkpoint:
            self.release_calls.append(checkpoint)
            self.events.append((self.role, "release", checkpoint))
            return None
        return super().release_cache_checkpoint(checkpoint)


class MutatingCheckpointTarget(FaultBackend):
    def create_cache_checkpoint(self):
        checkpoint = super().create_cache_checkpoint()
        if self.create_attempts == 2:
            self.decode(0)
        return checkpoint


class MutatingReleaseDraft(FaultBackend):
    def __init__(self, role, events):
        self.mutated_release = False
        super().__init__(role, events)

    def release_cache_checkpoint(self, checkpoint, /):
        result = super().release_cache_checkpoint(checkpoint)
        if checkpoint.allocation_id == 6 and not self.mutated_release:
            self.mutated_release = True
            self.decode(1)
        return result


def _replacement_for(proposal, accepted_count, *, offset):
    token_id = (proposal[accepted_count] + offset) % VOCAB_SIZE
    if token_id == proposal[accepted_count]:
        token_id = (token_id + 1) % VOCAB_SIZE
    return token_id


def _target_outcomes(proposal, accepted_count, *, replacement_offset, bonus):
    if accepted_count == len(proposal):
        return (*proposal, bonus), bonus
    replacement = _replacement_for(
        proposal,
        accepted_count,
        offset=replacement_offset,
    )
    return (*proposal[:accepted_count], replacement), replacement


def _prepared_backends(*, backend_type=RecordingBackend, target_type=None):
    events = []
    target_type = backend_type if target_type is None else target_type
    draft = backend_type("draft", events)
    target = target_type("target", events)
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    events.clear()
    return draft, target, draft_root, target_root, events


def _selectors_for(
    first_accepted,
    second_accepted,
    *,
    first_proposal=FIRST_PROPOSAL,
    second_proposal=SECOND_PROPOSAL,
):
    first_target, first_uncached = _target_outcomes(
        first_proposal,
        first_accepted,
        replacement_offset=4,
        bonus=7,
    )
    second_target, second_uncached = _target_outcomes(
        second_proposal,
        second_accepted,
        replacement_offset=3,
        bonus=0,
    )
    draft_selector = ScriptedSelector((*first_proposal, *second_proposal))
    target_selector = ScriptedSelector((*first_target, *second_target))
    return draft_selector, target_selector, first_uncached, second_uncached


def _coordinate(
    draft,
    target,
    draft_root,
    target_root,
    *,
    first_accepted,
    second_accepted,
    first_proposal=FIRST_PROPOSAL,
    second_proposal=SECOND_PROPOSAL,
):
    draft_selector, target_selector, first_uncached, second_uncached = _selectors_for(
        first_accepted,
        second_accepted,
        first_proposal=first_proposal,
        second_proposal=second_proposal,
    )
    result = coordinate_two_iteration_speculative_handoff(
        draft,
        target,
        CURRENT_TOKEN,
        proposal_length=len(first_proposal),
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    return (
        result,
        draft_selector,
        target_selector,
        first_uncached,
        second_uncached,
    )


def _result(
    proposal,
    accepted_count,
    *,
    initial_cache_length,
    uncached_next_token_id,
):
    replacement = (
        None
        if accepted_count == len(proposal)
        else uncached_next_token_id
    )
    return ContinuationAwareSpeculativeIterationResult(
        proposal_token_ids=proposal,
        accepted_count=accepted_count,
        replacement_token_id=replacement,
        initial_cache_length=initial_cache_length,
        final_cache_length=initial_cache_length + accepted_count + 1,
        uncached_next_token_id=uncached_next_token_id,
    )


def test_public_surface_and_exact_signature():
    import onyx_cuda

    current_module = importlib.import_module("onyx_cuda.speculative_handoff")
    public_names = (
        "SpeculativeHandoffCleanupError",
        "SpeculativeHandoffError",
        "SpeculativeHandoffInvariantError",
        "TwoIterationSpeculativeHandoffResult",
        "coordinate_two_iteration_speculative_handoff",
    )
    assert tuple(current_module.__all__) == public_names
    for name in public_names:
        symbol = getattr(current_module, name)
        assert getattr(onyx_cuda, name) is symbol
        assert name in onyx_cuda.__all__
        assert symbol.__module__ == "onyx_cuda.speculative_handoff"

    assert issubclass(SpeculativeHandoffError, BackendError)
    assert issubclass(SpeculativeHandoffInvariantError, SpeculativeHandoffError)
    assert issubclass(SpeculativeHandoffCleanupError, SpeculativeHandoffError)

    parameters = inspect.signature(
        coordinate_two_iteration_speculative_handoff
    ).parameters
    assert tuple(parameters) == (
        "draft_backend",
        "target_backend",
        "current_token_id",
        "proposal_length",
        "draft_select_token",
        "target_select_token",
        "draft_root_checkpoint",
        "target_root_checkpoint",
    )
    for name in tuple(parameters)[3:]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(
        inspect.signature(
            coordinate_continuation_aware_speculative_iteration
        ).parameters
    ) == tuple(parameters)
    assert tuple(field.name for field in fields(ContinuationAwareSpeculativeIterationResult)) == (
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "initial_cache_length",
        "final_cache_length",
        "uncached_next_token_id",
    )


def test_cleanup_error_freezes_ordered_evidence_and_sets_cause():
    original = RuntimeError("transaction failed")
    first_cleanup = ValueError("draft rollback failed")
    second_cleanup = CacheCheckpointStateError("target release failed")
    source = [
        ("draft initial root rollback", first_cleanup),
        ("target intermediate root release", second_cleanup),
    ]

    error = SpeculativeHandoffCleanupError(original, source)
    source.clear()

    assert error.original_failure is original
    assert error.cleanup_failures == (
        ("draft initial root rollback", first_cleanup),
        ("target intermediate root release", second_cleanup),
    )
    assert error.__cause__ is original
    with pytest.raises(ValueError, match="cannot be empty"):
        SpeculativeHandoffCleanupError(original, ())


@pytest.mark.parametrize(
    ("first_accepted", "second_accepted"),
    [(first, second) for first in range(4) for second in range(4)],
)
def test_complete_three_token_success_matrix_and_exact_handoff(
    monkeypatch,
    first_accepted,
    second_accepted,
):
    draft, target, draft_root, target_root, events = _prepared_backends()
    calls = []
    original = handoff_module.coordinate_continuation_aware_speculative_iteration

    def record_call(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        record_call,
    )
    (
        result,
        draft_selector,
        target_selector,
        first_uncached,
        second_uncached,
    ) = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        first_accepted=first_accepted,
        second_accepted=second_accepted,
    )

    first_output = FIRST_PROPOSAL[:first_accepted] + (first_uncached,)
    second_output = SECOND_PROPOSAL[:second_accepted] + (second_uncached,)
    intermediate_length = len(PROMPT) + first_accepted + 1
    final_length = intermediate_length + second_accepted + 1
    expected_cache = (
        PROMPT
        + (CURRENT_TOKEN,)
        + FIRST_PROPOSAL[:first_accepted]
        + (first_uncached,)
        + SECOND_PROPOSAL[:second_accepted]
    )

    assert len(calls) == 2
    assert calls[0][0] == (draft, target, CURRENT_TOKEN)
    assert calls[1][0] == (draft, target, first_uncached)
    for args, kwargs in calls:
        assert args[0] is draft
        assert args[1] is target
        assert kwargs["proposal_length"] == len(FIRST_PROPOSAL)
        assert kwargs["draft_select_token"] is draft_selector
        assert kwargs["target_select_token"] is target_selector
    assert calls[0][1]["draft_root_checkpoint"] is draft_root
    assert calls[0][1]["target_root_checkpoint"] is target_root
    draft_intermediate = calls[1][1]["draft_root_checkpoint"]
    target_intermediate = calls[1][1]["target_root_checkpoint"]
    assert draft_intermediate is not draft_root
    assert target_intermediate is not target_root
    assert draft_intermediate.cache_length == intermediate_length
    assert target_intermediate.cache_length == intermediate_length

    assert result.first_iteration.proposal_token_ids == FIRST_PROPOSAL
    assert result.first_iteration.accepted_count == first_accepted
    assert result.first_iteration.output_token_ids == first_output
    assert result.second_iteration.proposal_token_ids == SECOND_PROPOSAL
    assert result.second_iteration.accepted_count == second_accepted
    assert result.second_iteration.output_token_ids == second_output
    assert result.handoff_token_id == first_uncached
    assert result.output_token_ids == first_output + second_output
    assert result.uncached_next_token_id == second_uncached
    assert result.initial_cache_length == len(PROMPT)
    assert result.intermediate_cache_length == intermediate_length
    assert result.final_cache_length == final_length
    assert result.output_token_ids == (
        result.first_iteration.output_token_ids
        + result.second_iteration.output_token_ids
    )

    assert draft.cached_token_ids == expected_cache
    assert target.cached_token_ids == expected_cache
    assert draft.cache_length == target.cache_length == final_length
    assert expected_cache[-(second_accepted + 1)] == first_uncached
    assert expected_cache != (*expected_cache, second_uncached)
    assert len(draft_selector.calls) == 2 * len(FIRST_PROPOSAL)
    assert len(target_selector.calls) == first_accepted + second_accepted + 2
    assert draft.verify_calls == []
    assert target.verify_calls == [
        (CURRENT_TOKEN, FIRST_PROPOSAL),
        (first_uncached, SECOND_PROPOSAL),
    ]

    create_events = [event for event in events if event[1] == "create"]
    draft_index = next(
        index
        for index, event in enumerate(create_events)
        if event[2] is draft_intermediate
    )
    target_index = next(
        index
        for index, event in enumerate(create_events)
        if event[2] is target_intermediate
    )
    assert target_index == draft_index + 1
    assert [event[0] for event in create_events[draft_index : target_index + 1]] == [
        "draft",
        "target",
    ]
    release_events = [event for event in events if event[1] == "release"]
    assert release_events[-2:] == [
        ("draft", "release", draft_intermediate),
        ("target", "release", target_intermediate),
    ]
    assert draft.release_calls[-1] is draft_intermediate
    assert target.release_calls == [target_intermediate]
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1

    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT


@pytest.mark.parametrize(
    ("first_accepted", "second_accepted"),
    [(0, 0), (0, 1), (1, 0), (1, 1)],
)
def test_all_one_token_outcome_pairings(first_accepted, second_accepted):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    first_proposal = (1,)
    second_proposal = (2,)
    result, _, _, first_uncached, second_uncached = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        first_accepted=first_accepted,
        second_accepted=second_accepted,
        first_proposal=first_proposal,
        second_proposal=second_proposal,
    )

    assert result.output_token_ids == (
        first_proposal[:first_accepted]
        + (first_uncached,)
        + second_proposal[:second_accepted]
        + (second_uncached,)
    )
    assert result.final_cache_length == len(PROMPT) + first_accepted + second_accepted + 2
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_result_contract_is_frozen_slotted_minimal_and_validates_relationships():
    first = _result(
        FIRST_PROPOSAL,
        1,
        initial_cache_length=2,
        uncached_next_token_id=5,
    )
    second = _result(
        SECOND_PROPOSAL,
        3,
        initial_cache_length=4,
        uncached_next_token_id=0,
    )
    result = TwoIterationSpeculativeHandoffResult(first, second)

    assert tuple(field.name for field in fields(result)) == (
        "first_iteration",
        "second_iteration",
    )
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "next_current_token_id")
    assert result == TwoIterationSpeculativeHandoffResult(first, second)
    assert result.output_token_ids == (1, 5, 4, 5, 6, 0)
    with pytest.raises(FrozenInstanceError):
        result.first_iteration = second
    with pytest.raises(SpeculativeHandoffInvariantError, match="first_iteration"):
        TwoIterationSpeculativeHandoffResult(object(), second)
    with pytest.raises(SpeculativeHandoffInvariantError, match="same positive"):
        TwoIterationSpeculativeHandoffResult(
            first,
            _result(
                (4,),
                1,
                initial_cache_length=4,
                uncached_next_token_id=0,
            ),
        )
    with pytest.raises(SpeculativeHandoffInvariantError, match="first final"):
        TwoIterationSpeculativeHandoffResult(
            first,
            _result(
                SECOND_PROPOSAL,
                3,
                initial_cache_length=5,
                uncached_next_token_id=0,
            ),
        )

    unsafe = object.__new__(ContinuationAwareSpeculativeIterationResult)
    for field in fields(first):
        object.__setattr__(unsafe, field.name, getattr(first, field.name))
    object.__setattr__(unsafe, "uncached_next_token_id", 6)
    with pytest.raises(SpeculativeHandoffInvariantError, match="mismatch uncached"):
        TwoIterationSpeculativeHandoffResult(unsafe, second)


def test_numeric_token_equality_is_not_a_coordinator_added_duplicate():
    draft, target, draft_root, target_root, _ = _prepared_backends()
    first_proposal = (1, 2, 3)
    first_accepted = 0
    first_uncached = _replacement_for(first_proposal, first_accepted, offset=4)
    second_proposal = (first_uncached, 6, 7)
    result, _, _, _, second_uncached = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        first_accepted=first_accepted,
        second_accepted=1,
        first_proposal=first_proposal,
        second_proposal=second_proposal,
    )

    assert result.first_iteration.output_token_ids == (first_uncached,)
    assert result.second_iteration.output_token_ids == (
        first_uncached,
        second_uncached,
    )
    assert result.output_token_ids == (
        first_uncached,
        first_uncached,
        second_uncached,
    )
    assert result.output_token_ids.count(first_uncached) == 2
    assert draft.cached_token_ids == target.cached_token_ids == (
        PROMPT + (CURRENT_TOKEN, first_uncached, first_uncached)
    )


@pytest.mark.parametrize(
    ("draft_root", "target_root", "error_type", "message"),
    [
        (object(), CheckpointRecord(2), TypeError, "draft_root_checkpoint"),
        (CheckpointRecord(True), CheckpointRecord(2), TypeError, "integer"),
        (CheckpointRecord(0), CheckpointRecord(2), ValueError, "greater than zero"),
        (
            CheckpointRecord(2),
            CheckpointRecord(3),
            SpeculativeHandoffInvariantError,
            "lengths differ",
        ),
    ],
)
def test_read_only_root_preflight_rejects_before_any_backend_operation(
    draft_root,
    target_root,
    error_type,
    message,
):
    draft, target, _, _, events = _prepared_backends()
    draft_selector = ScriptedSelector(FIRST_PROPOSAL + SECOND_PROPOSAL)
    target_selector = ScriptedSelector((1,))

    with pytest.raises(error_type, match=message):
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert events == []
    assert draft_selector.calls == target_selector.calls == []


def test_root_preflight_reads_each_length_once(monkeypatch):
    class CountingCheckpoint:
        def __init__(self):
            self.reads = 0

        @property
        def cache_length(self):
            self.reads += 1
            return len(PROMPT)

    draft_root = CountingCheckpoint()
    target_root = CountingCheckpoint()
    stop = RuntimeError("first call boundary")

    def fail_first_call(*args, **kwargs):
        raise stop

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        fail_first_call,
    )
    with pytest.raises(RuntimeError) as raised:
        coordinate_two_iteration_speculative_handoff(
            object(),
            object(),
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=lambda row: 0,
            target_select_token=lambda row: 0,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is stop
    assert draft_root.reads == target_root.reads == 1


def test_first_call_failure_propagates_exactly_without_outer_cleanup(monkeypatch):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    failure = RuntimeError("first target selector failed")
    draft_selector = ScriptedSelector(FIRST_PROPOSAL + SECOND_PROPOSAL)
    target_selector = ScriptedSelector((failure,))
    calls = 0
    original = handoff_module.coordinate_continuation_aware_speculative_iteration

    def record_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        record_call,
    )
    with pytest.raises(RuntimeError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is failure
    assert calls == 1
    assert len(draft.rollback_calls) == len(target.rollback_calls) == 2
    assert target.create_calls == [target_root]
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_first_call_nested_cleanup_error_is_not_wrapped(monkeypatch):
    draft, target, draft_root, target_root, events = _prepared_backends()
    original = RuntimeError("D38 failure")
    nested_cleanup = ValueError("D38 cleanup failure")
    nested = SpeculativeIterationCleanupError(
        original,
        (("draft root rollback", nested_cleanup),),
    )

    def fail_first_call(*args, **kwargs):
        raise nested

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        fail_first_call,
    )
    with pytest.raises(SpeculativeIterationCleanupError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=lambda row: 0,
            target_select_token=lambda row: 0,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is nested
    assert events == []


@pytest.mark.parametrize(
    ("returned_result", "message"),
    [
        (object(), "ContinuationAwareSpeculativeIterationResult"),
        (
            _result(
                FIRST_PROPOSAL,
                0,
                initial_cache_length=3,
                uncached_next_token_id=5,
            ),
            "initial cache length",
        ),
    ],
)
def test_first_result_validation_failure_enters_outer_cleanup(
    monkeypatch,
    returned_result,
    message,
):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    calls = 0

    def return_invalid_first_result(*args, **kwargs):
        nonlocal calls
        calls += 1
        return returned_result

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        return_invalid_first_result,
    )
    with pytest.raises(SpeculativeHandoffInvariantError, match=message):
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=lambda row: 0,
            target_select_token=lambda row: 0,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert calls == 1
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


@pytest.mark.parametrize(("role", "attempt"), [("draft", 6), ("target", 2)])
def test_intermediate_creation_failure_restores_initial_roots_and_settles_peer(
    role,
    attempt,
):
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend
    )
    failure = RuntimeError(f"{role} intermediate creation failed")
    backend = draft if role == "draft" else target
    backend.create_failures[attempt] = failure
    draft_selector, target_selector, _, _ = _selectors_for(1, 1)

    with pytest.raises(RuntimeError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is failure
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    if role == "target":
        assert draft.release_calls[-1].cache_length == len(PROMPT) + 2


def test_returned_malformed_intermediate_is_owned_before_validation_and_settled():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend,
        target_type=SettledMalformedTarget,
    )
    draft_selector, target_selector, _, _ = _selectors_for(1, 1)

    with pytest.raises(SpeculativeHandoffInvariantError, match="reports cache length"):
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert target.release_calls[-1] is target.malformed_checkpoint
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_checkpoint_creation_cache_mutation_is_detected_and_cleaned_up():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend,
        target_type=MutatingCheckpointTarget,
    )
    draft_selector, target_selector, _, _ = _selectors_for(1, 1)

    with pytest.raises(
        SpeculativeHandoffInvariantError,
        match="target backend reported cache length",
    ):
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_second_call_failure_propagates_exactly_after_healthy_outer_cleanup(monkeypatch):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    failure = RuntimeError("second D38 call failed")
    calls = 0
    original = handoff_module.coordinate_continuation_aware_speculative_iteration

    def fail_second_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        fail_second_call,
    )
    draft_selector, target_selector, _, _ = _selectors_for(1, 1)
    with pytest.raises(RuntimeError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is failure
    assert calls == 2
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_nested_second_call_cleanup_error_is_preserved_after_outer_cleanup(monkeypatch):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    original = RuntimeError("second D38 failure")
    nested_cleanup = RuntimeError("second D38 cleanup failure")
    nested = SpeculativeIterationCleanupError(
        original,
        (("target root rollback", nested_cleanup),),
    )
    calls = 0
    d38 = handoff_module.coordinate_continuation_aware_speculative_iteration

    def fail_second_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise nested
        return d38(*args, **kwargs)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        fail_second_call,
    )
    draft_selector, target_selector, _, _ = _selectors_for(1, 1)
    with pytest.raises(SpeculativeIterationCleanupError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is nested
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_failed_second_selector_state_remains_consumed_without_retry():
    draft, target, draft_root, target_root, _ = _prepared_backends()
    failure = RuntimeError("second selector failed")
    draft_selector = ScriptedSelector((*FIRST_PROPOSAL, *SECOND_PROPOSAL))
    target_selector = ScriptedSelector(
        (*FIRST_PROPOSAL, 7, SECOND_PROPOSAL[0], failure)
    )

    with pytest.raises(RuntimeError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is failure
    assert len(draft_selector.calls) == 6
    assert len(target_selector.calls) == 6
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_malformed_second_result_restores_roots(monkeypatch):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    calls = 0
    original = handoff_module.coordinate_continuation_aware_speculative_iteration

    def corrupt_second_result(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            object.__setattr__(
                result,
                "uncached_next_token_id",
                (result.uncached_next_token_id + 1) % VOCAB_SIZE,
            )
        return result

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        corrupt_second_result,
    )
    draft_selector, target_selector, _, _ = _selectors_for(1, 0)
    with pytest.raises(SpeculativeHandoffInvariantError, match="mismatch uncached"):
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_success_path_release_failure_re_raises_original_after_cleanup():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend
    )
    failure = RuntimeError("draft intermediate release failed once")
    draft.release_failures_by_id[6] = [failure]
    draft_selector, target_selector, _, _ = _selectors_for(3, 3)

    with pytest.raises(RuntimeError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is failure
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert [checkpoint.allocation_id for checkpoint in draft.release_calls].count(6) == 2


def test_cache_mutation_during_success_release_is_detected_and_cleaned_up():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=MutatingReleaseDraft
    )
    draft_selector, target_selector, _, _ = _selectors_for(3, 3)

    with pytest.raises(
        SpeculativeHandoffInvariantError,
        match="draft backend reported cache length",
    ):
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1


def test_outer_cleanup_aggregates_all_failures_in_global_order():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend
    )
    original = RuntimeError("success-path draft release failed")
    draft_rollback = RuntimeError("draft initial rollback failed")
    target_rollback = RuntimeError("target initial rollback failed")
    draft_release = RuntimeError("draft settlement failed")
    target_release = RuntimeError("target settlement failed")
    draft.release_failures_by_id[6] = [original, draft_release]
    target.release_failures_by_id[2] = [target_release]
    draft.rollback_failures[3] = draft_rollback
    target.rollback_failures[3] = target_rollback
    draft_selector, target_selector, _, _ = _selectors_for(3, 3)

    with pytest.raises(SpeculativeHandoffCleanupError) as raised:
        coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    error = raised.value
    assert error.original_failure is original
    assert error.cleanup_failures == (
        ("draft initial root rollback", draft_rollback),
        ("target initial root rollback", target_rollback),
        ("draft intermediate root release", draft_release),
        ("target intermediate root release", target_release),
    )
    assert error.__cause__ is original
    assert draft.reset_calls == target.reset_calls == 2


def test_result_does_not_retain_backends_or_selectors():
    draft, target, draft_root, target_root, _ = _prepared_backends()
    draft_selector, target_selector, _, _ = _selectors_for(1, 3)
    draft_ref = weakref.ref(draft)
    target_ref = weakref.ref(target)
    draft_selector_ref = weakref.ref(draft_selector)
    target_selector_ref = weakref.ref(target_selector)

    result = coordinate_two_iteration_speculative_handoff(
        draft,
        target,
        CURRENT_TOKEN,
        proposal_length=3,
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    del draft, target, draft_root, target_root, draft_selector, target_selector
    gc.collect()

    assert result.first_iteration.proposal_token_ids == FIRST_PROPOSAL
    assert result.second_iteration.proposal_token_ids == SECOND_PROPOSAL
    assert draft_ref() is target_ref() is None
    assert draft_selector_ref() is target_selector_ref() is None


def test_one_thousand_reuses_have_stable_root_counts_and_allocation_growth():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FakeAutoregressiveBackendAdapter
    )
    proposal_length = len(FIRST_PROPOSAL)
    draft_allocations_per_operation = 2 * (proposal_length + 1) + 1

    for operation in range(1000):
        first_accepted = (operation // 4) % 4
        second_accepted = operation % 4
        draft_selector, target_selector, first_uncached, second_uncached = _selectors_for(
            first_accepted,
            second_accepted,
        )
        result = coordinate_two_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=proposal_length,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

        assert result.handoff_token_id == first_uncached
        assert result.uncached_next_token_id == second_uncached
        assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
        assert draft._next_checkpoint_id == (
            2 + (operation + 1) * draft_allocations_per_operation
        )
        assert target._next_checkpoint_id == 2 + operation + 1
        draft.rollback_cache(draft_root)
        target.rollback_cache(target_root)
        assert draft.cached_token_ids == target.cached_token_ids == PROMPT


class FakeAutoregressiveBackendAdapter(FakeAutoregressiveBackend):
    def __init__(self, role, events):
        super().__init__(_script(), model_id=f"{role}-fake")


def test_isolated_import_and_two_outcome_pairings_are_optional_runtime_free():
    package_root = Path(__file__).resolve().parents[1]
    script = (
        f"import sys\nsys.path.insert(0, {str(package_root / 'src')!r})\n"
        + textwrap.dedent(
        """
        from onyx_cuda import (
            SpeculativeHandoffCleanupError,
            SpeculativeHandoffError,
            SpeculativeHandoffInvariantError,
            TwoIterationSpeculativeHandoffResult,
            coordinate_two_iteration_speculative_handoff,
        )
        from onyx_cuda.testing import FakeAutoregressiveBackend

        def row():
            return tuple(0.0 for _ in range(8))

        class Selector:
            def __init__(self, values):
                self.values = iter(values)
            def __call__(self, logits):
                return next(self.values)

        cases = (
            (1, 3, (1, 5, 4, 5, 6, 0)),
            (3, 1, (1, 2, 3, 7, 4, 7)),
        )
        for first_accepted, second_accepted, expected in cases:
            draft = FakeAutoregressiveBackend((row(),) * 64)
            target = FakeAutoregressiveBackend((row(),) * 64)
            draft.prefill((6, 7))
            target.prefill((6, 7))
            draft_root = draft.create_cache_checkpoint()
            target_root = target.create_cache_checkpoint()
            first_target = (1, 5) if first_accepted == 1 else (1, 2, 3, 7)
            second_target = (4, 7) if second_accepted == 1 else (4, 5, 6, 0)
            result = coordinate_two_iteration_speculative_handoff(
                draft,
                target,
                0,
                proposal_length=3,
                draft_select_token=Selector((1, 2, 3, 4, 5, 6)),
                target_select_token=Selector(first_target + second_target),
                draft_root_checkpoint=draft_root,
                target_root_checkpoint=target_root,
            )
            assert result.output_token_ids == expected
            assert result.uncached_next_token_id == expected[-1]
            assert draft.active_checkpoint_count == target.active_checkpoint_count == 1

        forbidden = (
            "onyx", "mlx", "torch", "transformers", "tokenizers",
            "huggingface_hub", "bitsandbytes", "accelerate",
            "onnxruntime", "psutil", "onyx_cuda._grammar_native",
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in forbidden
        )
        """
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=package_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
