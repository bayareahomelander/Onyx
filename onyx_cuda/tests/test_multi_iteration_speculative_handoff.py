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
    ContinuationAwareSpeculativeIterationResult,
    MultiIterationSpeculativeHandoffResult,
    SpeculativeHandoffCleanupError,
    SpeculativeHandoffError,
    SpeculativeHandoffInvariantError,
    coordinate_continuation_aware_speculative_iteration,
    coordinate_multi_iteration_speculative_handoff,
    coordinate_two_iteration_speculative_handoff,
)
from onyx_cuda.testing import FakeAutoregressiveBackend


VOCAB_SIZE = 32
PROMPT = (30, 31)
CURRENT_TOKEN = 0


def _row(selected_token_id=0):
    return tuple(
        20.0 if token_id == selected_token_id else float(-token_id)
        for token_id in range(VOCAB_SIZE)
    )


def _script(row_count=512):
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
        self.active_counts = []
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
        self.active_counts.append(self.active_checkpoint_count)
        self.events.append((self.role, "create", checkpoint))
        return checkpoint

    def rollback_cache(self, checkpoint, /):
        self.rollback_calls.append(checkpoint)
        self.events.append((self.role, "rollback", checkpoint))
        return super().rollback_cache(checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        self.events.append((self.role, "release", checkpoint))
        result = super().release_cache_checkpoint(checkpoint)
        self.active_counts.append(self.active_checkpoint_count)
        return result

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
        result = FakeAutoregressiveBackend.release_cache_checkpoint(self, checkpoint)
        self.active_counts.append(self.active_checkpoint_count)
        return result


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    cache_length: int


class MalformedTarget(FaultBackend):
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


class MutatingTarget(FaultBackend):
    def create_cache_checkpoint(self):
        checkpoint = super().create_cache_checkpoint()
        if self.create_attempts == 2:
            self.decode(0)
        return checkpoint


def _proposals(iteration_count, proposal_length):
    return tuple(
        tuple(1 + ((iteration * proposal_length + offset) % 24) for offset in range(proposal_length))
        for iteration in range(iteration_count)
    )


def _selectors_for(proposals, accepted_counts):
    draft_outcomes = []
    target_outcomes = []
    uncached_tokens = []
    for position, (proposal, accepted_count) in enumerate(
        zip(proposals, accepted_counts)
    ):
        draft_outcomes.extend(proposal)
        if accepted_count == len(proposal):
            uncached = 25 + (position % 5)
            target_outcomes.extend((*proposal, uncached))
        else:
            uncached = (proposal[accepted_count] + 9 + position) % VOCAB_SIZE
            if uncached == proposal[accepted_count]:
                uncached = (uncached + 1) % VOCAB_SIZE
            target_outcomes.extend((*proposal[:accepted_count], uncached))
        uncached_tokens.append(uncached)
    return (
        ScriptedSelector(draft_outcomes),
        ScriptedSelector(target_outcomes),
        tuple(uncached_tokens),
    )


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


def _coordinate(
    draft,
    target,
    draft_root,
    target_root,
    *,
    accepted_counts,
    proposal_length=3,
):
    proposals = _proposals(len(accepted_counts), proposal_length)
    draft_selector, target_selector, uncached_tokens = _selectors_for(
        proposals,
        accepted_counts,
    )
    result = coordinate_multi_iteration_speculative_handoff(
        draft,
        target,
        CURRENT_TOKEN,
        iteration_count=len(accepted_counts),
        proposal_length=proposal_length,
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    return result, proposals, uncached_tokens, draft_selector, target_selector


def _result(
    proposal,
    accepted_count,
    *,
    initial_cache_length,
    uncached_next_token_id,
):
    return ContinuationAwareSpeculativeIterationResult(
        proposal_token_ids=proposal,
        accepted_count=accepted_count,
        replacement_token_id=(
            None if accepted_count == len(proposal) else uncached_next_token_id
        ),
        initial_cache_length=initial_cache_length,
        final_cache_length=initial_cache_length + accepted_count + 1,
        uncached_next_token_id=uncached_next_token_id,
    )


def _normalized_events(events):
    normalized = []
    for role, operation, evidence in events:
        if operation in {"create", "rollback", "release"}:
            normalized.append(
                (
                    role,
                    operation,
                    getattr(evidence, "allocation_id", None),
                    evidence.cache_length,
                )
            )
        else:
            normalized.append((role, operation, type(evidence), str(evidence)))
    return normalized


def test_public_surface_exact_signature_and_result_fields():
    import onyx_cuda

    current_module = importlib.import_module("onyx_cuda.speculative_handoff")
    public_names = (
        "SpeculativeHandoffCleanupError",
        "SpeculativeHandoffError",
        "SpeculativeHandoffInvariantError",
        "TwoIterationSpeculativeHandoffResult",
        "coordinate_two_iteration_speculative_handoff",
        "MultiIterationSpeculativeHandoffResult",
        "coordinate_multi_iteration_speculative_handoff",
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
    assert tuple(
        inspect.signature(coordinate_multi_iteration_speculative_handoff).parameters
    ) == (
        "draft_backend",
        "target_backend",
        "current_token_id",
        "iteration_count",
        "proposal_length",
        "draft_select_token",
        "target_select_token",
        "draft_root_checkpoint",
        "target_root_checkpoint",
    )
    parameters = inspect.signature(
        coordinate_multi_iteration_speculative_handoff
    ).parameters
    for name in tuple(parameters)[3:]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(field.name for field in fields(MultiIterationSpeculativeHandoffResult)) == (
        "iterations",
    )
    assert tuple(
        inspect.signature(coordinate_two_iteration_speculative_handoff).parameters
    ) == (
        "draft_backend",
        "target_backend",
        "current_token_id",
        "proposal_length",
        "draft_select_token",
        "target_select_token",
        "draft_root_checkpoint",
        "target_root_checkpoint",
    )


@pytest.mark.parametrize(
    ("iteration_count", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        (1.0, TypeError),
        ("2", TypeError),
        (object(), TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_invalid_count_fails_before_root_backend_selector_or_d38_work(
    monkeypatch,
    iteration_count,
    error_type,
):
    touched = []

    def fail_if_touched(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("work occurred before iteration_count validation")

    monkeypatch.setattr(
        handoff_module,
        "_validate_initial_root_metadata",
        fail_if_touched,
    )
    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        fail_if_touched,
    )
    with pytest.raises(error_type):
        coordinate_multi_iteration_speculative_handoff(
            object(),
            object(),
            CURRENT_TOKEN,
            iteration_count=iteration_count,
            proposal_length=1,
            draft_select_token=fail_if_touched,
            target_select_token=fail_if_touched,
            draft_root_checkpoint=object(),
            target_root_checkpoint=object(),
        )
    assert touched == []


def test_result_contract_is_frozen_slotted_minimal_and_validates_sequence():
    first = _result((1, 2), 1, initial_cache_length=2, uncached_next_token_id=9)
    second = _result((3, 4), 2, initial_cache_length=4, uncached_next_token_id=10)
    third = _result((5, 6), 0, initial_cache_length=7, uncached_next_token_id=11)
    result = MultiIterationSpeculativeHandoffResult((first, second, third))

    assert result.iterations == (first, second, third)
    assert result.output_token_ids == (1, 9, 3, 4, 10, 11)
    assert result.uncached_next_token_id == 11
    assert result.initial_cache_length == 2
    assert result.final_cache_length == 8
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "iteration_count")
    assert not hasattr(result, "handoff_token_id")
    with pytest.raises(FrozenInstanceError):
        result.iterations = ()
    with pytest.raises(TypeError):
        MultiIterationSpeculativeHandoffResult([first])
    with pytest.raises(ValueError):
        MultiIterationSpeculativeHandoffResult(())
    with pytest.raises(SpeculativeHandoffInvariantError):
        MultiIterationSpeculativeHandoffResult((object(),))
    with pytest.raises(SpeculativeHandoffInvariantError, match="same positive"):
        MultiIterationSpeculativeHandoffResult(
            (
                first,
                _result((3,), 1, initial_cache_length=4, uncached_next_token_id=10),
            )
        )
    with pytest.raises(SpeculativeHandoffInvariantError, match="continuous"):
        MultiIterationSpeculativeHandoffResult(
            (
                first,
                _result((3, 4), 1, initial_cache_length=5, uncached_next_token_id=10),
            )
        )


@pytest.mark.parametrize("accepted_count", range(4))
def test_count_one_is_observationally_equivalent_to_direct_d38(accepted_count):
    multi_state = _prepared_backends()
    direct_state = _prepared_backends()
    proposal = _proposals(1, 3)
    multi_selectors = _selectors_for(proposal, (accepted_count,))
    direct_selectors = _selectors_for(proposal, (accepted_count,))

    multi_result = coordinate_multi_iteration_speculative_handoff(
        multi_state[0],
        multi_state[1],
        CURRENT_TOKEN,
        iteration_count=1,
        proposal_length=3,
        draft_select_token=multi_selectors[0],
        target_select_token=multi_selectors[1],
        draft_root_checkpoint=multi_state[2],
        target_root_checkpoint=multi_state[3],
    )
    direct_result = coordinate_continuation_aware_speculative_iteration(
        direct_state[0],
        direct_state[1],
        CURRENT_TOKEN,
        proposal_length=3,
        draft_select_token=direct_selectors[0],
        target_select_token=direct_selectors[1],
        draft_root_checkpoint=direct_state[2],
        target_root_checkpoint=direct_state[3],
    )

    assert multi_result.iterations == (direct_result,)
    assert multi_result.output_token_ids == direct_result.output_token_ids
    assert multi_result.uncached_next_token_id == direct_result.uncached_next_token_id
    assert multi_state[0].cached_token_ids == direct_state[0].cached_token_ids
    assert multi_state[1].cached_token_ids == direct_state[1].cached_token_ids
    assert _normalized_events(multi_state[4]) == _normalized_events(direct_state[4])
    assert multi_state[0].active_checkpoint_count == 1
    assert multi_state[1].active_checkpoint_count == 1


@pytest.mark.parametrize("first_accepted", range(4))
@pytest.mark.parametrize("second_accepted", range(4))
def test_count_two_is_observationally_equivalent_to_d39(
    first_accepted,
    second_accepted,
):
    accepted_counts = (first_accepted, second_accepted)
    proposals = _proposals(2, 3)
    multi_state = _prepared_backends()
    d39_state = _prepared_backends()
    multi_selectors = _selectors_for(proposals, accepted_counts)
    d39_selectors = _selectors_for(proposals, accepted_counts)

    multi_result = coordinate_multi_iteration_speculative_handoff(
        multi_state[0],
        multi_state[1],
        CURRENT_TOKEN,
        iteration_count=2,
        proposal_length=3,
        draft_select_token=multi_selectors[0],
        target_select_token=multi_selectors[1],
        draft_root_checkpoint=multi_state[2],
        target_root_checkpoint=multi_state[3],
    )
    d39_result = coordinate_two_iteration_speculative_handoff(
        d39_state[0],
        d39_state[1],
        CURRENT_TOKEN,
        proposal_length=3,
        draft_select_token=d39_selectors[0],
        target_select_token=d39_selectors[1],
        draft_root_checkpoint=d39_state[2],
        target_root_checkpoint=d39_state[3],
    )

    assert multi_result.iterations == (
        d39_result.first_iteration,
        d39_result.second_iteration,
    )
    assert multi_result.output_token_ids == d39_result.output_token_ids
    assert multi_result.uncached_next_token_id == d39_result.uncached_next_token_id
    assert multi_state[0].cached_token_ids == d39_state[0].cached_token_ids
    assert multi_state[1].cached_token_ids == d39_state[1].cached_token_ids
    assert len(multi_selectors[0].calls) == len(d39_selectors[0].calls)
    assert len(multi_selectors[1].calls) == len(d39_selectors[1].calls)
    assert _normalized_events(multi_state[4]) == _normalized_events(d39_state[4])
    assert multi_state[0].active_checkpoint_count == 1
    assert multi_state[1].active_checkpoint_count == 1


@pytest.mark.parametrize(
    ("accepted_counts", "proposal_length"),
    [
        ((0, 1, 2), 3),
        ((3, 3, 0, 2), 3),
        ((1, 0, 1, 0, 1), 1),
        ((2, 2, 1, 3, 0), 3),
    ],
)
def test_multi_iteration_success_has_exact_state_output_and_accounting(
    monkeypatch,
    accepted_counts,
    proposal_length,
):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    calls = []
    returned_results = []
    original = handoff_module.coordinate_continuation_aware_speculative_iteration

    def recording_call(*args, **kwargs):
        calls.append((args, kwargs))
        result = original(*args, **kwargs)
        returned_results.append(result)
        return result

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        recording_call,
    )
    result, proposals, uncached_tokens, draft_selector, target_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        accepted_counts=accepted_counts,
        proposal_length=proposal_length,
    )

    expected_output = tuple(
        token
        for proposal, accepted_count, uncached in zip(
            proposals,
            accepted_counts,
            uncached_tokens,
        )
        for token in (*proposal[:accepted_count], uncached)
    )
    current_tokens = (CURRENT_TOKEN, *uncached_tokens[:-1])
    expected_cache = tuple(PROMPT) + tuple(
        token
        for current, proposal, accepted_count in zip(
            current_tokens,
            proposals,
            accepted_counts,
        )
        for token in (current, *proposal[:accepted_count])
    )
    assert len(calls) == len(accepted_counts)
    assert tuple(call[0][2] for call in calls) == current_tokens
    assert all(call[0][0] is draft and call[0][1] is target for call in calls)
    assert all(call[1]["draft_select_token"] is draft_selector for call in calls)
    assert all(call[1]["target_select_token"] is target_selector for call in calls)
    assert all(call[1]["proposal_length"] == proposal_length for call in calls)
    assert all(
        stored is returned
        for stored, returned in zip(result.iterations, returned_results)
    )
    assert result.output_token_ids == expected_output
    assert result.uncached_next_token_id == uncached_tokens[-1]
    assert draft.cached_token_ids == expected_cache
    assert target.cached_token_ids == expected_cache
    assert result.final_cache_length == len(expected_cache)
    assert len(draft_selector.calls) == len(accepted_counts) * proposal_length
    assert len(target_selector.calls) == len(accepted_counts) + sum(accepted_counts)
    assert len(draft.create_calls) == (
        1
        + len(accepted_counts) * (proposal_length + 1)
        + len(accepted_counts)
        - 1
    )
    assert len(target.create_calls) == len(accepted_counts)
    assert len(draft.release_calls) == (
        len(accepted_counts) * (proposal_length + 1)
        + len(accepted_counts)
        - 1
    )
    assert len(target.release_calls) == len(accepted_counts) - 1
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert draft.reset_calls == 2
    assert target.reset_calls == 2


def test_root_rotation_order_is_bounded_and_never_reuses_retired_roots():
    draft, target, draft_root, target_root, events = _prepared_backends()
    result, _, _, _, _ = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        accepted_counts=(0, 1, 1, 0),
        proposal_length=1,
    )
    assert len(result.iterations) == 4

    draft_intermediate_ids = (4, 7, 10)
    target_intermediate_ids = (2, 3, 4)
    root_events = [
        (
            role,
            operation,
            evidence.allocation_id,
            evidence.cache_length,
        )
        for role, operation, evidence in events
        if operation in {"create", "release"}
        and (
            (role == "draft" and evidence.allocation_id in draft_intermediate_ids)
            or (role == "target" and evidence.allocation_id in target_intermediate_ids)
        )
    ]
    assert [(role, operation, allocation_id) for role, operation, allocation_id, _ in root_events] == [
        ("draft", "create", 4),
        ("target", "create", 2),
        ("draft", "create", 7),
        ("target", "create", 3),
        ("draft", "release", 4),
        ("target", "release", 2),
        ("draft", "create", 10),
        ("target", "create", 4),
        ("draft", "release", 7),
        ("target", "release", 3),
        ("draft", "release", 10),
        ("target", "release", 4),
    ]
    used_draft_roots = [
        checkpoint.allocation_id
        for checkpoint in draft.rollback_calls
        if checkpoint.allocation_id in draft_intermediate_ids
    ]
    used_target_roots = [
        checkpoint.allocation_id
        for checkpoint in target.rollback_calls
        if checkpoint.allocation_id in target_intermediate_ids
    ]
    assert used_draft_roots == list(draft_intermediate_ids)
    assert used_target_roots == [2, 3, 4, 4]
    assert max(target.active_counts) == 3
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_first_d38_failure_is_not_retried_or_outer_cleaned(monkeypatch):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    failure = RuntimeError("first transaction failed")
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        fail,
    )
    with pytest.raises(RuntimeError) as raised:
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=3,
            proposal_length=1,
            draft_select_token=ScriptedSelector((1,)),
            target_select_token=ScriptedSelector((2,)),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert raised.value is failure
    assert calls == 1
    assert draft.rollback_calls == []
    assert target.rollback_calls == []
    assert draft.release_calls == []
    assert target.release_calls == []


def test_first_returned_result_validation_failure_restores_initial_roots(monkeypatch):
    draft, target, draft_root, target_root, _ = _prepared_backends()

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        lambda *args, **kwargs: object(),
    )
    with pytest.raises(SpeculativeHandoffInvariantError):
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=3,
            proposal_length=1,
            draft_select_token=ScriptedSelector((1,)),
            target_select_token=ScriptedSelector((2,)),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.rollback_calls == [draft_root]
    assert target.rollback_calls == [target_root]
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_later_d38_failure_restores_initial_roots_and_settles_current_pair(
    monkeypatch,
):
    draft, target, draft_root, target_root, _ = _prepared_backends()
    proposals = _proposals(4, 1)
    selectors = _selectors_for(proposals, (1, 0, 1, 0))
    original = handoff_module.coordinate_continuation_aware_speculative_iteration
    failure = RuntimeError("third transaction failed")
    calls = 0

    def fail_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_continuation_aware_speculative_iteration",
        fail_third,
    )
    with pytest.raises(RuntimeError) as raised:
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=4,
            proposal_length=1,
            draft_select_token=selectors[0],
            target_select_token=selectors[1],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert raised.value is failure
    assert calls == 3
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_partial_first_next_pair_is_owned_before_validation_and_settled():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend,
        target_type=MalformedTarget,
    )
    proposals = _proposals(2, 1)
    selectors = _selectors_for(proposals, (0, 0))

    with pytest.raises(SpeculativeHandoffInvariantError, match="reports cache length"):
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=2,
            proposal_length=1,
            draft_select_token=selectors[0],
            target_select_token=selectors[1],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert target.malformed_checkpoint in target.release_calls
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_checkpoint_creation_cache_mutation_is_detected_and_cleaned_up():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend,
        target_type=MutatingTarget,
    )
    proposals = _proposals(2, 1)
    selectors = _selectors_for(proposals, (0, 0))

    with pytest.raises(SpeculativeHandoffInvariantError, match="target backend"):
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=2,
            proposal_length=1,
            draft_select_token=selectors[0],
            target_select_token=selectors[1],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_later_rotation_creation_failure_settles_current_pair():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend
    )
    failure = RuntimeError("later draft root creation failed")
    draft.create_failures[7] = failure
    proposals = _proposals(3, 1)
    selectors = _selectors_for(proposals, (0, 1, 0))

    with pytest.raises(RuntimeError) as raised:
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=3,
            proposal_length=1,
            draft_select_token=selectors[0],
            target_select_token=selectors[1],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert raised.value is failure
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_retirement_failure_keeps_current_and_next_pairs_for_cleanup():
    draft, target, draft_root, target_root, events = _prepared_backends(
        backend_type=FaultBackend
    )
    failure = RuntimeError("current draft retirement failed")
    draft.release_failures_by_id[4] = [failure]
    proposals = _proposals(3, 1)
    selectors = _selectors_for(proposals, (0, 1, 0))

    with pytest.raises(RuntimeError) as raised:
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=3,
            proposal_length=1,
            draft_select_token=selectors[0],
            target_select_token=selectors[1],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert raised.value is failure
    failure_index = events.index(("draft", "release", draft.create_calls[3]))
    cleanup_trace = [
        (role, operation, getattr(evidence, "allocation_id", None))
        for role, operation, evidence in events[failure_index + 1 :]
    ]
    assert cleanup_trace[:6] == [
        ("draft", "rollback", 1),
        ("target", "rollback", 1),
        ("draft", "release", 4),
        ("target", "release", 2),
        ("draft", "release", 7),
        ("target", "release", 3),
    ]
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_outer_cleanup_aggregates_six_failures_in_global_order():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend
    )
    original = RuntimeError("retirement failed")
    cleanup_errors = [
        RuntimeError("draft root rollback failed"),
        RuntimeError("target root rollback failed"),
        RuntimeError("draft current release failed"),
        RuntimeError("target current release failed"),
        RuntimeError("draft next release failed"),
        RuntimeError("target next release failed"),
    ]
    draft.release_failures_by_id[4] = [original, cleanup_errors[2]]
    target.release_failures_by_id[2] = [cleanup_errors[3]]
    draft.release_failures_by_id[7] = [cleanup_errors[4]]
    target.release_failures_by_id[3] = [cleanup_errors[5]]
    draft.rollback_failures[4] = cleanup_errors[0]
    target.rollback_failures[4] = cleanup_errors[1]
    proposals = _proposals(3, 1)
    selectors = _selectors_for(proposals, (0, 1, 0))

    with pytest.raises(SpeculativeHandoffCleanupError) as raised:
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=3,
            proposal_length=1,
            draft_select_token=selectors[0],
            target_select_token=selectors[1],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    error = raised.value
    assert error.original_failure is original
    assert error.__cause__ is original
    assert tuple(operation for operation, _ in error.cleanup_failures) == (
        "draft initial root rollback",
        "target initial root rollback",
        "draft intermediate root release",
        "target intermediate root release",
        "draft next intermediate root release",
        "target next intermediate root release",
    )
    assert tuple(failure for _, failure in error.cleanup_failures) == tuple(
        cleanup_errors
    )


def test_final_pair_release_failure_restores_initial_roots_and_retries_settlement():
    draft, target, draft_root, target_root, _ = _prepared_backends(
        backend_type=FaultBackend
    )
    failure = RuntimeError("final draft root release failed")
    draft.release_failures_by_id[7] = [failure]
    proposals = _proposals(3, 1)
    selectors = _selectors_for(proposals, (0, 1, 0))

    with pytest.raises(RuntimeError) as raised:
        coordinate_multi_iteration_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN,
            iteration_count=3,
            proposal_length=1,
            draft_select_token=selectors[0],
            target_select_token=selectors[1],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert raised.value is failure
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_result_does_not_retain_backends_selectors_or_roots():
    draft, target, draft_root, target_root, _ = _prepared_backends()
    result, _, _, draft_selector, target_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        accepted_counts=(0, 1, 1),
        proposal_length=1,
    )
    references = (
        weakref.ref(draft),
        weakref.ref(target),
        weakref.ref(draft_selector),
        weakref.ref(target_selector),
    )
    del draft, target, draft_selector, target_selector, draft_root, target_root
    gc.collect()

    assert all(reference() is None for reference in references)
    assert len(result.iterations) == 3
    assert tuple(field.name for field in fields(result)) == ("iterations",)


def test_one_thousand_reuses_have_exact_growth_and_stable_active_roots():
    draft, target, draft_root, target_root, _ = _prepared_backends()
    expected_draft_allocations = 1
    expected_target_allocations = 1

    for operation in range(1000):
        count = 1 + (operation % 3)
        accepted_counts = tuple((operation + position) % 2 for position in range(count))
        result, _, _, _, _ = _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            accepted_counts=accepted_counts,
            proposal_length=1,
        )
        expected_draft_allocations += count * 2 + count - 1
        expected_target_allocations += count - 1
        assert result.output_token_ids[-1] == result.uncached_next_token_id
        assert draft.create_calls[-1].allocation_id <= expected_draft_allocations
        assert target.create_calls[-1].allocation_id <= expected_target_allocations
        assert draft.active_checkpoint_count == 1
        assert target.active_checkpoint_count == 1
        draft.rollback_cache(draft_root)
        target.rollback_cache(target_root)
        assert draft.cached_token_ids == PROMPT
        assert target.cached_token_ids == PROMPT

    assert draft.create_calls[-1].allocation_id == expected_draft_allocations
    assert target.create_calls[-1].allocation_id == expected_target_allocations


def test_isolated_import_and_multi_count_execution_are_optional_runtime_free():
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(source_root)!r})
        from onyx_cuda import (
            MultiIterationSpeculativeHandoffResult,
            coordinate_multi_iteration_speculative_handoff,
        )
        from onyx_cuda.testing import FakeAutoregressiveBackend

        def row():
            return tuple(float(-token) for token in range(16))

        class Selector:
            def __init__(self, values):
                self.values = iter(values)
            def __call__(self, logits):
                return next(self.values)

        for count in (1, 2, 4):
            draft = FakeAutoregressiveBackend([row()] * 128, model_id="draft")
            target = FakeAutoregressiveBackend([row()] * 128, model_id="target")
            draft.prefill((14, 15))
            target.prefill((14, 15))
            draft_root = draft.create_cache_checkpoint()
            target_root = target.create_cache_checkpoint()
            result = coordinate_multi_iteration_speculative_handoff(
                draft,
                target,
                0,
                iteration_count=count,
                proposal_length=1,
                draft_select_token=Selector([1] * count),
                target_select_token=Selector([2] * count),
                draft_root_checkpoint=draft_root,
                target_root_checkpoint=target_root,
            )
            assert isinstance(result, MultiIterationSpeculativeHandoffResult)
            assert len(result.iterations) == count
            assert result.output_token_ids == (2,) * count
            assert result.uncached_next_token_id == 2
            assert draft.active_checkpoint_count == 1
            assert target.active_checkpoint_count == 1

        forbidden = (
            "onyx", "mlx", "torch", "transformers", "tokenizers",
            "huggingface_hub", "bitsandbytes", "accelerate", "onnxruntime",
            "psutil", "onyx_cuda._grammar_native",
        )
        assert not any(
            name == blocked or name.startswith(blocked + ".")
            for name in sys.modules
            for blocked in forbidden
        )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
