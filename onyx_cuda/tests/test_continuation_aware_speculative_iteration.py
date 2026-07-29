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

import onyx_cuda.speculative_iteration as iteration_module
from onyx_cuda import (
    BackendError,
    BatchedTargetVerificationResult,
    CacheCheckpointStateError,
    ContinuationAwareSpeculativeIterationResult,
    ModelStep,
    PostIterationContinuationResult,
    SpeculativeIterationCleanupError,
    SpeculativeIterationError,
    SpeculativeIterationInvariantError,
    SpeculativeIterationResult,
    coordinate_continuation_aware_speculative_iteration,
    coordinate_speculative_iteration,
    select_highest_logit,
)
from onyx_cuda.testing import FakeAutoregressiveBackend


VOCAB_SIZE = 8
PROMPT = (6, 7)
CURRENT_TOKEN = 5
PROPOSAL = (1, 2, 3)


def _row(selected_token_id):
    return tuple(
        20.0 if token_id == selected_token_id else float(-token_id)
        for token_id in range(VOCAB_SIZE)
    )


def _draft_script(proposal=PROPOSAL):
    return (_row(0), *(_row(token_id) for token_id in proposal), _row(0), *(_row(0),) * 8)


def _target_script(mismatch_position, proposal=PROPOSAL, *, bonus=7):
    selected = list(proposal)
    if mismatch_position is not None:
        selected[mismatch_position] = (proposal[mismatch_position] + 4) % VOCAB_SIZE
    return (_row(0), *(_row(token_id) for token_id in selected), _row(bonus), *(_row(0),) * 8)


class TupleSubclass(tuple):
    pass


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    cache_length: int


class MinimumBackend:
    model_id = "minimum"
    vocab_size = VOCAB_SIZE
    cache_length = len(PROMPT)

    def prefill(self, prompt_token_ids, /):
        return ModelStep(logits=_row(0), cache_length=len(prompt_token_ids))

    def decode(self, token_id, /):
        return ModelStep(logits=_row(0), cache_length=self.cache_length + 1)

    def reset(self):
        return None


class CheckpointOnlyBackend(MinimumBackend):
    def create_cache_checkpoint(self):
        return CheckpointRecord(self.cache_length)

    def rollback_cache(self, checkpoint, /):
        return None

    def release_cache_checkpoint(self, checkpoint, /):
        return None


class VerificationOnlyBackend(MinimumBackend):
    def verify_proposal(self, current_token_id, proposal_token_ids, /):
        return BatchedTargetVerificationResult(
            logit_rows=tuple(_row(0) for _ in range(len(proposal_token_ids) + 1)),
            cache_length=self.cache_length + len(proposal_token_ids) + 1,
        )


class RecordingSelector:
    __slots__ = ("calls", "__weakref__")

    def __init__(self):
        self.calls = []

    def __call__(self, row):
        self.calls.append(row)
        return select_highest_logit(row)


class StatefulSelector:
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
    def __init__(self, scripted_logits):
        self.decode_token_ids = []
        self.verify_calls = []
        self.verification_results = []
        self.rollback_calls = []
        self.release_calls = []
        self.reset_calls = 0
        super().__init__(scripted_logits)

    def decode(self, token_id, /):
        self.decode_token_ids.append(token_id)
        return super().decode(token_id)

    def verify_proposal(self, current_token_id, proposal_token_ids, /):
        self.verify_calls.append((current_token_id, proposal_token_ids))
        result = super().verify_proposal(current_token_id, proposal_token_ids)
        self.verification_results.append(result)
        return result

    def rollback_cache(self, checkpoint, /):
        self.rollback_calls.append(checkpoint)
        return super().rollback_cache(checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        return super().release_cache_checkpoint(checkpoint)

    def reset(self):
        self.reset_calls += 1
        return super().reset()


class FaultBackend(RecordingBackend):
    def __init__(self, scripted_logits):
        self.rollback_failures = {}
        self.release_failures = {}
        super().__init__(scripted_logits)

    def rollback_cache(self, checkpoint, /):
        call_number = len(self.rollback_calls) + 1
        self.rollback_calls.append(checkpoint)
        failure = self.rollback_failures.get(call_number)
        if failure is not None:
            raise failure
        return FakeAutoregressiveBackend.rollback_cache(self, checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        call_number = len(self.release_calls) + 1
        self.release_calls.append(checkpoint)
        failure = self.release_failures.get(call_number)
        if failure is not None:
            raise failure
        return FakeAutoregressiveBackend.release_cache_checkpoint(self, checkpoint)


def _prepared_backends(
    mismatch_position,
    *,
    proposal=PROPOSAL,
    bonus=7,
    backend_type=RecordingBackend,
):
    draft = backend_type(_draft_script(proposal))
    target = backend_type(_target_script(mismatch_position, proposal, bonus=bonus))
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    return draft, target, draft_root, target_root


def _coordinate(
    draft,
    target,
    draft_root,
    target_root,
    *,
    proposal=PROPOSAL,
    draft_selector=None,
    target_selector=None,
):
    draft_selector = RecordingSelector() if draft_selector is None else draft_selector
    target_selector = RecordingSelector() if target_selector is None else target_selector
    result = coordinate_continuation_aware_speculative_iteration(
        draft,
        target,
        CURRENT_TOKEN,
        proposal_length=len(proposal),
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    return result, draft_selector, target_selector


def _unsafe_continuation(output_token_ids, uncached_next_token_id):
    result = object.__new__(PostIterationContinuationResult)
    object.__setattr__(result, "output_token_ids", output_token_ids)
    object.__setattr__(result, "uncached_next_token_id", uncached_next_token_id)
    return result


def test_public_surface_is_additive_and_preserves_d35_contract():
    import onyx_cuda

    current_module = importlib.import_module("onyx_cuda.speculative_iteration")
    for symbol_name in (
        "ContinuationAwareSpeculativeIterationResult",
        "coordinate_continuation_aware_speculative_iteration",
    ):
        symbol = getattr(current_module, symbol_name)
        assert getattr(onyx_cuda, symbol_name) is symbol
        assert symbol_name in current_module.__all__
        assert symbol_name in onyx_cuda.__all__

    assert issubclass(SpeculativeIterationError, BackendError)
    assert not hasattr(current_module, "ContinuationAwareSpeculativeIterationError")

    parameters = inspect.signature(
        coordinate_continuation_aware_speculative_iteration
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

    d35_parameters = inspect.signature(coordinate_speculative_iteration).parameters
    assert tuple(d35_parameters) == tuple(parameters)
    assert tuple(field.name for field in fields(SpeculativeIterationResult)) == (
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "initial_cache_length",
        "final_cache_length",
    )


def test_result_is_frozen_slotted_minimal_and_immutable():
    result = ContinuationAwareSpeculativeIterationResult(
        PROPOSAL,
        1,
        6,
        2,
        4,
        6,
    )

    assert tuple(field.name for field in fields(result)) == (
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "initial_cache_length",
        "final_cache_length",
        "uncached_next_token_id",
    )
    assert result == ContinuationAwareSpeculativeIterationResult(PROPOSAL, 1, 6, 2, 4, 6)
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "next_current_token_id")
    with pytest.raises(FrozenInstanceError):
        result.accepted_count = 2


@pytest.mark.parametrize("accepted_count", [0, 1, 2, 3])
def test_result_properties_cover_every_outcome(accepted_count):
    fully_accepted = accepted_count == len(PROPOSAL)
    replacement = None if fully_accepted else (PROPOSAL[accepted_count] + 4) % VOCAB_SIZE
    uncached = 3 if fully_accepted else replacement
    final_length = 2 + (len(PROPOSAL) + 1 if fully_accepted else accepted_count + 1)
    result = ContinuationAwareSpeculativeIterationResult(
        PROPOSAL,
        accepted_count,
        replacement,
        2,
        final_length,
        uncached,
    )

    assert result.fully_accepted is fully_accepted
    assert result.accepted_token_ids == PROPOSAL[:accepted_count]
    assert result.rejected_proposal_token_id == (
        None if fully_accepted else PROPOSAL[accepted_count]
    )
    assert result.output_token_ids == PROPOSAL[:accepted_count] + (uncached,)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"proposal_token_ids": TupleSubclass(PROPOSAL)}, TypeError, "must be a tuple"),
        ({"proposal_token_ids": []}, TypeError, "must be a tuple"),
        ({"proposal_token_ids": ()}, ValueError, "cannot be empty"),
        ({"proposal_token_ids": (1, True)}, TypeError, "must be an integer"),
        ({"proposal_token_ids": (1, -1)}, ValueError, "cannot be negative"),
        ({"accepted_count": True}, TypeError, "must be an integer"),
        ({"accepted_count": -1}, SpeculativeIterationInvariantError, "within"),
        ({"accepted_count": 4}, SpeculativeIterationInvariantError, "within"),
        ({"replacement_token_id": None}, SpeculativeIterationInvariantError, "must contain"),
        ({"replacement_token_id": 2}, SpeculativeIterationInvariantError, "must differ"),
        ({"uncached_next_token_id": True}, TypeError, "must be an integer"),
        ({"uncached_next_token_id": -1}, ValueError, "cannot be negative"),
        ({"uncached_next_token_id": 7}, SpeculativeIterationInvariantError, "must equal"),
        (
            {"accepted_count": 3, "replacement_token_id": 6, "final_cache_length": 6},
            SpeculativeIterationInvariantError,
            "cannot contain",
        ),
        ({"initial_cache_length": True}, SpeculativeIterationInvariantError, "integer"),
        (
            {"initial_cache_length": 0, "final_cache_length": 2},
            SpeculativeIterationInvariantError,
            "greater",
        ),
        ({"final_cache_length": -1}, SpeculativeIterationInvariantError, "negative"),
        ({"final_cache_length": 5}, SpeculativeIterationInvariantError, "expected"),
    ],
)
def test_result_rejects_malformed_or_impossible_metadata(changes, error, message):
    values = {
        "proposal_token_ids": PROPOSAL,
        "accepted_count": 1,
        "replacement_token_id": 6,
        "initial_cache_length": 2,
        "final_cache_length": 4,
        "uncached_next_token_id": 6,
    }
    values.update(changes)
    with pytest.raises(error, match=message):
        ContinuationAwareSpeculativeIterationResult(**values)


def test_direct_result_construction_has_no_vocabulary_upper_bound():
    result = ContinuationAwareSpeculativeIterationResult((1000,), 1, None, 1, 3, 1000)
    assert result.output_token_ids == (1000, 1000)


@pytest.mark.parametrize("mismatch_position", [0, 1, 2, None])
def test_exact_transaction_output_selector_order_and_cache_outcome(
    mismatch_position,
    monkeypatch,
):
    draft, target, draft_root, target_root = _prepared_backends(mismatch_position)
    continuation_calls = []
    original_continuation = iteration_module.decide_post_iteration_continuation

    def recording_continuation(*args, **kwargs):
        continuation_calls.append((args, kwargs))
        return original_continuation(*args, **kwargs)

    monkeypatch.setattr(
        iteration_module,
        "decide_post_iteration_continuation",
        recording_continuation,
    )
    result, draft_selector, target_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
    )

    accepted_count = len(PROPOSAL) if mismatch_position is None else mismatch_position
    uncached = 7 if mismatch_position is None else (PROPOSAL[mismatch_position] + 4) % VOCAB_SIZE
    expected_prefix = PROMPT + (CURRENT_TOKEN, *PROPOSAL[:accepted_count])
    expected_output = PROPOSAL[:accepted_count] + (uncached,)
    target_rows = _target_script(mismatch_position)[1 : len(PROPOSAL) + 2]
    expected_selector_rows = target_rows[:accepted_count]
    if mismatch_position is not None:
        expected_selector_rows += (target_rows[mismatch_position],)
    else:
        expected_selector_rows += (target_rows[-1],)

    assert result == ContinuationAwareSpeculativeIterationResult(
        PROPOSAL,
        accepted_count,
        None if mismatch_position is None else uncached,
        len(PROMPT),
        len(expected_prefix),
        uncached,
    )
    assert result.output_token_ids == expected_output
    assert draft.cached_token_ids == expected_prefix
    assert target.cached_token_ids == expected_prefix
    assert draft.decode_token_ids == [CURRENT_TOKEN, *PROPOSAL]
    assert target.decode_token_ids == (
        [] if mismatch_position is None else [CURRENT_TOKEN, *PROPOSAL[:accepted_count]]
    )
    assert len(target.verify_calls) == 1
    assert target.verify_calls[0][0] == CURRENT_TOKEN
    assert target.verify_calls[0][1] is result.proposal_token_ids
    assert len(draft_selector.calls) == len(PROPOSAL)
    assert target_selector.calls == list(expected_selector_rows)
    assert len(continuation_calls) == 1
    assert continuation_calls[0][0][0] is result.proposal_token_ids
    assert continuation_calls[0][0][1] is target.verification_results[0].logit_rows
    assert continuation_calls[0][1]["select_token"] is target_selector
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert [checkpoint.cache_length for checkpoint in draft.release_calls] == [2, 3, 4, 5]
    assert all(checkpoint is not draft_root for checkpoint in draft.release_calls)
    assert target.release_calls == []

    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT


@pytest.mark.parametrize(("mismatch_position", "expected_output"), [(0, (5,)), (None, (1, 7))])
def test_one_token_transactions_do_not_depend_on_multi_token_indexing(
    mismatch_position,
    expected_output,
):
    proposal = (1,)
    draft, target, draft_root, target_root = _prepared_backends(
        mismatch_position,
        proposal=proposal,
    )
    result, _, _ = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        proposal=proposal,
    )
    assert result.output_token_ids == expected_output


def test_bonus_may_equal_the_final_proposal_token_without_entering_either_cache_twice():
    draft, target, draft_root, target_root = _prepared_backends(None, bonus=PROPOSAL[-1])
    result, _, _ = _coordinate(draft, target, draft_root, target_root)
    expected_cache = PROMPT + (CURRENT_TOKEN, *PROPOSAL)

    assert result.output_token_ids == (*PROPOSAL, PROPOSAL[-1])
    assert result.uncached_next_token_id == PROPOSAL[-1]
    assert draft.cached_token_ids == expected_cache
    assert target.cached_token_ids == expected_cache


def test_same_stateful_selector_session_spans_d33_then_d37_without_retry():
    draft, target, draft_root, target_root = _prepared_backends(None)
    target_rows = _target_script(None)[1 : len(PROPOSAL) + 2]
    selector = StatefulSelector((*PROPOSAL, 7))

    result, _, returned_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        target_selector=selector,
    )

    assert returned_selector is selector
    assert selector.calls == list(target_rows)
    assert result.output_token_ids == (*PROPOSAL, 7)


def test_mismatch_d37_consumes_no_additional_selector_state():
    draft, target, draft_root, target_root = _prepared_backends(1)
    selector = StatefulSelector((1, 6, RuntimeError("must remain unconsumed")))

    result, _, _ = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        target_selector=selector,
    )
    assert result.output_token_ids == (1, 6)
    assert len(selector.calls) == 2
    with pytest.raises(RuntimeError, match="unconsumed"):
        selector(_row(0))


@pytest.mark.parametrize("bonus", [True, -1, VOCAB_SIZE])
def test_invalid_bonus_consumes_one_draw_then_restores_both_roots(bonus):
    draft, target, draft_root, target_root = _prepared_backends(None)
    selector = StatefulSelector((*PROPOSAL, bonus, 0))

    with pytest.raises((TypeError, ValueError)):
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            target_selector=selector,
        )

    assert len(selector.calls) == len(PROPOSAL) + 1
    assert selector(_row(0)) == 0
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_final_row_selector_exception_propagates_unchanged_and_is_not_retried():
    draft, target, draft_root, target_root = _prepared_backends(None)
    failure = RuntimeError("bonus draw failed")
    selector = StatefulSelector((*PROPOSAL, failure, 0))

    with pytest.raises(RuntimeError) as captured:
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            target_selector=selector,
        )

    assert captured.value is failure
    assert len(selector.calls) == len(PROPOSAL) + 1
    assert selector(_row(0)) == 0
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT


@pytest.mark.parametrize(
    ("continuation", "message"),
    [
        (object(), "PostIterationContinuationResult"),
        (_unsafe_continuation([1, 2, 3, 7], 7), "exact tuple"),
        (_unsafe_continuation((1, 2, 7), 7), "expected"),
        (_unsafe_continuation((1, 2, 4, 7), 7), "exact proposal"),
        (_unsafe_continuation((1, 2, 3, 7), 6), "final output"),
        (_unsafe_continuation((1, 2, 3, VOCAB_SIZE), VOCAB_SIZE), "outside vocabulary"),
    ],
)
def test_malformed_composed_continuation_is_rejected_and_restored(
    monkeypatch,
    continuation,
    message,
):
    draft, target, draft_root, target_root = _prepared_backends(None)
    monkeypatch.setattr(
        iteration_module,
        "decide_post_iteration_continuation",
        lambda *args, **kwargs: continuation,
    )

    with pytest.raises(SpeculativeIterationInvariantError, match=message):
        _coordinate(draft, target, draft_root, target_root)

    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_mismatch_continuation_must_agree_with_d33_evidence(monkeypatch):
    draft, target, draft_root, target_root = _prepared_backends(1)
    malformed = _unsafe_continuation((1, 7), 7)
    monkeypatch.setattr(
        iteration_module,
        "decide_post_iteration_continuation",
        lambda *args, **kwargs: malformed,
    )

    with pytest.raises(SpeculativeIterationInvariantError, match="disagrees"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT


def test_result_construction_failure_restores_roots_and_settles_handles(monkeypatch):
    draft, target, draft_root, target_root = _prepared_backends(None)
    failure = RuntimeError("result construction failed")

    def fail_result(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        iteration_module,
        "ContinuationAwareSpeculativeIterationResult",
        fail_result,
    )
    with pytest.raises(RuntimeError) as captured:
        _coordinate(draft, target, draft_root, target_root)

    assert captured.value is failure
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_success_path_release_failure_is_transactional_after_continuation_selection():
    draft, target, draft_root, target_root = _prepared_backends(
        None,
        backend_type=FaultBackend,
    )
    failure = RuntimeError("first result checkpoint release failed")
    draft.release_failures[2] = failure
    selector = RecordingSelector()

    with pytest.raises(RuntimeError) as captured:
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            target_selector=selector,
        )

    assert captured.value is failure
    assert len(selector.calls) == len(PROPOSAL) + 1
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert len(draft.release_calls) == 5


def test_cleanup_failures_are_aggregated_after_a_d37_failure_in_fixed_order():
    draft, target, draft_root, target_root = _prepared_backends(
        None,
        backend_type=FaultBackend,
    )
    original = RuntimeError("bonus failed")
    draft.rollback_failures[2] = RuntimeError("draft root failed")
    target.rollback_failures[2] = RuntimeError("target root failed")
    draft.release_failures[2] = RuntimeError("checkpoint zero failed")
    draft.release_failures[4] = RuntimeError("checkpoint two failed")
    selector = StatefulSelector((*PROPOSAL, original))

    with pytest.raises(SpeculativeIterationCleanupError) as captured:
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            target_selector=selector,
        )

    error = captured.value
    assert error.original_failure is original
    assert error.__cause__ is original
    assert tuple(operation for operation, _ in error.cleanup_failures) == (
        "draft root rollback",
        "target root rollback",
        "draft proposal checkpoint 0 release",
        "draft proposal checkpoint 2 release",
    )


@pytest.mark.parametrize(
    ("draft_backend", "target_backend", "error", "message"),
    [
        (CheckpointOnlyBackend(), CheckpointOnlyBackend(), TypeError, "BatchedTargetVerification"),
        (MinimumBackend(), VerificationOnlyBackend(), TypeError, "Checkpointable"),
    ],
)
def test_missing_backend_capabilities_fail_before_proposal(
    draft_backend,
    target_backend,
    error,
    message,
):
    with pytest.raises(error, match=message):
        coordinate_continuation_aware_speculative_iteration(
            draft_backend,
            target_backend,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=CheckpointRecord(len(PROMPT)),
            target_root_checkpoint=CheckpointRecord(len(PROMPT)),
        )


def test_same_backend_and_invalid_selector_fail_before_mutation():
    draft, target, draft_root, target_root = _prepared_backends(0)
    before = (
        draft.cached_token_ids,
        target.cached_token_ids,
        draft.active_checkpoint_count,
        target.active_checkpoint_count,
    )
    with pytest.raises(ValueError, match="distinct"):
        coordinate_continuation_aware_speculative_iteration(
            draft,
            draft,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=draft_root,
        )
    with pytest.raises(TypeError, match="target_select_token"):
        coordinate_continuation_aware_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=None,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert before == (
        draft.cached_token_ids,
        target.cached_token_ids,
        draft.active_checkpoint_count,
        target.active_checkpoint_count,
    )


def test_foreign_and_stale_roots_are_rejected_before_proposal():
    draft, target, draft_root, target_root = _prepared_backends(0)
    other = RecordingBackend(_draft_script())
    other.prefill(PROMPT)
    foreign_root = other.create_cache_checkpoint()

    with pytest.raises(CacheCheckpointStateError, match="another backend"):
        coordinate_continuation_aware_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=foreign_root,
            target_root_checkpoint=target_root,
        )

    draft.prefill(PROMPT)
    with pytest.raises(CacheCheckpointStateError, match="stale"):
        coordinate_continuation_aware_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )


def test_retained_result_does_not_retain_backends_selectors_or_transaction_evidence(
    monkeypatch,
):
    draft, target, draft_root, target_root = _prepared_backends(None)
    draft_selector = RecordingSelector()
    target_selector = RecordingSelector()
    draft_reference = weakref.ref(draft)
    target_reference = weakref.ref(target)
    draft_selector_reference = weakref.ref(draft_selector)
    target_selector_reference = weakref.ref(target_selector)
    evidence_references = []
    original_continuation = iteration_module.decide_post_iteration_continuation

    class WeakContinuation(PostIterationContinuationResult):
        pass

    def recording_continuation(*args, **kwargs):
        result = original_continuation(*args, **kwargs)
        weak_result = WeakContinuation(
            result.output_token_ids,
            result.uncached_next_token_id,
        )
        evidence_references.append(weakref.ref(weak_result))
        return weak_result

    monkeypatch.setattr(
        iteration_module,
        "decide_post_iteration_continuation",
        recording_continuation,
    )
    result, returned_draft_selector, returned_target_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
        draft_selector=draft_selector,
        target_selector=target_selector,
    )
    del draft_selector
    del target_selector
    del returned_draft_selector
    del returned_target_selector
    del draft
    del target
    del draft_root
    del target_root
    gc.collect()

    assert draft_selector_reference() is None
    assert target_selector_reference() is None
    assert evidence_references[0]() is None
    assert draft_reference() is None
    assert target_reference() is None
    assert result.output_token_ids == (*PROPOSAL, 7)


def test_one_thousand_root_restored_transactions_have_bounded_state():
    draft, target, draft_root, target_root = _prepared_backends(None)
    initial_draft_epoch = draft._epoch
    initial_target_epoch = target._epoch

    for iteration in range(1000):
        mismatch_position = (0, 1, 2, None)[iteration % 4]
        target._scripted_logits = _target_script(mismatch_position)
        result, _, _ = _coordinate(draft, target, draft_root, target_root)
        expected_accepted = 3 if mismatch_position is None else mismatch_position
        assert result.accepted_count == expected_accepted
        assert draft.active_checkpoint_count == 1
        assert target.active_checkpoint_count == 1
        draft.rollback_cache(draft_root)
        target.rollback_cache(target_root)
        assert draft.cached_token_ids == PROMPT
        assert target.cached_token_ids == PROMPT

    assert draft._epoch == initial_draft_epoch
    assert target._epoch == initial_target_epoch
    assert draft._next_checkpoint_id == 4002
    assert target._next_checkpoint_id == 2


def test_replacement_prefill_epochs_cannot_reuse_old_roots():
    draft = RecordingBackend(_draft_script())
    target = RecordingBackend(_target_script(0))
    old_roots = []

    for _ in range(10):
        draft.prefill(PROMPT)
        target.prefill(PROMPT)
        draft_root = draft.create_cache_checkpoint()
        target_root = target.create_cache_checkpoint()
        result, _, _ = _coordinate(draft, target, draft_root, target_root)
        assert result.accepted_count == 0
        old_roots.append((draft_root, target_root))

    for draft_root, target_root in old_roots[:-1]:
        with pytest.raises(CacheCheckpointStateError, match="stale"):
            draft.rollback_cache(draft_root)
        with pytest.raises(CacheCheckpointStateError, match="stale"):
            target.rollback_cache(target_root)


def test_isolated_source_import_runs_both_outcomes_without_optional_runtimes():
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(source_root)!r})

        from onyx_cuda import (
            coordinate_continuation_aware_speculative_iteration,
            select_highest_logit,
        )
        from onyx_cuda.testing import FakeAutoregressiveBackend

        def row(selected):
            return tuple(20.0 if token == selected else float(-token) for token in range(8))

        prompt = (6, 7)
        proposal = (1, 2, 3)
        draft_script = (row(0), row(1), row(2), row(3), row(0), *(row(0),) * 8)
        for selected, expected in (((5, 2, 3), (5,)), (proposal, (1, 2, 3, 7))):
            target_script = (row(0), *(row(token) for token in selected), row(7), *(row(0),) * 8)
            draft = FakeAutoregressiveBackend(draft_script)
            target = FakeAutoregressiveBackend(target_script)
            draft.prefill(prompt)
            target.prefill(prompt)
            draft_root = draft.create_cache_checkpoint()
            target_root = target.create_cache_checkpoint()
            result = coordinate_continuation_aware_speculative_iteration(
                draft,
                target,
                5,
                proposal_length=3,
                draft_select_token=select_highest_logit,
                target_select_token=select_highest_logit,
                draft_root_checkpoint=draft_root,
                target_root_checkpoint=target_root,
            )
            assert result.output_token_ids == expected
            assert result.uncached_next_token_id == expected[-1]

        forbidden = (
            "onyx", "mlx", "torch", "transformers", "tokenizers", "huggingface_hub",
            "bitsandbytes", "accelerate", "onnxruntime", "psutil",
        )
        loaded = tuple(sys.modules)
        assert "onyx_cuda._grammar_native" not in loaded
        assert not any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for module_name in loaded
            for prefix in forbidden
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

    assert completed.returncode == 0, completed.stdout + completed.stderr
