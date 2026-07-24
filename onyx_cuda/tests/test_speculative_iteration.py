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
    DraftProposalCleanupError,
    ModelStep,
    SpeculativeIterationCleanupError,
    SpeculativeIterationError,
    SpeculativeIterationInvariantError,
    SpeculativeIterationResult,
    coordinate_speculative_iteration,
    select_highest_logit,
)
from onyx_cuda.testing import FakeAutoregressiveBackend, ScriptExhaustedError


VOCAB_SIZE = 8
PROMPT = (6, 7)
CURRENT_TOKEN = 5
PROPOSAL = (1, 2, 3)
_UNSET = object()


def _row(selected_token_id):
    return tuple(
        20.0 if token_id == selected_token_id else float(-token_id)
        for token_id in range(VOCAB_SIZE)
    )


def _draft_script():
    return (_row(0), *(_row(token_id) for token_id in PROPOSAL), _row(0), *(_row(0),) * 8)


def _target_script(mismatch_position):
    selected = list(PROPOSAL)
    if mismatch_position is not None:
        selected[mismatch_position] = (PROPOSAL[mismatch_position] + 4) % VOCAB_SIZE
    return (_row(0), *(_row(token_id) for token_id in selected), _row(7), *(_row(0),) * 8)


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


class RecordingBackend(FakeAutoregressiveBackend):
    def __init__(self, scripted_logits):
        self.decode_token_ids = []
        self.verify_calls = []
        self.rollback_calls = []
        self.release_calls = []
        self.reset_calls = 0
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

    def decode(self, token_id, /):
        self.decode_token_ids.append(token_id)
        return super().decode(token_id)

    def verify_proposal(self, current_token_id, proposal_token_ids, /):
        self.verify_calls.append((current_token_id, proposal_token_ids))
        return super().verify_proposal(current_token_id, proposal_token_ids)

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
        self.decode_failures = {}
        self.non_step_decode_calls = set()
        self.wrong_step_length_calls = set()
        self.terminal_verify_failure = None
        super().__init__(scripted_logits)

    def decode(self, token_id, /):
        call_number = len(self.decode_token_ids) + 1
        failure = self.decode_failures.get(call_number)
        if failure is not None:
            self.decode_token_ids.append(token_id)
            raise failure
        step = super().decode(token_id)
        if call_number in self.non_step_decode_calls:
            return object()
        if call_number in self.wrong_step_length_calls:
            return ModelStep(logits=step.logits, cache_length=step.cache_length + 1)
        return step

    def verify_proposal(self, current_token_id, proposal_token_ids, /):
        if self.terminal_verify_failure is not None:
            self.verify_calls.append((current_token_id, proposal_token_ids))
            failure = self.terminal_verify_failure
            self.reset()
            raise failure
        return super().verify_proposal(current_token_id, proposal_token_ids)

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


class TerminalDraftBackend(FaultBackend):
    def __init__(self, scripted_logits):
        self.terminal_decode_failure = None
        super().__init__(scripted_logits)

    def decode(self, token_id, /):
        if self.terminal_decode_failure is not None:
            self.decode_token_ids.append(token_id)
            failure = self.terminal_decode_failure
            self.terminal_decode_failure = None
            self.reset()
            raise failure
        return super().decode(token_id)


def _complete_state(backend):
    return (
        backend.cache_length,
        backend.cached_token_ids,
        backend._next_row,
        backend._epoch,
        backend._next_checkpoint_id,
        tuple(backend._cache_checkpoints.items()),
    )


def _sequence_state(backend):
    return (
        backend.cache_length,
        backend.cached_token_ids,
        backend._next_row,
        backend._epoch,
        tuple(backend._cache_checkpoints.items()),
    )


def _prepared_backends(mismatch_position, backend_type=RecordingBackend):
    draft = backend_type(_draft_script())
    target = backend_type(_target_script(mismatch_position))
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
    draft_selector=None,
    target_selector=None,
):
    draft_selector = RecordingSelector() if draft_selector is None else draft_selector
    target_selector = RecordingSelector() if target_selector is None else target_selector
    result = coordinate_speculative_iteration(
        draft,
        target,
        CURRENT_TOKEN,
        proposal_length=len(PROPOSAL),
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    return result, draft_selector, target_selector


def test_public_surface_signature_and_error_hierarchy():
    import onyx_cuda

    current_module = importlib.import_module("onyx_cuda.speculative_iteration")
    symbol_names = (
        "SpeculativeIterationCleanupError",
        "SpeculativeIterationError",
        "SpeculativeIterationInvariantError",
        "SpeculativeIterationResult",
        "coordinate_speculative_iteration",
    )
    for symbol_name in symbol_names:
        symbol = getattr(current_module, symbol_name)
        assert symbol.__module__ == "onyx_cuda.speculative_iteration"
        assert getattr(onyx_cuda, symbol_name) is symbol
        assert symbol_name in onyx_cuda.__all__

    assert issubclass(SpeculativeIterationError, BackendError)
    assert issubclass(SpeculativeIterationInvariantError, SpeculativeIterationError)
    assert issubclass(SpeculativeIterationCleanupError, SpeculativeIterationError)

    parameters = inspect.signature(coordinate_speculative_iteration).parameters
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


def test_result_is_frozen_slotted_minimal_and_has_no_iterative_handoff_alias():
    result = SpeculativeIterationResult(
        proposal_token_ids=PROPOSAL,
        accepted_count=1,
        replacement_token_id=6,
        initial_cache_length=2,
        final_cache_length=4,
    )

    assert tuple(field.name for field in fields(result)) == (
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "initial_cache_length",
        "final_cache_length",
    )
    assert result == SpeculativeIterationResult(PROPOSAL, 1, 6, 2, 4)
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "next_current_token_id")
    with pytest.raises(FrozenInstanceError):
        result.accepted_count = 2


@pytest.mark.parametrize("accepted_count", [0, 1, 2, 3])
def test_result_properties_and_cache_relationships(accepted_count):
    fully_accepted = accepted_count == len(PROPOSAL)
    replacement = None if fully_accepted else (PROPOSAL[accepted_count] + 4) % VOCAB_SIZE
    final_length = 2 + (len(PROPOSAL) + 1 if fully_accepted else accepted_count + 1)
    result = SpeculativeIterationResult(
        PROPOSAL,
        accepted_count,
        replacement,
        2,
        final_length,
    )

    assert result.fully_accepted is fully_accepted
    assert result.accepted_token_ids == PROPOSAL[:accepted_count]
    assert result.rejected_proposal_token_id == (
        None if fully_accepted else PROPOSAL[accepted_count]
    )
    assert result.output_token_ids == (
        PROPOSAL if fully_accepted else PROPOSAL[:accepted_count] + (replacement,)
    )
    assert result.uncached_next_token_id == replacement


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"proposal_token_ids": TupleSubclass(PROPOSAL)}, TypeError, "must be a tuple"),
        ({"proposal_token_ids": ()}, ValueError, "cannot be empty"),
        ({"proposal_token_ids": (1, True)}, TypeError, "must be an integer"),
        ({"proposal_token_ids": (1, -1)}, ValueError, "cannot be negative"),
        ({"accepted_count": True}, TypeError, "must be an integer"),
        ({"accepted_count": -1}, SpeculativeIterationInvariantError, "within"),
        ({"accepted_count": 4}, SpeculativeIterationInvariantError, "within"),
        ({"replacement_token_id": None}, SpeculativeIterationInvariantError, "must contain"),
        ({"replacement_token_id": 2}, SpeculativeIterationInvariantError, "must differ"),
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
    }
    values.update(changes)
    with pytest.raises(error, match=message):
        SpeculativeIterationResult(**values)


def test_cleanup_error_retains_original_and_ordered_tuple_payload():
    original = RuntimeError("primary")
    failures = [
        ("draft root rollback", RuntimeError("draft")),
        ("target root rollback", RuntimeError("target")),
    ]
    error = SpeculativeIterationCleanupError(original, failures)

    assert error.original_failure is original
    assert error.cleanup_failures == tuple(failures)
    assert "draft root rollback also failed" in str(error)
    failures.clear()
    assert len(error.cleanup_failures) == 2
    with pytest.raises(ValueError, match="cannot be empty"):
        SpeculativeIterationCleanupError(original, ())


@pytest.mark.parametrize("mismatch_position", [0, 1, 2, None])
def test_exact_one_iteration_composition_and_cache_outcome(mismatch_position):
    draft, target, draft_root, target_root = _prepared_backends(mismatch_position)
    result, draft_selector, target_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
    )
    accepted_count = len(PROPOSAL) if mismatch_position is None else mismatch_position
    expected_prefix = PROMPT + (CURRENT_TOKEN, *PROPOSAL[:accepted_count])

    assert result.proposal_token_ids == PROPOSAL
    assert result.accepted_count == accepted_count
    assert result.initial_cache_length == len(PROMPT)
    assert result.final_cache_length == len(expected_prefix)
    assert draft.cached_token_ids == expected_prefix
    assert target.cached_token_ids == expected_prefix
    assert draft.decode_token_ids == [CURRENT_TOKEN, *PROPOSAL]
    assert len(target.verify_calls) == 1
    assert target.verify_calls[0][0] == CURRENT_TOKEN
    assert target.verify_calls[0][1] is result.proposal_token_ids
    assert len(draft_selector.calls) == len(PROPOSAL)
    assert len(target_selector.calls) == (
        len(PROPOSAL) if mismatch_position is None else mismatch_position + 1
    )
    assert target_selector.calls == list(
        _target_script(mismatch_position)[1:][: len(target_selector.calls)]
    )
    assert target.decode_token_ids == (
        [] if mismatch_position is None else [CURRENT_TOKEN, *PROPOSAL[:mismatch_position]]
    )
    assert result.uncached_next_token_id == (
        None if mismatch_position is None else (PROPOSAL[mismatch_position] + 4) % VOCAB_SIZE
    )
    if mismatch_position is not None:
        assert result.uncached_next_token_id not in draft.cached_token_ids[len(expected_prefix) :]
        assert result.uncached_next_token_id not in target.cached_token_ids[len(expected_prefix) :]

    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT


def test_full_acceptance_performs_no_reconciliation_and_selects_no_final_row():
    draft, target, draft_root, target_root = _prepared_backends(None)
    result, _, target_selector = _coordinate(draft, target, draft_root, target_root)

    assert result.fully_accepted
    assert result.uncached_next_token_id is None
    assert target.decode_token_ids == []
    assert len(draft.rollback_calls) == 1
    assert len(target.rollback_calls) == 1
    assert _row(7) not in target_selector.calls


@pytest.mark.parametrize("mismatch_position", [0, 1, 2, None])
def test_success_releases_owned_handles_in_position_order_but_never_roots(
    mismatch_position,
):
    draft, target, draft_root, target_root = _prepared_backends(mismatch_position)
    result, _, _ = _coordinate(draft, target, draft_root, target_root)

    assert [checkpoint.cache_length for checkpoint in draft.release_calls] == [2, 3, 4, 5]
    assert all(checkpoint is not draft_root for checkpoint in draft.release_calls)
    assert target.release_calls == []
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert result.final_cache_length >= len(PROMPT) + 1


def test_fresh_selector_sessions_replay_after_explicit_caller_root_rollback():
    draft, target, draft_root, target_root = _prepared_backends(1)
    first, first_draft_selector, first_target_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
    )
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    second, second_draft_selector, second_target_selector = _coordinate(
        draft,
        target,
        draft_root,
        target_root,
    )

    assert second == first
    assert len(first_draft_selector.calls) == len(second_draft_selector.calls) == 3
    assert len(first_target_selector.calls) == len(second_target_selector.calls) == 2


def test_failed_stateful_selector_is_consumed_but_not_rewound():
    draft, target, draft_root, target_root = _prepared_backends(None)
    failure = RuntimeError("selector failed on its second call")

    class FailingStatefulSelector:
        def __init__(self):
            self.calls = []

        def __call__(self, row):
            self.calls.append(row)
            if len(self.calls) == 2:
                raise failure
            return select_highest_logit(row)

    selector = FailingStatefulSelector()
    with pytest.raises(RuntimeError) as captured:
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            target_selector=selector,
        )

    assert captured.value is failure
    assert len(selector.calls) == 2
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT


@pytest.mark.parametrize(
    ("proposal_length", "error"),
    [(True, TypeError), (1.0, TypeError), ("1", TypeError), (0, ValueError), (-1, ValueError)],
)
def test_invalid_proposal_length_fails_before_root_or_backend_work(proposal_length, error):
    draft, target, draft_root, target_root = _prepared_backends(0)
    before = (_complete_state(draft), _complete_state(target))
    with pytest.raises(error):
        coordinate_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=proposal_length,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert (_complete_state(draft), _complete_state(target)) == before
    assert draft.rollback_calls == []
    assert target.rollback_calls == []


@pytest.mark.parametrize("selector_name", ["draft", "target"])
def test_noncallable_selector_fails_before_root_validation(selector_name):
    draft, target, draft_root, target_root = _prepared_backends(0)
    selectors = {"draft": select_highest_logit, "target": select_highest_logit}
    selectors[selector_name] = None
    with pytest.raises(TypeError, match=f"{selector_name}_select_token"):
        coordinate_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=selectors["draft"],
            target_select_token=selectors["target"],
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.rollback_calls == []
    assert target.rollback_calls == []


def test_roles_must_be_distinct_and_satisfy_the_required_capabilities():
    draft, target, draft_root, target_root = _prepared_backends(0)
    with pytest.raises(ValueError, match="distinct"):
        coordinate_speculative_iteration(
            draft,
            draft,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=draft_root,
        )
    with pytest.raises(TypeError, match="draft_backend"):
        coordinate_speculative_iteration(
            MinimumBackend(),
            target,
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    with pytest.raises(TypeError, match="BatchedTargetVerificationBackend"):
        coordinate_speculative_iteration(
            draft,
            CheckpointOnlyBackend(),
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    with pytest.raises(TypeError, match="CheckpointableAutoregressiveBackend"):
        coordinate_speculative_iteration(
            draft,
            VerificationOnlyBackend(),
            CURRENT_TOKEN,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )


@pytest.mark.parametrize("role", ["draft", "target"])
@pytest.mark.parametrize("vocab_size", [True, 1.0, "8", 0, -1])
def test_invalid_vocabulary_metadata_fails_before_root_validation(role, vocab_size):
    draft, target, draft_root, target_root = _prepared_backends(0)
    if role == "draft":
        draft.vocab_override = vocab_size
    else:
        target.vocab_override = vocab_size
    with pytest.raises(SpeculativeIterationInvariantError, match="vocab_size"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.rollback_calls == []
    assert target.rollback_calls == []


def test_unequal_vocabularies_and_cache_lengths_fail_before_root_validation():
    draft, target, draft_root, target_root = _prepared_backends(0)
    target.vocab_override = VOCAB_SIZE + 1
    with pytest.raises(SpeculativeIterationInvariantError, match="vocabulary sizes differ"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.rollback_calls == []

    target.vocab_override = _UNSET
    target.cache_override = len(PROMPT) + 1
    with pytest.raises(SpeculativeIterationInvariantError, match="cache lengths differ"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.rollback_calls == []


@pytest.mark.parametrize(
    ("current_token", "error"),
    [(True, TypeError), (1.0, TypeError), ("1", TypeError), (-1, ValueError), (8, ValueError)],
)
def test_invalid_current_token_fails_before_root_validation(current_token, error):
    draft, target, draft_root, target_root = _prepared_backends(0)
    with pytest.raises(error):
        coordinate_speculative_iteration(
            draft,
            target,
            current_token,
            proposal_length=3,
            draft_select_token=select_highest_logit,
            target_select_token=select_highest_logit,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.rollback_calls == []
    assert target.rollback_calls == []


def test_root_metadata_and_actual_ownership_are_validated_before_proposal():
    draft, target, draft_root, target_root = _prepared_backends(0)
    wrong_length = CheckpointRecord(len(PROMPT) + 1)
    with pytest.raises(SpeculativeIterationInvariantError, match="reports cache length"):
        _coordinate(draft, target, wrong_length, target_root)
    assert draft.decode_token_ids == []

    foreign = FakeAutoregressiveBackend(_draft_script())
    foreign.prefill(PROMPT)
    foreign_root = foreign.create_cache_checkpoint()
    with pytest.raises(CacheCheckpointStateError, match="another backend"):
        _coordinate(draft, target, foreign_root, target_root)
    assert draft.decode_token_ids == []

    draft.release_cache_checkpoint(draft_root)
    with pytest.raises(CacheCheckpointStateError, match="unknown|released"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.decode_token_ids == []


@pytest.mark.parametrize("role", ["draft", "target"])
def test_non_checkpoint_roots_fail_before_any_root_operation(role):
    draft, target, draft_root, target_root = _prepared_backends(0)
    if role == "draft":
        draft_root = object()
    else:
        target_root = object()

    with pytest.raises(TypeError, match="CacheCheckpoint"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.rollback_calls == []
    assert target.rollback_calls == []
    assert draft.decode_token_ids == []
    assert target.verify_calls == []


def test_same_position_root_validation_preserves_peers_and_exact_state(monkeypatch):
    draft, target, draft_root, target_root = _prepared_backends(0)
    draft_peer = draft.create_cache_checkpoint()
    target_peer = target.create_cache_checkpoint()
    before_draft = _complete_state(draft)
    before_target = _complete_state(target)
    failure = RuntimeError("stop immediately after root validation")

    def fail_proposal(*args, **kwargs):
        raise failure

    monkeypatch.setattr(iteration_module, "generate_draft_proposal", fail_proposal)
    with pytest.raises(RuntimeError) as captured:
        _coordinate(draft, target, draft_root, target_root)
    assert captured.value is failure
    assert _complete_state(draft) == before_draft
    assert _complete_state(target) == before_target
    draft.rollback_cache(draft_peer)
    target.rollback_cache(target_peer)


@pytest.mark.parametrize("phase", ["draft-selector", "verification", "target-selector"])
def test_healthy_phase_failures_restore_both_roots_and_preserve_original_exception(phase):
    mismatch = 1
    draft, target, draft_root, target_root = _prepared_backends(
        mismatch,
        backend_type=FaultBackend,
    )
    failure = RuntimeError(f"injected {phase} failure")

    def fail_selector(row):
        raise failure

    draft_selector = fail_selector if phase == "draft-selector" else select_highest_logit
    target_selector = fail_selector if phase == "target-selector" else select_highest_logit
    if phase == "verification":
        target._scripted_logits = target._scripted_logits[:2]

    expected_error = RuntimeError if phase != "verification" else ScriptExhaustedError
    with pytest.raises(expected_error) as captured:
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            draft_selector=draft_selector,
            target_selector=target_selector,
        )
    if phase != "verification":
        assert captured.value is failure
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert draft.reset_calls == 2
    assert target.reset_calls == 2


@pytest.mark.parametrize("fault", ["exception", "non-step", "wrong-length"])
def test_target_replay_failures_restore_both_roots(fault):
    draft, target, draft_root, target_root = _prepared_backends(
        2,
        backend_type=FaultBackend,
    )
    if fault == "exception":
        target.decode_failures[2] = RuntimeError("replay failed")
        expected_error = RuntimeError
    elif fault == "non-step":
        target.non_step_decode_calls.add(2)
        expected_error = SpeculativeIterationInvariantError
    else:
        target.wrong_step_length_calls.add(2)
        expected_error = SpeculativeIterationInvariantError

    with pytest.raises(expected_error):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_out_of_range_d33_replacement_is_rejected_and_restored():
    draft, target, draft_root, target_root = _prepared_backends(0)

    with pytest.raises(SpeculativeIterationInvariantError, match="outside vocabulary"):
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            target_selector=lambda row: VOCAB_SIZE,
        )
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1


def test_success_path_release_failure_becomes_transaction_failure_and_retries_cleanup():
    draft, target, draft_root, target_root = _prepared_backends(
        None,
        backend_type=FaultBackend,
    )
    failure = RuntimeError("first result checkpoint release failed")
    draft.release_failures[2] = failure

    with pytest.raises(RuntimeError) as captured:
        _coordinate(draft, target, draft_root, target_root)
    assert captured.value is failure
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert len(draft.release_calls) == 5


def test_cleanup_failures_are_aggregated_in_deterministic_operation_order():
    draft, target, draft_root, target_root = _prepared_backends(
        0,
        backend_type=FaultBackend,
    )
    original = RuntimeError("selector failed")
    draft.rollback_failures[2] = RuntimeError("draft root failed")
    target.rollback_failures[2] = RuntimeError("target root failed")
    draft.release_failures[2] = RuntimeError("checkpoint zero failed")
    draft.release_failures[4] = RuntimeError("checkpoint two failed")

    with pytest.raises(SpeculativeIterationCleanupError) as captured:
        _coordinate(
            draft,
            target,
            draft_root,
            target_root,
            target_selector=lambda row: (_ for _ in ()).throw(original),
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
    assert len(draft.release_calls) == 4


def test_terminal_target_failure_restores_healthy_peer_and_reports_stale_target_root():
    draft, target, draft_root, target_root = _prepared_backends(
        0,
        backend_type=FaultBackend,
    )
    terminal = RuntimeError("terminal target failure")
    target.terminal_verify_failure = terminal

    with pytest.raises(SpeculativeIterationCleanupError) as captured:
        _coordinate(draft, target, draft_root, target_root)

    error = captured.value
    assert error.original_failure is terminal
    assert tuple(operation for operation, _ in error.cleanup_failures) == ("target root rollback",)
    assert draft.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.cache_length == 0
    assert target.cached_token_ids == ()


def test_nested_d32_terminal_cleanup_error_is_preserved_by_d35_cleanup():
    draft = TerminalDraftBackend(_draft_script())
    target = FaultBackend(_target_script(0))
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    terminal = RuntimeError("terminal draft failure")
    draft.terminal_decode_failure = terminal

    with pytest.raises(SpeculativeIterationCleanupError) as captured:
        _coordinate(draft, target, draft_root, target_root)

    outer = captured.value
    assert isinstance(outer.original_failure, DraftProposalCleanupError)
    assert outer.original_failure.original_failure is terminal
    assert tuple(operation for operation, _ in outer.cleanup_failures) == ("draft root rollback",)
    assert draft.cache_length == 0
    assert target.cached_token_ids == PROMPT


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("proposal_token_ids", TupleSubclass(PROPOSAL), "exact tuple"),
        ("rollback_checkpoints", None, "exact tuple"),
        ("initial_cache_length", 2.0, "must be an integer"),
        ("final_cache_length", 6.0, "must be an integer"),
    ],
)
def test_malformed_d32_results_release_acquired_handles_and_restore_roots(
    monkeypatch,
    field_name,
    replacement,
    message,
):
    draft, target, draft_root, target_root = _prepared_backends(0)
    original = iteration_module.generate_draft_proposal

    def malformed_proposal(*args, **kwargs):
        result = original(*args, **kwargs)
        value = list(result.rollback_checkpoints) if replacement is None else replacement
        object.__setattr__(result, field_name, value)
        return result

    monkeypatch.setattr(iteration_module, "generate_draft_proposal", malformed_proposal)
    with pytest.raises(SpeculativeIterationInvariantError, match=message):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_malformed_composed_results_are_rejected_and_restored(monkeypatch):
    draft, target, draft_root, target_root = _prepared_backends(0)
    original_verify = target.verify_proposal

    def malformed_verification(current_token_id, proposal_token_ids, /):
        result = original_verify(current_token_id, proposal_token_ids)
        malformed = object.__new__(BatchedTargetVerificationResult)
        object.__setattr__(malformed, "logit_rows", result.logit_rows[:-1])
        object.__setattr__(malformed, "cache_length", result.cache_length)
        return malformed

    target.verify_proposal = malformed_verification
    with pytest.raises(SpeculativeIterationInvariantError, match="rows"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1


def test_malformed_d30_cache_metadata_is_rejected_and_restored():
    draft, target, draft_root, target_root = _prepared_backends(0)
    original_verify = target.verify_proposal

    def malformed_verification(current_token_id, proposal_token_ids, /):
        result = original_verify(current_token_id, proposal_token_ids)
        object.__setattr__(result, "cache_length", float(result.cache_length))
        return result

    target.verify_proposal = malformed_verification
    with pytest.raises(SpeculativeIterationInvariantError, match="must be an integer"):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("proposal_token_ids", list(PROPOSAL), "exact tuple"),
        ("accepted_count", True, "must be an integer"),
        ("accepted_count", 4, "within"),
        ("replacement_token_id", 7, "fully accepted"),
    ],
)
def test_malformed_d33_results_are_rejected_and_restored(
    monkeypatch,
    field_name,
    replacement,
    message,
):
    draft, target, draft_root, target_root = _prepared_backends(None)
    original = iteration_module.decide_match_replace_acceptance

    def malformed_decision(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(result, field_name, replacement)
        return result

    monkeypatch.setattr(
        iteration_module,
        "decide_match_replace_acceptance",
        malformed_decision,
    )
    with pytest.raises(SpeculativeIterationInvariantError, match=message):
        _coordinate(draft, target, draft_root, target_root)
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1


def test_result_construction_failure_restores_roots_and_releases_handles(monkeypatch):
    draft, target, draft_root, target_root = _prepared_backends(1)
    failure = RuntimeError("result construction failed")

    def fail_result(*args, **kwargs):
        raise failure

    monkeypatch.setattr(iteration_module, "SpeculativeIterationResult", fail_result)
    with pytest.raises(RuntimeError) as captured:
        _coordinate(draft, target, draft_root, target_root)
    assert captured.value is failure
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_result_and_selectors_are_not_retained_after_success():
    draft, target, draft_root, target_root = _prepared_backends(None)
    draft_selector = RecordingSelector()
    target_selector = RecordingSelector()
    draft_reference = weakref.ref(draft)
    target_reference = weakref.ref(target)
    draft_selector_reference = weakref.ref(draft_selector)
    target_selector_reference = weakref.ref(target_selector)

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
    gc.collect()

    assert draft_selector_reference() is None
    assert target_selector_reference() is None
    assert draft_reference() is draft
    assert target_reference() is target
    assert tuple(field.name for field in fields(result)) == (
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "initial_cache_length",
        "final_cache_length",
    )


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
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_replacement_prefill_epochs_cannot_reuse_old_roots():
    draft = RecordingBackend(_draft_script())
    target = RecordingBackend(_target_script(0))
    old_roots = []

    for _ in range(25):
        draft.prefill(PROMPT)
        target.prefill(PROMPT)
        draft_root = draft.create_cache_checkpoint()
        target_root = target.create_cache_checkpoint()
        result, _, _ = _coordinate(draft, target, draft_root, target_root)
        assert result.accepted_count == 0
        assert draft.active_checkpoint_count == 1
        assert target.active_checkpoint_count == 1
        old_roots.append((draft_root, target_root))

    for draft_root, target_root in old_roots[:-1]:
        with pytest.raises(CacheCheckpointStateError, match="stale"):
            draft.rollback_cache(draft_root)
        with pytest.raises(CacheCheckpointStateError, match="stale"):
            target.rollback_cache(target_root)


def test_isolated_source_import_runs_mismatch_and_full_acceptance_without_optional_runtimes():
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(source_root)!r})

        from onyx_cuda import coordinate_speculative_iteration, select_highest_logit
        from onyx_cuda.testing import FakeAutoregressiveBackend

        def row(selected):
            return tuple(20.0 if token == selected else float(-token) for token in range(8))

        prompt = (6, 7)
        proposal = (1, 2, 3)
        draft_script = (row(0), row(1), row(2), row(3), row(0), *(row(0),) * 8)
        for selected in ((5, 2, 3), proposal):
            target_script = (row(0), *(row(token) for token in selected), row(7), *(row(0),) * 8)
            draft = FakeAutoregressiveBackend(draft_script)
            target = FakeAutoregressiveBackend(target_script)
            draft.prefill(prompt)
            target.prefill(prompt)
            draft_root = draft.create_cache_checkpoint()
            target_root = target.create_cache_checkpoint()
            result = coordinate_speculative_iteration(
                draft,
                target,
                5,
                proposal_length=3,
                draft_select_token=select_highest_logit,
                target_select_token=select_highest_logit,
                draft_root_checkpoint=draft_root,
                target_root_checkpoint=target_root,
            )
            assert result.proposal_token_ids == proposal

        forbidden = (
            "onyx", "mlx", "torch", "transformers", "tokenizers", "huggingface_hub",
            "bitsandbytes", "accelerate", "onnxruntime", "psutil",
            "onyx_cuda._grammar_native",
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in forbidden
        )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
