import gc
import inspect
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_speculative_iteration as iteration_module
from onyx_cuda import (
    BackendError,
    GrammarMaskedDraftProposalResult,
    GrammarMaskedPostAcceptanceContinuationResult,
    GrammarMaskedSelectionResult,
    GrammarMaskedTargetAcceptanceResult,
    GrammarMaskedSpeculativeIterationCleanupError,
    GrammarMaskedSpeculativeIterationError,
    GrammarMaskedSpeculativeIterationInvariantError,
    GrammarMaskedSpeculativeIterationResult,
    SpeculativeIterationError,
    coordinate_grammar_masked_speculative_iteration,
)
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    FakeGrammarConstraint,
    FakeGrammarProgram,
)


VOCAB_SIZE = 7
SCRIPT = tuple(
    tuple(float(row * VOCAB_SIZE + token) for token in range(VOCAB_SIZE))
    for row in range(48)
)
PROMPT = (6,)
CURRENT_TOKEN_ID = 0
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class RecordingBackend(FakeAutoregressiveBackend):
    def __init__(self, *, model_id):
        self.decode_calls = []
        self.verify_calls = []
        self.create_calls = []
        self.rollback_calls = []
        self.release_calls = []
        super().__init__(SCRIPT, model_id=model_id)

    def decode(self, token_id, /):
        self.decode_calls.append(token_id)
        return super().decode(token_id)

    def verify_proposal(self, current_token_id, proposal_token_ids, /):
        self.verify_calls.append((current_token_id, proposal_token_ids))
        return super().verify_proposal(current_token_id, proposal_token_ids)

    def create_cache_checkpoint(self):
        checkpoint = super().create_cache_checkpoint()
        self.create_calls.append(checkpoint)
        return checkpoint

    def rollback_cache(self, checkpoint, /):
        self.rollback_calls.append(checkpoint)
        return super().rollback_cache(checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        return super().release_cache_checkpoint(checkpoint)


class RecordingMask:
    def __init__(self):
        self.calls = []

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    def apply(self, logits, valid_token_ids, /):
        self.calls.append((logits, valid_token_ids))
        return logits


class RecordingSelector:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, row):
        self.calls.append(row)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class NoneStateConstraint:
    def __init__(self, starting_state, *, support, child_support=()):
        self.starting_state = starting_state
        self.support = tuple(support)
        self.child_support = tuple(child_support)
        self.live_start = True
        self.live_none = starting_state is None
        self.release_calls = []

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @property
    def grammar_type(self):
        return "regex"

    def init_state(self):
        return self.starting_state

    def advance_state(self, state, token_id, /):
        assert state is self.starting_state
        assert token_id in self.support
        self.live_none = True
        return None

    def get_valid_token_ids(self, state, /):
        self._require_live(state)
        return self.support if state is self.starting_state else self.child_support

    def is_match_state(self, state, /):
        self._require_live(state)
        return state is None

    def is_dead_state(self, state, /):
        self._require_live(state)
        return False

    def release_state(self, state, /):
        self._require_live(state)
        self.release_calls.append(state)
        if state is self.starting_state:
            self.live_start = False
        else:
            self.live_none = False

    def release_states(self, states, /):
        for state in states:
            self.release_state(state)

    def reset(self):
        self.live_start = False
        self.live_none = False

    def _require_live(self, state):
        if state is self.starting_state:
            if not self.live_start:
                raise RuntimeError("starting state is released")
            return
        if state is None and self.live_none:
            return
        raise RuntimeError("unknown state")


def _constraint(*, final_support=(5,), grammar_type="regex"):
    valid_token_ids = [
        ("s0", (1, 3)),
        ("s1", (2, 4)),
        ("x", ()),
        ("y", ()),
    ]
    transitions = [
        ("s0", 1, "s1"),
        ("s0", 3, "x"),
        ("s1", 2, "s2"),
        ("s1", 4, "y"),
    ]
    match_states = {"x", "y"}
    if final_support:
        valid_token_ids.extend((("s2", final_support), ("z", ())))
        transitions.append(("s2", final_support[0], "z"))
        match_states.add("z")
    else:
        valid_token_ids.append(("s2", ()))
        match_states.add("s2")
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=tuple(transitions),
        valid_token_ids=tuple(valid_token_ids),
        match_states=frozenset(match_states),
    )
    return FakeGrammarConstraint(
        tuple(bytes((token,)) for token in range(VOCAB_SIZE)),
        grammar_type=grammar_type,
        program=program,
    )


def _empty_constraint(*, grammar_type="regex"):
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(),
        valid_token_ids=(("s0", ()),),
        match_states=frozenset({"s0"}),
    )
    return FakeGrammarConstraint(
        tuple(bytes((token,)) for token in range(VOCAB_SIZE)),
        grammar_type=grammar_type,
        program=program,
    )


def _no_decision_constraint(position):
    if position == 0:
        return _empty_constraint()
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"),),
        valid_token_ids=(("s0", (1,)), ("s1", ())),
        match_states=frozenset({"s1"}),
    )
    return FakeGrammarConstraint(
        tuple(bytes((token,)) for token in range(VOCAB_SIZE)),
        grammar_type="regex",
        program=program,
    )


def _prefilled_pair():
    draft = RecordingBackend(model_id="draft")
    target = RecordingBackend(model_id="target")
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    return draft, target, draft.create_cache_checkpoint(), target.create_cache_checkpoint()


def _coordinate(
    constraint,
    *,
    draft_outcomes,
    target_outcomes,
    proposal_bound=2,
    draft=None,
    target=None,
    draft_root=None,
    target_root=None,
):
    if draft is None:
        draft, target, draft_root, target_root = _prefilled_pair()
    starting_state = constraint.init_state()
    draft_mask = RecordingMask()
    target_mask = RecordingMask()
    draft_selector = RecordingSelector(draft_outcomes)
    target_selector = RecordingSelector(target_outcomes)
    result = coordinate_grammar_masked_speculative_iteration(
        draft,
        target,
        CURRENT_TOKEN_ID,
        constraint,
        starting_state,
        draft_mask,
        target_mask,
        proposal_bound=proposal_bound,
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    return (
        result,
        starting_state,
        draft,
        target,
        draft_root,
        target_root,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
    )


def _install_mock_d44(monkeypatch, proposal_token_ids, *, shortening_selection=None):
    proposal = tuple(proposal_token_ids)

    def generate(
        backend,
        current_token_id,
        constraint,
        starting_state,
        logit_mask,
        *,
        proposal_bound,
        select_token,
    ):
        del constraint, starting_state, logit_mask, select_token
        initial_cache_length = backend.cache_length
        backend.decode(current_token_id)
        checkpoints = []
        for token_id in proposal:
            checkpoints.append(backend.create_cache_checkpoint())
            backend.decode(token_id)
        return GrammarMaskedDraftProposalResult(
            proposal_token_ids=proposal,
            rollback_checkpoints=tuple(checkpoints),
            initial_cache_length=initial_cache_length,
            final_cache_length=initial_cache_length + 1 + len(proposal),
            shortening_selection=shortening_selection,
        )

    monkeypatch.setattr(
        iteration_module,
        "generate_grammar_masked_draft_proposal",
        generate,
    )


def test_public_surface_signature_result_and_error_hierarchy():
    assert iteration_module.__all__ == [
        "GrammarMaskedSpeculativeIterationCleanupError",
        "GrammarMaskedSpeculativeIterationError",
        "GrammarMaskedSpeculativeIterationInvariantError",
        "GrammarMaskedSpeculativeIterationResult",
        "coordinate_grammar_masked_speculative_iteration",
    ]
    assert issubclass(GrammarMaskedSpeculativeIterationError, SpeculativeIterationError)
    assert issubclass(GrammarMaskedSpeculativeIterationError, BackendError)
    assert issubclass(
        GrammarMaskedSpeculativeIterationInvariantError,
        GrammarMaskedSpeculativeIterationError,
    )
    assert issubclass(
        GrammarMaskedSpeculativeIterationCleanupError,
        GrammarMaskedSpeculativeIterationError,
    )

    signature = inspect.signature(coordinate_grammar_masked_speculative_iteration)
    assert tuple(signature.parameters) == (
        "draft_backend",
        "target_backend",
        "current_token_id",
        "constraint",
        "starting_state",
        "draft_logit_mask",
        "target_logit_mask",
        "proposal_bound",
        "draft_select_token",
        "target_select_token",
        "draft_root_checkpoint",
        "target_root_checkpoint",
    )
    kinds = [parameter.kind for parameter in signature.parameters.values()]
    assert kinds == [
        *([inspect.Parameter.POSITIONAL_OR_KEYWORD] * 7),
        *([inspect.Parameter.KEYWORD_ONLY] * 5),
    ]
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in tuple(signature.parameters.values())[7:]
    )
    assert [field.name for field in fields(GrammarMaskedSpeculativeIterationResult)] == [
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "initial_cache_length",
        "final_cache_length",
        "uncached_next_token_id",
        "shortening_selection",
        "acceptance_no_decision_selection",
        "final_row_no_decision_selection",
        "committed_state",
        "committed_state_is_match",
    ]


def test_direct_results_cover_all_five_routes_and_derived_properties():
    shortening = GrammarMaskedSelectionResult((), True, None)
    acceptance_empty = GrammarMaskedSelectionResult((), False, None)
    final_empty = GrammarMaskedSelectionResult((), True, None)
    state = object()
    zero = GrammarMaskedSpeculativeIterationResult(
        (), 0, None, 2, 3, None, shortening, None, None, None, True
    )
    no_decision = GrammarMaskedSpeculativeIterationResult(
        (1, 2), 1, None, 2, 4, None, None, acceptance_empty, None, state, False
    )
    mismatch = GrammarMaskedSpeculativeIterationResult(
        (1, 2), 1, 3, 2, 4, 3, None, None, None, state, False
    )
    bonus = GrammarMaskedSpeculativeIterationResult(
        (1, 2), 2, None, 2, 5, 4, None, None, None, state, True
    )
    final_no_decision = GrammarMaskedSpeculativeIterationResult(
        (1, 2), 2, None, 2, 5, None, None, None, final_empty, state, True
    )

    assert zero.output_token_ids == ()
    assert zero.shortened is True
    assert zero.acceptance_decision_made is False
    assert zero.fully_accepted is False
    assert no_decision.output_token_ids == (1,)
    assert no_decision.rejected_proposal_token_id is None
    assert mismatch.accepted_token_ids == (1,)
    assert mismatch.rejected_proposal_token_id == 2
    assert mismatch.output_token_ids == (1, 3)
    assert bonus.fully_accepted is True
    assert bonus.output_token_ids == (1, 2, 4)
    assert final_no_decision.output_token_ids == (1, 2)
    assert final_no_decision.final_row_no_decision_selection is final_empty
    assert zero.committed_state is None
    assert not hasattr(zero, "__dict__")
    with pytest.raises(FrozenInstanceError):
        zero.final_cache_length = 4


@pytest.mark.parametrize(
    "args,error",
    [
        (([], 0, None, 1, 2, None, None, None, None, object(), False), TypeError),
        (
            ((), 0, None, 1, 2, None, None, None, None, object(), False),
            GrammarMaskedSpeculativeIterationInvariantError,
        ),
        (((True,), 1, None, 1, 3, 2, None, None, None, object(), False), TypeError),
        (((1,), True, None, 1, 3, 2, None, None, None, object(), False), TypeError),
        (
            ((1,), 0, 1, 1, 2, 1, None, None, None, object(), False),
            GrammarMaskedSpeculativeIterationInvariantError,
        ),
        (
            (
                (1,),
                0,
                None,
                1,
                2,
                None,
                None,
                GrammarMaskedSelectionResult((), False, None),
                GrammarMaskedSelectionResult((), False, None),
                object(),
                False,
            ),
            GrammarMaskedSpeculativeIterationInvariantError,
        ),
    ],
)
def test_direct_result_rejects_malformed_routes(args, error):
    with pytest.raises(error):
        GrammarMaskedSpeculativeIterationResult(*args)


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
def test_full_acceptance_selects_uncached_bonus_and_preserves_full_caches(grammar_type):
    constraint = _constraint(grammar_type=grammar_type)
    (
        result,
        starting_state,
        draft,
        target,
        draft_root,
        target_root,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
    ) = _coordinate(
        constraint,
        draft_outcomes=(1, 2),
        target_outcomes=(1, 2, 5),
    )

    assert result.output_token_ids == (1, 2, 5)
    assert result.uncached_next_token_id == 5
    assert result.final_cache_length == 4
    assert draft.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1, 2)
    assert target.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1, 2)
    assert draft_selector.calls == [call[0] for call in draft_mask.calls]
    assert target_selector.calls == [call[0] for call in target_mask.calls]
    assert len(target.verify_calls) == 1
    assert target.verify_calls[0][1] is result.proposal_token_ids
    assert constraint.active_state_count == 1
    assert result.committed_state is not starting_state
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls
    constraint.release_state(result.committed_state)


def test_mismatch_reconciles_both_caches_before_returning_replacement():
    constraint = _constraint()
    result, _, draft, target, *rest = _coordinate(
        constraint,
        draft_outcomes=(1, 2),
        target_outcomes=(1, 4),
    )
    del rest

    assert result.accepted_count == 1
    assert result.replacement_token_id == 4
    assert result.output_token_ids == (1, 4)
    assert result.uncached_next_token_id == 4
    assert result.final_cache_length == 3
    assert draft.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1)
    assert target.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1)
    assert target.decode_calls[-2:] == [CURRENT_TOKEN_ID, 1]
    assert len(target.verify_calls) == 1
    constraint.release_state(result.committed_state)


def test_full_acceptance_final_row_empty_returns_no_fabricated_token():
    constraint = _constraint(final_support=())
    result, _, draft, target, *rest = _coordinate(
        constraint,
        draft_outcomes=(1, 2),
        target_outcomes=(1, 2),
    )
    del rest

    assert result.fully_accepted is True
    assert result.output_token_ids == (1, 2)
    assert result.uncached_next_token_id is None
    assert result.final_row_no_decision_selection is not None
    assert result.final_row_no_decision_selection.valid_token_ids == ()
    assert draft.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1, 2)
    assert target.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1, 2)
    constraint.release_state(result.committed_state)


def test_zero_proposal_skips_target_batch_and_both_target_decisions():
    constraint = _empty_constraint()
    (
        result,
        starting_state,
        draft,
        target,
        _draft_root,
        _target_root,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
    ) = _coordinate(
        constraint,
        draft_outcomes=(),
        target_outcomes=(),
    )

    assert result.proposal_token_ids == ()
    assert result.committed_state is starting_state
    assert result.output_token_ids == ()
    assert result.shortening_selection is not None
    assert draft.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID,)
    assert target.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID,)
    assert target.verify_calls == []
    assert target.decode_calls == [CURRENT_TOKEN_ID]
    assert draft_mask.calls == []
    assert target_mask.calls == []
    assert draft_selector.calls == []
    assert target_selector.calls == []
    assert constraint.active_state_count == 1
    constraint.release_state(result.committed_state)


def test_zero_proposal_transfers_opaque_none_input_state():
    constraint = NoneStateConstraint(None, support=())
    result, starting_state, *_rest = _coordinate(
        constraint,
        draft_outcomes=(),
        target_outcomes=(),
    )

    assert starting_state is None
    assert result.committed_state is None
    assert constraint.release_calls == []
    constraint.release_state(result.committed_state)


def test_mismatch_transfers_opaque_none_downstream_state(monkeypatch):
    _install_mock_d44(monkeypatch, (1,))
    starting_state = object()
    constraint = NoneStateConstraint(starting_state, support=(1, 2))
    draft, target, draft_root, target_root = _prefilled_pair()

    result = coordinate_grammar_masked_speculative_iteration(
        draft,
        target,
        CURRENT_TOKEN_ID,
        constraint,
        starting_state,
        RecordingMask(),
        RecordingMask(),
        proposal_bound=1,
        draft_select_token=RecordingSelector(()),
        target_select_token=RecordingSelector((2,)),
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )

    assert result.replacement_token_id == 2
    assert result.committed_state is None
    assert constraint.release_calls == [starting_state]
    constraint.release_state(result.committed_state)


@pytest.mark.parametrize("position", [0, 1])
def test_d45_no_decision_skips_d46_and_aligns_to_accepted_prefix(
    monkeypatch,
    position,
):
    _install_mock_d44(monkeypatch, (1, 2))
    constraint = _no_decision_constraint(position)
    target_outcomes = () if position == 0 else (1,)
    result, starting_state, draft, target, *rest = _coordinate(
        constraint,
        draft_outcomes=(),
        target_outcomes=target_outcomes,
    )
    del rest

    assert result.acceptance_decision_made is False
    assert result.acceptance_no_decision_selection is not None
    assert result.final_row_no_decision_selection is None
    assert result.output_token_ids == (1,)[:position]
    assert result.uncached_next_token_id is None
    assert result.final_cache_length == len(PROMPT) + 1 + position
    assert draft.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1)[: position + 1]
    assert target.cached_token_ids == PROMPT + (CURRENT_TOKEN_ID, 1)[: position + 1]
    if position == 0:
        assert result.committed_state is starting_state
    else:
        assert result.committed_state is not starting_state
    constraint.release_state(result.committed_state)


def test_post_preflight_failure_restores_roots_and_releases_owned_input():
    class FailingVerificationBackend(RecordingBackend):
        def __init__(self, failure):
            self.failure = failure
            super().__init__(model_id="target")

        def verify_proposal(self, current_token_id, proposal_token_ids, /):
            self.verify_calls.append((current_token_id, proposal_token_ids))
            raise self.failure

    failure = RuntimeError("verification failed")
    draft = RecordingBackend(model_id="draft")
    target = FailingVerificationBackend(failure)
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    constraint = _constraint()
    starting_state = constraint.init_state()

    with pytest.raises(RuntimeError) as raised:
        coordinate_grammar_masked_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=2,
            draft_select_token=RecordingSelector((1, 2)),
            target_select_token=RecordingSelector(()),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value is failure
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_malformed_d45_result_releases_acquired_downstream_state(
    monkeypatch,
):
    constraint = _constraint()
    starting_state_holder = []

    def malformed_acceptance(
        proposal_token_ids,
        target_logit_rows,
        received_constraint,
        starting_state,
        logit_mask,
        *,
        vocab_size,
        select_token,
    ):
        del target_logit_rows, logit_mask, vocab_size, select_token
        starting_state_holder.append(starting_state)
        child = received_constraint.advance_state(starting_state, 1)
        distinct_equal_proposal = tuple(list(proposal_token_ids))
        return GrammarMaskedTargetAcceptanceResult(
            distinct_equal_proposal,
            0,
            3,
            None,
            child,
            False,
        )

    monkeypatch.setattr(
        iteration_module,
        "decide_grammar_masked_target_acceptance",
        malformed_acceptance,
    )
    draft, target, draft_root, target_root = _prefilled_pair()
    starting_state = constraint.init_state()

    with pytest.raises(
        GrammarMaskedSpeculativeIterationInvariantError,
        match="exact proposal-token tuple",
    ):
        coordinate_grammar_masked_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=2,
            draft_select_token=RecordingSelector((1, 2)),
            target_select_token=RecordingSelector(()),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert starting_state_holder == [starting_state]
    assert constraint.active_state_count == 0
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_malformed_d46_final_state_evidence_releases_acquired_child(
    monkeypatch,
):
    constraint = _constraint()

    def malformed_continuation(
        proposal_token_ids,
        target_logit_rows,
        acceptance_result,
        received_constraint,
        logit_mask,
        *,
        vocab_size,
        select_token,
    ):
        del target_logit_rows, logit_mask, vocab_size, select_token
        parent = acceptance_result.committed_state
        child = received_constraint.advance_state(parent, 5)
        received_constraint.release_state(parent)
        return GrammarMaskedPostAcceptanceContinuationResult(
            proposal_token_ids + (5,),
            5,
            None,
            child,
            False,
        )

    monkeypatch.setattr(
        iteration_module,
        "decide_grammar_masked_post_acceptance_continuation",
        malformed_continuation,
    )
    draft, target, draft_root, target_root = _prefilled_pair()
    starting_state = constraint.init_state()

    with pytest.raises(
        GrammarMaskedSpeculativeIterationInvariantError,
        match="match status changed",
    ):
        coordinate_grammar_masked_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=2,
            draft_select_token=RecordingSelector((1, 2)),
            target_select_token=RecordingSelector((1, 2)),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert constraint.active_state_count == 0
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1


def test_preflight_failure_leaves_starting_state_caller_owned():
    draft, target, draft_root, target_root = _prefilled_pair()
    constraint = _constraint()
    starting_state = constraint.init_state()

    with pytest.raises(ValueError, match="proposal_bound"):
        coordinate_grammar_masked_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=0,
            draft_select_token=RecordingSelector(()),
            target_select_token=RecordingSelector(()),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert constraint.active_state_count == 1
    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    constraint.release_state(starting_state)


def test_cleanup_error_preserves_original_cause_and_ordered_failures(monkeypatch):
    class CleanupFailingBackend(RecordingBackend):
        def rollback_cache(self, checkpoint, /):
            self.rollback_calls.append(checkpoint)
            if len(self.rollback_calls) > 1:
                raise RuntimeError(f"{self.model_id} rollback failed")
            return FakeAutoregressiveBackend.rollback_cache(self, checkpoint)

    class ReleaseFailingConstraint(FakeGrammarConstraint):
        def release_state(self, state, /):
            raise RuntimeError("state release failed")

    draft = CleanupFailingBackend(model_id="draft")
    target = CleanupFailingBackend(model_id="target")
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(),
        valid_token_ids=(("s0", ()),),
        match_states=frozenset({"s0"}),
    )
    constraint = ReleaseFailingConstraint(
        tuple(bytes((token,)) for token in range(VOCAB_SIZE)),
        grammar_type="regex",
        program=program,
    )
    starting_state = constraint.init_state()
    original = RuntimeError("D44 failed")

    def fail_d44(*args, **kwargs):
        del args, kwargs
        raise original

    monkeypatch.setattr(
        iteration_module,
        "generate_grammar_masked_draft_proposal",
        fail_d44,
    )
    with pytest.raises(GrammarMaskedSpeculativeIterationCleanupError) as raised:
        coordinate_grammar_masked_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=1,
            draft_select_token=RecordingSelector(()),
            target_select_token=RecordingSelector(()),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )

    assert raised.value.original_failure is original
    assert raised.value.__cause__ is original
    assert tuple(label for label, _ in raised.value.cleanup_failures) == (
        "draft root rollback",
        "target root rollback",
        "starting state release",
    )


def test_cleanup_error_constructor_requires_and_freezes_evidence():
    original = RuntimeError("original")
    cleanup = RuntimeError("cleanup")
    with pytest.raises(ValueError, match="cannot be empty"):
        GrammarMaskedSpeculativeIterationCleanupError(original, ())
    error = GrammarMaskedSpeculativeIterationCleanupError(
        original,
        [("operation", cleanup)],
    )
    assert error.original_failure is original
    assert error.cleanup_failures == (("operation", cleanup),)
    assert error.__cause__ is original


def test_one_thousand_alternating_transactions_reuse_roots_without_growth(
    monkeypatch,
):
    draft, target, draft_root, target_root = _prefilled_pair()
    zero_constraint = _empty_constraint()
    no_decision_constraint = _constraint()
    mismatch_constraint = _constraint()
    bonus_constraint = _constraint()
    final_empty_constraint = _constraint(final_support=())
    original_acceptance = iteration_module.decide_grammar_masked_target_acceptance
    no_decision_selection = GrammarMaskedSelectionResult((), False, None)

    def acceptance_dispatch(
        proposal_token_ids,
        target_logit_rows,
        constraint,
        starting_state,
        logit_mask,
        *,
        vocab_size,
        select_token,
    ):
        if constraint is no_decision_constraint:
            return GrammarMaskedTargetAcceptanceResult(
                proposal_token_ids,
                0,
                None,
                no_decision_selection,
                None,
                None,
            )
        return original_acceptance(
            proposal_token_ids,
            target_logit_rows,
            constraint,
            starting_state,
            logit_mask,
            vocab_size=vocab_size,
            select_token=select_token,
        )

    monkeypatch.setattr(
        iteration_module,
        "decide_grammar_masked_target_acceptance",
        acceptance_dispatch,
    )
    routes = (
        (zero_constraint, (), ()),
        (no_decision_constraint, (1, 2), ()),
        (mismatch_constraint, (1, 2), (1, 4)),
        (bonus_constraint, (1, 2), (1, 2, 5)),
        (final_empty_constraint, (1, 2), (1, 2)),
    )

    for iteration in range(1_000):
        constraint, draft_outcomes, target_outcomes = routes[iteration % len(routes)]
        starting_state = constraint.init_state()
        result = coordinate_grammar_masked_speculative_iteration(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=2,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
        constraint.release_state(result.committed_state)
        draft.rollback_cache(draft_root)
        target.rollback_cache(target_root)
        assert draft.active_checkpoint_count == 1
        assert target.active_checkpoint_count == 1

    assert draft.cached_token_ids == PROMPT
    assert target.cached_token_ids == PROMPT
    for constraint, *_ in routes:
        assert constraint.active_state_count == 0
    draft_allocation_ids = [checkpoint.allocation_id for checkpoint in draft.create_calls]
    assert draft_allocation_ids == sorted(draft_allocation_ids)
    assert len(draft_allocation_ids) == len(set(draft_allocation_ids))


def test_isolated_model_free_execution_loads_no_optional_runtime():
    code = (
        f"import sys\nsys.path.insert(0, {str(PACKAGE_ROOT / 'src')!r})\n"
        + textwrap.dedent(
        """
        from onyx_cuda import coordinate_grammar_masked_speculative_iteration
        from onyx_cuda.testing import (
            FakeAutoregressiveBackend,
            FakeGrammarConstraint,
            FakeGrammarProgram,
        )

        program = FakeGrammarProgram(
            initial_state="match",
            transitions=(),
            valid_token_ids=(("match", ()),),
            match_states=frozenset({"match"}),
        )
        constraint = FakeGrammarConstraint(
            (b"a", b"b"),
            grammar_type="regex",
            program=program,
        )
        state = constraint.init_state()
        rows = ((0.0, 1.0),) * 8
        draft = FakeAutoregressiveBackend(rows, model_id="draft")
        target = FakeAutoregressiveBackend(rows, model_id="target")
        draft.prefill((1,))
        target.prefill((1,))
        draft_root = draft.create_cache_checkpoint()
        target_root = target.create_cache_checkpoint()

        class UnusedMask:
            vocab_size = 2
            def apply(self, logits, valid_token_ids, /):
                raise AssertionError("zero route must not apply a mask")

        def unused_selector(_row):
            raise AssertionError("zero route must not select")

        result = coordinate_grammar_masked_speculative_iteration(
            draft,
            target,
            0,
            constraint,
            state,
            UnusedMask(),
            UnusedMask(),
            proposal_bound=2,
            draft_select_token=unused_selector,
            target_select_token=unused_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
        assert result.output_token_ids == ()
        assert draft.cached_token_ids == target.cached_token_ids == (1, 0)
        forbidden = (
            "onyx", "mlx", "torch", "transformers", "bitsandbytes", "accelerate"
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
        [sys.executable, "-I", "-c", code],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_successful_result_does_not_retain_transaction_components():
    constraint = _constraint()
    (
        result,
        _starting_state,
        draft,
        target,
        _draft_root,
        _target_root,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
    ) = _coordinate(
        constraint,
        draft_outcomes=(1, 2),
        target_outcomes=(1, 2, 5),
    )
    component_refs = tuple(
        weakref.ref(component)
        for component in (
            draft,
            target,
            constraint,
            draft_mask,
            target_mask,
            draft_selector,
            target_selector,
        )
    )
    constraint.release_state(result.committed_state)
    del (
        constraint,
        draft,
        target,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
    )
    gc.collect()

    assert all(reference() is None for reference in component_refs)
    assert result.output_token_ids == (1, 2, 5)
