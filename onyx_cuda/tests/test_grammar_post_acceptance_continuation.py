import gc
import importlib
import inspect
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_continuation as grammar_continuation_module
from onyx_cuda import (
    GrammarMaskedDraftProposalResult,
    GrammarMaskedPostAcceptanceContinuationCleanupError,
    GrammarMaskedPostAcceptanceContinuationError,
    GrammarMaskedPostAcceptanceContinuationInvariantError,
    GrammarMaskedPostAcceptanceContinuationResult,
    GrammarMaskedSelectionResult,
    GrammarMaskedTargetAcceptanceResult,
    GrammarMaskedTransitionCleanupError,
    GrammarMaskedTransitionResult,
    MatchReplaceAcceptanceResult,
    PostIterationContinuationError,
    decide_grammar_masked_post_acceptance_continuation,
    decide_grammar_masked_target_acceptance,
    decide_post_iteration_continuation,
    generate_grammar_masked_draft_proposal,
)
from onyx_cuda.testing import FakeAutoregressiveBackend, FakeGrammarConstraint, FakeGrammarProgram


VOCAB_SIZE = 7
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class OpaqueState:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        raise AssertionError("D46 must compare grammar states only by identity")

    def __hash__(self):
        raise AssertionError("D46 must not hash opaque grammar states")


class OpaqueRow:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name):
        self.name = name

    def __iter__(self):
        raise AssertionError("D46 must not inspect target rows")

    def __bool__(self):
        raise AssertionError("D46 must not inspect target-row truthiness")

    def __eq__(self, other):
        raise AssertionError("D46 must not compare target rows")


class RecordingMask:
    def __init__(self):
        self.calls = []
        self.masked_rows = []

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    def apply(self, logits, valid_token_ids, /):
        masked = OpaqueRow(f"masked-{len(self.calls)}")
        self.calls.append((logits, valid_token_ids))
        self.masked_rows.append(masked)
        return masked


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


class TrackingConstraint:
    def __init__(
        self,
        supports,
        *,
        matches=None,
        states=None,
        grammar_type="regex",
    ):
        self.supports = tuple(supports)
        self.matches = tuple(matches or (False,) * len(self.supports))
        self.states = list(
            states
            if states is not None
            else (OpaqueState(f"state-{index}") for index in range(len(self.supports)))
        )
        self.grammar_type = grammar_type
        self.live_states = list(self.states[:1])
        self.advance_calls = []
        self.valid_calls = []
        self.dead_calls = []
        self.match_calls = []
        self.release_calls = []
        self.release_outcomes = []
        self.dead_outcomes = []
        self.match_outcomes = []
        self.init_calls = 0
        self.bulk_release_calls = 0
        self.reset_calls = 0
        self.peak_live_states = len(self.live_states)

    @property
    def parent(self):
        return self.states[0]

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @property
    def child(self):
        return self.states[1]

    def init_state(self):
        self.init_calls += 1
        raise AssertionError("D46 must not initialize grammar state")

    def advance_state(self, state, token_id, /):
        position = self._position(state)
        self.advance_calls.append((state, token_id))
        child = self.states[position + 1]
        if not any(child is live for live in self.live_states):
            self.live_states.append(child)
        self.peak_live_states = max(self.peak_live_states, len(self.live_states))
        return child

    def get_valid_token_ids(self, state, /):
        position = self._position(state)
        self.valid_calls.append(state)
        return self.supports[position]

    def is_match_state(self, state, /):
        position = self._position(state)
        self.match_calls.append(state)
        outcome = self._take_outcome(self.match_outcomes, state)
        if outcome is not None:
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return self.matches[position]

    def is_dead_state(self, state, /):
        self._position(state)
        self.dead_calls.append(state)
        outcome = self._take_outcome(self.dead_outcomes, state)
        if outcome is not None:
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return False

    def release_state(self, state, /):
        self._position(state)
        self.release_calls.append(state)
        outcome = self._take_outcome(self.release_outcomes, state)
        if outcome is not None:
            if isinstance(outcome, BaseException):
                raise outcome
        self.live_states = [live for live in self.live_states if live is not state]

    def release_states(self, states, /):
        self.bulk_release_calls += 1
        raise AssertionError("D46 must not bulk-release grammar states")

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("D46 must not reset the grammar constraint")

    def set_release_outcomes(self, state, *outcomes):
        self.release_outcomes.append([state, list(outcomes)])

    def set_dead_outcomes(self, state, *outcomes):
        self.dead_outcomes.append([state, list(outcomes)])

    def _take_outcome(self, records, state):
        for candidate, outcomes in records:
            if candidate is state and outcomes:
                return outcomes.pop(0)
        return None

    def _position(self, state):
        for position, candidate in enumerate(self.states):
            if state is candidate and any(state is live for live in self.live_states):
                return position
        raise RuntimeError("state is not live")


class HostileComponent:
    def __getattribute__(self, name):
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError("mismatch must not touch grammar components")


def _acceptance(proposal, accepted_count, replacement, state, is_match=False):
    return GrammarMaskedTargetAcceptanceResult(
        proposal,
        accepted_count,
        replacement,
        None,
        state,
        is_match,
    )


def _rows(proposal):
    return tuple(OpaqueRow(f"row-{index}") for index in range(len(proposal) + 1))


def _run_full(
    proposal,
    constraint,
    selector,
    *,
    rows=None,
    mask=None,
    parent_is_match=False,
):
    target_rows = _rows(proposal) if rows is None else rows
    target_mask = RecordingMask() if mask is None else mask
    acceptance = _acceptance(
        proposal,
        len(proposal),
        None,
        constraint.parent,
        parent_is_match,
    )
    result = decide_grammar_masked_post_acceptance_continuation(
        proposal,
        target_rows,
        acceptance,
        constraint,
        target_mask,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    return result, target_rows, target_mask, acceptance


def _forged_selection(valid_token_ids, is_match, selected_token_id):
    selection = object.__new__(GrammarMaskedSelectionResult)
    object.__setattr__(selection, "valid_token_ids", valid_token_ids)
    object.__setattr__(selection, "is_match", is_match)
    object.__setattr__(selection, "selected_token_id", selected_token_id)
    return selection


def _forged_acceptance(**changes):
    values = {
        "proposal_token_ids": (1, 2),
        "accepted_count": 2,
        "replacement_token_id": None,
        "no_decision_selection": None,
        "committed_state": OpaqueState("parent"),
        "committed_state_is_match": False,
    }
    values.update(changes)
    result = object.__new__(GrammarMaskedTargetAcceptanceResult)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _forged_transition(selection, child_state, child_is_match):
    result = object.__new__(GrammarMaskedTransitionResult)
    object.__setattr__(result, "selection", selection)
    object.__setattr__(result, "child_state", child_state)
    object.__setattr__(result, "child_is_match", child_is_match)
    return result


def test_public_surface_signature_result_and_error_hierarchy():
    assert grammar_continuation_module.__all__ == [
        "GrammarMaskedPostAcceptanceContinuationCleanupError",
        "GrammarMaskedPostAcceptanceContinuationError",
        "GrammarMaskedPostAcceptanceContinuationInvariantError",
        "GrammarMaskedPostAcceptanceContinuationResult",
        "decide_grammar_masked_post_acceptance_continuation",
    ]
    assert issubclass(
        GrammarMaskedPostAcceptanceContinuationError,
        PostIterationContinuationError,
    )
    assert issubclass(
        GrammarMaskedPostAcceptanceContinuationInvariantError,
        GrammarMaskedPostAcceptanceContinuationError,
    )
    assert issubclass(
        GrammarMaskedPostAcceptanceContinuationCleanupError,
        GrammarMaskedPostAcceptanceContinuationError,
    )
    signature = inspect.signature(decide_grammar_masked_post_acceptance_continuation)
    assert tuple(signature.parameters) == (
        "proposal_token_ids",
        "target_logit_rows",
        "acceptance_result",
        "constraint",
        "logit_mask",
        "vocab_size",
        "select_token",
    )
    for name in (
        "proposal_token_ids",
        "target_logit_rows",
        "acceptance_result",
        "constraint",
        "logit_mask",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("vocab_size", "select_token"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is inspect.Parameter.empty
    assert [field.name for field in fields(GrammarMaskedPostAcceptanceContinuationResult)] == [
        "output_token_ids",
        "uncached_next_token_id",
        "final_row_no_decision_selection",
        "committed_state",
        "committed_state_is_match",
    ]


def test_direct_results_are_frozen_slotted_and_allow_opaque_none_state():
    selected = GrammarMaskedPostAcceptanceContinuationResult(
        (1, 2), 2, None, None, False
    )
    empty_selection = GrammarMaskedSelectionResult((), True, None)
    empty = GrammarMaskedPostAcceptanceContinuationResult(
        (1,), None, empty_selection, None, True
    )

    assert selected == GrammarMaskedPostAcceptanceContinuationResult(
        (1, 2), 2, None, None, False
    )
    assert selected.committed_state is None
    assert empty.final_row_no_decision_selection is empty_selection
    with pytest.raises(FrozenInstanceError):
        selected.uncached_next_token_id = 1
    assert not hasattr(selected, "__dict__")


@pytest.mark.parametrize(
    "args,error",
    [
        (([], 1, None, object(), False), TypeError),
        (((), 1, None, object(), False), ValueError),
        (((True,), True, None, object(), False), TypeError),
        (((-1,), -1, None, object(), False), ValueError),
        (((1,), None, None, object(), False), TypeError),
        (((1,), 2, None, object(), False), GrammarMaskedPostAcceptanceContinuationInvariantError),
        (((1,), 1, None, object(), 0), TypeError),
        (
            ((1,), 1, GrammarMaskedSelectionResult((), False, None), object(), False),
            GrammarMaskedPostAcceptanceContinuationInvariantError,
        ),
        (
            ((1,), None, GrammarMaskedSelectionResult((), True, None), object(), False),
            GrammarMaskedPostAcceptanceContinuationInvariantError,
        ),
    ],
)
def test_direct_result_rejects_malformed_relationships(args, error):
    with pytest.raises(error):
        GrammarMaskedPostAcceptanceContinuationResult(*args)


@pytest.mark.parametrize(
    "selection,error",
    [
        (_forged_selection([], False, None), TypeError),
        (_forged_selection((1,), False, None), ValueError),
        (_forged_selection((), 0, None), TypeError),
        (_forged_selection((), False, 1), ValueError),
    ],
)
def test_direct_result_rejects_malformed_empty_support(selection, error):
    with pytest.raises(error):
        GrammarMaskedPostAcceptanceContinuationResult(
            (1,), None, selection, object(), False
        )


def test_direct_construction_has_no_vocabulary_upper_bound_but_operation_does():
    direct = GrammarMaskedPostAcceptanceContinuationResult(
        (1000,), 1000, None, object(), False
    )
    assert direct.uncached_next_token_id == 1000
    with pytest.raises(ValueError, match="outside vocabulary"):
        decide_grammar_masked_post_acceptance_continuation(
            (VOCAB_SIZE,),
            (object(), object()),
            object(),
            object(),
            object(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 0,
        )


@pytest.mark.parametrize(
    "vocab_size,proposal,rows,acceptance,selector,error",
    [
        (True, (1,), (object(), object()), object(), lambda row: 1, TypeError),
        (1.0, (1,), (object(), object()), object(), lambda row: 1, TypeError),
        (0, (1,), (object(), object()), object(), lambda row: 1, ValueError),
        (7, [], (object(), object()), object(), lambda row: 1, TypeError),
        (7, (), (object(),), object(), lambda row: 1, ValueError),
        (7, (True,), (object(), object()), object(), lambda row: 1, TypeError),
        (7, (-1,), (object(), object()), object(), lambda row: 1, ValueError),
        (7, (7,), (object(), object()), object(), lambda row: 1, ValueError),
        (7, (1,), [object(), object()], object(), lambda row: 1, TypeError),
        (
            7,
            (1,),
            (object(),),
            object(),
            lambda row: 1,
            GrammarMaskedPostAcceptanceContinuationInvariantError,
        ),
        (
            7,
            (1,),
            (object(), object(), object()),
            object(),
            lambda row: 1,
            GrammarMaskedPostAcceptanceContinuationInvariantError,
        ),
        (7, (1,), (object(), object()), object(), lambda row: 1, TypeError),
    ],
)
def test_structural_preflight_precedes_d43_and_ownership(
    monkeypatch,
    vocab_size,
    proposal,
    rows,
    acceptance,
    selector,
    error,
):
    calls = []
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(error):
        decide_grammar_masked_post_acceptance_continuation(
            proposal,
            rows,
            acceptance,
            HostileComponent(),
            HostileComponent(),
            vocab_size=vocab_size,
            select_token=selector,
        )
    assert calls == []


@pytest.mark.parametrize(
    "acceptance",
    [
        _forged_acceptance(proposal_token_ids=[1, 2]),
        _forged_acceptance(proposal_token_ids=(1, True)),
        _forged_acceptance(proposal_token_ids=(1, 7)),
        _forged_acceptance(proposal_token_ids=(1, 3)),
        _forged_acceptance(accepted_count=True),
        _forged_acceptance(accepted_count=3),
        _forged_acceptance(accepted_count=1, replacement_token_id=None),
        _forged_acceptance(accepted_count=1, replacement_token_id=True),
        _forged_acceptance(accepted_count=1, replacement_token_id=7),
        _forged_acceptance(accepted_count=1, replacement_token_id=2),
        _forged_acceptance(accepted_count=2, replacement_token_id=3),
        _forged_acceptance(committed_state_is_match=None),
    ],
)
def test_malformed_d45_evidence_is_rejected_before_ownership(monkeypatch, acceptance):
    calls = []
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(GrammarMaskedPostAcceptanceContinuationInvariantError):
        decide_grammar_masked_post_acceptance_continuation(
            (1, 2),
            (object(), object(), object()),
            acceptance,
            HostileComponent(),
            HostileComponent(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 1,
        )
    assert calls == []


@pytest.mark.parametrize(
    "missing_field",
    [
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "no_decision_selection",
        "committed_state",
        "committed_state_is_match",
    ],
)
def test_unreadable_d45_fields_are_rejected_before_ownership(missing_field):
    acceptance = _forged_acceptance()
    object.__delattr__(acceptance, missing_field)
    with pytest.raises(
        GrammarMaskedPostAcceptanceContinuationInvariantError,
        match="could not be read",
    ):
        decide_grammar_masked_post_acceptance_continuation(
            (1, 2),
            _rows((1, 2)),
            acceptance,
            HostileComponent(),
            HostileComponent(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 1,
        )


def test_noncallable_selector_is_rejected_on_mismatch_without_state_work():
    parent = OpaqueState("parent")
    acceptance = _acceptance((1,), 0, 2, parent)
    with pytest.raises(TypeError, match="select_token"):
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            (object(), object()),
            acceptance,
            HostileComponent(),
            HostileComponent(),
            vocab_size=VOCAB_SIZE,
            select_token=None,
        )


@pytest.mark.parametrize("position", [0, 1])
def test_d45_no_decision_is_rejected_before_ownership(monkeypatch, position):
    proposal = (1, 2)
    selection = GrammarMaskedSelectionResult((), position == 1, None)
    state = OpaqueState("prefix") if position else None
    match = True if position else None
    acceptance = GrammarMaskedTargetAcceptanceResult(
        proposal,
        position,
        None,
        selection,
        state,
        match,
    )
    calls = []
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(
        GrammarMaskedPostAcceptanceContinuationInvariantError,
        match="outside D46",
    ):
        decide_grammar_masked_post_acceptance_continuation(
            proposal,
            _rows(proposal),
            acceptance,
            HostileComponent(),
            HostileComponent(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 1,
        )
    assert calls == []


def test_genuine_zero_token_d44_result_is_rejected_before_acceptance_or_rows():
    draft_result = GrammarMaskedDraftProposalResult(
        (),
        (),
        3,
        4,
        GrammarMaskedSelectionResult((), True, None),
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        decide_grammar_masked_post_acceptance_continuation(
            draft_result.proposal_token_ids,
            (HostileComponent(),),
            object(),
            HostileComponent(),
            HostileComponent(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 1,
        )


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("parent_is_match", [False, True])
@pytest.mark.parametrize(
    "proposal,accepted_count,replacement",
    [
        ((1,), 0, 2),
        ((1, 2, 3), 0, 4),
        ((1, 2, 3), 1, 4),
        ((1, 2, 3), 2, 4),
    ],
)
def test_mismatch_reuses_exact_d45_output_and_state_with_zero_new_work(
    monkeypatch,
    grammar_type,
    parent_is_match,
    proposal,
    accepted_count,
    replacement,
):
    parent = OpaqueState("parent")
    acceptance = _acceptance(
        proposal,
        accepted_count,
        replacement,
        parent,
        parent_is_match,
    )
    calls = []
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = decide_grammar_masked_post_acceptance_continuation(
        proposal,
        _rows(proposal),
        acceptance,
        HostileComponent(),
        HostileComponent(),
        vocab_size=VOCAB_SIZE,
        select_token=lambda row: (_ for _ in ()).throw(
            AssertionError("mismatch must not select")
        ),
    )

    assert result.output_token_ids == proposal[:accepted_count] + (replacement,)
    assert result.uncached_next_token_id == replacement
    assert result.final_row_no_decision_selection is None
    assert result.committed_state is parent
    assert result.committed_state_is_match is parent_is_match
    assert calls == []
    assert grammar_type in ("regex", "json_schema")


def test_mismatch_transfers_opaque_none_state_without_a_none_ownership_check():
    proposal = (1,)
    result = decide_grammar_masked_post_acceptance_continuation(
        proposal,
        _rows(proposal),
        _acceptance(proposal, 0, 2, None, True),
        HostileComponent(),
        HostileComponent(),
        vocab_size=VOCAB_SIZE,
        select_token=lambda row: 1,
    )
    assert result.committed_state is None
    assert result.committed_state_is_match is True


def test_decided_token_relationships_match_d37_without_d46_calling_it():
    proposal = (1, 2)
    rows = _rows(proposal)
    d37_mismatch = decide_post_iteration_continuation(
        proposal,
        rows,
        MatchReplaceAcceptanceResult(proposal, 1, 3),
        vocab_size=VOCAB_SIZE,
        select_token=lambda row: (_ for _ in ()).throw(
            AssertionError("D37 mismatch must not select")
        ),
    )
    parent = OpaqueState("mismatch-parent")
    d46_mismatch = decide_grammar_masked_post_acceptance_continuation(
        proposal,
        rows,
        _acceptance(proposal, 1, 3, parent),
        HostileComponent(),
        HostileComponent(),
        vocab_size=VOCAB_SIZE,
        select_token=lambda row: 3,
    )
    assert d46_mismatch.output_token_ids == d37_mismatch.output_token_ids
    assert d46_mismatch.uncached_next_token_id == d37_mismatch.uncached_next_token_id

    d37_full = decide_post_iteration_continuation(
        proposal,
        rows,
        MatchReplaceAcceptanceResult(proposal, 2, None),
        vocab_size=VOCAB_SIZE,
        select_token=RecordingSelector([3]),
    )
    constraint = TrackingConstraint(((3,), ()))
    d46_full, _rows_value, _mask, _acceptance_result = _run_full(
        proposal,
        constraint,
        RecordingSelector([3]),
        rows=rows,
    )
    assert d46_full.output_token_ids == d37_full.output_token_ids
    assert d46_full.uncached_next_token_id == d37_full.uncached_next_token_id
    constraint.release_state(d46_full.committed_state)


def test_mismatch_result_failure_consumes_and_settles_parent(monkeypatch):
    original = RuntimeError("result failed")

    class FailingResult:
        def __init__(self, **kwargs):
            raise original

    monkeypatch.setattr(
        grammar_continuation_module,
        "GrammarMaskedPostAcceptanceContinuationResult",
        FailingResult,
    )
    constraint = TrackingConstraint(((1,),))
    acceptance = _acceptance((1,), 0, 2, constraint.parent)
    with pytest.raises(RuntimeError) as captured:
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            acceptance,
            constraint,
            HostileComponent(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 1,
        )
    assert captured.value is original
    assert constraint.release_calls == [constraint.parent]
    assert constraint.live_states == []


def test_composed_result_failure_consumes_and_settles_parent(monkeypatch):
    parent = OpaqueState("parent")
    constraint = TrackingConstraint(((1,),), states=(parent,))

    class MalformedResult:
        def __init__(self, **kwargs):
            self.output_token_ids = (6,)
            self.uncached_next_token_id = kwargs["uncached_next_token_id"]
            self.final_row_no_decision_selection = kwargs[
                "final_row_no_decision_selection"
            ]
            self.committed_state = kwargs["committed_state"]
            self.committed_state_is_match = kwargs["committed_state_is_match"]

    monkeypatch.setattr(
        grammar_continuation_module,
        "GrammarMaskedPostAcceptanceContinuationResult",
        MalformedResult,
    )
    with pytest.raises(
        GrammarMaskedPostAcceptanceContinuationInvariantError,
        match="output-token tuple",
    ):
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 0, 2, parent),
            constraint,
            HostileComponent(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 1,
        )
    assert constraint.release_calls == [parent]
    assert constraint.live_states == []


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("parent_is_match", [False, True])
@pytest.mark.parametrize("child_is_match", [False, True])
@pytest.mark.parametrize("proposal", [(1,), (1, 2, 3)])
def test_full_acceptance_selects_one_bonus_and_rotates_state(
    grammar_type,
    parent_is_match,
    child_is_match,
    proposal,
):
    bonus = proposal[-1]
    constraint = TrackingConstraint(
        ((bonus,), ()),
        matches=(parent_is_match, child_is_match),
        grammar_type=grammar_type,
    )
    selector = RecordingSelector([bonus])

    result, rows, mask, _acceptance_result = _run_full(
        proposal,
        constraint,
        selector,
        parent_is_match=parent_is_match,
    )

    assert result.output_token_ids == proposal + (bonus,)
    assert result.uncached_next_token_id == bonus
    assert result.final_row_no_decision_selection is None
    assert result.committed_state is constraint.child
    assert result.committed_state_is_match is child_is_match
    assert mask.calls == [(rows[-1], (bonus,))]
    assert selector.calls == mask.masked_rows
    assert constraint.advance_calls == [(constraint.parent, bonus)]
    assert constraint.release_calls == [constraint.parent]
    assert constraint.live_states == [constraint.child]
    assert constraint.bulk_release_calls == 0
    assert constraint.reset_calls == 0
    constraint.release_state(result.committed_state)
    assert constraint.live_states == []


def test_full_acceptance_passes_only_final_row_to_one_exact_d43_call(monkeypatch):
    proposal = (1, 2, 3)
    rows = _rows(proposal)
    constraint = TrackingConstraint(((4,), ()), matches=(True, False))
    mask = RecordingMask()
    selector = RecordingSelector([4])
    calls = []
    original = grammar_continuation_module.select_and_advance_grammar_state

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        record,
    )
    result, _, _, acceptance = _run_full(
        proposal,
        constraint,
        selector,
        rows=rows,
        mask=mask,
        parent_is_match=True,
    )

    assert len(calls) == 1
    assert calls[0][0] == (constraint, acceptance.committed_state, rows[-1], mask)
    assert calls[0][0][2] is rows[-1]
    assert calls[0][1] == {"vocab_size": VOCAB_SIZE, "select_token": selector}
    assert all(calls[0][0][2] is not row for row in rows[:-1])
    constraint.release_state(result.committed_state)


def test_selected_opaque_none_child_is_owned_revalidated_and_transferred():
    parent = OpaqueState("parent")
    constraint = TrackingConstraint(
        ((3,), ()),
        states=(parent, None),
        matches=(False, True),
    )
    result, _rows_value, _mask, _acceptance_result = _run_full(
        (1,),
        constraint,
        RecordingSelector([3]),
    )
    assert result.committed_state is None
    assert result.committed_state_is_match is True
    assert constraint.release_calls == [parent]
    assert constraint.live_states == [None]
    constraint.release_state(result.committed_state)


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("parent_is_match", [False, True])
def test_empty_final_row_support_is_explicit_unclassified_and_transfers_parent(
    grammar_type,
    parent_is_match,
):
    proposal = (1, 2)
    constraint = TrackingConstraint(
        ((),),
        matches=(parent_is_match,),
        grammar_type=grammar_type,
    )
    selector = RecordingSelector([])

    result, rows, mask, _acceptance_result = _run_full(
        proposal,
        constraint,
        selector,
        parent_is_match=parent_is_match,
    )

    assert result.output_token_ids is proposal
    assert result.uncached_next_token_id is None
    assert result.final_row_no_decision_selection.valid_token_ids == ()
    assert result.final_row_no_decision_selection.selected_token_id is None
    assert result.final_row_no_decision_selection.is_match is parent_is_match
    assert result.committed_state is constraint.parent
    assert result.committed_state_is_match is parent_is_match
    assert constraint.valid_calls == [constraint.parent]
    assert mask.calls == []
    assert selector.calls == []
    assert constraint.advance_calls == []
    assert constraint.release_calls == []
    assert rows[-1] is not None
    assert not hasattr(result, "finish_reason")
    constraint.release_state(result.committed_state)


def test_empty_support_retains_exact_d43_selection_identity(monkeypatch):
    proposal = (1,)
    parent = OpaqueState("parent")
    selection = GrammarMaskedSelectionResult((), True, None)
    transition = GrammarMaskedTransitionResult(selection, None, None)
    calls = []
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: (calls.append((args, kwargs)), transition)[1],
    )
    constraint = TrackingConstraint(((),), states=(parent,), matches=(True,))
    result = decide_grammar_masked_post_acceptance_continuation(
        proposal,
        _rows(proposal),
        _acceptance(proposal, 1, None, parent, True),
        constraint,
        RecordingMask(),
        vocab_size=VOCAB_SIZE,
        select_token=RecordingSelector([]),
    )
    assert result.final_row_no_decision_selection is selection
    assert len(calls) == 1
    constraint.release_state(parent)


def test_parent_match_evidence_change_is_rejected_and_both_states_are_settled(monkeypatch):
    parent = OpaqueState("parent")
    child = OpaqueState("child")
    constraint = TrackingConstraint(
        ((2,), ()), states=(parent, child), matches=(False, False)
    )
    constraint.live_states.append(child)
    selection = GrammarMaskedSelectionResult((2,), True, 2)
    transition = GrammarMaskedTransitionResult(selection, child, False)
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )
    with pytest.raises(
        GrammarMaskedPostAcceptanceContinuationInvariantError,
        match="parent-match evidence changed",
    ):
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 1, None, parent, False),
            constraint,
            RecordingMask(),
            vocab_size=VOCAB_SIZE,
            select_token=RecordingSelector([]),
        )
    assert constraint.release_calls == [parent, child]
    assert constraint.live_states == []


@pytest.mark.parametrize(
    "transition",
    [
        object(),
        _forged_transition(_forged_selection([], False, None), None, None),
        _forged_transition(_forged_selection((2,), False, None), None, None),
        _forged_transition(_forged_selection((2,), False, True), OpaqueState("a"), False),
        _forged_transition(_forged_selection((7,), False, 7), OpaqueState("b"), False),
        _forged_transition(_forged_selection((2, 2), False, 2), OpaqueState("c"), False),
        _forged_transition(_forged_selection((2,), False, 2), OpaqueState("d"), None),
        _forged_transition(_forged_selection((), False, None), OpaqueState("e"), None),
    ],
)
def test_malformed_d43_evidence_releases_parent_and_any_distinct_child(
    monkeypatch,
    transition,
):
    parent = OpaqueState("parent")
    states = [parent]
    child = getattr(transition, "child_state", None)
    if child is not None:
        states.append(child)
    constraint = TrackingConstraint(
        ((),) * len(states),
        states=states,
        matches=(False,) * len(states),
    )
    if child is not None:
        constraint.live_states.append(child)
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )
    with pytest.raises(
        (GrammarMaskedPostAcceptanceContinuationInvariantError, TypeError)
    ):
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 1, None, parent),
            constraint,
            RecordingMask(),
            vocab_size=VOCAB_SIZE,
            select_token=RecordingSelector([]),
        )
    expected = [parent]
    if child is not None and child is not parent:
        expected.append(child)
    assert constraint.release_calls == expected


@pytest.mark.parametrize("missing_field", ["selection", "child_is_match"])
def test_distinct_child_is_owned_before_later_transition_field_read_fails(
    monkeypatch,
    missing_field,
):
    parent = OpaqueState("parent")
    child = OpaqueState("child")
    transition = object.__new__(GrammarMaskedTransitionResult)
    object.__setattr__(transition, "child_state", child)
    if missing_field != "selection":
        object.__setattr__(
            transition,
            "selection",
            GrammarMaskedSelectionResult((2,), False, 2),
        )
    if missing_field != "child_is_match":
        object.__setattr__(transition, "child_is_match", False)
    constraint = TrackingConstraint(
        ((2,), ()), states=(parent, child), matches=(False, False)
    )
    constraint.live_states.append(child)
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )

    with pytest.raises(GrammarMaskedPostAcceptanceContinuationInvariantError):
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 1, None, parent),
            constraint,
            RecordingMask(),
            vocab_size=VOCAB_SIZE,
            select_token=RecordingSelector([]),
        )

    assert constraint.release_calls == [parent, child]
    assert constraint.live_states == []


def test_none_child_with_malformed_match_evidence_is_not_abandoned(monkeypatch):
    parent = OpaqueState("parent")
    constraint = TrackingConstraint(
        ((), ()),
        states=(parent, None),
        matches=(False, False),
    )
    constraint.live_states.append(None)
    transition = _forged_transition(
        GrammarMaskedSelectionResult((), False, None),
        None,
        False,
    )
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )

    with pytest.raises(GrammarMaskedPostAcceptanceContinuationInvariantError):
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 1, None, parent),
            constraint,
            RecordingMask(),
            vocab_size=VOCAB_SIZE,
            select_token=RecordingSelector([]),
        )

    assert constraint.release_calls == [parent, None]
    assert constraint.live_states == []


def test_parent_alias_is_rejected_without_double_release(monkeypatch):
    parent = OpaqueState("parent")
    constraint = TrackingConstraint(((2,),), states=(parent,))
    transition = GrammarMaskedTransitionResult(
        GrammarMaskedSelectionResult((2,), False, 2),
        parent,
        False,
    )
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )
    with pytest.raises(
        GrammarMaskedPostAcceptanceContinuationInvariantError,
        match="aliases",
    ):
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 1, None, parent),
            constraint,
            RecordingMask(),
            vocab_size=VOCAB_SIZE,
            select_token=RecordingSelector([]),
        )
    assert constraint.release_calls == [parent]


def test_d43_failure_releases_parent_and_propagates_exact_exception(monkeypatch):
    original = RuntimeError("selection failed")
    parent = OpaqueState("parent")
    constraint = TrackingConstraint(((),), states=(parent,))
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(original),
    )
    with pytest.raises(RuntimeError) as captured:
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 1, None, parent),
            constraint,
            RecordingMask(),
            vocab_size=VOCAB_SIZE,
            select_token=RecordingSelector([]),
        )
    assert captured.value is original
    assert constraint.release_calls == [parent]


def test_nested_d43_cleanup_error_remains_the_exact_original(monkeypatch):
    transition_failure = RuntimeError("transition failed")
    transition_cleanup = RuntimeError("transition cleanup failed")
    nested = GrammarMaskedTransitionCleanupError(
        transition_failure,
        (("child state release", transition_cleanup),),
    )
    parent = OpaqueState("parent")
    constraint = TrackingConstraint(((),), states=(parent,))
    monkeypatch.setattr(
        grammar_continuation_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(nested),
    )
    with pytest.raises(GrammarMaskedTransitionCleanupError) as captured:
        decide_grammar_masked_post_acceptance_continuation(
            (1,),
            _rows((1,)),
            _acceptance((1,), 1, None, parent),
            constraint,
            RecordingMask(),
            vocab_size=VOCAB_SIZE,
            select_token=RecordingSelector([]),
        )
    assert captured.value is nested
    assert constraint.release_calls == [parent]


def test_success_parent_release_failure_gets_one_cleanup_retry_then_child_cleanup():
    original = RuntimeError("parent release failed")
    constraint = TrackingConstraint(((2,), ()))
    constraint.set_release_outcomes(constraint.parent, original, None)
    with pytest.raises(RuntimeError) as captured:
        _run_full((1,), constraint, RecordingSelector([2]))
    assert captured.value is original
    assert constraint.release_calls == [
        constraint.parent,
        constraint.parent,
        constraint.child,
    ]
    assert constraint.live_states == []


def test_multiple_cleanup_failures_have_global_order_identity_and_cause():
    original = RuntimeError("parent release failed")
    parent_retry = RuntimeError("parent retry failed")
    child_cleanup = RuntimeError("child cleanup failed")
    constraint = TrackingConstraint(((2,), ()))
    constraint.set_release_outcomes(constraint.parent, original, parent_retry)
    constraint.set_release_outcomes(constraint.child, child_cleanup)
    with pytest.raises(
        GrammarMaskedPostAcceptanceContinuationCleanupError
    ) as captured:
        _run_full((1,), constraint, RecordingSelector([2]))
    error = captured.value
    assert error.original_failure is original
    assert error.__cause__ is original
    assert error.cleanup_failures == (
        ("committed parent state release", parent_retry),
        ("bonus child state release", child_cleanup),
    )
    assert type(error.cleanup_failures) is tuple


def test_child_revalidation_failure_releases_only_the_still_owned_child():
    original = RuntimeError("child query failed")
    constraint = TrackingConstraint(((2,), ()))
    constraint.set_dead_outcomes(constraint.child, False, original)
    with pytest.raises(RuntimeError) as captured:
        _run_full((1,), constraint, RecordingSelector([2]))
    assert captured.value is original
    assert constraint.release_calls == [constraint.parent, constraint.child]
    assert constraint.live_states == []


def test_selected_result_construction_failure_settles_parent_then_child(monkeypatch):
    original = RuntimeError("result construction failed")

    class FailingResult:
        def __init__(self, **kwargs):
            raise original

    monkeypatch.setattr(
        grammar_continuation_module,
        "GrammarMaskedPostAcceptanceContinuationResult",
        FailingResult,
    )
    constraint = TrackingConstraint(((2,), ()))
    with pytest.raises(RuntimeError) as captured:
        _run_full((1,), constraint, RecordingSelector([2]))
    assert captured.value is original
    assert constraint.release_calls == [constraint.parent, constraint.child]
    assert constraint.live_states == []


def test_empty_support_result_construction_failure_settles_parent(monkeypatch):
    original = RuntimeError("empty result construction failed")

    class FailingResult:
        def __init__(self, **kwargs):
            raise original

    monkeypatch.setattr(
        grammar_continuation_module,
        "GrammarMaskedPostAcceptanceContinuationResult",
        FailingResult,
    )
    constraint = TrackingConstraint(((),), matches=(True,))
    with pytest.raises(RuntimeError) as captured:
        _run_full(
            (1,),
            constraint,
            RecordingSelector([]),
            parent_is_match=True,
        )
    assert captured.value is original
    assert constraint.release_calls == [constraint.parent]
    assert constraint.live_states == []


def test_cleanup_error_constructor_preserves_exact_immutable_evidence():
    original = RuntimeError("original")
    cleanup = RuntimeError("cleanup")
    failures = [("committed parent state release", cleanup)]
    error = GrammarMaskedPostAcceptanceContinuationCleanupError(original, failures)
    failures.clear()
    assert error.original_failure is original
    assert error.cleanup_failures == (("committed parent state release", cleanup),)
    assert error.cleanup_failures[0][1] is cleanup
    assert error.__cause__ is original
    with pytest.raises(ValueError, match="cannot be empty"):
        GrammarMaskedPostAcceptanceContinuationCleanupError(original, ())


def test_selector_session_continues_from_d45_to_final_row_in_exact_order():
    proposal = (1, 2)
    rows = _rows(proposal)
    constraint = TrackingConstraint(
        ((1,), (2,), (3,), ()),
        matches=(False, False, False, True),
    )
    selector = RecordingSelector([1, 2, 3])
    mask = RecordingMask()
    acceptance = decide_grammar_masked_target_acceptance(
        proposal,
        rows,
        constraint,
        constraint.parent,
        mask,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    result = decide_grammar_masked_post_acceptance_continuation(
        proposal,
        rows,
        acceptance,
        constraint,
        mask,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    assert result.output_token_ids == (1, 2, 3)
    assert [call[0] for call in mask.calls] == list(rows)
    assert selector.calls == mask.masked_rows
    assert constraint.release_calls == [
        constraint.states[1],
        constraint.states[2],
    ]
    constraint.release_state(result.committed_state)
    constraint.release_state(constraint.parent)
    assert constraint.live_states == []


def test_genuine_d44_d30_d45_d46_composition_is_cache_and_checkpoint_neutral():
    script = tuple(
        tuple(float(row * VOCAB_SIZE + token) for token in range(VOCAB_SIZE))
        for row in range(12)
    )
    draft_backend = FakeAutoregressiveBackend(script, model_id="fake-draft")
    target_backend = FakeAutoregressiveBackend(script, model_id="fake-target")
    draft_backend.prefill((6,))
    target_backend.prefill((6,))
    draft_root = draft_backend.create_cache_checkpoint()
    target_root = target_backend.create_cache_checkpoint()
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(
            ("s0", 1, "s1"),
            ("s1", 2, "s2"),
            ("s2", 3, "s3"),
        ),
        valid_token_ids=(
            ("s0", (1,)),
            ("s1", (2,)),
            ("s2", (3,)),
            ("s3", ()),
        ),
        match_states=frozenset({"s3"}),
    )
    constraint = FakeGrammarConstraint(
        tuple(bytes((token,)) for token in range(VOCAB_SIZE)),
        grammar_type="regex",
        program=program,
    )
    start = constraint.init_state()
    draft_result = generate_grammar_masked_draft_proposal(
        draft_backend,
        0,
        constraint,
        start,
        RecordingMask(),
        proposal_bound=2,
        select_token=RecordingSelector([1, 2]),
    )
    verification = target_backend.verify_proposal(
        0,
        draft_result.proposal_token_ids,
    )
    selector = RecordingSelector([1, 2, 3])
    mask = RecordingMask()
    acceptance = decide_grammar_masked_target_acceptance(
        draft_result.proposal_token_ids,
        verification.logit_rows,
        constraint,
        start,
        mask,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    snapshots = (
        (
            draft_backend.cache_length,
            draft_backend.cached_token_ids,
            draft_backend._next_row,
            tuple(draft_backend._cache_checkpoints),
            draft_backend._next_checkpoint_id,
        ),
        (
            target_backend.cache_length,
            target_backend.cached_token_ids,
            target_backend._next_row,
            tuple(target_backend._cache_checkpoints),
            target_backend._next_checkpoint_id,
        ),
    )
    rollback_identities = tuple(id(value) for value in draft_result.rollback_checkpoints)
    result = decide_grammar_masked_post_acceptance_continuation(
        draft_result.proposal_token_ids,
        verification.logit_rows,
        acceptance,
        constraint,
        mask,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    assert result.output_token_ids == (1, 2, 3)
    assert snapshots == (
        (
            draft_backend.cache_length,
            draft_backend.cached_token_ids,
            draft_backend._next_row,
            tuple(draft_backend._cache_checkpoints),
            draft_backend._next_checkpoint_id,
        ),
        (
            target_backend.cache_length,
            target_backend.cached_token_ids,
            target_backend._next_row,
            tuple(target_backend._cache_checkpoints),
            target_backend._next_checkpoint_id,
        ),
    )
    assert rollback_identities == tuple(
        id(value) for value in draft_result.rollback_checkpoints
    )
    assert constraint.active_state_count == 2
    constraint.release_state(result.committed_state)
    constraint.release_state(start)
    for checkpoint in draft_result.rollback_checkpoints:
        draft_backend.release_cache_checkpoint(checkpoint)
    draft_backend.release_cache_checkpoint(draft_root)
    target_backend.release_cache_checkpoint(target_root)


def test_success_result_does_not_retain_execution_evidence_or_released_parent():
    class WeakAcceptance(GrammarMaskedTargetAcceptanceResult):
        __slots__ = ("__weakref__",)

    proposal = (1,)
    rows = _rows(proposal)
    constraint = TrackingConstraint(((2,), ()))
    mask = RecordingMask()
    selector = RecordingSelector([2])
    parent = constraint.parent
    acceptance = WeakAcceptance(proposal, 1, None, None, parent, False)
    result = decide_grammar_masked_post_acceptance_continuation(
        proposal,
        rows,
        acceptance,
        constraint,
        mask,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    references = [
        weakref.ref(value)
        for value in (*rows, mask, selector, parent, *mask.masked_rows, acceptance)
    ]
    constraint.release_state(result.committed_state)
    del constraint, rows, mask, selector, parent, acceptance
    gc.collect()
    assert all(reference() is None for reference in references)
    assert result.output_token_ids == (1, 2)


def test_one_thousand_alternating_outcomes_leave_bounded_state_ownership():
    for iteration in range(1000):
        if iteration % 3 == 0:
            constraint = TrackingConstraint(((2,),))
            result = decide_grammar_masked_post_acceptance_continuation(
                (1,),
                _rows((1,)),
                _acceptance((1,), 0, 2, constraint.parent),
                constraint,
                HostileComponent(),
                vocab_size=VOCAB_SIZE,
                select_token=lambda row: 2,
            )
            assert result.committed_state is constraint.parent
            assert constraint.release_calls == []
            constraint.release_state(result.committed_state)
            assert constraint.live_states == []
        elif iteration % 3 == 1:
            constraint = TrackingConstraint(((2,), ()))
            result, _rows_value, _mask, _acceptance_result = _run_full(
                (1,), constraint, RecordingSelector([2])
            )
            assert constraint.peak_live_states <= 2
            constraint.release_state(result.committed_state)
            assert constraint.live_states == []
        else:
            constraint = TrackingConstraint(((),), matches=(True,))
            result, _rows_value, _mask, _acceptance_result = _run_full(
                (1,), constraint, RecordingSelector([]), parent_is_match=True
            )
            constraint.release_state(result.committed_state)
            assert constraint.live_states == []


def test_module_has_no_mutable_state_registry():
    allowed_mutable_names = {"__all__", "__builtins__"}
    mutable_globals = {
        name
        for name, value in vars(grammar_continuation_module).items()
        if isinstance(value, (dict, list, set)) and name not in allowed_mutable_names
    }
    assert mutable_globals == set()


def test_normal_and_isolated_execution_remain_optional_runtime_free():
    imported = importlib.import_module("onyx_cuda.grammar_continuation")
    assert imported.decide_grammar_masked_post_acceptance_continuation is (
        decide_grammar_masked_post_acceptance_continuation
    )
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        import onyx_cuda
        from onyx_cuda import GrammarMaskedTargetAcceptanceResult

        class Constraint:
            def __init__(self, support):
                self.support = support
                self.released = []
                self.vocab_size = 3
                self.grammar_type = 'regex'
            def init_state(self):
                raise AssertionError
            def get_valid_token_ids(self, state):
                return self.support
            def is_dead_state(self, state):
                return False
            def is_match_state(self, state):
                return False
            def advance_state(self, state, token):
                return object()
            def release_state(self, state):
                self.released.append(state)
            def release_states(self, states):
                raise AssertionError
            def reset(self):
                raise AssertionError

        class Mask:
            vocab_size = 3
            def apply(self, row, valid):
                return row

        def run(accepted, support, selected):
            parent = object()
            constraint = Constraint(support)
            acceptance = GrammarMaskedTargetAcceptanceResult(
                (1,), accepted, None if accepted else 2, None, parent, False
            )
            result = onyx_cuda.decide_grammar_masked_post_acceptance_continuation(
                (1,), (object(), object()), acceptance, constraint,
                Mask(),
                vocab_size=3, select_token=lambda row: selected,
            )
            constraint.release_state(result.committed_state)

        run(0, (), 2)
        run(1, (2,), 2)
        run(1, (), 2)
        forbidden = (
            'onyx', 'mlx', 'torch', 'transformers', 'tokenizers',
            'huggingface_hub', 'bitsandbytes', 'accelerate', 'onnxruntime', 'psutil',
        )
        assert not any(
            name == prefix or name.startswith(prefix + '.')
            for name in sys.modules
            for prefix in forbidden
        )
        assert 'onyx_cuda._grammar_native' not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PACKAGE_ROOT,
        env={"PYTHONPATH": source_root},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
