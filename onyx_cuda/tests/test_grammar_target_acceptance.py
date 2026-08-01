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

import onyx_cuda.grammar_acceptance as grammar_acceptance_module
from onyx_cuda import (
    GrammarMaskedSelectionResult,
    GrammarMaskedDraftProposalResult,
    GrammarMaskedTargetAcceptanceCleanupError,
    GrammarMaskedTargetAcceptanceError,
    GrammarMaskedTargetAcceptanceInvariantError,
    GrammarMaskedTargetAcceptanceResult,
    GrammarMaskedTransitionCleanupError,
    GrammarMaskedTransitionResult,
    MatchReplaceAcceptanceError,
    MatchReplaceAcceptanceResult,
    decide_grammar_masked_target_acceptance,
    generate_grammar_masked_draft_proposal,
)
from onyx_cuda.testing import FakeAutoregressiveBackend, FakeGrammarConstraint, FakeGrammarProgram


VOCAB_SIZE = 5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class OpaqueState:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        raise AssertionError("D45 must compare grammar states only by identity")

    def __hash__(self):
        raise AssertionError("D45 must not hash opaque grammar states")


class OpaqueRow:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name):
        self.name = name

    def __iter__(self):
        raise AssertionError("D45 must not inspect target rows")

    def __bool__(self):
        raise AssertionError("D45 must not inspect target-row truthiness")

    def __eq__(self, other):
        raise AssertionError("D45 must not compare target rows")


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
        grammar_type="regex",
        matches=None,
        children=None,
    ):
        self.supports = tuple(supports)
        self.matches = tuple(matches or (False,) * len(self.supports))
        self._grammar_type = grammar_type
        self.starting_state = OpaqueState("start")
        child_count = max(0, len(self.supports) - 1)
        self.children = list(
            children
            if children is not None
            else (OpaqueState(f"child-{position}") for position in range(child_count))
        )
        self.live_children = []
        self.advance_calls = []
        self.release_calls = []
        self.release_outcomes = {}
        self.init_calls = 0
        self.bulk_release_calls = 0
        self.reset_calls = 0
        self.peak_live_children = 0

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @property
    def grammar_type(self):
        return self._grammar_type

    def init_state(self):
        self.init_calls += 1
        raise AssertionError("D45 must not initialize grammar state")

    def advance_state(self, state, token_id, /):
        position = self._position(state)
        self.advance_calls.append((state, token_id))
        child = self.children[position]
        if not any(child is live for live in self.live_children):
            self.live_children.append(child)
        self.peak_live_children = max(self.peak_live_children, len(self.live_children))
        return child

    def get_valid_token_ids(self, state, /):
        return self.supports[self._position(state)]

    def is_match_state(self, state, /):
        return self.matches[self._position(state)]

    def is_dead_state(self, state, /):
        self._position(state)
        return False

    def release_state(self, state, /):
        position = self._position(state)
        self.release_calls.append(state)
        outcomes = self.release_outcomes.get(position)
        if outcomes:
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        self.live_children = [live for live in self.live_children if live is not state]

    def release_states(self, states, /):
        self.bulk_release_calls += 1
        raise AssertionError("D45 must not bulk-release grammar states")

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("D45 must not reset the grammar constraint")

    def _position(self, state):
        if state is self.starting_state:
            return 0
        for position, child in enumerate(self.children, start=1):
            if state is child and any(state is live for live in self.live_children):
                return position
        raise RuntimeError("state is not live")


def _run(
    proposal,
    constraint,
    selector,
    *,
    rows=None,
    mask=None,
):
    target_rows = (
        tuple(OpaqueRow(f"row-{position}") for position in range(len(proposal) + 1))
        if rows is None
        else rows
    )
    target_mask = RecordingMask() if mask is None else mask
    result = decide_grammar_masked_target_acceptance(
        proposal,
        target_rows,
        constraint,
        constraint.starting_state,
        target_mask,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    return result, target_rows, target_mask


def _release_result(constraint, result):
    if result.committed_state_transferred:
        constraint.release_state(result.committed_state)


def _forged_selection(valid_token_ids, is_match, selected_token_id):
    result = object.__new__(GrammarMaskedSelectionResult)
    object.__setattr__(result, "valid_token_ids", valid_token_ids)
    object.__setattr__(result, "is_match", is_match)
    object.__setattr__(result, "selected_token_id", selected_token_id)
    return result


def test_public_surface_signature_result_and_error_hierarchy():
    assert grammar_acceptance_module.__all__ == [
        "GrammarMaskedTargetAcceptanceCleanupError",
        "GrammarMaskedTargetAcceptanceError",
        "GrammarMaskedTargetAcceptanceInvariantError",
        "GrammarMaskedTargetAcceptanceResult",
        "decide_grammar_masked_target_acceptance",
    ]
    assert issubclass(GrammarMaskedTargetAcceptanceError, MatchReplaceAcceptanceError)
    assert issubclass(
        GrammarMaskedTargetAcceptanceInvariantError,
        GrammarMaskedTargetAcceptanceError,
    )
    assert issubclass(
        GrammarMaskedTargetAcceptanceCleanupError,
        GrammarMaskedTargetAcceptanceError,
    )
    signature = inspect.signature(decide_grammar_masked_target_acceptance)
    assert tuple(signature.parameters) == (
        "proposal_token_ids",
        "target_logit_rows",
        "constraint",
        "starting_state",
        "logit_mask",
        "vocab_size",
        "select_token",
    )
    for name in (
        "proposal_token_ids",
        "target_logit_rows",
        "constraint",
        "starting_state",
        "logit_mask",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("vocab_size", "select_token"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is inspect.Parameter.empty
    assert [field.name for field in fields(GrammarMaskedTargetAcceptanceResult)] == [
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
        "no_decision_selection",
        "committed_state",
        "committed_state_is_match",
    ]


def test_direct_results_expose_every_derived_outcome_and_compose_with_d33():
    child = OpaqueState("committed")
    mismatch = GrammarMaskedTargetAcceptanceResult(
        (1, 2, 3), 1, 4, None, child, False
    )
    full = GrammarMaskedTargetAcceptanceResult((1, 2), 2, None, None, child, True)
    later_selection = GrammarMaskedSelectionResult((), True, None)
    later_empty = GrammarMaskedTargetAcceptanceResult(
        (1, 2), 1, None, later_selection, child, True
    )
    first_selection = GrammarMaskedSelectionResult((), False, None)
    first_empty = GrammarMaskedTargetAcceptanceResult(
        (1,), 0, None, first_selection, None, None
    )
    none_child = GrammarMaskedTargetAcceptanceResult(
        (1,), 0, 2, None, None, False
    )

    assert mismatch.decision_made is True
    assert mismatch.fully_accepted is False
    assert mismatch.accepted_token_ids == (1,)
    assert mismatch.committed_token_ids == (1, 4)
    assert mismatch.committed_state_transferred is True
    assert full.fully_accepted is True
    assert full.committed_token_ids == (1, 2)
    assert later_empty.decision_made is False
    assert later_empty.committed_token_ids == (1,)
    assert later_empty.committed_state_transferred is True
    assert first_empty.committed_token_ids == ()
    assert first_empty.committed_state_transferred is False
    assert none_child.committed_state is None
    assert none_child.committed_state_transferred is True
    assert MatchReplaceAcceptanceResult(
        mismatch.proposal_token_ids,
        mismatch.accepted_count,
        mismatch.replacement_token_id,
    ) == MatchReplaceAcceptanceResult((1, 2, 3), 1, 4)
    assert MatchReplaceAcceptanceResult(
        full.proposal_token_ids,
        full.accepted_count,
        full.replacement_token_id,
    ) == MatchReplaceAcceptanceResult((1, 2), 2, None)
    with pytest.raises(FrozenInstanceError):
        mismatch.accepted_count = 2
    assert not hasattr(mismatch, "__dict__")


@pytest.mark.parametrize(
    "args,error",
    [
        (([], 0, 1, None, object(), False), TypeError),
        (((), 0, None, None, object(), False), ValueError),
        (((True,), 0, 1, None, object(), False), TypeError),
        (((-1,), 0, 1, None, object(), False), ValueError),
        (((1,), True, 2, None, object(), False), TypeError),
        (((1,), -1, 2, None, object(), False), GrammarMaskedTargetAcceptanceInvariantError),
        (((1,), 2, None, None, object(), False), GrammarMaskedTargetAcceptanceInvariantError),
        (((1,), 1, 2, None, object(), False), GrammarMaskedTargetAcceptanceInvariantError),
        (((1,), 0, None, None, object(), False), GrammarMaskedTargetAcceptanceInvariantError),
        (((1,), 0, 1, None, object(), False), GrammarMaskedTargetAcceptanceInvariantError),
        (((1,), 0, 2, None, object(), None), GrammarMaskedTargetAcceptanceInvariantError),
        (((1,), 0, None, object(), None, None), TypeError),
        (
            ((1,), 0, None, GrammarMaskedSelectionResult((), False, None), object(), False),
            GrammarMaskedTargetAcceptanceInvariantError,
        ),
        (
            ((1,), 0, None, GrammarMaskedSelectionResult((), False, None), None, False),
            GrammarMaskedTargetAcceptanceInvariantError,
        ),
        (
            ((1, 2), 1, None, GrammarMaskedSelectionResult((), True, None), object(), False),
            GrammarMaskedTargetAcceptanceInvariantError,
        ),
    ],
)
def test_direct_result_rejects_malformed_relationships(args, error):
    with pytest.raises(error):
        GrammarMaskedTargetAcceptanceResult(*args)


@pytest.mark.parametrize(
    "selection,error",
    [
        (_forged_selection([], False, None), TypeError),
        (_forged_selection((1,), False, None), ValueError),
        (_forged_selection((), 0, None), TypeError),
        (_forged_selection((), False, 1), ValueError),
    ],
)
def test_direct_no_decision_rejects_malformed_nested_selection(selection, error):
    with pytest.raises(error):
        GrammarMaskedTargetAcceptanceResult((1,), 0, None, selection, None, None)


@pytest.mark.parametrize(
    "vocab_size,proposal,rows,selector,error",
    [
        (True, (1,), (object(), object()), lambda row: 1, TypeError),
        (1.0, (1,), (object(), object()), lambda row: 1, TypeError),
        (0, (1,), (object(), object()), lambda row: 1, ValueError),
        (5, [], (object(), object()), lambda row: 1, TypeError),
        (5, (), (object(),), lambda row: 1, ValueError),
        (5, (True,), (object(), object()), lambda row: 1, TypeError),
        (5, (-1,), (object(), object()), lambda row: 1, ValueError),
        (5, (5,), (object(), object()), lambda row: 1, ValueError),
        (5, (1,), [object(), object()], lambda row: 1, TypeError),
        (5, (1,), (object(),), lambda row: 1, GrammarMaskedTargetAcceptanceInvariantError),
        (
            5,
            (1,),
            (object(), object(), object()),
            lambda row: 1,
            GrammarMaskedTargetAcceptanceInvariantError,
        ),
        (5, (1,), (object(), object()), None, TypeError),
    ],
)
def test_structural_validation_precedes_d43_and_state_work(
    monkeypatch,
    vocab_size,
    proposal,
    rows,
    selector,
    error,
):
    calls = []
    monkeypatch.setattr(
        grammar_acceptance_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(error):
        decide_grammar_masked_target_acceptance(
            proposal,
            rows,
            object(),
            object(),
            object(),
            vocab_size=vocab_size,
            select_token=selector,
        )
    assert calls == []


def test_genuine_d44_zero_token_result_is_rejected_before_d43(monkeypatch):
    shortening = GrammarMaskedSelectionResult((), True, None)
    d44_result = GrammarMaskedDraftProposalResult((), (), 3, 4, shortening)
    calls = []
    monkeypatch.setattr(
        grammar_acceptance_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        decide_grammar_masked_target_acceptance(
            d44_result.proposal_token_ids,
            (object(),),
            object(),
            object(),
            object(),
            vocab_size=VOCAB_SIZE,
            select_token=lambda row: 1,
        )

    assert calls == []


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize(
    "selected,accepted_count,replacement",
    [
        ((4,), 0, 4),
        ((1, 4), 1, 4),
        ((1, 2, 4), 2, 4),
        ((1, 2, 3), 3, None),
    ],
)
def test_matching_and_mismatching_target_rows_transfer_exact_final_child(
    grammar_type,
    selected,
    accepted_count,
    replacement,
):
    proposal = (1, 2, 3)
    constraint = TrackingConstraint(
        ((0, 1, 2, 3, 4),) * 4,
        grammar_type=grammar_type,
        matches=(False, True, False, True),
    )
    selector = RecordingSelector(selected)

    result, rows, mask = _run(proposal, constraint, selector)

    final_position = len(selected) - 1
    assert result.proposal_token_ids is proposal
    assert result.accepted_count == accepted_count
    assert result.replacement_token_id == replacement
    assert result.no_decision_selection is None
    assert result.committed_state is constraint.children[final_position]
    assert result.committed_state_is_match is constraint.matches[final_position + 1]
    assert result.committed_token_ids == (
        proposal if replacement is None else proposal[:accepted_count] + (replacement,)
    )
    assert len(mask.calls) == len(selected)
    assert len(selector.calls) == len(selected)
    assert len(constraint.advance_calls) == len(selected)
    assert all(call[0] is rows[position] for position, call in enumerate(mask.calls))
    assert constraint.release_calls == constraint.children[:final_position]
    assert constraint.peak_live_children <= 2
    assert constraint.init_calls == 0
    assert constraint.bulk_release_calls == 0
    assert constraint.reset_calls == 0
    _release_result(constraint, result)
    assert constraint.live_children == []


def test_every_inspected_row_uses_exact_d43_arguments_and_final_row_is_unused(monkeypatch):
    proposal = (1, 2, 3)
    rows = tuple(OpaqueRow(f"row-{position}") for position in range(4))
    constraint = TrackingConstraint(((0, 1, 2, 3, 4),) * 4)
    mask = RecordingMask()
    selector = RecordingSelector([1, 4])
    calls = []
    original = grammar_acceptance_module.select_and_advance_grammar_state

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        grammar_acceptance_module,
        "select_and_advance_grammar_state",
        record,
    )
    result, _, _ = _run(
        proposal,
        constraint,
        selector,
        rows=rows,
        mask=mask,
    )

    assert len(calls) == 2
    assert calls[0][0] == (constraint, constraint.starting_state, rows[0], mask)
    assert calls[1][0] == (constraint, constraint.children[0], rows[1], mask)
    assert all(
        kwargs == {"vocab_size": VOCAB_SIZE, "select_token": selector}
        for _args, kwargs in calls
    )
    assert all(args[2] is not rows[-1] for args, _kwargs in calls)
    _release_result(constraint, result)


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("empty_position", [0, 1, 2, 3])
@pytest.mark.parametrize("terminal_match", [False, True])
def test_empty_support_returns_unclassified_evidence_at_every_position(
    grammar_type,
    empty_position,
    terminal_match,
):
    proposal = (1, 1, 1, 1)
    supports = ((1,),) * empty_position + ((),)
    matches = (False,) * empty_position + (terminal_match,)
    constraint = TrackingConstraint(
        supports,
        grammar_type=grammar_type,
        matches=matches,
    )
    selector = RecordingSelector([1] * empty_position)

    result, _rows, mask = _run(proposal, constraint, selector)

    assert result.decision_made is False
    assert result.fully_accepted is False
    assert result.accepted_count == empty_position
    assert result.replacement_token_id is None
    assert result.no_decision_selection.valid_token_ids is supports[-1]
    assert result.no_decision_selection.selected_token_id is None
    assert result.no_decision_selection.is_match is terminal_match
    assert result.committed_token_ids == proposal[:empty_position]
    assert len(mask.calls) == empty_position
    assert len(selector.calls) == empty_position
    assert len(constraint.advance_calls) == empty_position
    if empty_position == 0:
        assert result.committed_state is None
        assert result.committed_state_is_match is None
        assert result.committed_state_transferred is False
        assert constraint.release_calls == []
    else:
        assert result.committed_state is constraint.children[empty_position - 1]
        assert result.committed_state_is_match is terminal_match
        assert result.committed_state_transferred is True
        assert constraint.release_calls == constraint.children[: empty_position - 1]
        _release_result(constraint, result)
    assert constraint.live_children == []


def test_none_is_a_legal_transferred_child_and_controls_later_selection():
    second_child = OpaqueState("second")
    constraint = TrackingConstraint(
        ((1,), (2,), ()),
        children=(None, second_child),
        matches=(False, True, False),
    )
    result, _rows, _mask = _run((1, 3), constraint, RecordingSelector([1, 2]))

    assert constraint.advance_calls[0][0] is constraint.starting_state
    assert constraint.advance_calls[1][0] is None
    assert result.accepted_count == 1
    assert result.replacement_token_id == 2
    assert result.committed_state is second_child
    assert constraint.release_calls == [None]
    _release_result(constraint, result)


def test_d43_failure_after_an_accepted_token_releases_prefix_child_and_propagates_exactly():
    original = RuntimeError("selector failed")
    constraint = TrackingConstraint(((1,), (2,), ()))
    selector = RecordingSelector([1, original])

    with pytest.raises(RuntimeError) as captured:
        _run((1, 2), constraint, selector)

    assert captured.value is original
    assert constraint.release_calls == [constraint.children[0]]
    assert constraint.live_children == []
    assert len(selector.calls) == 2


def test_success_path_ancestor_release_failure_retries_once_in_cleanup():
    original = RuntimeError("ancestor release failed")
    constraint = TrackingConstraint(((1,), (2,), ()))
    constraint.release_outcomes[1] = [original, None]

    with pytest.raises(RuntimeError) as captured:
        _run((1, 2), constraint, RecordingSelector([1, 2]))

    assert captured.value is original
    assert constraint.release_calls == [
        constraint.children[0],
        constraint.children[0],
        constraint.children[1],
    ]
    assert constraint.live_children == []


def test_multiple_cleanup_failures_preserve_labels_order_identities_and_cause():
    original = RuntimeError("ancestor release failed")
    retry_failure = RuntimeError("ancestor retry failed")
    final_failure = RuntimeError("final release failed")
    constraint = TrackingConstraint(((1,), (2,), ()))
    constraint.release_outcomes[1] = [original, retry_failure]
    constraint.release_outcomes[2] = [final_failure]

    with pytest.raises(GrammarMaskedTargetAcceptanceCleanupError) as captured:
        _run((1, 2), constraint, RecordingSelector([1, 2]))

    error = captured.value
    assert error.original_failure is original
    assert error.__cause__ is original
    assert error.cleanup_failures == (
        ("target state release at position 0", retry_failure),
        ("target state release at position 1", final_failure),
    )
    assert type(error.cleanup_failures) is tuple
    assert all(type(entry) is tuple for entry in error.cleanup_failures)


def test_nested_d43_cleanup_error_remains_exact_without_d45_owned_state(monkeypatch):
    transition_failure = RuntimeError("transition failed")
    transition_cleanup = RuntimeError("transition cleanup failed")
    nested = GrammarMaskedTransitionCleanupError(
        transition_failure,
        (("child state release", transition_cleanup),),
    )
    monkeypatch.setattr(
        grammar_acceptance_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(nested),
    )
    constraint = TrackingConstraint(((1,), ()))

    with pytest.raises(GrammarMaskedTransitionCleanupError) as captured:
        _run((1,), constraint, RecordingSelector([1]))

    assert captured.value is nested
    assert constraint.release_calls == []


def test_result_construction_failure_releases_untransferred_final_child(monkeypatch):
    original = RuntimeError("result construction failed")

    class FailingResult:
        def __init__(self, **kwargs):
            raise original

    monkeypatch.setattr(
        grammar_acceptance_module,
        "GrammarMaskedTargetAcceptanceResult",
        FailingResult,
    )
    constraint = TrackingConstraint(((1,), ()))

    with pytest.raises(RuntimeError) as captured:
        _run((1,), constraint, RecordingSelector([1]))

    assert captured.value is original
    assert constraint.release_calls == [constraint.children[0]]
    assert constraint.live_children == []


def test_transferred_child_is_owned_before_malformed_evidence_is_rejected(monkeypatch):
    constraint = TrackingConstraint(((1,), ()))
    child = constraint.children[0]
    constraint.live_children.append(child)
    selection = GrammarMaskedSelectionResult((1,), False, 1)
    transition = object.__new__(GrammarMaskedTransitionResult)
    object.__setattr__(transition, "selection", selection)
    object.__setattr__(transition, "child_state", child)
    object.__setattr__(transition, "child_is_match", "not-a-boolean")
    monkeypatch.setattr(
        grammar_acceptance_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )

    with pytest.raises(
        GrammarMaskedTargetAcceptanceInvariantError,
        match="boolean child_is_match",
    ):
        _run((1,), constraint, RecordingSelector([1]))

    assert constraint.release_calls == [child]
    assert constraint.live_children == []


def test_retained_final_state_query_failure_releases_untransferred_child():
    original = RuntimeError("retained child query failed")

    class FailingRetainedQueryConstraint(TrackingConstraint):
        def __init__(self):
            super().__init__(((1,), ()))
            self.child_dead_queries = 0

        def is_dead_state(self, state, /):
            if state is self.children[0]:
                self.child_dead_queries += 1
                if self.child_dead_queries == 2:
                    raise original
            return super().is_dead_state(state)

    constraint = FailingRetainedQueryConstraint()

    with pytest.raises(RuntimeError) as captured:
        _run((1,), constraint, RecordingSelector([1]))

    assert captured.value is original
    assert constraint.release_calls == [constraint.children[0]]
    assert constraint.live_children == []


@pytest.mark.parametrize("alias_kind", ["start", "parent", "earlier"])
def test_child_aliases_are_rejected_by_identity_without_double_release(
    monkeypatch,
    alias_kind,
):
    constraint = TrackingConstraint(((1,), (1,), (1,), ()))
    calls = 0
    created = []

    def transition(_constraint, parent, *args, **kwargs):
        nonlocal calls
        selection = GrammarMaskedSelectionResult((1,), False, 1)
        if calls == 0 and alias_kind == "start":
            child = constraint.starting_state
        elif calls > 0 and alias_kind == "parent":
            child = parent
        elif calls == 2 and alias_kind == "earlier":
            child = created[0]
        else:
            child = constraint.children[calls]
            constraint.live_children.append(child)
            created.append(child)
        calls += 1
        return GrammarMaskedTransitionResult(selection, child, False)

    monkeypatch.setattr(
        grammar_acceptance_module,
        "select_and_advance_grammar_state",
        transition,
    )
    length = {"start": 1, "parent": 2, "earlier": 3}[alias_kind]

    with pytest.raises(GrammarMaskedTargetAcceptanceInvariantError, match="aliases"):
        _run((1,) * length, constraint, RecordingSelector([1] * length))

    assert all(state is not constraint.starting_state for state in constraint.release_calls)
    assert len({id(state) for state in constraint.release_calls}) == len(
        constraint.release_calls
    )


def test_cleanup_error_constructor_preserves_exact_immutable_evidence():
    original = RuntimeError("original")
    cleanup = RuntimeError("cleanup")
    failures = [("target state release at position 0", cleanup)]
    error = GrammarMaskedTargetAcceptanceCleanupError(original, failures)
    failures.clear()

    assert error.original_failure is original
    assert error.cleanup_failures == (("target state release at position 0", cleanup),)
    assert error.cleanup_failures[0][1] is cleanup
    assert error.__cause__ is original
    with pytest.raises(ValueError, match="cannot be empty"):
        GrammarMaskedTargetAcceptanceCleanupError(original, ())


def test_success_result_does_not_retain_rows_mask_selector_or_released_ancestors():
    proposal = (1, 2)
    constraint = TrackingConstraint(((1,), (2,), ()))
    rows = tuple(OpaqueRow(f"row-{position}") for position in range(3))
    mask = RecordingMask()
    selector = RecordingSelector([1, 2])
    first_child = constraint.children[0]

    result, _, _ = _run(
        proposal,
        constraint,
        selector,
        rows=rows,
        mask=mask,
    )
    references = [
        weakref.ref(value)
        for value in (*rows, mask, selector, first_child, *mask.masked_rows)
    ]
    _release_result(constraint, result)
    del constraint, rows, mask, selector, first_child, _
    gc.collect()

    assert all(reference() is None for reference in references)
    assert result.proposal_token_ids is proposal


def test_repeated_outcomes_leave_bounded_state_ownership():
    start = OpaqueState("stable-start")
    for iteration in range(1000):
        if iteration % 3 == 0:
            constraint = TrackingConstraint(((1, 2), ()))
            selector = RecordingSelector([2])
            proposal = (1,)
        elif iteration % 3 == 1:
            constraint = TrackingConstraint(((1,), ()))
            selector = RecordingSelector([1])
            proposal = (1,)
        else:
            constraint = TrackingConstraint(((),), matches=(True,))
            selector = RecordingSelector([])
            proposal = (1,)
        constraint.starting_state = start
        result, _rows, _mask = _run(proposal, constraint, selector)
        assert constraint.peak_live_children <= 1
        _release_result(constraint, result)
        assert constraint.live_children == []
        assert constraint.is_dead_state(start) is False


def test_genuine_d44_d30_and_d45_evidence_compose_without_cache_mutation():
    script = tuple(
        tuple(float(row * VOCAB_SIZE + token) for token in range(VOCAB_SIZE))
        for row in range(12)
    )
    draft_backend = FakeAutoregressiveBackend(script, model_id="fake-draft")
    target_backend = FakeAutoregressiveBackend(script, model_id="fake-target")
    draft_backend.prefill((4,))
    target_backend.prefill((4,))
    draft_root = draft_backend.create_cache_checkpoint()
    target_root = target_backend.create_cache_checkpoint()
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "s1"), ("s1", 2, "s2")),
        valid_token_ids=(("s0", (1,)), ("s1", (2,)), ("s2", ())),
        match_states=frozenset({"s2"}),
    )
    constraint = FakeGrammarConstraint(
        tuple(bytes((token,)) for token in range(VOCAB_SIZE)),
        grammar_type="regex",
        program=program,
    )
    starting_state = constraint.init_state()
    draft_result = generate_grammar_masked_draft_proposal(
        draft_backend,
        0,
        constraint,
        starting_state,
        RecordingMask(),
        proposal_bound=2,
        select_token=RecordingSelector([1, 2]),
    )
    verification = target_backend.verify_proposal(0, draft_result.proposal_token_ids)
    draft_snapshot = (
        draft_backend.cache_length,
        draft_backend.cached_token_ids,
        draft_backend._next_row,
        tuple(draft_backend._cache_checkpoints),
    )
    target_snapshot = (
        target_backend.cache_length,
        target_backend.cached_token_ids,
        target_backend._next_row,
        tuple(target_backend._cache_checkpoints),
    )
    checkpoint_identities = tuple(id(value) for value in draft_result.rollback_checkpoints)
    next_state_id = constraint._next_state_id

    result = decide_grammar_masked_target_acceptance(
        draft_result.proposal_token_ids,
        verification.logit_rows,
        constraint,
        starting_state,
        RecordingMask(),
        vocab_size=VOCAB_SIZE,
        select_token=RecordingSelector([1, 2]),
    )

    assert result.proposal_token_ids is draft_result.proposal_token_ids
    assert result.fully_accepted is True
    assert result.committed_token_ids == (1, 2)
    assert result.committed_state.value >= next_state_id
    assert constraint.active_state_count == 2
    assert draft_snapshot == (
        draft_backend.cache_length,
        draft_backend.cached_token_ids,
        draft_backend._next_row,
        tuple(draft_backend._cache_checkpoints),
    )
    assert target_snapshot == (
        target_backend.cache_length,
        target_backend.cached_token_ids,
        target_backend._next_row,
        tuple(target_backend._cache_checkpoints),
    )
    assert checkpoint_identities == tuple(
        id(value) for value in draft_result.rollback_checkpoints
    )

    constraint.release_state(result.committed_state)
    constraint.release_state(starting_state)
    for checkpoint in draft_result.rollback_checkpoints:
        draft_backend.release_cache_checkpoint(checkpoint)
    draft_backend.release_cache_checkpoint(draft_root)
    target_backend.release_cache_checkpoint(target_root)
    assert constraint.active_state_count == 0


def test_reused_selector_continues_while_fresh_selector_replays():
    shared = RecordingSelector([1, 2])
    first_constraint = TrackingConstraint(((1, 2), ()))
    first, _rows, _mask = _run((1,), first_constraint, shared)
    assert first.fully_accepted is True
    _release_result(first_constraint, first)

    second_constraint = TrackingConstraint(((1, 2), ()))
    second, _rows, _mask = _run((1,), second_constraint, shared)
    assert second.replacement_token_id == 2
    _release_result(second_constraint, second)

    replay_constraint = TrackingConstraint(((1, 2), ()))
    replay, _rows, _mask = _run((1,), replay_constraint, RecordingSelector([1, 2]))
    assert replay.fully_accepted is True
    _release_result(replay_constraint, replay)
    assert len(shared.calls) == 2


def test_module_has_no_mutable_state_registry():
    allowed_mutable_names = {"__all__", "__builtins__"}
    mutable_globals = {
        name
        for name, value in vars(grammar_acceptance_module).items()
        if isinstance(value, (dict, list, set)) and name not in allowed_mutable_names
    }
    assert mutable_globals == set()


def test_normal_and_isolated_imports_remain_optional_runtime_free():
    imported = importlib.import_module("onyx_cuda.grammar_acceptance")
    assert imported.decide_grammar_masked_target_acceptance is (
        decide_grammar_masked_target_acceptance
    )
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        import onyx_cuda
        assert onyx_cuda.decide_grammar_masked_target_acceptance.__module__ == (
            "onyx_cuda.grammar_acceptance"
        )
        forbidden = (
            "onyx", "mlx", "torch", "transformers", "tokenizers",
            "huggingface_hub", "bitsandbytes", "accelerate", "onnxruntime", "psutil",
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in forbidden
        )
        assert "onyx_cuda._grammar_native" not in sys.modules
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
