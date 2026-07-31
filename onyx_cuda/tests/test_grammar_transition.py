import gc
import importlib
import inspect
import math
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_transition as transition_module
from onyx_cuda import (
    GrammarError,
    GrammarMaskedSelectionResult,
    GrammarMaskedTransitionCleanupError,
    GrammarMaskedTransitionError,
    GrammarMaskedTransitionInvariantError,
    GrammarMaskedTransitionResult,
    GrammarStateError,
    TemperatureTopPSelection,
    create_reference_sampler,
    select_and_advance_grammar_state,
)
from onyx_cuda.testing import FakeGrammarConstraint, FakeGrammarProgram


VOCAB_SIZE = 5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class OpaqueState:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        raise AssertionError("D43 must compare grammar states only by identity")

    def __hash__(self):
        raise AssertionError("D43 must not hash opaque grammar states")


class OpaqueRow:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name):
        self.name = name

    def __iter__(self):
        raise AssertionError("D43 must not inspect backend-native logits rows")

    def __eq__(self, other):
        raise AssertionError("D43 must not compare backend-native logits rows")


def _next_outcome(outcomes):
    outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


class RecordingConstraint:
    def __init__(
        self,
        *,
        grammar_type="regex",
        support=(1, 3),
        parent=None,
        child=None,
        events=None,
    ):
        self._vocab_size = VOCAB_SIZE
        self._grammar_type = grammar_type
        self.support = support
        self.parent = parent if parent is not None else OpaqueState("parent")
        self.child = child if child is not None else OpaqueState("child")
        self.events = events if events is not None else []
        self.parent_dead_outcomes = [False, False, False]
        self.parent_match_outcomes = [False, False, False]
        self.child_dead_outcomes = [False]
        self.child_match_outcomes = [False]
        self.support_outcomes = [support]
        self.advance_outcomes = [self.child]
        self.release_outcomes = [None]
        self.advance_calls = []
        self.release_calls = []
        self.bulk_release_calls = []
        self.reset_calls = 0
        self.init_calls = 0

    @property
    def vocab_size(self):
        self.events.append("constraint.vocab_size")
        return self._vocab_size

    @property
    def grammar_type(self):
        self.events.append("constraint.grammar_type")
        return self._grammar_type

    def init_state(self):
        self.init_calls += 1
        raise AssertionError("D43 must not initialize grammar state")

    def advance_state(self, state, token_id, /):
        self.events.append(("advance_state", state, token_id))
        self.advance_calls.append((state, token_id))
        return _next_outcome(self.advance_outcomes)

    def get_valid_token_ids(self, state, /):
        self.events.append(("get_valid_token_ids", state))
        return _next_outcome(self.support_outcomes)

    def is_match_state(self, state, /):
        self.events.append(("is_match_state", state))
        outcomes = (
            self.parent_match_outcomes
            if state is self.parent
            else self.child_match_outcomes
        )
        return _next_outcome(outcomes)

    def is_dead_state(self, state, /):
        self.events.append(("is_dead_state", state))
        outcomes = (
            self.parent_dead_outcomes if state is self.parent else self.child_dead_outcomes
        )
        return _next_outcome(outcomes)

    def release_state(self, state, /):
        self.events.append(("release_state", state))
        outcome = _next_outcome(self.release_outcomes)
        self.release_calls.append(state)
        return outcome

    def release_states(self, states, /):
        self.bulk_release_calls.append(states)
        raise AssertionError("D43 must not bulk-release grammar states")

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("D43 must not reset the grammar constraint")


class RecordingMask:
    def __init__(self, *, masked_row=None, events=None):
        self._vocab_size = VOCAB_SIZE
        self.masked_row = masked_row if masked_row is not None else OpaqueRow("masked")
        self.events = events if events is not None else []
        self.calls = []

    @property
    def vocab_size(self):
        self.events.append("mask.vocab_size")
        return self._vocab_size

    def apply(self, logits, valid_token_ids, /):
        self.events.append("mask.apply")
        self.calls.append((logits, valid_token_ids))
        return self.masked_row


class RecordingSelector:
    def __init__(self, outcomes, *, events=None):
        self.outcomes = list(outcomes)
        self.events = events if events is not None else []
        self.calls = []

    def __call__(self, logits):
        self.events.append("select_token")
        self.calls.append(logits)
        return _next_outcome(self.outcomes)


def _run(
    constraint,
    *,
    state=None,
    row=None,
    mask=None,
    selector=None,
    vocab_size=VOCAB_SIZE,
):
    return select_and_advance_grammar_state(
        constraint,
        constraint.parent if state is None else state,
        OpaqueRow("input") if row is None else row,
        RecordingMask() if mask is None else mask,
        vocab_size=vocab_size,
        select_token=RecordingSelector([1]) if selector is None else selector,
    )


def _forged_selection(valid_token_ids, is_match, selected_token_id):
    result = object.__new__(GrammarMaskedSelectionResult)
    object.__setattr__(result, "valid_token_ids", valid_token_ids)
    object.__setattr__(result, "is_match", is_match)
    object.__setattr__(result, "selected_token_id", selected_token_id)
    return result


def _fake_constraint(
    *,
    grammar_type,
    support=(1, 3),
    child_match=False,
    constraint_class=FakeGrammarConstraint,
):
    return constraint_class(
        tuple(bytes((token_id,)) for token_id in range(VOCAB_SIZE)),
        grammar_type=grammar_type,
        program=FakeGrammarProgram(
            initial_state="parent",
            transitions=tuple(
                [
                    *(("parent", token_id, "child") for token_id in support),
                    *(("child", token_id, "child") for token_id in support),
                ]
            ),
            valid_token_ids=(("parent", support), ("child", support)),
            match_states=frozenset({"child"} if child_match else {"parent"}),
        ),
    )


def test_public_surface_signature_result_and_hierarchy():
    assert transition_module.__all__ == [
        "GrammarMaskedTransitionCleanupError",
        "GrammarMaskedTransitionError",
        "GrammarMaskedTransitionInvariantError",
        "GrammarMaskedTransitionResult",
        "select_and_advance_grammar_state",
    ]
    assert issubclass(GrammarMaskedTransitionError, GrammarError)
    assert issubclass(
        GrammarMaskedTransitionInvariantError,
        GrammarMaskedTransitionError,
    )
    assert issubclass(
        GrammarMaskedTransitionCleanupError,
        GrammarMaskedTransitionError,
    )
    signature = inspect.signature(select_and_advance_grammar_state)
    assert tuple(signature.parameters) == (
        "constraint",
        "state",
        "logits",
        "logit_mask",
        "vocab_size",
        "select_token",
    )
    for name in ("constraint", "state", "logits", "logit_mask"):
        assert signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("vocab_size", "select_token"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    assert [field.name for field in fields(GrammarMaskedTransitionResult)] == [
        "selection",
        "child_state",
        "child_is_match",
    ]

    empty_selection = GrammarMaskedSelectionResult((), True, None)
    empty = GrammarMaskedTransitionResult(empty_selection, None, None)
    child = OpaqueState("child")
    selected = GrammarMaskedTransitionResult(
        GrammarMaskedSelectionResult((1, 3), False, 3),
        child,
        True,
    )
    none_child = GrammarMaskedTransitionResult(
        GrammarMaskedSelectionResult((1,), False, 1),
        None,
        False,
    )
    assert empty.transitioned is False
    assert selected.transitioned is True
    assert selected.child_state is child
    assert none_child.transitioned is True
    assert none_child.child_state is None
    with pytest.raises(FrozenInstanceError):
        selected.child_state = None
    assert not hasattr(selected, "__dict__")


@pytest.mark.parametrize(
    "selection,child,child_is_match,error",
    [
        (object(), None, None, TypeError),
        (_forged_selection([], False, None), None, None, TypeError),
        (_forged_selection((True,), False, True), None, False, TypeError),
        (_forged_selection((1, 1), False, 1), None, False, ValueError),
        (_forged_selection((), 0, None), None, None, TypeError),
        (_forged_selection((), False, 1), None, None, ValueError),
        (_forged_selection((1,), False, None), None, False, TypeError),
        (_forged_selection((1,), False, 2), None, False, ValueError),
        (GrammarMaskedSelectionResult((), False, None), OpaqueState("child"), None, ValueError),
        (GrammarMaskedSelectionResult((1,), False, 1), OpaqueState("child"), None, TypeError),
    ],
)
def test_direct_result_rejects_malformed_nested_evidence(
    selection,
    child,
    child_is_match,
    error,
):
    with pytest.raises(error):
        GrammarMaskedTransitionResult(selection, child, child_is_match)


def test_cleanup_error_retains_exact_immutable_evidence_and_cause():
    original = RuntimeError("transition failed")
    release_failure = RuntimeError("release failed")
    failures = [("child state release", release_failure)]

    error = GrammarMaskedTransitionCleanupError(original, failures)
    failures.clear()

    assert error.original_failure is original
    assert error.cleanup_failures == (("child state release", release_failure),)
    assert type(error.cleanup_failures) is tuple
    assert error.cleanup_failures[0][1] is release_failure
    assert error.__cause__ is original
    with pytest.raises(ValueError, match="cannot be empty"):
        GrammarMaskedTransitionCleanupError(original, ())


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("parent_match", [False, True])
@pytest.mark.parametrize("support,selected", [((1,), 1), ((0, 2, 4), 2)])
@pytest.mark.parametrize("child_match", [False, True])
def test_successful_transition_has_exact_order_identity_and_transfer(
    grammar_type,
    parent_match,
    support,
    selected,
    child_match,
):
    events = []
    parent = OpaqueState("parent")
    child = OpaqueState("child")
    row = OpaqueRow("input")
    masked_row = OpaqueRow("masked")
    constraint = RecordingConstraint(
        grammar_type=grammar_type,
        support=support,
        parent=parent,
        child=child,
        events=events,
    )
    constraint.parent_match_outcomes = [parent_match, parent_match, parent_match]
    constraint.child_match_outcomes = [child_match]
    mask = RecordingMask(masked_row=masked_row, events=events)
    selector = RecordingSelector([selected], events=events)

    result = _run(
        constraint,
        state=parent,
        row=row,
        mask=mask,
        selector=selector,
    )

    assert events == [
        "constraint.vocab_size",
        "constraint.grammar_type",
        "mask.vocab_size",
        ("is_dead_state", parent),
        ("is_match_state", parent),
        ("get_valid_token_ids", parent),
        "mask.apply",
        "select_token",
        ("is_dead_state", parent),
        ("is_match_state", parent),
        ("advance_state", parent, selected),
        ("is_dead_state", child),
        ("is_match_state", child),
        ("is_dead_state", parent),
        ("is_match_state", parent),
    ]
    assert len(mask.calls) == 1
    assert mask.calls[0][0] is row
    assert mask.calls[0][1] is support
    assert selector.calls == [masked_row]
    assert constraint.advance_calls == [(parent, selected)]
    assert constraint.release_calls == []
    assert result.selection.valid_token_ids is support
    assert result.child_state is child
    assert result.child_is_match is child_match
    assert result.transitioned is True
    assert constraint.bulk_release_calls == []
    assert constraint.reset_calls == 0


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("parent_match", [False, True])
def test_empty_support_preserves_parent_without_transition_or_release(
    grammar_type,
    parent_match,
):
    events = []
    constraint = RecordingConstraint(
        grammar_type=grammar_type,
        support=(),
        events=events,
    )
    constraint.parent_match_outcomes = [parent_match, parent_match]
    mask = RecordingMask(events=events)
    selector = RecordingSelector([1], events=events)

    result = _run(constraint, mask=mask, selector=selector)

    assert events == [
        "constraint.vocab_size",
        "constraint.grammar_type",
        "mask.vocab_size",
        ("is_dead_state", constraint.parent),
        ("is_match_state", constraint.parent),
        ("get_valid_token_ids", constraint.parent),
        ("is_dead_state", constraint.parent),
        ("is_match_state", constraint.parent),
    ]
    assert result.selection.valid_token_ids is constraint.support
    assert result.selection.is_match is parent_match
    assert result.child_state is None
    assert result.child_is_match is None
    assert result.transitioned is False
    assert mask.calls == []
    assert selector.calls == []
    assert constraint.advance_calls == []
    assert constraint.release_calls == []


def test_d42_is_called_once_with_exact_arguments_and_result_is_retained(monkeypatch):
    constraint = RecordingConstraint()
    state = constraint.parent
    row = OpaqueRow("input")
    mask = RecordingMask()
    selector = RecordingSelector([3])
    selection = GrammarMaskedSelectionResult((1, 3), False, 3)
    calls = []

    def fake_select(*args, **kwargs):
        calls.append((args, kwargs))
        return selection

    monkeypatch.setattr(transition_module, "select_grammar_masked_token", fake_select)
    result = _run(
        constraint,
        state=state,
        row=row,
        mask=mask,
        selector=selector,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (constraint, state, row, mask)
    assert kwargs == {"vocab_size": VOCAB_SIZE, "select_token": selector}
    assert result.selection is selection
    assert result.selection.valid_token_ids is selection.valid_token_ids


def test_d42_failure_propagates_exactly_without_state_or_cleanup(monkeypatch):
    failure = RuntimeError("D42 failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(transition_module, "select_grammar_masked_token", fail)
    constraint = RecordingConstraint()

    with pytest.raises(RuntimeError) as captured:
        _run(constraint)

    assert captured.value is failure
    assert constraint.events == []
    assert constraint.advance_calls == []
    assert constraint.release_calls == []


@pytest.mark.parametrize(
    "selection",
    [
        object(),
        _forged_selection([1], False, 1),
        _forged_selection((True,), False, True),
        _forged_selection((1, 1), False, 1),
        _forged_selection((VOCAB_SIZE,), False, VOCAB_SIZE),
        _forged_selection((), 0, None),
        _forged_selection((), False, 1),
        _forged_selection((1,), False, None),
        _forged_selection((1,), False, 2),
    ],
)
def test_malformed_d42_evidence_fails_before_parent_or_transition(
    monkeypatch,
    selection,
):
    monkeypatch.setattr(
        transition_module,
        "select_grammar_masked_token",
        lambda *args, **kwargs: selection,
    )
    constraint = RecordingConstraint()

    with pytest.raises(GrammarMaskedTransitionInvariantError):
        _run(constraint)

    assert not any(
        isinstance(event, tuple) and event[0].startswith("is_")
        for event in constraint.events
    )
    assert constraint.advance_calls == []
    assert constraint.release_calls == []


@pytest.mark.parametrize(
    "dead_outcomes,match_outcomes",
    [
        ([False, True], [False]),
        ([False, 0], [False]),
        ([False, False], [False, True]),
        ([False, False], [False, 0]),
    ],
)
def test_post_selection_parent_revalidation_fails_without_child_cleanup(
    dead_outcomes,
    match_outcomes,
):
    constraint = RecordingConstraint()
    constraint.parent_dead_outcomes = dead_outcomes
    constraint.parent_match_outcomes = match_outcomes

    with pytest.raises(GrammarMaskedTransitionInvariantError):
        _run(constraint)

    assert constraint.advance_calls == []
    assert constraint.release_calls == []


@pytest.mark.parametrize(
    "failure",
    [GrammarStateError("invalid JSON transition"), RuntimeError("advance failed")],
)
def test_transition_failure_before_return_propagates_without_cleanup(failure):
    constraint = RecordingConstraint(grammar_type="json_schema")
    constraint.advance_outcomes = [failure]

    with pytest.raises(type(failure)) as captured:
        _run(constraint)

    assert captured.value is failure
    assert constraint.advance_calls == [(constraint.parent, 1)]
    assert constraint.release_calls == []


def test_parent_alias_is_rejected_without_equality_hashing_or_release():
    parent = OpaqueState("parent")
    constraint = RecordingConstraint(parent=parent, child=parent)

    with pytest.raises(
        GrammarMaskedTransitionInvariantError,
        match="independent child",
    ):
        _run(constraint)

    assert constraint.advance_calls == [(parent, 1)]
    assert constraint.release_calls == []


@pytest.mark.parametrize(
    "child_dead_outcomes,child_match_outcomes",
    [
        ([True], [False]),
        ([0], [False]),
        ([False], [0]),
        ([RuntimeError("dead query failed")], [False]),
        ([False], [RuntimeError("match query failed")]),
    ],
)
def test_malformed_child_or_child_query_failure_releases_owned_child(
    child_dead_outcomes,
    child_match_outcomes,
):
    constraint = RecordingConstraint()
    constraint.child_dead_outcomes = child_dead_outcomes
    constraint.child_match_outcomes = child_match_outcomes

    with pytest.raises(
        (
            GrammarMaskedTransitionInvariantError,
            RuntimeError,
        )
    ):
        _run(constraint)

    assert constraint.release_calls == [constraint.child]
    assert constraint.advance_calls == [(constraint.parent, 1)]


@pytest.mark.parametrize(
    "dead_outcomes,match_outcomes",
    [
        ([False, False, True], [False, False]),
        ([False, False, 0], [False, False]),
        ([False, False, False], [False, False, True]),
        ([False, False, False], [False, False, 0]),
        ([False, False, RuntimeError("parent query failed")], [False, False]),
    ],
)
def test_post_transition_parent_failure_releases_child(
    dead_outcomes,
    match_outcomes,
):
    constraint = RecordingConstraint()
    constraint.parent_dead_outcomes = dead_outcomes
    constraint.parent_match_outcomes = match_outcomes

    with pytest.raises(
        (
            GrammarMaskedTransitionInvariantError,
            RuntimeError,
        )
    ):
        _run(constraint)

    assert constraint.release_calls == [constraint.child]


@pytest.mark.parametrize("record_before_raise", [False, True])
def test_cleanup_failure_aggregates_exact_evidence(record_before_raise):
    original = RuntimeError("child match query failed")
    release_failure = RuntimeError("child release failed")

    class CleanupFailingConstraint(RecordingConstraint):
        def release_state(self, state, /):
            self.events.append(("release_state", state))
            if record_before_raise:
                self.release_calls.append(state)
            raise release_failure

    constraint = CleanupFailingConstraint()
    constraint.child_match_outcomes = [original]

    with pytest.raises(GrammarMaskedTransitionCleanupError) as captured:
        _run(constraint)

    error = captured.value
    assert error.original_failure is original
    assert error.cleanup_failures == (("child state release", release_failure),)
    assert error.cleanup_failures[0][1] is release_failure
    assert error.__cause__ is original
    assert constraint.release_calls == ([constraint.child] if record_before_raise else [])
    assert len(
        [event for event in constraint.events if event == ("release_state", constraint.child)]
    ) == 1


def test_selected_result_construction_failure_releases_child(monkeypatch):
    failure = RuntimeError("result construction failed")

    class FailingResult:
        def __init__(self, *args, **kwargs):
            raise failure

    monkeypatch.setattr(transition_module, "GrammarMaskedTransitionResult", FailingResult)
    constraint = RecordingConstraint()

    with pytest.raises(RuntimeError) as captured:
        _run(constraint)

    assert captured.value is failure
    assert constraint.release_calls == [constraint.child]


def test_empty_result_construction_failure_has_no_cleanup(monkeypatch):
    failure = RuntimeError("result construction failed")

    class FailingResult:
        def __init__(self, *args, **kwargs):
            raise failure

    monkeypatch.setattr(transition_module, "GrammarMaskedTransitionResult", FailingResult)
    constraint = RecordingConstraint(support=())

    with pytest.raises(RuntimeError) as captured:
        _run(constraint)

    assert captured.value is failure
    assert constraint.advance_calls == []
    assert constraint.release_calls == []


def test_malformed_selected_result_composition_releases_child(monkeypatch):
    class WrongResult:
        def __init__(self, selection, child_state, child_is_match):
            self.selection = selection
            self.child_state = OpaqueState("wrong")
            self.child_is_match = child_is_match
            self.transitioned = True

    monkeypatch.setattr(transition_module, "GrammarMaskedTransitionResult", WrongResult)
    constraint = RecordingConstraint()

    with pytest.raises(
        GrammarMaskedTransitionInvariantError,
        match="exact child state",
    ):
        _run(constraint)

    assert constraint.release_calls == [constraint.child]


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("child_match", [False, True])
def test_fake_grammar_success_transfers_independent_caller_owned_child(
    grammar_type,
    child_match,
):
    constraint = _fake_constraint(
        grammar_type=grammar_type,
        child_match=child_match,
    )
    parent = constraint.init_state()

    result = select_and_advance_grammar_state(
        constraint,
        parent,
        OpaqueRow("input"),
        RecordingMask(),
        vocab_size=VOCAB_SIZE,
        select_token=RecordingSelector([3]),
    )

    assert result.child_state is not parent
    assert result.child_is_match is child_match
    assert constraint.is_dead_state(parent) is False
    assert constraint.active_state_count == 2
    constraint.release_state(result.child_state)
    assert constraint.active_state_count == 1
    constraint.release_state(parent)
    assert constraint.active_state_count == 0


def test_stateful_selector_consumption_continues_after_transition_failure():
    transition_failure = GrammarStateError("transition failed")
    constraint = RecordingConstraint(grammar_type="json_schema")
    constraint.advance_outcomes = [transition_failure, constraint.child]
    selector = RecordingSelector([1, 3])

    with pytest.raises(GrammarStateError) as captured:
        _run(constraint, selector=selector)
    result = _run(constraint, selector=selector)

    assert captured.value is transition_failure
    assert len(selector.calls) == 2
    assert result.selection.selected_token_id == 3
    assert constraint.advance_calls == [
        (constraint.parent, 1),
        (constraint.parent, 3),
    ]


def test_fresh_same_seed_selector_sessions_replay_through_transition():
    policy = TemperatureTopPSelection(temperature=0.8, top_p=1.0, seed=42)
    first = create_reference_sampler(policy)
    replay = create_reference_sampler(policy)
    constraint = _fake_constraint(
        grammar_type="regex",
        support=(0, 1, 3),
    )
    parent = constraint.init_state()
    mask = RecordingMask(
        masked_row=(1.0, 2.0, -math.inf, 3.0, -math.inf),
    )

    def draw(selector):
        selected = []
        for _ in range(5):
            result = select_and_advance_grammar_state(
                constraint,
                parent,
                OpaqueRow("input"),
                mask,
                vocab_size=VOCAB_SIZE,
                select_token=selector,
            )
            selected.append(result.selection.selected_token_id)
            constraint.release_state(result.child_state)
        return tuple(selected)

    assert draw(first) == draw(replay)
    assert constraint.active_state_count == 1
    constraint.release_state(parent)


def test_selected_result_retains_no_borrowed_inputs():
    constraint = RecordingConstraint()
    parent = constraint.parent
    child = constraint.child
    row = OpaqueRow("input")
    masked = OpaqueRow("masked")
    mask = RecordingMask(masked_row=masked)
    selector = RecordingSelector([1])
    result = _run(
        constraint,
        state=parent,
        row=row,
        mask=mask,
        selector=selector,
    )
    references = [
        weakref.ref(value) for value in (constraint, parent, row, masked, mask, selector)
    ]
    mask.calls.clear()
    selector.calls.clear()
    constraint.events.clear()
    constraint.advance_calls.clear()
    constraint.release_calls.clear()
    constraint.parent = None
    constraint.child = None

    del constraint
    del parent
    del row
    del masked
    del mask
    del selector
    gc.collect()

    assert all(reference() is None for reference in references)
    assert result.child_state is child
    assert result.selection == GrammarMaskedSelectionResult((1, 3), False, 1)


def test_empty_result_retains_no_borrowed_inputs_or_state():
    constraint = RecordingConstraint(support=())
    parent = constraint.parent
    row = OpaqueRow("input")
    mask = RecordingMask()
    selector = RecordingSelector([1])
    result = _run(
        constraint,
        state=parent,
        row=row,
        mask=mask,
        selector=selector,
    )
    references = [
        weakref.ref(value) for value in (constraint, parent, row, mask, selector)
    ]
    constraint.events.clear()
    constraint.parent = None
    constraint.child = None

    del constraint
    del parent
    del row
    del mask
    del selector
    gc.collect()

    assert all(reference() is None for reference in references)
    assert result == GrammarMaskedTransitionResult(
        GrammarMaskedSelectionResult((), False, None),
        None,
        None,
    )


def test_one_thousand_transfer_and_caller_release_cycles_are_bounded():
    class CountingConstraint(FakeGrammarConstraint):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.advance_count = 0
            self.support_scan_count = 0
            self.release_count = 0

        def advance_state(self, state, token_id, /):
            self.advance_count += 1
            return super().advance_state(state, token_id)

        def get_valid_token_ids(self, state, /):
            self.support_scan_count += 1
            return super().get_valid_token_ids(state)

        def release_state(self, state, /):
            self.release_count += 1
            return super().release_state(state)

    constraint = _fake_constraint(
        grammar_type="regex",
        constraint_class=CountingConstraint,
    )
    parent = constraint.init_state()
    mask = RecordingMask()
    selector = RecordingSelector([3])

    for _ in range(1000):
        result = select_and_advance_grammar_state(
            constraint,
            parent,
            OpaqueRow("input"),
            mask,
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )
        assert constraint.active_state_count == 2
        constraint.release_state(result.child_state)
        assert constraint.active_state_count == 1

    assert len(mask.calls) == 1000
    assert len(selector.calls) == 1000
    assert constraint.support_scan_count == 1000
    assert constraint.advance_count == 1000
    assert constraint.release_count == 1000
    constraint.release_state(parent)
    assert constraint.release_count == 1001
    assert constraint.active_state_count == 0


def test_one_thousand_empty_support_calls_allocate_no_state():
    constraint = FakeGrammarConstraint(
        tuple(bytes((token_id,)) for token_id in range(VOCAB_SIZE)),
        grammar_type="json_schema",
        program=FakeGrammarProgram(
            initial_state="done",
            transitions=(),
            valid_token_ids=(("done", ()),),
            match_states=frozenset({"done"}),
        ),
    )
    parent = constraint.init_state()
    mask = RecordingMask()
    selector = RecordingSelector([1])

    for _ in range(1000):
        result = select_and_advance_grammar_state(
            constraint,
            parent,
            OpaqueRow("input"),
            mask,
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )
        assert result.transitioned is False
        assert constraint.active_state_count == 1

    assert mask.calls == []
    assert selector.calls == []
    constraint.release_state(parent)
    assert constraint.active_state_count == 0


def test_isolated_both_branch_execution_remains_optional_runtime_free():
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        from onyx_cuda import select_and_advance_grammar_state
        from onyx_cuda.testing import FakeGrammarConstraint, FakeGrammarProgram

        class Mask:
            vocab_size = 4
            def apply(self, logits, valid_token_ids, /):
                return tuple(
                    value if index in valid_token_ids else float("-inf")
                    for index, value in enumerate(logits)
                )

        vocabulary = (b"a", b"b", b"c", b"d")
        selected_constraint = FakeGrammarConstraint(
            vocabulary,
            grammar_type="regex",
            program=FakeGrammarProgram(
                initial_state="parent",
                transitions=(("parent", 1, "child"), ("parent", 3, "child")),
                valid_token_ids=(("parent", (1, 3)), ("child", ())),
                match_states=frozenset({{"child"}}),
            ),
        )
        empty_constraint = FakeGrammarConstraint(
            vocabulary,
            grammar_type="json_schema",
            program=FakeGrammarProgram(
                initial_state="done",
                transitions=(),
                valid_token_ids=(("done", ()),),
                match_states=frozenset({{"done"}}),
            ),
        )
        selected_parent = selected_constraint.init_state()
        empty_parent = empty_constraint.init_state()
        selected = select_and_advance_grammar_state(
            selected_constraint,
            selected_parent,
            (9.0, 3.0, 8.0, 5.0),
            Mask(),
            vocab_size=4,
            select_token=lambda row: max(range(len(row)), key=row.__getitem__),
        )
        empty = select_and_advance_grammar_state(
            empty_constraint,
            empty_parent,
            (0.0, 0.0, 0.0, 0.0),
            Mask(),
            vocab_size=4,
            select_token=lambda row: 0,
        )
        assert selected.transitioned is True
        assert selected.selection.selected_token_id == 3
        assert selected.child_is_match is True
        assert empty.transitioned is False
        assert empty.child_state is None
        assert selected_constraint.active_state_count == 2
        assert empty_constraint.active_state_count == 1
        selected_constraint.release_state(selected.child_state)
        selected_constraint.release_state(selected_parent)
        empty_constraint.release_state(empty_parent)
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
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_package_root_exports_are_same_objects():
    package = importlib.import_module("onyx_cuda")
    module = importlib.import_module("onyx_cuda.grammar_transition")
    for name in module.__all__:
        assert getattr(package, name) is getattr(module, name)
        assert name in package.__all__
