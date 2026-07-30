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

import onyx_cuda.grammar_selection as selection_module
from onyx_cuda import (
    GrammarError,
    GrammarMaskedSelectionError,
    GrammarMaskedSelectionInvariantError,
    GrammarMaskedSelectionResult,
    GrammarStateError,
    TemperatureTopPSelection,
    create_reference_sampler,
    select_grammar_masked_token,
)
from onyx_cuda.testing import FakeGrammarConstraint, FakeGrammarProgram


VOCAB_SIZE = 5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class State:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name="live"):
        self.name = name


class OpaqueRow:
    __slots__ = ("name", "__weakref__")

    def __init__(self, name):
        self.name = name

    def __iter__(self):
        raise AssertionError("D42 must not inspect a backend-native logits row")

    def __eq__(self, other):
        raise AssertionError("D42 must not compare a backend-native logits row")


class RecordingConstraint:
    def __init__(
        self,
        *,
        vocab_size=VOCAB_SIZE,
        grammar_type="regex",
        valid_token_ids=(1, 3),
        is_dead=False,
        is_match=False,
        events=None,
    ):
        self._vocab_size = vocab_size
        self._grammar_type = grammar_type
        self.valid_token_ids = valid_token_ids
        self.dead_result = is_dead
        self.match_result = is_match
        self.events = events if events is not None else []
        self.metadata_failures = {}
        self.query_failures = {}
        self.forbidden_calls = []

    @property
    def vocab_size(self):
        self.events.append("constraint.vocab_size")
        failure = self.metadata_failures.get("vocab_size")
        if failure is not None:
            raise failure
        return self._vocab_size

    @property
    def grammar_type(self):
        self.events.append("constraint.grammar_type")
        failure = self.metadata_failures.get("grammar_type")
        if failure is not None:
            raise failure
        return self._grammar_type

    def init_state(self):
        self.forbidden_calls.append("init_state")
        return State()

    def advance_state(self, state, token_id, /):
        self.forbidden_calls.append("advance_state")
        return State()

    def get_valid_token_ids(self, state, /):
        self.events.append(("get_valid_token_ids", state))
        failure = self.query_failures.get("get_valid_token_ids")
        if failure is not None:
            raise failure
        return self.valid_token_ids

    def is_match_state(self, state, /):
        self.events.append(("is_match_state", state))
        failure = self.query_failures.get("is_match_state")
        if failure is not None:
            raise failure
        return self.match_result

    def is_dead_state(self, state, /):
        self.events.append(("is_dead_state", state))
        failure = self.query_failures.get("is_dead_state")
        if failure is not None:
            raise failure
        return self.dead_result

    def release_state(self, state, /):
        self.forbidden_calls.append("release_state")

    def release_states(self, states, /):
        self.forbidden_calls.append("release_states")

    def reset(self):
        self.forbidden_calls.append("reset")


class RecordingMask:
    def __init__(
        self,
        *,
        vocab_size=VOCAB_SIZE,
        masked_logits=None,
        failure=None,
        events=None,
    ):
        self._vocab_size = vocab_size
        self.masked_logits = masked_logits if masked_logits is not None else OpaqueRow("masked")
        self.failure = failure
        self.events = events if events is not None else []
        self.calls = []
        self.timing_calls = 0
        self.metadata_failure = None

    @property
    def vocab_size(self):
        self.events.append("logit_mask.vocab_size")
        if self.metadata_failure is not None:
            raise self.metadata_failure
        return self._vocab_size

    def apply(self, logits, valid_token_ids, /):
        self.events.append("mask.apply")
        self.calls.append((logits, valid_token_ids))
        if self.failure is not None:
            raise self.failure
        return self.masked_logits

    def apply_with_timing(self, *args):
        self.timing_calls += 1
        raise AssertionError("D42 must use the ordinary apply() method")


class RecordingSelector:
    def __init__(self, selected_token_id, *, failure=None, events=None):
        self.selected_token_id = selected_token_id
        self.failure = failure
        self.events = events if events is not None else []
        self.calls = []

    def __call__(self, logits):
        self.events.append("select_token")
        self.calls.append(logits)
        if self.failure is not None:
            raise self.failure
        return self.selected_token_id


def _run(
    constraint,
    state,
    *,
    row=None,
    mask=None,
    selector=None,
    vocab_size=VOCAB_SIZE,
):
    return select_grammar_masked_token(
        constraint,
        state,
        row if row is not None else OpaqueRow("input"),
        mask if mask is not None else RecordingMask(),
        vocab_size=vocab_size,
        select_token=selector if selector is not None else RecordingSelector(1),
    )


def _state_events(events):
    return [
        event
        for event in events
        if isinstance(event, tuple)
        and event[0] in {"is_dead_state", "is_match_state", "get_valid_token_ids"}
    ]


def _fake_constraint(*, grammar_type="regex", valid_token_ids=(1, 3), match=False):
    transitions = tuple(("s", token_id, "s") for token_id in valid_token_ids)
    program = FakeGrammarProgram(
        initial_state="s",
        transitions=transitions,
        valid_token_ids=(("s", valid_token_ids),),
        match_states=frozenset({"s"} if match else ()),
    )
    return FakeGrammarConstraint(
        tuple(bytes((token_id,)) for token_id in range(VOCAB_SIZE)),
        grammar_type=grammar_type,
        program=program,
    )


def test_public_surface_signature_result_and_hierarchy():
    assert selection_module.__all__ == [
        "GrammarMaskedSelectionError",
        "GrammarMaskedSelectionInvariantError",
        "GrammarMaskedSelectionResult",
        "select_grammar_masked_token",
    ]
    assert issubclass(GrammarMaskedSelectionError, GrammarError)
    assert issubclass(
        GrammarMaskedSelectionInvariantError,
        GrammarMaskedSelectionError,
    )
    signature = inspect.signature(select_grammar_masked_token)
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
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is inspect.Parameter.empty
    assert [field.name for field in fields(GrammarMaskedSelectionResult)] == [
        "valid_token_ids",
        "is_match",
        "selected_token_id",
    ]

    empty = GrammarMaskedSelectionResult((), False, None)
    selected = GrammarMaskedSelectionResult((1, 3), True, 3)
    assert empty == GrammarMaskedSelectionResult((), False, None)
    assert selected.valid_token_ids == (1, 3)
    assert selected.is_match is True
    assert selected.selected_token_id == 3
    with pytest.raises(FrozenInstanceError):
        selected.selected_token_id = 1
    assert not hasattr(selected, "__dict__")


class TupleSubclass(tuple):
    pass


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "valid_token_ids",
    [[1], TupleSubclass((1,)), (True,), (IntSubclass(1),)],
)
def test_direct_result_rejects_nonexact_support_types(valid_token_ids):
    with pytest.raises(TypeError):
        GrammarMaskedSelectionResult(valid_token_ids, False, 1)


@pytest.mark.parametrize("valid_token_ids", [(-1,), (1, 1), (2, 1)])
def test_direct_result_rejects_invalid_support_values(valid_token_ids):
    with pytest.raises(ValueError):
        GrammarMaskedSelectionResult(valid_token_ids, False, valid_token_ids[-1])


@pytest.mark.parametrize("selected_token_id", [None, True, 1.0, IntSubclass(1)])
def test_direct_result_requires_exact_integer_selection_for_nonempty_support(
    selected_token_id,
):
    with pytest.raises(TypeError):
        GrammarMaskedSelectionResult((1,), False, selected_token_id)


@pytest.mark.parametrize(
    "valid_token_ids,selected_token_id",
    [((), 0), ((1,), -1), ((1, 3), 2)],
)
def test_direct_result_rejects_selection_inconsistent_with_support(
    valid_token_ids,
    selected_token_id,
):
    with pytest.raises(ValueError):
        GrammarMaskedSelectionResult(valid_token_ids, False, selected_token_id)


@pytest.mark.parametrize("is_match", [0, 1, None, "yes"])
def test_direct_result_requires_exact_match_boolean(is_match):
    with pytest.raises(TypeError, match="is_match"):
        GrammarMaskedSelectionResult((), is_match, None)


@pytest.mark.parametrize("vocab_size", [True, False, 1.0, "5", None, IntSubclass(5)])
def test_invalid_explicit_vocab_type_fails_before_component_or_state_work(vocab_size):
    events = []
    constraint = RecordingConstraint(events=events)
    mask = RecordingMask(events=events)

    with pytest.raises(TypeError, match="vocab_size"):
        _run(constraint, State(), mask=mask, vocab_size=vocab_size)

    assert events == []
    assert mask.calls == []
    assert constraint.forbidden_calls == []


@pytest.mark.parametrize("vocab_size", [0, -1])
def test_nonpositive_explicit_vocab_fails_before_component_or_state_work(vocab_size):
    events = []
    constraint = RecordingConstraint(events=events)

    with pytest.raises(ValueError, match="greater than zero"):
        _run(constraint, State(), vocab_size=vocab_size)

    assert events == []
    assert constraint.forbidden_calls == []


def test_plainly_nonconforming_constraint_and_mask_are_type_errors():
    with pytest.raises(TypeError, match="constraint"):
        _run(object(), State())

    constraint = RecordingConstraint()
    with pytest.raises(TypeError, match="logit_mask"):
        _run(constraint, State(), mask=object())
    assert _state_events(constraint.events) == []


def test_protocol_conformance_exceptions_are_chained_invariant_errors(monkeypatch):
    failure = RuntimeError("protocol inspection failed")

    class ExplodingMeta(type):
        def __instancecheck__(cls, instance):
            raise failure

    class ExplodingProtocol(metaclass=ExplodingMeta):
        pass

    monkeypatch.setattr(selection_module, "GrammarConstraint", ExplodingProtocol)
    with pytest.raises(
        GrammarMaskedSelectionInvariantError,
        match="constraint runtime conformance",
    ) as captured:
        _run(RecordingConstraint(), State())
    assert captured.value.__cause__ is failure

    monkeypatch.undo()
    monkeypatch.setattr(selection_module, "GrammarLogitMask", ExplodingProtocol)
    with pytest.raises(
        GrammarMaskedSelectionInvariantError,
        match="logit_mask runtime conformance",
    ) as captured:
        _run(RecordingConstraint(), State())
    assert captured.value.__cause__ is failure


def test_noncallable_selector_fails_before_metadata_and_state_work():
    events = []
    constraint = RecordingConstraint(events=events)
    mask = RecordingMask(events=events)

    with pytest.raises(TypeError, match="select_token"):
        _run(constraint, State(), mask=mask, selector=object())

    assert events == []
    assert mask.calls == []


@pytest.mark.parametrize("value", [True, 5.0, "5", None, IntSubclass(5), 0, -1])
@pytest.mark.parametrize("component", ["constraint", "logit_mask"])
def test_malformed_component_vocab_metadata_fails_before_state_work(component, value):
    events = []
    constraint = RecordingConstraint(events=events)
    mask = RecordingMask(events=events)
    if component == "constraint":
        constraint._vocab_size = value
    else:
        mask._vocab_size = value

    with pytest.raises(
        GrammarMaskedSelectionInvariantError,
        match=rf"{component} vocab_size",
    ):
        _run(constraint, State(), mask=mask)

    assert _state_events(events) == []
    assert mask.calls == []


@pytest.mark.parametrize(
    "component,attribute",
    [
        ("constraint", "vocab_size"),
        ("constraint", "grammar_type"),
        ("logit_mask", "vocab_size"),
    ],
)
def test_unreadable_component_metadata_chains_the_original_failure(component, attribute):
    events = []
    failure = RuntimeError("unreadable metadata")
    constraint = RecordingConstraint(events=events)
    mask = RecordingMask(events=events)
    if component == "constraint":
        constraint.metadata_failures[attribute] = failure
    else:
        mask.metadata_failure = failure

    with pytest.raises(
        GrammarMaskedSelectionInvariantError,
        match=rf"{component} {attribute} could not be read",
    ) as captured:
        _run(constraint, State(), mask=mask)

    assert captured.value.__cause__ is failure
    assert _state_events(events) == []
    assert mask.calls == []


@pytest.mark.parametrize("grammar_type", [None, True, "json", IntSubclass(1)])
def test_invalid_grammar_type_fails_before_mask_metadata_and_state_work(grammar_type):
    events = []
    constraint = RecordingConstraint(grammar_type=grammar_type, events=events)
    mask = RecordingMask(events=events)

    with pytest.raises(
        GrammarMaskedSelectionInvariantError,
        match="grammar_type",
    ):
        _run(constraint, State(), mask=mask)

    assert events == ["constraint.vocab_size", "constraint.grammar_type"]
    assert mask.calls == []


@pytest.mark.parametrize(
    "constraint_vocab_size,mask_vocab_size",
    [(4, 5), (5, 4), (4, 4)],
)
def test_vocabulary_disagreement_fails_after_all_metadata_and_before_state_work(
    constraint_vocab_size,
    mask_vocab_size,
):
    events = []
    constraint = RecordingConstraint(vocab_size=constraint_vocab_size, events=events)
    mask = RecordingMask(vocab_size=mask_vocab_size, events=events)

    with pytest.raises(
        GrammarMaskedSelectionInvariantError,
        match="must match exactly",
    ):
        _run(constraint, State(), mask=mask)

    assert events == [
        "constraint.vocab_size",
        "constraint.grammar_type",
        "logit_mask.vocab_size",
    ]
    assert mask.calls == []


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
def test_successful_nonempty_selection_has_exact_order_and_identity_flow(grammar_type):
    events = []
    state = State()
    input_row = OpaqueRow("input")
    masked_row = OpaqueRow("masked")
    support = (0, 2, 4)
    constraint = RecordingConstraint(
        grammar_type=grammar_type,
        valid_token_ids=support,
        is_match=True,
        events=events,
    )
    mask = RecordingMask(masked_logits=masked_row, events=events)
    selector = RecordingSelector(2, events=events)

    result = _run(
        constraint,
        state,
        row=input_row,
        mask=mask,
        selector=selector,
    )

    assert events == [
        "constraint.vocab_size",
        "constraint.grammar_type",
        "logit_mask.vocab_size",
        ("is_dead_state", state),
        ("is_match_state", state),
        ("get_valid_token_ids", state),
        "mask.apply",
        "select_token",
    ]
    assert len(mask.calls) == 1
    assert mask.calls[0][0] is input_row
    assert mask.calls[0][1] is support
    assert selector.calls == [masked_row]
    assert mask.timing_calls == 0
    assert result.valid_token_ids is support
    assert result.is_match is True
    assert result.selected_token_id == 2
    assert constraint.forbidden_calls == []


@pytest.mark.parametrize("is_match", [False, True])
def test_empty_support_is_successful_without_mask_or_selector_calls(is_match):
    events = []
    state = State()
    support = ()
    constraint = RecordingConstraint(
        valid_token_ids=support,
        is_match=is_match,
        events=events,
    )
    mask = RecordingMask(events=events)
    selector = RecordingSelector(1, events=events)

    result = _run(constraint, state, mask=mask, selector=selector)

    assert result == GrammarMaskedSelectionResult(support, is_match, None)
    assert result.valid_token_ids is support
    assert mask.calls == []
    assert selector.calls == []
    assert constraint.forbidden_calls == []


def test_dead_state_fails_before_match_scan_mask_and_selector():
    events = []
    state = State()
    constraint = RecordingConstraint(is_dead=True, events=events)
    mask = RecordingMask(events=events)
    selector = RecordingSelector(1, events=events)

    with pytest.raises(GrammarMaskedSelectionInvariantError, match="must not be dead"):
        _run(constraint, state, mask=mask, selector=selector)

    assert _state_events(events) == [("is_dead_state", state)]
    assert mask.calls == []
    assert selector.calls == []
    assert constraint.forbidden_calls == []


@pytest.mark.parametrize("operation", ["is_dead_state", "is_match_state"])
@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_state_queries_require_exact_booleans(operation, value):
    events = []
    constraint = RecordingConstraint(events=events)
    if operation == "is_dead_state":
        constraint.dead_result = value
    else:
        constraint.match_result = value

    with pytest.raises(GrammarMaskedSelectionInvariantError, match=operation):
        _run(constraint, State())

    state_operations = [event[0] for event in _state_events(events)]
    expected = ["is_dead_state"] if operation == "is_dead_state" else [
        "is_dead_state",
        "is_match_state",
    ]
    assert state_operations == expected


@pytest.mark.parametrize(
    "operation",
    ["is_dead_state", "is_match_state", "get_valid_token_ids"],
)
def test_constraint_query_failures_propagate_by_identity_without_cleanup(operation):
    failure = GrammarStateError(f"{operation} failed")
    constraint = RecordingConstraint()
    constraint.query_failures[operation] = failure

    with pytest.raises(GrammarStateError) as captured:
        _run(constraint, State())

    assert captured.value is failure
    assert constraint.forbidden_calls == []


@pytest.mark.parametrize(
    "support",
    [
        [1],
        TupleSubclass((1,)),
        (True,),
        (IntSubclass(1),),
        (1.0,),
        ("1",),
        (-1,),
        (VOCAB_SIZE,),
        (1, 1),
        (3, 1),
    ],
)
def test_invalid_native_support_is_rejected_without_mask_or_selector(support):
    constraint = RecordingConstraint(valid_token_ids=support)
    mask = RecordingMask()
    selector = RecordingSelector(1)

    with pytest.raises(GrammarMaskedSelectionInvariantError):
        _run(constraint, State(), mask=mask, selector=selector)

    assert mask.calls == []
    assert selector.calls == []
    assert constraint.forbidden_calls == []


@pytest.mark.parametrize(
    "support,selected",
    [((0,), 0), ((2,), 2), ((0, 1, 2, 3, 4), 4)],
)
def test_one_token_multi_token_and_full_domain_support(support, selected):
    constraint = RecordingConstraint(valid_token_ids=support)
    mask = RecordingMask()

    result = _run(
        constraint,
        State(),
        mask=mask,
        selector=RecordingSelector(selected),
    )

    assert result.valid_token_ids is support
    assert result.selected_token_id == selected
    assert mask.calls[0][1] is support


def test_support_is_not_sorted_deduplicated_filtered_or_given_eos_policy():
    support = (0, 2, 4)
    constraint = RecordingConstraint(valid_token_ids=support, is_match=True)
    mask = RecordingMask()

    result = _run(
        constraint,
        State(),
        mask=mask,
        selector=RecordingSelector(4),
    )

    assert mask.calls[0][1] is support
    assert result.valid_token_ids is support
    assert result.valid_token_ids == (0, 2, 4)


def test_mask_failure_propagates_by_identity_and_skips_selector():
    failure = RuntimeError("mask failed")
    mask = RecordingMask(failure=failure)
    selector = RecordingSelector(1)
    constraint = RecordingConstraint()

    with pytest.raises(RuntimeError) as captured:
        _run(constraint, State(), mask=mask, selector=selector)

    assert captured.value is failure
    assert len(mask.calls) == 1
    assert selector.calls == []
    assert constraint.forbidden_calls == []


def test_selector_failure_propagates_by_identity_without_retry():
    failure = RuntimeError("selector failed")
    selector = RecordingSelector(1, failure=failure)
    constraint = RecordingConstraint()

    with pytest.raises(RuntimeError) as captured:
        _run(constraint, State(), selector=selector)

    assert captured.value is failure
    assert len(selector.calls) == 1
    assert constraint.forbidden_calls == []


@pytest.mark.parametrize(
    "selected_token_id",
    [True, 1.0, "1", IntSubclass(1), -1, VOCAB_SIZE, 2],
)
def test_invalid_selected_token_is_rejected_after_one_mask_and_selector_call(
    selected_token_id,
):
    mask = RecordingMask()
    selector = RecordingSelector(selected_token_id)
    constraint = RecordingConstraint(valid_token_ids=(1, 3))

    with pytest.raises(GrammarMaskedSelectionInvariantError):
        _run(constraint, State(), mask=mask, selector=selector)

    assert len(mask.calls) == 1
    assert len(selector.calls) == 1
    assert constraint.forbidden_calls == []


def test_stateful_selector_consumption_continues_after_an_invalid_draw():
    class StatefulSelector:
        def __init__(self):
            self.draws = iter((2, 3))
            self.call_count = 0

        def __call__(self, logits):
            self.call_count += 1
            return next(self.draws)

    selector = StatefulSelector()
    constraint = RecordingConstraint(valid_token_ids=(1, 3))
    state = State()

    with pytest.raises(GrammarMaskedSelectionInvariantError, match="exact grammar support"):
        _run(constraint, state, selector=selector)
    result = _run(constraint, state, selector=selector)

    assert selector.call_count == 2
    assert result.selected_token_id == 3


def test_stateful_selector_consumption_continues_after_a_failed_draw():
    failure = RuntimeError("draw failed after consuming RNG state")

    class StatefulSelector:
        def __init__(self):
            self.outcomes = iter((failure, 3))
            self.call_count = 0

        def __call__(self, logits):
            self.call_count += 1
            outcome = next(self.outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    selector = StatefulSelector()
    constraint = RecordingConstraint(valid_token_ids=(1, 3))
    state = State()

    with pytest.raises(RuntimeError) as captured:
        _run(constraint, state, selector=selector)
    result = _run(constraint, state, selector=selector)

    assert captured.value is failure
    assert selector.call_count == 2
    assert result.selected_token_id == 3


def test_fresh_same_seed_reference_sampler_replays_without_d42_owning_rng():
    policy = TemperatureTopPSelection(temperature=0.8, top_p=1.0, seed=42)
    first = create_reference_sampler(policy)
    replay = create_reference_sampler(policy)
    masked_row = (1.0, 2.0, -math.inf, 3.0, -math.inf)
    constraint = RecordingConstraint(valid_token_ids=(0, 1, 3))
    mask = RecordingMask(masked_logits=masked_row)
    state = State()

    first_results = tuple(
        _run(constraint, state, mask=mask, selector=first).selected_token_id
        for _ in range(5)
    )
    replay_results = tuple(
        _run(constraint, state, mask=mask, selector=replay).selected_token_id
        for _ in range(5)
    )

    assert first_results == replay_results


def test_fake_constraint_composition_preserves_state_and_input_row():
    constraint = _fake_constraint(grammar_type="json_schema", match=True)
    state = constraint.init_state()
    row = [1.0, 2.0, 3.0, 4.0, 5.0]
    original_row = list(row)

    class SequenceMask:
        vocab_size = VOCAB_SIZE

        def apply(self, logits, valid_token_ids, /):
            return tuple(
                value if token_id in valid_token_ids else -math.inf
                for token_id, value in enumerate(logits)
            )

    result = _run(
        constraint,
        state,
        row=row,
        mask=SequenceMask(),
        selector=lambda logits: max(range(len(logits)), key=logits.__getitem__),
    )

    assert result == GrammarMaskedSelectionResult((1, 3), True, 3)
    assert row == original_row
    assert constraint.active_state_count == 1
    assert constraint.is_match_state(state) is True
    constraint.release_state(state)


def test_foreign_released_and_stale_fake_states_remain_constraint_errors():
    for kind in ("foreign", "released", "stale"):
        constraint = _fake_constraint()
        state = constraint.init_state()
        if kind == "foreign":
            other = _fake_constraint()
            state = other.init_state()
        elif kind == "released":
            constraint.release_state(state)
        else:
            constraint.reset()

        with pytest.raises(GrammarStateError):
            _run(constraint, state)


def test_result_construction_failure_does_not_trigger_cleanup(monkeypatch):
    failure = RuntimeError("result construction failed")

    class FailingResult:
        def __init__(self, **kwargs):
            raise failure

    monkeypatch.setattr(selection_module, "GrammarMaskedSelectionResult", FailingResult)
    constraint = RecordingConstraint()

    with pytest.raises(RuntimeError) as captured:
        _run(constraint, State())

    assert captured.value is failure
    assert constraint.forbidden_calls == []


def test_result_retains_no_borrowed_component_or_row_objects():
    events = []
    constraint = RecordingConstraint(events=events)
    state = State()
    input_row = OpaqueRow("input")
    masked_row = OpaqueRow("masked")
    mask = RecordingMask(masked_logits=masked_row, events=events)
    selector = RecordingSelector(1, events=events)
    result = _run(
        constraint,
        state,
        row=input_row,
        mask=mask,
        selector=selector,
    )
    references = tuple(
        weakref.ref(value)
        for value in (constraint, state, input_row, masked_row, mask, selector)
    )
    events.clear()
    constraint.events = []
    mask.events = []
    mask.calls.clear()
    selector.events = []
    selector.calls.clear()

    del constraint
    del state
    del input_row
    del masked_row
    del mask
    del selector
    gc.collect()

    assert all(reference() is None for reference in references)
    assert result == GrammarMaskedSelectionResult((1, 3), False, 1)


def test_one_thousand_reuses_have_linear_calls_and_no_state_growth():
    constraint = _fake_constraint()
    state = constraint.init_state()
    input_row = OpaqueRow("input")
    masked_row = OpaqueRow("masked")
    mask = RecordingMask(masked_logits=masked_row)
    selector = RecordingSelector(3)

    for _ in range(1000):
        result = _run(
            constraint,
            state,
            row=input_row,
            mask=mask,
            selector=selector,
        )
        assert result.selected_token_id == 3
        assert constraint.active_state_count == 1

    assert len(mask.calls) == 1000
    assert len(selector.calls) == 1000
    constraint.release_state(state)
    assert constraint.active_state_count == 0


def test_isolated_execution_remains_optional_runtime_free():
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        from onyx_cuda import (
            GrammarMaskedSelectionError,
            GrammarMaskedSelectionInvariantError,
            GrammarMaskedSelectionResult,
            select_grammar_masked_token,
        )
        from onyx_cuda.testing import FakeGrammarConstraint, FakeGrammarProgram

        class Mask:
            vocab_size = 4

            def apply(self, logits, valid_token_ids, /):
                return tuple(
                    value if index in valid_token_ids else float("-inf")
                    for index, value in enumerate(logits)
                )

        vocabulary = (b"a", b"b", b"c", b"d")
        regex = FakeGrammarConstraint(
            vocabulary,
            grammar_type="regex",
            program=FakeGrammarProgram(
                initial_state="live",
                transitions=(("live", 1, "live"), ("live", 3, "live")),
                valid_token_ids=(("live", (1, 3)),),
            ),
        )
        json_constraint = FakeGrammarConstraint(
            vocabulary,
            grammar_type="json_schema",
            program=FakeGrammarProgram(
                initial_state="done",
                transitions=(),
                valid_token_ids=(("done", ()),),
                match_states=frozenset({{"done"}}),
            ),
        )
        regex_state = regex.init_state()
        json_state = json_constraint.init_state()
        selected = select_grammar_masked_token(
            regex,
            regex_state,
            (9.0, 3.0, 8.0, 5.0),
            Mask(),
            vocab_size=4,
            select_token=lambda row: max(range(len(row)), key=row.__getitem__),
        )
        empty = select_grammar_masked_token(
            json_constraint,
            json_state,
            (0.0, 0.0, 0.0, 0.0),
            Mask(),
            vocab_size=4,
            select_token=lambda row: 0,
        )
        assert selected == GrammarMaskedSelectionResult((1, 3), False, 3)
        assert empty == GrammarMaskedSelectionResult((), True, None)
        assert regex.active_state_count == 1
        assert json_constraint.active_state_count == 1
        regex.release_state(regex_state)
        json_constraint.release_state(json_state)
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
    imported = importlib.import_module("onyx_cuda")
    current_module = importlib.import_module("onyx_cuda.grammar_selection")
    for name in current_module.__all__:
        assert getattr(imported, name) is getattr(current_module, name)
        assert name in imported.__all__
