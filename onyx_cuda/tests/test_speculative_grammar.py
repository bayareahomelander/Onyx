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

import onyx_cuda.speculative_grammar as grammar_module
from onyx_cuda import (
    ContinuationAwareSpeculativeIterationResult,
    GrammarError,
    GrammarStateError,
    SpeculativeGrammarReconciliationCleanupError,
    SpeculativeGrammarReconciliationError,
    SpeculativeGrammarReconciliationInvariantError,
    SpeculativeGrammarReconciliationResult,
    coordinate_continuation_aware_speculative_iteration,
    reconcile_speculative_grammar_state,
    select_highest_logit,
)
from onyx_cuda.testing import FakeAutoregressiveBackend


VOCAB_SIZE = 8
PROPOSAL = (1, 2, 3)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class State:
    __slots__ = ("path", "dead", "__weakref__")

    def __init__(self, path=(), *, dead=False):
        self.path = path
        self.dead = dead

    def __eq__(self, other):
        raise AssertionError("opaque grammar states must not be compared")

    def __hash__(self):
        raise AssertionError("opaque grammar states must not be hashed")


class RecordingConstraint:
    def __init__(
        self,
        *,
        vocab_size=VOCAB_SIZE,
        grammar_type="regex",
        dead_tokens=(),
        match_paths=(),
    ):
        self._vocab_size = vocab_size
        self._grammar_type = grammar_type
        self.dead_tokens = set(dead_tokens)
        self.match_paths = set(match_paths)
        self.active = []
        self.known = []
        self.advance_events = []
        self.query_events = []
        self.release_events = []
        self.aliases = {}
        self.advance_failures = {}
        self.release_failures = {}
        self.release_call_failures = {}
        self.dead_failures = {}
        self.dead_results = {}
        self.match_failures = {}
        self.match_results = {}
        self.dead_call_count = 0
        self.match_call_count = 0
        self.forbidden_calls = []

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def grammar_type(self):
        return self._grammar_type

    @property
    def active_state_count(self):
        return len(self.active)

    def init_state(self):
        self.forbidden_calls.append("init_state")
        state = State()
        self.active.append(state)
        self.known.append(state)
        return state

    def make_start(self, path=(), *, dead=False):
        state = State(path, dead=dead)
        self.active.append(state)
        self.known.append(state)
        return state

    def advance_state(self, state, token_id, /):
        self._require_active(state)
        call_number = len(self.advance_events) + 1
        failure = self.advance_failures.get(call_number)
        if failure is not None:
            raise failure
        alias = self.aliases.get(call_number)
        if alias is not None:
            child = alias
        else:
            child = State(
                state.path + (token_id,),
                dead=state.dead or token_id in self.dead_tokens,
            )
            self.active.append(child)
            self.known.append(child)
        self.advance_events.append((state, token_id, child))
        return child

    def get_valid_token_ids(self, state, /):
        self.forbidden_calls.append("get_valid_token_ids")
        self._require_active(state)
        return ()

    def is_match_state(self, state, /):
        self._require_active(state)
        self.match_call_count += 1
        self.query_events.append(("match", state))
        failure = self.match_failures.get(self.match_call_count)
        if failure is not None:
            raise failure
        if self.match_call_count in self.match_results:
            return self.match_results[self.match_call_count]
        return state.path in self.match_paths

    def is_dead_state(self, state, /):
        self._require_active(state)
        self.dead_call_count += 1
        self.query_events.append(("dead", state))
        failure = self.dead_failures.get(self.dead_call_count)
        if failure is not None:
            raise failure
        if self.dead_call_count in self.dead_results:
            return self.dead_results[self.dead_call_count]
        return state.dead

    def release_state(self, state, /):
        if not self._contains_identity(self.known, state):
            raise GrammarStateError("unknown state")
        self.release_events.append(state)
        call_failure_spec = self.release_call_failures.pop(
            len(self.release_events),
            None,
        )
        if call_failure_spec is not None:
            failure, settle = call_failure_spec
            if settle:
                self._remove_identity(self.active, state)
            raise failure
        failure_spec = self.release_failures.get(state.path)
        if failure_spec is not None:
            failure, settle, remaining = failure_spec
            if remaining > 0:
                self.release_failures[state.path] = (
                    failure,
                    settle,
                    remaining - 1,
                )
                if settle:
                    self._remove_identity(self.active, state)
                raise failure
        self._remove_identity(self.active, state)

    def release_states(self, states, /):
        self.forbidden_calls.append("release_states")

    def reset(self):
        self.forbidden_calls.append("reset")

    @staticmethod
    def _contains_identity(states, candidate):
        return any(state is candidate for state in states)

    @staticmethod
    def _remove_identity(states, candidate):
        for position, state in enumerate(states):
            if state is candidate:
                states.pop(position)
                return

    def _require_active(self, state):
        if not self._contains_identity(self.active, state):
            raise GrammarStateError("state is unknown or released")


class JsonConstraint(RecordingConstraint):
    def __init__(self, invalid_paths):
        super().__init__(grammar_type="json_schema")
        self.invalid_paths = set(invalid_paths)
        self.transition_error = GrammarStateError("invalid JSON transition")

    def advance_state(self, state, token_id, /):
        if state.path + (token_id,) in self.invalid_paths:
            self._require_active(state)
            raise self.transition_error
        return super().advance_state(state, token_id)


def _result(
    *,
    proposal=PROPOSAL,
    accepted_count=1,
    replacement_token_id=4,
    uncached_next_token_id=None,
    initial_cache_length=2,
):
    if accepted_count == len(proposal):
        replacement_token_id = None
        if uncached_next_token_id is None:
            uncached_next_token_id = 7
    elif uncached_next_token_id is None:
        uncached_next_token_id = replacement_token_id
    return ContinuationAwareSpeculativeIterationResult(
        proposal_token_ids=proposal,
        accepted_count=accepted_count,
        replacement_token_id=replacement_token_id,
        initial_cache_length=initial_cache_length,
        final_cache_length=initial_cache_length + accepted_count + 1,
        uncached_next_token_id=uncached_next_token_id,
    )


def _unsafe_result(**changes):
    values = {
        "proposal_token_ids": PROPOSAL,
        "accepted_count": 1,
        "replacement_token_id": 4,
        "initial_cache_length": 2,
        "final_cache_length": 4,
        "uncached_next_token_id": 4,
    }
    values.update(changes)
    result = object.__new__(ContinuationAwareSpeculativeIterationResult)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _run(constraint, start, result):
    return reconcile_speculative_grammar_state(
        constraint,
        start,
        result,
        vocab_size=VOCAB_SIZE,
    )


def _row(selected_token_id):
    return tuple(
        20.0 if token_id == selected_token_id else float(-token_id)
        for token_id in range(VOCAB_SIZE)
    )


def test_public_surface_signature_result_and_hierarchy():
    assert grammar_module.__all__ == [
        "SpeculativeGrammarReconciliationCleanupError",
        "SpeculativeGrammarReconciliationError",
        "SpeculativeGrammarReconciliationInvariantError",
        "SpeculativeGrammarReconciliationResult",
        "reconcile_speculative_grammar_state",
    ]
    assert issubclass(SpeculativeGrammarReconciliationError, GrammarError)
    assert issubclass(
        SpeculativeGrammarReconciliationInvariantError,
        SpeculativeGrammarReconciliationError,
    )
    assert issubclass(
        SpeculativeGrammarReconciliationCleanupError,
        SpeculativeGrammarReconciliationError,
    )
    signature = inspect.signature(reconcile_speculative_grammar_state)
    assert tuple(signature.parameters) == (
        "constraint",
        "starting_state",
        "iteration_result",
        "vocab_size",
    )
    assert signature.parameters["vocab_size"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["vocab_size"].default is inspect.Parameter.empty
    assert [field.name for field in fields(SpeculativeGrammarReconciliationResult)] == [
        "committed_state",
        "is_match",
    ]

    state = State()
    result = SpeculativeGrammarReconciliationResult(state, True)
    assert result.committed_state is state
    assert result.is_match is True
    with pytest.raises(TypeError, match="is_match"):
        SpeculativeGrammarReconciliationResult(state, 1)
    with pytest.raises(FrozenInstanceError):
        result.is_match = False
    assert not hasattr(result, "__dict__")


def test_cleanup_error_retains_exact_immutable_evidence_and_cause():
    original = GrammarStateError("transition")
    first = RuntimeError("first cleanup")
    second = ValueError("second cleanup")
    error = SpeculativeGrammarReconciliationCleanupError(
        original,
        [("draft state release at position 0", first), ("later", second)],
    )

    assert error.original_failure is original
    assert error.cleanup_failures == (
        ("draft state release at position 0", first),
        ("later", second),
    )
    assert error.cleanup_failures[0][1] is first
    assert error.cleanup_failures[1][1] is second
    assert error.__cause__ is original
    with pytest.raises(ValueError, match="cannot be empty"):
        SpeculativeGrammarReconciliationCleanupError(original, ())


@pytest.mark.parametrize("vocab_size", [True, False, 1.5, "8", None])
def test_invalid_vocab_size_type_fails_before_grammar_work(vocab_size):
    constraint = RecordingConstraint()
    start = constraint.make_start()

    with pytest.raises(TypeError, match="vocab_size"):
        reconcile_speculative_grammar_state(
            constraint,
            start,
            _result(),
            vocab_size=vocab_size,
        )

    assert constraint.advance_events == []
    assert constraint.release_events == []


@pytest.mark.parametrize("vocab_size", [0, -1])
def test_nonpositive_vocab_size_fails_before_grammar_work(vocab_size):
    constraint = RecordingConstraint()
    start = constraint.make_start()

    with pytest.raises(ValueError, match="greater than zero"):
        reconcile_speculative_grammar_state(
            constraint,
            start,
            _result(),
            vocab_size=vocab_size,
        )

    assert constraint.advance_events == []
    assert constraint.release_events == []


def test_constraint_and_vocabulary_metadata_are_validated_before_transitions():
    with pytest.raises(TypeError, match="GrammarConstraint"):
        reconcile_speculative_grammar_state(
            object(),
            object(),
            _result(),
            vocab_size=VOCAB_SIZE,
        )

    for constraint, message in (
        (RecordingConstraint(vocab_size=True), "positive integer"),
        (RecordingConstraint(vocab_size=0), "positive integer"),
        (RecordingConstraint(vocab_size=VOCAB_SIZE - 1), "expected"),
        (RecordingConstraint(grammar_type="other"), "grammar_type"),
    ):
        start = constraint.make_start()
        with pytest.raises(
            SpeculativeGrammarReconciliationInvariantError,
            match=message,
        ):
            _run(constraint, start, _result())
        assert constraint.advance_events == []
        assert constraint.release_events == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"proposal_token_ids": [1, 2]}, "could not be read"),
        ({"proposal_token_ids": ()}, "proposal_token_ids"),
        ({"proposal_token_ids": (True,)}, "proposal token"),
        ({"proposal_token_ids": (VOCAB_SIZE,)}, "outside vocabulary"),
        ({"accepted_count": True}, "accepted_count"),
        ({"accepted_count": 4}, "within"),
        ({"replacement_token_id": None}, "replacement_token_id"),
        ({"replacement_token_id": 2}, "differ"),
        ({"initial_cache_length": 0}, "greater than zero"),
        ({"final_cache_length": 5}, "expected"),
        ({"uncached_next_token_id": VOCAB_SIZE}, "outside vocabulary"),
        ({"uncached_next_token_id": 5}, "must equal"),
    ],
)
def test_tampered_d38_evidence_is_rejected_before_transitions(changes, message):
    constraint = RecordingConstraint()
    start = constraint.make_start()

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match=message,
    ):
        _run(constraint, start, _unsafe_result(**changes))

    assert constraint.advance_events == []
    assert constraint.release_events == []


def test_foreign_or_dead_start_fails_without_cleanup():
    constraint = RecordingConstraint()
    foreign_constraint = RecordingConstraint()
    foreign = foreign_constraint.make_start()

    with pytest.raises(GrammarStateError) as raised:
        _run(constraint, foreign, _result())
    assert str(raised.value) == "state is unknown or released"
    assert constraint.release_events == []

    dead = constraint.make_start(dead=True)
    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="must not be dead",
    ):
        _run(constraint, dead, _result())
    assert constraint.release_events == []


@pytest.mark.parametrize(
    ("query", "result"),
    [
        ("dead", 0),
        ("match", 1),
    ],
)
def test_non_boolean_start_query_fails_without_cleanup(query, result):
    constraint = RecordingConstraint()
    start = constraint.make_start()
    getattr(constraint, f"{query}_results")[1] = result

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="must return a boolean",
    ):
        _run(constraint, start, _result())

    assert constraint.advance_events == []
    assert constraint.release_events == []
    constraint.release_state(start)


@pytest.mark.parametrize("accepted_count", [0, 1, 2, 3])
def test_every_mismatch_and_full_acceptance_replays_independent_branches(
    accepted_count,
):
    full_acceptance = accepted_count == len(PROPOSAL)
    output = PROPOSAL + (7,) if full_acceptance else PROPOSAL[:accepted_count] + (4,)
    constraint = RecordingConstraint(match_paths=(("current", *output),))
    start = constraint.make_start(("current",))
    result = _result(
        accepted_count=accepted_count,
        replacement_token_id=None if full_acceptance else 4,
        uncached_next_token_id=7 if full_acceptance else 4,
    )

    reconciled = _run(constraint, start, result)

    assert [event[1] for event in constraint.advance_events] == [
        *PROPOSAL,
        *output,
    ]
    draft_events = constraint.advance_events[: len(PROPOSAL)]
    committed_events = constraint.advance_events[len(PROPOSAL) :]
    assert draft_events[0][0] is start
    assert committed_events[0][0] is start
    for position in range(accepted_count):
        assert draft_events[position][1] == committed_events[position][1]
        assert draft_events[position][2] is not committed_events[position][2]
    assert reconciled.committed_state is committed_events[-1][2]
    assert reconciled.committed_state.path == ("current", *output)
    assert reconciled.is_match is True
    assert constraint.release_events == [
        *(event[2] for event in draft_events),
        *(event[2] for event in committed_events[:-1]),
    ]
    assert constraint.active_state_count == 2
    assert constraint.forbidden_calls == []

    constraint.release_state(reconciled.committed_state)
    constraint.release_state(start)
    assert constraint.active_state_count == 0


@pytest.mark.parametrize(
    ("result", "advance_count"),
    [
        (_result(), 5),
        (_result(accepted_count=3, uncached_next_token_id=7), 7),
    ],
)
def test_failure_at_every_transition_boundary_releases_prior_children(
    result,
    advance_count,
):
    for call_number in range(1, advance_count + 1):
        constraint = RecordingConstraint()
        start = constraint.make_start()
        transition_failure = GrammarStateError(f"transition {call_number}")
        constraint.advance_failures[call_number] = transition_failure

        with pytest.raises(GrammarStateError) as raised:
            _run(constraint, start, result)

        assert raised.value is transition_failure
        assert constraint.release_events == [
            event[2] for event in constraint.advance_events
        ]
        assert constraint.active == [start]
        constraint.release_state(start)


def test_child_and_final_query_results_are_validated_and_cleaned_up():
    child_constraint = RecordingConstraint()
    child_start = child_constraint.make_start()
    child_constraint.dead_results[2] = 0

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="must return a boolean",
    ):
        _run(child_constraint, child_start, _result())
    assert child_constraint.active == [child_start]
    child_constraint.release_state(child_start)

    final_constraint = RecordingConstraint()
    final_start = final_constraint.make_start()
    final_constraint.match_results[3] = 1

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="must return a boolean",
    ):
        _run(final_constraint, final_start, _result())
    assert final_constraint.active == [final_start]
    final_constraint.release_state(final_start)


def test_genuine_public_d38_result_composes_without_cache_or_selector_access():
    prompt = (6, 7)
    draft = FakeAutoregressiveBackend(
        (_row(0), _row(1), _row(2), _row(3), _row(0), _row(0))
    )
    target = FakeAutoregressiveBackend(
        (_row(0), _row(1), _row(4), _row(3), _row(7), _row(0))
    )
    draft.prefill(prompt)
    target.prefill(prompt)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    iteration = coordinate_continuation_aware_speculative_iteration(
        draft,
        target,
        5,
        proposal_length=3,
        draft_select_token=select_highest_logit,
        target_select_token=select_highest_logit,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    draft_cache = draft.cached_token_ids
    target_cache = target.cached_token_ids
    constraint = RecordingConstraint()
    start = constraint.make_start(("current",))

    result = _run(constraint, start, iteration)

    assert iteration.output_token_ids == (1, 4)
    assert result.committed_state.path == ("current", 1, 4)
    assert draft.cached_token_ids == draft_cache
    assert target.cached_token_ids == target_cache
    constraint.release_state(result.committed_state)
    constraint.release_state(start)


def test_match_flag_does_not_terminate_or_release_the_final_state():
    constraint = RecordingConstraint(match_paths=((1,), (1, 2), (1, 2, 3), (1, 4)))
    start = constraint.make_start()

    result = _run(constraint, start, _result())

    assert [event[1] for event in constraint.advance_events] == [1, 2, 3, 1, 4]
    assert result.is_match is True
    assert RecordingConstraint._contains_identity(
        constraint.active,
        result.committed_state,
    )
    constraint.release_state(result.committed_state)
    constraint.release_state(start)


def test_dead_regex_draft_suffix_is_discarded_but_committed_path_stays_live():
    constraint = RecordingConstraint(dead_tokens=(2,))
    start = constraint.make_start()

    result = _run(constraint, start, _result(accepted_count=1))

    draft_children = [event[2] for event in constraint.advance_events[:3]]
    assert [child.dead for child in draft_children] == [False, True, True]
    assert result.committed_state.dead is False
    assert constraint.release_events[:3] == draft_children
    constraint.release_state(result.committed_state)
    constraint.release_state(start)


@pytest.mark.parametrize("accepted_count", [1, 2, 3])
def test_dead_committed_state_is_rejected_and_all_children_are_released(
    accepted_count,
):
    dead_token = PROPOSAL[accepted_count - 1]
    constraint = RecordingConstraint(dead_tokens=(dead_token,))
    start = constraint.make_start()

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="committed state",
    ):
        _run(
            constraint,
            start,
            _result(
                accepted_count=accepted_count,
                replacement_token_id=None if accepted_count == 3 else 4,
                uncached_next_token_id=7 if accepted_count == 3 else 4,
            ),
        )

    assert constraint.active == [start]
    assert constraint.release_events == [
        *(event[2] for event in constraint.advance_events[:3]),
        *(event[2] for event in constraint.advance_events[3:]),
    ]
    constraint.release_state(start)


@pytest.mark.parametrize(
    "invalid_path",
    [(1,), (1, 2), (1, 2, 3), (1, 4)],
)
def test_json_transition_failure_preserves_original_and_releases_acquired_children(
    invalid_path,
):
    constraint = JsonConstraint((invalid_path,))
    start = constraint.make_start()

    with pytest.raises(GrammarStateError) as raised:
        _run(constraint, start, _result())

    assert raised.value is constraint.transition_error
    assert constraint.active == [start]
    assert constraint.forbidden_calls == []
    constraint.release_state(start)


@pytest.mark.parametrize("alias_kind", ["start", "parent", "draft", "committed"])
def test_returned_child_aliases_are_rejected_without_double_release(alias_kind):
    constraint = RecordingConstraint()
    start = constraint.make_start()
    if alias_kind == "start":
        constraint.aliases[1] = start
    elif alias_kind == "parent":
        first = State((1,))
        constraint.active.append(first)
        constraint.known.append(first)
        constraint.aliases[1] = first
        constraint.aliases[2] = first
    elif alias_kind == "draft":
        first = State((1,))
        constraint.active.append(first)
        constraint.known.append(first)
        constraint.aliases[1] = first
        constraint.aliases[3] = first
    else:
        first_committed = State((1,))
        constraint.active.append(first_committed)
        constraint.known.append(first_committed)
        constraint.aliases[4] = first_committed
        constraint.aliases[5] = first_committed

    with pytest.raises(SpeculativeGrammarReconciliationInvariantError, match="aliases"):
        _run(constraint, start, _result())

    assert all(state is not start for state in constraint.release_events)
    released_ids = [id(state) for state in constraint.release_events]
    assert len(released_ids) == len(set(released_ids))
    if RecordingConstraint._contains_identity(constraint.active, start):
        constraint.release_state(start)


@pytest.mark.parametrize("settle_before_raise", [False, True])
@pytest.mark.parametrize("release_position", range(1, 7))
def test_every_success_release_failure_is_retried_and_preserved(
    settle_before_raise,
    release_position,
):
    constraint = RecordingConstraint()
    start = constraint.make_start()
    release_failure = RuntimeError(f"release {release_position} failed")
    constraint.release_call_failures[release_position] = (
        release_failure,
        settle_before_raise,
    )

    with pytest.raises(RuntimeError) as raised:
        _run(
            constraint,
            start,
            _result(accepted_count=3, uncached_next_token_id=7),
        )

    assert raised.value is release_failure
    failed_state = constraint.release_events[release_position - 1]
    assert sum(state is failed_state for state in constraint.release_events) == 2
    assert constraint.active == [start]
    constraint.release_state(start)


def test_cleanup_aggregates_exact_draft_then_committed_release_order():
    constraint = RecordingConstraint(dead_tokens=(4,))
    start = constraint.make_start()
    failures = [
        RuntimeError("draft zero"),
        RuntimeError("draft one"),
        RuntimeError("draft two"),
        RuntimeError("committed zero"),
    ]
    for path, failure in zip(
        ((1,), (1, 2), (1, 2, 3), (4,)),
        failures,
    ):
        constraint.release_failures[path] = (failure, False, 1)

    with pytest.raises(SpeculativeGrammarReconciliationCleanupError) as raised:
        _run(constraint, start, _result(accepted_count=0))

    error = raised.value
    assert isinstance(error.original_failure, SpeculativeGrammarReconciliationInvariantError)
    assert error.__cause__ is error.original_failure
    assert error.cleanup_failures == (
        ("draft state release at position 0", failures[0]),
        ("draft state release at position 1", failures[1]),
        ("draft state release at position 2", failures[2]),
        ("committed state release at position 0", failures[3]),
    )
    assert RecordingConstraint._contains_identity(constraint.active, start)
    for state in tuple(constraint.active):
        constraint.release_state(state)


def test_result_construction_failure_releases_untransferred_final(monkeypatch):
    constraint = RecordingConstraint()
    start = constraint.make_start()
    construction_failure = RuntimeError("result construction failed")

    def fail_result_construction(*args, **kwargs):
        raise construction_failure

    monkeypatch.setattr(
        grammar_module,
        "SpeculativeGrammarReconciliationResult",
        fail_result_construction,
    )

    with pytest.raises(RuntimeError) as raised:
        _run(constraint, start, _result())

    assert raised.value is construction_failure
    assert constraint.active == [start]
    constraint.release_state(start)


def test_result_composition_must_retain_exact_state_and_boolean(monkeypatch):
    constraint = RecordingConstraint()
    start = constraint.make_start()
    wrong_state = State(("wrong",))

    class WrongResult:
        def __init__(self, *, committed_state, is_match):
            self.committed_state = wrong_state
            self.is_match = is_match

    monkeypatch.setattr(
        grammar_module,
        "SpeculativeGrammarReconciliationResult",
        WrongResult,
    )

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="exact final committed state",
    ):
        _run(constraint, start, _result())

    assert constraint.active == [start]
    constraint.release_state(start)


def test_unreadable_result_composition_is_an_invariant_and_cleans_final(
    monkeypatch,
):
    constraint = RecordingConstraint()
    start = constraint.make_start()

    class UnreadableResult:
        def __init__(self, *, committed_state, is_match):
            pass

    monkeypatch.setattr(
        grammar_module,
        "SpeculativeGrammarReconciliationResult",
        UnreadableResult,
    )

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="fields must be readable",
    ):
        _run(constraint, start, _result())

    assert constraint.active == [start]
    constraint.release_state(start)


def test_post_release_retained_state_change_releases_only_untransferred_final():
    constraint = RecordingConstraint()
    start = constraint.make_start()
    constraint.dead_results[9] = True

    with pytest.raises(
        SpeculativeGrammarReconciliationInvariantError,
        match="starting_state dead status changed",
    ):
        _run(constraint, start, _result())

    assert constraint.active == [start]
    assert len(constraint.release_events) == 5
    constraint.release_state(start)


def test_result_does_not_retain_constraint_evidence_or_released_ancestors():
    constraint = RecordingConstraint()
    start = constraint.make_start()
    iteration = _result()
    result = _run(constraint, start, iteration)
    draft_children = [event[2] for event in constraint.advance_events[:3]]
    committed_ancestor = constraint.advance_events[3][2]
    constraint_ref = weakref.ref(constraint)
    start_ref = weakref.ref(start)
    draft_refs = [weakref.ref(state) for state in draft_children]
    ancestor_ref = weakref.ref(committed_ancestor)
    constraint.known.clear()
    constraint.advance_events.clear()
    constraint.query_events.clear()
    constraint.release_events.clear()
    constraint.active = [start, result.committed_state]

    del iteration
    del draft_children
    del committed_ancestor
    del constraint
    del start
    gc.collect()

    assert constraint_ref() is None
    assert start_ref() is None
    assert all(reference() is None for reference in draft_refs)
    assert ancestor_ref() is None
    assert result.committed_state.path == (1, 4)


def test_bounded_reuse_keeps_only_caller_start_and_each_transferred_final():
    constraint = RecordingConstraint(dead_tokens=(6,))
    start = constraint.make_start(("current",))
    cases = (
        _result(accepted_count=0),
        _result(accepted_count=1),
        _result(accepted_count=2, replacement_token_id=4),
        _result(accepted_count=3, uncached_next_token_id=7),
        _result(proposal=(1, 6, 3), accepted_count=1),
    )

    for index in range(1000):
        result = _run(constraint, start, cases[index % len(cases)])
        assert constraint.active_state_count == 2
        assert start.path == ("current",)
        constraint.release_state(result.committed_state)
        assert constraint.active == [start]

    constraint.release_state(start)
    assert constraint.active_state_count == 0
    assert constraint.forbidden_calls == []


def test_isolated_execution_remains_optional_runtime_free():
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        from onyx_cuda import (
            ContinuationAwareSpeculativeIterationResult,
            SpeculativeGrammarReconciliationCleanupError,
            SpeculativeGrammarReconciliationError,
            SpeculativeGrammarReconciliationInvariantError,
            SpeculativeGrammarReconciliationResult,
            reconcile_speculative_grammar_state,
        )
        from onyx_cuda.testing import FakeGrammarConstraint, FakeGrammarProgram

        vocabulary = tuple(bytes((token,)) for token in range(5))
        program = FakeGrammarProgram(
            initial_state="s",
            transitions=(
                ("s", 1, "a"),
                ("a", 2, "b"),
                ("a", 4, "done"),
            ),
            valid_token_ids=(("s", (1,)), ("a", (2, 4))),
            match_states=("b", "done"),
        )
        constraint = FakeGrammarConstraint(
            vocabulary,
            grammar_type="regex",
            program=program,
        )
        start = constraint.init_state()
        evidence = ContinuationAwareSpeculativeIterationResult(
            proposal_token_ids=(1, 2),
            accepted_count=1,
            replacement_token_id=4,
            initial_cache_length=2,
            final_cache_length=4,
            uncached_next_token_id=4,
        )
        result = reconcile_speculative_grammar_state(
            constraint,
            start,
            evidence,
            vocab_size=5,
        )
        assert result.is_match is True
        assert constraint.active_state_count == 2
        constraint.release_state(result.committed_state)
        constraint.release_state(start)
        assert constraint.active_state_count == 0
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
    current_module = importlib.import_module("onyx_cuda.speculative_grammar")
    for name in current_module.__all__:
        assert getattr(imported, name) is getattr(current_module, name)
        assert name in imported.__all__
