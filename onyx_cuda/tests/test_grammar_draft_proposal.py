import importlib
import inspect
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_draft as grammar_draft_module
from onyx_cuda import (
    BackendError,
    DraftProposalError,
    GrammarMaskedDraftProposalCleanupError,
    GrammarMaskedDraftProposalError,
    GrammarMaskedDraftProposalInvariantError,
    GrammarMaskedDraftProposalResult,
    GrammarMaskedSelectionResult,
    GrammarMaskedTransitionCleanupError,
    GrammarMaskedTransitionResult,
    generate_grammar_masked_draft_proposal,
)
from onyx_cuda.testing import FakeAutoregressiveBackend


VOCAB_SIZE = 5
SCRIPT = tuple(
    tuple(float(row * VOCAB_SIZE + token) for token in range(VOCAB_SIZE))
    for row in range(32)
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    cache_length: int


class OpaqueState:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        raise AssertionError("D44 must compare grammar states only by identity")

    def __hash__(self):
        raise AssertionError("D44 must not hash opaque grammar states")


class RecordingBackend(FakeAutoregressiveBackend):
    def __init__(self, scripted_logits=SCRIPT, *, events=None):
        self.events = events if events is not None else []
        self.decode_calls = []
        self.created_checkpoints = []
        self.rollback_calls = []
        self.release_calls = []
        self.reset_calls = 0
        self.peak_checkpoint_count = 0
        super().__init__(scripted_logits)

    def decode(self, token_id, /):
        self.events.append(("decode", token_id))
        self.decode_calls.append(token_id)
        return super().decode(token_id)

    def create_cache_checkpoint(self):
        checkpoint = super().create_cache_checkpoint()
        self.events.append(("checkpoint.create", checkpoint.cache_length))
        self.created_checkpoints.append(checkpoint)
        self.peak_checkpoint_count = max(
            self.peak_checkpoint_count,
            self.active_checkpoint_count,
        )
        return checkpoint

    def rollback_cache(self, checkpoint, /):
        self.events.append(("checkpoint.rollback", checkpoint.cache_length))
        self.rollback_calls.append(checkpoint)
        return super().rollback_cache(checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.events.append(("checkpoint.release", checkpoint.cache_length))
        self.release_calls.append(checkpoint)
        return super().release_cache_checkpoint(checkpoint)

    def reset(self):
        self.reset_calls += 1
        return super().reset()


class FaultingCleanupBackend(RecordingBackend):
    def rollback_cache(self, checkpoint, /):
        self.events.append(("checkpoint.rollback", checkpoint.cache_length))
        self.rollback_calls.append(checkpoint)
        raise RuntimeError("rollback cleanup failed")

    def release_cache_checkpoint(self, checkpoint, /):
        self.events.append(("checkpoint.release", checkpoint.cache_length))
        self.release_calls.append(checkpoint)
        raise RuntimeError(f"checkpoint {checkpoint.cache_length} release failed")


class OpaqueMask:
    def __init__(self, *, events=None):
        self.events = events if events is not None else []
        self.calls = []
        self.masked_rows = []

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    def apply(self, logits, valid_token_ids, /):
        masked_row = object()
        self.events.append(("mask", logits, valid_token_ids))
        self.calls.append((logits, valid_token_ids))
        self.masked_rows.append(masked_row)
        return masked_row


class RecordingSelector:
    def __init__(self, outcomes, *, events=None):
        self.outcomes = list(outcomes)
        self.events = events if events is not None else []
        self.calls = []

    def __call__(self, row):
        self.events.append(("select", row))
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
        events=None,
    ):
        self._grammar_type = grammar_type
        self.supports = tuple(supports)
        self.starting_state = OpaqueState("start")
        child_count = max(0, len(self.supports) - 1)
        self.children = list(
            children
            if children is not None
            else (OpaqueState(f"child-{position}") for position in range(child_count))
        )
        self.matches = tuple(matches or (False,) * len(self.supports))
        self.events = events if events is not None else []
        self.advance_calls = []
        self.release_calls = []
        self.release_failures = {}
        self._live_children = []
        self.init_calls = 0
        self.bulk_release_calls = 0
        self.reset_calls = 0
        self.peak_owned_children = 0

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @property
    def grammar_type(self):
        return self._grammar_type

    def init_state(self):
        self.init_calls += 1
        raise AssertionError("D44 must not initialize grammar state")

    def advance_state(self, state, token_id, /):
        position = self._position(state)
        self.events.append(("advance", state, token_id))
        self.advance_calls.append((state, token_id))
        child = self.children[position]
        if not any(child is live for live in self._live_children):
            self._live_children.append(child)
        self.peak_owned_children = max(
            self.peak_owned_children,
            len(self._live_children),
        )
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
        self.events.append(("release_state", state))
        self.release_calls.append(state)
        remaining_failures = self.release_failures.get(position, 0)
        if remaining_failures:
            self.release_failures[position] = remaining_failures - 1
            raise RuntimeError(f"state {position} release failed")
        self._live_children = [
            live for live in self._live_children if live is not state
        ]

    def release_states(self, states, /):
        self.bulk_release_calls += 1
        raise AssertionError("D44 must not bulk-release grammar states")

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("D44 must not reset the grammar constraint")

    @property
    def live_child_count(self):
        return len(self._live_children)

    def _position(self, state):
        if state is self.starting_state:
            return 0
        for position, child in enumerate(self.children, start=1):
            if state is child and any(state is live for live in self._live_children):
                return position
        raise RuntimeError("state is not live")


def _prefilled(backend_class=RecordingBackend, *, events=None):
    backend = backend_class(events=events)
    backend.prefill((4, 3, 2))
    backend.events.clear()
    backend.reset_calls = 0
    return backend


def _run(
    backend,
    constraint,
    *,
    proposal_bound,
    mask=None,
    selector=None,
):
    return generate_grammar_masked_draft_proposal(
        backend,
        0,
        constraint,
        constraint.starting_state,
        OpaqueMask() if mask is None else mask,
        proposal_bound=proposal_bound,
        select_token=(
            RecordingSelector([1] * proposal_bound)
            if selector is None
            else selector
        ),
    )


def _release_result(backend, result):
    for checkpoint in result.rollback_checkpoints:
        backend.release_cache_checkpoint(checkpoint)


def test_public_contract_signature_result_and_error_hierarchy():
    assert grammar_draft_module.__all__ == [
        "GrammarMaskedDraftProposalCleanupError",
        "GrammarMaskedDraftProposalError",
        "GrammarMaskedDraftProposalInvariantError",
        "GrammarMaskedDraftProposalResult",
        "generate_grammar_masked_draft_proposal",
    ]
    assert issubclass(GrammarMaskedDraftProposalError, DraftProposalError)
    assert issubclass(GrammarMaskedDraftProposalError, BackendError)
    assert issubclass(
        GrammarMaskedDraftProposalInvariantError,
        GrammarMaskedDraftProposalError,
    )
    assert issubclass(
        GrammarMaskedDraftProposalCleanupError,
        GrammarMaskedDraftProposalError,
    )

    signature = inspect.signature(generate_grammar_masked_draft_proposal)
    assert list(signature.parameters) == [
        "backend",
        "current_token_id",
        "constraint",
        "starting_state",
        "logit_mask",
        "proposal_bound",
        "select_token",
    ]
    assert [parameter.kind for parameter in signature.parameters.values()] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.KEYWORD_ONLY,
    ]
    assert [field.name for field in fields(GrammarMaskedDraftProposalResult)] == [
        "proposal_token_ids",
        "rollback_checkpoints",
        "initial_cache_length",
        "final_cache_length",
        "shortening_selection",
    ]

    result = GrammarMaskedDraftProposalResult(
        (1,),
        (CheckpointRecord(4),),
        3,
        5,
        None,
    )
    assert result.shortened is False
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.final_cache_length = 6


@pytest.mark.parametrize("is_match", [False, True])
def test_zero_token_result_requires_exact_empty_support(is_match):
    selection = GrammarMaskedSelectionResult((), is_match, None)
    result = GrammarMaskedDraftProposalResult((), (), 3, 4, selection)
    assert result.shortened is True
    assert result.shortening_selection is selection

    with pytest.raises(ValueError, match="empty proposal requires"):
        GrammarMaskedDraftProposalResult((), (), 3, 4, None)
    with pytest.raises(ValueError, match="empty valid_token_ids"):
        GrammarMaskedDraftProposalResult(
            (1,),
            (CheckpointRecord(4),),
            3,
            5,
            GrammarMaskedSelectionResult((1,), is_match, 1),
        )


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("proposal_bound", [1, 3])
def test_full_bound_preserves_d32_mapping_and_settles_children(
    grammar_type,
    proposal_bound,
):
    events = []
    backend = _prefilled(events=events)
    supports = ((1, 3),) * (proposal_bound + 1)
    constraint = TrackingConstraint(
        supports,
        grammar_type=grammar_type,
        matches=(False,) * proposal_bound + (True,),
        events=events,
    )
    mask = OpaqueMask(events=events)
    selector = RecordingSelector([1] * proposal_bound, events=events)
    initial_length = backend.cache_length

    result = _run(
        backend,
        constraint,
        proposal_bound=proposal_bound,
        mask=mask,
        selector=selector,
    )

    assert result.proposal_token_ids == (1,) * proposal_bound
    assert tuple(cp.cache_length for cp in result.rollback_checkpoints) == tuple(
        initial_length + 1 + position for position in range(proposal_bound)
    )
    assert result.initial_cache_length == initial_length
    assert result.final_cache_length == initial_length + proposal_bound + 1
    assert result.shortening_selection is None
    assert result.shortened is False
    assert backend.decode_calls == [0, *([1] * proposal_bound)]
    assert backend.cached_token_ids == (4, 3, 2, 0, *([1] * proposal_bound))
    assert len(mask.calls) == proposal_bound
    assert [call[0] for call in mask.calls] == list(backend._scripted_logits[1:])[
        :proposal_bound
    ]
    assert selector.calls == mask.masked_rows
    assert len(constraint.advance_calls) == proposal_bound
    assert constraint.release_calls == constraint.children[:proposal_bound]
    assert constraint.live_child_count == 0
    assert constraint.peak_owned_children <= 2
    assert constraint.init_calls == 0
    assert constraint.bulk_release_calls == 0
    assert constraint.reset_calls == 0
    assert backend.peak_checkpoint_count == proposal_bound + 1
    assert backend.active_checkpoint_count == proposal_bound
    assert backend.reset_calls == 0

    _release_result(backend, result)
    assert backend.active_checkpoint_count == 0
    assert constraint.is_dead_state(constraint.starting_state) is False


@pytest.mark.parametrize("grammar_type", ["regex", "json_schema"])
@pytest.mark.parametrize("is_match", [False, True])
def test_empty_support_at_first_row_returns_exact_zero_token_evidence(
    grammar_type,
    is_match,
):
    backend = _prefilled()
    constraint = TrackingConstraint(
        ((),),
        grammar_type=grammar_type,
        matches=(is_match,),
    )
    mask = OpaqueMask()
    selector = RecordingSelector([1])

    result = _run(
        backend,
        constraint,
        proposal_bound=3,
        mask=mask,
        selector=selector,
    )

    assert result.proposal_token_ids == ()
    assert result.rollback_checkpoints == ()
    assert result.final_cache_length == result.initial_cache_length + 1
    assert result.shortened is True
    assert result.shortening_selection.valid_token_ids is constraint.supports[0]
    assert result.shortening_selection.is_match is is_match
    assert result.shortening_selection.selected_token_id is None
    assert backend.decode_calls == [0]
    assert backend.active_checkpoint_count == 0
    assert mask.calls == []
    assert selector.calls == []
    assert constraint.advance_calls == []
    assert constraint.release_calls == []


@pytest.mark.parametrize("shortening_position", [1, 2, 3])
def test_later_empty_support_shortens_at_exact_position(shortening_position):
    backend = _prefilled()
    supports = ((1,),) * shortening_position + ((),)
    constraint = TrackingConstraint(
        supports,
        matches=(False,) * shortening_position + (True,),
    )
    mask = OpaqueMask()
    selector = RecordingSelector([1] * shortening_position)

    result = _run(
        backend,
        constraint,
        proposal_bound=4,
        mask=mask,
        selector=selector,
    )

    assert result.proposal_token_ids == (1,) * shortening_position
    assert len(result.rollback_checkpoints) == shortening_position
    assert result.final_cache_length == result.initial_cache_length + shortening_position + 1
    assert result.shortening_selection.valid_token_ids == ()
    assert len(backend.decode_calls) == shortening_position + 1
    assert len(mask.calls) == shortening_position
    assert len(selector.calls) == shortening_position
    assert len(constraint.advance_calls) == shortening_position
    assert len(constraint.release_calls) == shortening_position
    assert backend.peak_checkpoint_count == shortening_position + 2
    assert backend.active_checkpoint_count == shortening_position
    assert backend.release_calls[0].cache_length == result.final_cache_length

    _release_result(backend, result)


def test_none_is_a_legal_transferred_child_and_controls_the_next_parent():
    backend = _prefilled()
    second_child = OpaqueState("second")
    constraint = TrackingConstraint(
        ((1,), (1,), ()),
        children=(None, second_child),
    )

    result = _run(backend, constraint, proposal_bound=2)

    assert result.proposal_token_ids == (1, 1)
    assert constraint.advance_calls[0][0] is constraint.starting_state
    assert constraint.advance_calls[1][0] is None
    assert constraint.release_calls == [None, second_child]
    assert constraint.live_child_count == 0
    _release_result(backend, result)


def test_every_inspected_row_uses_one_exact_d43_call_and_the_final_row_is_uninspected(
    monkeypatch,
):
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), (1,), ()))
    mask = OpaqueMask()
    selector = RecordingSelector([1, 1])
    calls = []
    original = grammar_draft_module.select_and_advance_grammar_state

    def record_call(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        grammar_draft_module,
        "select_and_advance_grammar_state",
        record_call,
    )

    result = _run(
        backend,
        constraint,
        proposal_bound=2,
        mask=mask,
        selector=selector,
    )

    assert len(calls) == 2
    assert calls[0][0] == (
        constraint,
        constraint.starting_state,
        backend._scripted_logits[1],
        mask,
    )
    assert calls[1][0] == (
        constraint,
        constraint.children[0],
        backend._scripted_logits[2],
        mask,
    )
    assert all(
        kwargs == {"vocab_size": VOCAB_SIZE, "select_token": selector}
        for _args, kwargs in calls
    )
    assert backend._scripted_logits[3] not in [args[2] for args, _kwargs in calls]
    _release_result(backend, result)


def test_matching_nonempty_parent_continues_to_the_bound():
    backend = _prefilled()
    constraint = TrackingConstraint(
        ((1,), ()),
        matches=(True, True),
    )

    result = _run(backend, constraint, proposal_bound=1)

    assert result.proposal_token_ids == (1,)
    assert result.shortening_selection is None
    assert constraint.advance_calls == [(constraint.starting_state, 1)]
    _release_result(backend, result)


@pytest.mark.parametrize("accepted_count", [0, 1, 2])
def test_returned_checkpoints_restore_exact_proposal_prefixes(accepted_count):
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), (1,), (1,), ()))
    result = _run(backend, constraint, proposal_bound=3)

    checkpoint = result.rollback_checkpoints[accepted_count]
    backend.rollback_cache(checkpoint)

    assert backend.cached_token_ids == (
        4,
        3,
        2,
        0,
        *((1,) * accepted_count),
    )
    assert backend.cache_length == result.initial_cache_length + 1 + accepted_count
    _release_result(backend, result)


@pytest.mark.parametrize(
    ("proposal_bound", "expected_error"),
    [
        (True, TypeError),
        (1.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_invalid_bound_fails_before_backend_or_grammar_work(
    proposal_bound,
    expected_error,
):
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), ()))
    before = (
        backend.cache_length,
        backend.cached_token_ids,
        backend.active_checkpoint_count,
    )

    with pytest.raises(expected_error):
        _run(backend, constraint, proposal_bound=proposal_bound)

    assert (
        backend.cache_length,
        backend.cached_token_ids,
        backend.active_checkpoint_count,
    ) == before
    assert backend.decode_calls == []
    assert constraint.advance_calls == []


def test_component_failure_after_current_decode_restores_exact_backend_state():
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), ()))
    original = RuntimeError("selector failed")
    selector = RecordingSelector([original])
    initial = (backend.cache_length, backend.cached_token_ids, backend._next_row)

    with pytest.raises(RuntimeError) as captured:
        _run(
            backend,
            constraint,
            proposal_bound=2,
            selector=selector,
        )

    assert captured.value is original
    assert (backend.cache_length, backend.cached_token_ids, backend._next_row) == initial
    assert backend.active_checkpoint_count == 0
    assert backend.decode_calls == [0]
    assert len(selector.calls) == 1
    assert constraint.release_calls == []


def test_transferred_child_is_owned_before_malformed_evidence_is_rejected(monkeypatch):
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), ()))
    child = constraint.children[0]
    selection = GrammarMaskedSelectionResult((1,), False, 1)
    transition = object.__new__(GrammarMaskedTransitionResult)
    object.__setattr__(transition, "selection", selection)
    object.__setattr__(transition, "child_state", child)
    object.__setattr__(transition, "child_is_match", "not-a-boolean")
    constraint._live_children.append(child)
    monkeypatch.setattr(
        grammar_draft_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )

    with pytest.raises(
        GrammarMaskedDraftProposalInvariantError,
        match="boolean child_is_match",
    ):
        _run(backend, constraint, proposal_bound=1)

    assert constraint.release_calls == [child]
    assert constraint.live_child_count == 0
    assert backend.active_checkpoint_count == 0
    assert backend.cached_token_ids == (4, 3, 2)


@pytest.mark.parametrize("alias_kind", ["start", "parent", "earlier"])
def test_child_aliases_are_rejected_by_identity_without_releasing_borrowed_start(
    monkeypatch,
    alias_kind,
):
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), (1,), ()))
    calls = 0

    def transition(_constraint, parent, *args, **kwargs):
        nonlocal calls
        selection = GrammarMaskedSelectionResult((1,), False, 1)
        if calls == 0 and alias_kind == "start":
            child = constraint.starting_state
        elif calls == 0:
            child = constraint.children[0]
            constraint._live_children.append(child)
        elif calls == 1 and alias_kind == "earlier":
            child = constraint.children[1]
            constraint._live_children.append(child)
        elif alias_kind == "parent":
            child = parent
        else:
            child = constraint.children[0]
        calls += 1
        return GrammarMaskedTransitionResult(selection, child, False)

    monkeypatch.setattr(
        grammar_draft_module,
        "select_and_advance_grammar_state",
        transition,
    )

    with pytest.raises(GrammarMaskedDraftProposalInvariantError, match="aliases"):
        _run(
            backend,
            constraint,
            proposal_bound={"start": 1, "parent": 2, "earlier": 3}[alias_kind],
        )

    assert all(
        state is not constraint.starting_state for state in constraint.release_calls
    )
    assert constraint.is_dead_state(constraint.starting_state) is False


def test_cross_domain_cleanup_reports_exact_global_order_and_identities():
    backend = _prefilled(FaultingCleanupBackend)
    constraint = TrackingConstraint(((1,), (1,), ()))
    constraint.release_failures[1] = 1
    original = RuntimeError("second selector failed")
    selector = RecordingSelector([1, original])

    with pytest.raises(GrammarMaskedDraftProposalCleanupError) as captured:
        _run(
            backend,
            constraint,
            proposal_bound=3,
            selector=selector,
        )

    error = captured.value
    assert error.original_failure is original
    assert error.__cause__ is original
    assert tuple(label for label, _failure in error.cleanup_failures) == (
        "start checkpoint rollback",
        "rollback checkpoint 0 release",
        "rollback checkpoint 1 release",
        "start checkpoint release",
        "grammar state release at position 0",
    )
    assert type(error.cleanup_failures) is tuple
    assert all(type(entry) is tuple for entry in error.cleanup_failures)


def test_nested_d43_cleanup_failure_remains_the_exact_original_failure(monkeypatch):
    backend = _prefilled(FaultingCleanupBackend)
    constraint = TrackingConstraint(((1,), ()))
    transition_failure = RuntimeError("transition failed")
    transition_cleanup = RuntimeError("transition cleanup failed")
    nested = GrammarMaskedTransitionCleanupError(
        transition_failure,
        (("child state release", transition_cleanup),),
    )

    def fail_transition(*args, **kwargs):
        raise nested

    monkeypatch.setattr(
        grammar_draft_module,
        "select_and_advance_grammar_state",
        fail_transition,
    )

    with pytest.raises(GrammarMaskedDraftProposalCleanupError) as captured:
        _run(backend, constraint, proposal_bound=1)

    assert captured.value.original_failure is nested
    assert captured.value.__cause__ is nested
    assert all(failure is not transition_cleanup for _label, failure in captured.value.cleanup_failures)


def test_result_construction_failure_restores_cache_and_settles_child(monkeypatch):
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), ()))
    original = RuntimeError("result construction failed")

    class FailingResult:
        def __init__(self, **kwargs):
            raise original

    monkeypatch.setattr(
        grammar_draft_module,
        "GrammarMaskedDraftProposalResult",
        FailingResult,
    )

    with pytest.raises(RuntimeError) as captured:
        _run(backend, constraint, proposal_bound=1)

    assert captured.value is original
    assert backend.cached_token_ids == (4, 3, 2)
    assert backend.active_checkpoint_count == 0
    assert constraint.release_calls == [constraint.children[0]]
    assert constraint.live_child_count == 0


def test_success_path_state_release_failure_is_retried_only_in_failure_cleanup():
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), (1,), ()))
    constraint.release_failures[1] = 1

    with pytest.raises(RuntimeError, match="state 1 release failed"):
        _run(backend, constraint, proposal_bound=2)

    first_child, second_child = constraint.children
    assert constraint.release_calls == [first_child, first_child, second_child]
    assert constraint.live_child_count == 0
    assert backend.cached_token_ids == (4, 3, 2)
    assert backend.active_checkpoint_count == 0


def test_final_state_release_failure_composes_with_failed_cleanup_retry():
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), ()))
    constraint.release_failures[1] = 2

    with pytest.raises(GrammarMaskedDraftProposalCleanupError) as captured:
        _run(backend, constraint, proposal_bound=1)

    error = captured.value
    assert isinstance(error.original_failure, RuntimeError)
    assert str(error.original_failure) == "state 1 release failed"
    assert tuple(label for label, _failure in error.cleanup_failures) == (
        "grammar state release at position 0",
    )
    assert constraint.release_calls == [
        constraint.children[0],
        constraint.children[0],
    ]
    assert backend.cached_token_ids == (4, 3, 2)
    assert backend.active_checkpoint_count == 0


def test_cleanup_error_rejects_empty_failure_sequence_and_preserves_members():
    original = RuntimeError("original")
    cleanup = RuntimeError("cleanup")
    error = GrammarMaskedDraftProposalCleanupError(
        original,
        (("operation", cleanup),),
    )
    assert error.original_failure is original
    assert error.cleanup_failures == (("operation", cleanup),)
    assert error.cleanup_failures[0][1] is cleanup
    assert error.__cause__ is original
    with pytest.raises(ValueError, match="cannot be empty"):
        GrammarMaskedDraftProposalCleanupError(original, ())


def test_normal_import_keeps_optional_runtimes_out_of_the_module_graph():
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        import onyx_cuda
        assert onyx_cuda.generate_grammar_masked_draft_proposal.__module__ == (
            "onyx_cuda.grammar_draft"
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
        env={"PYTHONPATH": str(PACKAGE_ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_module_can_be_reimported_without_optional_runtime_side_effects():
    imported = importlib.import_module("onyx_cuda.grammar_draft")
    assert imported.generate_grammar_masked_draft_proposal is (
        generate_grammar_masked_draft_proposal
    )


def test_model_step_failure_after_decode_is_failure_atomic():
    class BadStepBackend(RecordingBackend):
        def decode(self, token_id, /):
            super().decode(token_id)
            return object()

    backend = _prefilled(BadStepBackend)
    constraint = TrackingConstraint(((1,), ()))

    with pytest.raises(
        GrammarMaskedDraftProposalInvariantError,
        match="must return a ModelStep",
    ):
        _run(backend, constraint, proposal_bound=1)

    assert backend.cached_token_ids == (4, 3, 2)
    assert backend.cache_length == 3
    assert backend.active_checkpoint_count == 0


def test_forged_non_transition_with_nonempty_support_is_rejected(monkeypatch):
    backend = _prefilled()
    constraint = TrackingConstraint(((1,), ()))
    selection = GrammarMaskedSelectionResult((1,), False, 1)
    transition = GrammarMaskedTransitionResult(selection, object(), False)
    object.__setattr__(selection, "selected_token_id", None)
    monkeypatch.setattr(
        grammar_draft_module,
        "select_and_advance_grammar_state",
        lambda *args, **kwargs: transition,
    )

    with pytest.raises(
        GrammarMaskedDraftProposalInvariantError,
        match="nonempty support requires an integer selected token",
    ):
        _run(backend, constraint, proposal_bound=1)

    assert backend.cached_token_ids == (4, 3, 2)
    assert backend.active_checkpoint_count == 0


def test_result_checkpoint_positions_and_cache_formula_are_enforced():
    with pytest.raises(ValueError, match="one checkpoint per proposal token"):
        GrammarMaskedDraftProposalResult((1,), (), 3, 5, None)
    with pytest.raises(
        GrammarMaskedDraftProposalInvariantError,
        match="expected 5",
    ):
        GrammarMaskedDraftProposalResult(
            (1,),
            (CheckpointRecord(4),),
            3,
            6,
            None,
        )
    with pytest.raises(
        GrammarMaskedDraftProposalInvariantError,
        match="reports cache length 3; expected 4",
    ):
        GrammarMaskedDraftProposalResult(
            (1,),
            (CheckpointRecord(3),),
            3,
            5,
            None,
        )


def test_repeated_full_and_shortened_calls_leave_bounded_owned_resources():
    backend = RecordingBackend(SCRIPT)
    backend.prefill((4, 3, 2))
    backend.reset_calls = 0
    first_discarded_checkpoint = None
    for iteration in range(1000):
        root = backend.create_cache_checkpoint()
        if iteration % 2:
            constraint = TrackingConstraint(((1,), (1,), ()))
            result = _run(backend, constraint, proposal_bound=2)
            assert len(result.proposal_token_ids) == 2
        else:
            constraint = TrackingConstraint(((),), matches=(True,))
            result = _run(backend, constraint, proposal_bound=2)
            assert result.proposal_token_ids == ()
        if first_discarded_checkpoint is None and result.rollback_checkpoints:
            first_discarded_checkpoint = result.rollback_checkpoints[0]
        backend.rollback_cache(root)
        _release_result(backend, result)
        backend.release_cache_checkpoint(root)
        assert backend.cached_token_ids == (4, 3, 2)
        assert backend.cache_length == 3
        assert backend._next_row == 1
        assert backend.active_checkpoint_count == 0
        assert constraint.live_child_count == 0
        assert constraint.is_dead_state(constraint.starting_state) is False
    assert backend._next_checkpoint_id == 3501
    assert backend.reset_calls == 0
    with pytest.raises(RuntimeError):
        backend.rollback_cache(first_discarded_checkpoint)
