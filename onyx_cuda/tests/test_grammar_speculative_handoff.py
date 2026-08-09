import gc
import inspect
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_speculative_handoff as handoff_module
import onyx_cuda.grammar_speculative_iteration as iteration_module
from onyx_cuda import (
    BackendError,
    GrammarMaskedDraftProposalResult,
    GrammarMaskedSelectionResult,
    GrammarMaskedSpeculativeHandoffCleanupError,
    GrammarMaskedSpeculativeHandoffError,
    GrammarMaskedSpeculativeHandoffInvariantError,
    GrammarMaskedSpeculativeHandoffResult,
    GrammarMaskedSpeculativeIterationResult,
    GrammarMaskedSpeculativeOutcomeError,
    GrammarMaskedSpeculativeOutcomeResult,
    coordinate_grammar_masked_speculative_handoff,
)
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    FakeGrammarConstraint,
    FakeGrammarProgram,
)


VOCAB_SIZE = 8
PROMPT = (7,)
CURRENT_TOKEN_ID = 0
PROPOSAL_BOUND = 2
SCRIPT = tuple(
    tuple(float(row * VOCAB_SIZE + token) for token in range(VOCAB_SIZE))
    for row in range(128)
)
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


class MalformedIntermediateTarget(RecordingBackend):
    class MalformedCheckpoint:
        pass

    def __init__(self, *, model_id):
        self.malformed_checkpoint = None
        self.malformed_release_calls = []
        super().__init__(model_id=model_id)

    def create_cache_checkpoint(self):
        if len(self.create_calls) == 1:
            self.malformed_checkpoint = self.MalformedCheckpoint()
            return self.malformed_checkpoint
        return super().create_cache_checkpoint()

    def release_cache_checkpoint(self, checkpoint, /):
        if checkpoint is self.malformed_checkpoint:
            self.malformed_release_calls.append(checkpoint)
            return None
        return super().release_cache_checkpoint(checkpoint)


class MutatingIntermediateTarget(RecordingBackend):
    def create_cache_checkpoint(self):
        checkpoint = super().create_cache_checkpoint()
        if len(self.create_calls) == 2:
            self.decode(0)
        return checkpoint


class RecordingConstraint(FakeGrammarConstraint):
    def __init__(self, *args, force_nonmatch=False, **kwargs):
        self.advance_calls = []
        self.release_calls = []
        self.force_nonmatch = force_nonmatch
        super().__init__(*args, **kwargs)

    def advance_state(self, state, token_id, /):
        self.advance_calls.append((state, token_id))
        return super().advance_state(state, token_id)

    def release_state(self, state, /):
        self.release_calls.append(state)
        return super().release_state(state)

    def is_match_state(self, state, /):
        value = super().is_match_state(state)
        return False if self.force_nonmatch else value


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
    def __init__(self):
        self.starting_state = object()
        self.live_start = True
        self.live_none = False
        self.advance_calls = []
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
        self._require_live(state)
        self.advance_calls.append((state, token_id))
        if state is not self.starting_state or token_id not in {1, 3}:
            raise RuntimeError("unsupported transition")
        self.live_none = True
        return None

    def get_valid_token_ids(self, state, /):
        self._require_live(state)
        if state is self.starting_state:
            return (1, 3)
        return ()

    def is_match_state(self, state, /):
        self._require_live(state)
        return state is None

    def is_dead_state(self, state, /):
        self._require_live(state)
        return False

    def release_state(self, state, /):
        self.release_calls.append(state)
        if state is self.starting_state:
            self.live_start = False
        elif state is None:
            self.live_none = False

    def release_states(self, states, /):
        for state in states:
            self.release_state(state)

    def reset(self):
        self.live_start = False
        self.live_none = False

    def _require_live(self, state):
        if state is self.starting_state and self.live_start:
            return
        if state is None and self.live_none:
            return
        raise RuntimeError("unknown or released state")


def _vocabulary():
    return tuple(bytes((token,)) for token in range(VOCAB_SIZE))


def _constraint(program, *, force_nonmatch=False):
    return RecordingConstraint(
        _vocabulary(),
        grammar_type="regex",
        program=program,
        force_nonmatch=force_nonmatch,
    )


def _terminal_constraint(route, *, is_match):
    if route == "zero":
        program = FakeGrammarProgram(
            initial_state="s0",
            transitions=(),
            valid_token_ids=(("s0", ()),),
            match_states=frozenset({"s0"}),
        )
        return _constraint(program, force_nonmatch=not is_match), [], []
    if route == "final_empty":
        program = FakeGrammarProgram(
            initial_state="s0",
            transitions=(("s0", 1, "s1"), ("s1", 2, "done")),
            valid_token_ids=(("s0", (1,)), ("s1", (2,)), ("done", ())),
            match_states=frozenset({"done"}),
        )
        return _constraint(program, force_nonmatch=not is_match), [1, 2], [1, 2]
    if route == "no_decision_0":
        program = FakeGrammarProgram(
            initial_state="s0",
            transitions=(),
            valid_token_ids=(("s0", ()),),
            match_states=frozenset({"s0"}),
        )
        return _constraint(program, force_nonmatch=not is_match), [], []
    if route == "no_decision_1":
        program = FakeGrammarProgram(
            initial_state="s0",
            transitions=(("s0", 1, "s1"),),
            valid_token_ids=(("s0", (1,)), ("s1", ())),
            match_states=frozenset({"s1"}),
        )
        return _constraint(program, force_nonmatch=not is_match), [], [1]
    raise AssertionError(f"unsupported terminal route: {route}")


def _two_stage_constraint(second_route, *, second_match=True):
    transitions = [
        ("s0", 1, "d1"),
        ("s0", 3, "handoff"),
        ("d1", 2, "d2"),
    ]
    supports = [("s0", (1, 3)), ("d1", (2,)), ("d2", ())]
    match_states = {"d2"}
    draft_outcomes = [1, 2]
    target_outcomes = [3]

    if second_route == "zero":
        supports.append(("handoff", ()))
        match_states.add("handoff")
    elif second_route in {"final_empty", "repeat"}:
        first_token = 3 if second_route == "repeat" else 4
        transitions.extend(
            (
                ("handoff", first_token, "q1"),
                ("q1", 5, "done"),
            )
        )
        supports.extend(
            (
                ("handoff", (first_token,)),
                ("q1", (5,)),
                ("done", ()),
            )
        )
        draft_outcomes.extend((first_token, 5))
        target_outcomes.extend((first_token, 5))
        match_states.add("done")
    elif second_route == "mismatch":
        transitions.extend(
            (
                ("handoff", 4, "q1"),
                ("handoff", 6, "replacement"),
                ("q1", 5, "q2"),
            )
        )
        supports.extend(
            (
                ("handoff", (4, 6)),
                ("q1", (5,)),
                ("q2", ()),
                ("replacement", ()),
            )
        )
        draft_outcomes.extend((4, 5))
        target_outcomes.append(6)
        match_states.update(("q2", "replacement"))
    elif second_route == "bonus":
        transitions.extend(
            (
                ("handoff", 4, "q1"),
                ("q1", 5, "q2"),
                ("q2", 6, "bonus"),
            )
        )
        supports.extend(
            (
                ("handoff", (4,)),
                ("q1", (5,)),
                ("q2", (6,)),
                ("bonus", ()),
            )
        )
        draft_outcomes.extend((4, 5))
        target_outcomes.extend((4, 5, 6))
        match_states.add("bonus")
    else:
        raise AssertionError(f"unsupported second route: {second_route}")

    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=tuple(transitions),
        valid_token_ids=tuple(supports),
        match_states=frozenset(match_states),
    )
    force_nonmatch = not second_match and second_route in {"zero", "final_empty"}
    return (
        _constraint(program, force_nonmatch=force_nonmatch),
        draft_outcomes,
        target_outcomes,
    )


def _first_mismatch_constraint(position):
    proposal = (1, 2, 4)
    states = ("s0", "s1", "s2")
    transitions = []
    supports = []
    match_states = {"s3"}
    for index, (state, proposal_token) in enumerate(zip(states, proposal)):
        next_state = f"s{index + 1}"
        handoff_state = f"handoff_{index}"
        transitions.extend(
            (
                (state, proposal_token, next_state),
                (state, 6, handoff_state),
            )
        )
        supports.append((state, (proposal_token, 6)))
        supports.append((handoff_state, ()))
        match_states.add(handoff_state)
    supports.append(("s3", ()))
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=tuple(transitions),
        valid_token_ids=tuple(supports),
        match_states=frozenset(match_states),
    )
    target_outcomes = list(proposal[:position]) + [6]
    return _constraint(program), list(proposal), target_outcomes


def _first_bonus_constraint():
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(
            ("s0", 1, "s1"),
            ("s1", 2, "s2"),
            ("s2", 3, "handoff"),
        ),
        valid_token_ids=(
            ("s0", (1,)),
            ("s1", (2,)),
            ("s2", (3,)),
            ("handoff", ()),
        ),
        match_states=frozenset({"handoff"}),
    )
    return _constraint(program), [1, 2], [1, 2, 3]


def _second_no_decision_constraint(position, *, is_match):
    transitions = [
        ("s0", 1, "d1"),
        ("s0", 3, "handoff"),
        ("d1", 2, "d2"),
    ]
    supports = [("s0", (1, 3)), ("d1", (2,)), ("d2", ())]
    match_states = {"d2"}
    target_outcomes = [3]
    if position == 0:
        supports.append(("handoff", ()))
        match_states.add("handoff")
    else:
        transitions.append(("handoff", 4, "terminal"))
        supports.extend((('handoff', (4,)), ("terminal", ())))
        match_states.add("terminal")
        target_outcomes.append(4)
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=tuple(transitions),
        valid_token_ids=tuple(supports),
        match_states=frozenset(match_states),
    )
    return (
        _constraint(program, force_nonmatch=not is_match),
        [1, 2],
        target_outcomes,
    )


def _prefilled_pair(*, draft_type=RecordingBackend, target_type=RecordingBackend):
    draft = draft_type(model_id="draft")
    target = target_type(model_id="target")
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    return draft, target, draft_root, target_root


def _coordinate(
    constraint,
    *,
    draft_outcomes,
    target_outcomes,
    proposal_bound=PROPOSAL_BOUND,
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
    result = coordinate_grammar_masked_speculative_handoff(
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


def _selection(is_match):
    return GrammarMaskedSelectionResult((), is_match, None)


def _direct_iteration(route, *, state=object(), initial_cache_length=1):
    if route == "handoff":
        return GrammarMaskedSpeculativeIterationResult(
            proposal_token_ids=(1,),
            accepted_count=0,
            replacement_token_id=2,
            initial_cache_length=initial_cache_length,
            final_cache_length=initial_cache_length + 1,
            uncached_next_token_id=2,
            shortening_selection=None,
            acceptance_no_decision_selection=None,
            final_row_no_decision_selection=None,
            committed_state=state,
            committed_state_is_match=False,
        )
    if route == "terminal":
        terminal = _selection(True)
        return GrammarMaskedSpeculativeIterationResult(
            proposal_token_ids=(),
            accepted_count=0,
            replacement_token_id=None,
            initial_cache_length=initial_cache_length,
            final_cache_length=initial_cache_length + 1,
            uncached_next_token_id=None,
            shortening_selection=terminal,
            acceptance_no_decision_selection=None,
            final_row_no_decision_selection=None,
            committed_state=state,
            committed_state_is_match=True,
        )
    raise AssertionError(route)


def _install_mock_d44(monkeypatch, proposal_token_ids):
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
            shortening_selection=None if len(proposal) == proposal_bound else _selection(False),
        )

    monkeypatch.setattr(
        iteration_module,
        "generate_grammar_masked_draft_proposal",
        generate,
    )


def _install_second_mock_d44(monkeypatch, proposal_token_ids):
    real_generate = iteration_module.generate_grammar_masked_draft_proposal
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
        if current_token_id != 3:
            return real_generate(
                backend,
                current_token_id,
                constraint,
                starting_state,
                logit_mask,
                proposal_bound=proposal_bound,
                select_token=select_token,
            )
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
            shortening_selection=None,
        )

    monkeypatch.setattr(
        iteration_module,
        "generate_grammar_masked_draft_proposal",
        generate,
    )


def _record_composed_calls(monkeypatch):
    d47_calls = []
    d48_calls = []
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    real_d48 = handoff_module.classify_grammar_masked_speculative_outcome

    def record_d47(*args, **kwargs):
        result = real_d47(*args, **kwargs)
        d47_calls.append((args, kwargs, result))
        return result

    def record_d48(result):
        outcome = real_d48(result)
        d48_calls.append((result, outcome))
        return outcome

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        record_d47,
    )
    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        record_d48,
    )
    return d47_calls, d48_calls


def test_public_surface_signature_result_and_error_hierarchy():
    assert handoff_module.__all__ == [
        "GrammarMaskedSpeculativeHandoffCleanupError",
        "GrammarMaskedSpeculativeHandoffError",
        "GrammarMaskedSpeculativeHandoffInvariantError",
        "GrammarMaskedSpeculativeHandoffResult",
        "coordinate_grammar_masked_speculative_handoff",
        "coordinate_multi_iteration_grammar_masked_speculative_handoff",
    ]
    assert issubclass(
        GrammarMaskedSpeculativeHandoffError,
        GrammarMaskedSpeculativeOutcomeError,
    )
    assert issubclass(
        GrammarMaskedSpeculativeHandoffInvariantError,
        GrammarMaskedSpeculativeHandoffError,
    )
    assert issubclass(
        GrammarMaskedSpeculativeHandoffCleanupError,
        GrammarMaskedSpeculativeHandoffError,
    )
    assert issubclass(GrammarMaskedSpeculativeHandoffError, BackendError)

    signature = inspect.signature(coordinate_grammar_masked_speculative_handoff)
    assert list(signature.parameters) == [
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
    ]
    parameters = tuple(signature.parameters.values())
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters[:7]
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters[7:]
    )


def test_direct_result_is_frozen_slotted_minimal_and_validates_relationships():
    iteration = _direct_iteration("handoff")
    outcome = GrammarMaskedSpeculativeOutcomeResult("handoff_available")
    result = GrammarMaskedSpeculativeHandoffResult((7, 2), iteration, outcome)

    assert [field.name for field in fields(result)] == [
        "output_token_ids",
        "final_iteration",
        "final_outcome",
    ]
    assert not hasattr(result, "__dict__")
    assert result.final_iteration is iteration
    assert result.final_outcome is outcome
    with pytest.raises(FrozenInstanceError):
        result.output_token_ids = ()

    terminal = _direct_iteration("terminal")
    GrammarMaskedSpeculativeHandoffResult(
        (),
        terminal,
        GrammarMaskedSpeculativeOutcomeResult("grammar_complete"),
    )
    with pytest.raises(TypeError, match="exact tuple"):
        GrammarMaskedSpeculativeHandoffResult(
            [2],
            iteration,
            outcome,
        )
    with pytest.raises(TypeError, match="position 0"):
        GrammarMaskedSpeculativeHandoffResult((True,), iteration, outcome)
    with pytest.raises(ValueError, match="cannot be negative"):
        GrammarMaskedSpeculativeHandoffResult((-1, 2), iteration, outcome)
    with pytest.raises(TypeError, match="final_iteration"):
        GrammarMaskedSpeculativeHandoffResult((2,), object(), outcome)
    with pytest.raises(TypeError, match="final_outcome"):
        GrammarMaskedSpeculativeHandoffResult((2,), iteration, object())
    with pytest.raises(GrammarMaskedSpeculativeHandoffInvariantError, match="suffix"):
        GrammarMaskedSpeculativeHandoffResult((1,), iteration, outcome)
    with pytest.raises(GrammarMaskedSpeculativeHandoffInvariantError, match="disagrees"):
        GrammarMaskedSpeculativeHandoffResult(
            (2,),
            iteration,
            GrammarMaskedSpeculativeOutcomeResult("grammar_complete"),
        )


def test_cleanup_error_freezes_ordered_evidence_and_sets_cause():
    original = RuntimeError("original")
    failures = [("one", RuntimeError("cleanup"))]
    error = GrammarMaskedSpeculativeHandoffCleanupError(original, failures)
    failures.append(("two", RuntimeError("later")))
    assert error.original_failure is original
    assert error.__cause__ is original
    assert error.cleanup_failures == (("one", error.cleanup_failures[0][1]),)
    with pytest.raises(ValueError, match="cannot be empty"):
        GrammarMaskedSpeculativeHandoffCleanupError(original, [])


@pytest.mark.parametrize(
    "route",
    ["zero", "no_decision_0", "no_decision_1", "final_empty"],
)
@pytest.mark.parametrize("is_match", [False, True])
def test_every_first_terminal_route_returns_after_one_classification(
    monkeypatch,
    route,
    is_match,
):
    if route.startswith("no_decision"):
        _install_mock_d44(monkeypatch, (1, 2))
    d47_calls, d48_calls = _record_composed_calls(monkeypatch)
    constraint, draft_outcomes, target_outcomes = _terminal_constraint(
        route,
        is_match=is_match,
    )
    coordinated = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
    )
    result, _, draft, target, draft_root, target_root, *rest = coordinated

    assert len(d47_calls) == 1
    assert len(d48_calls) == 1
    assert result.final_iteration is d47_calls[0][2]
    assert result.final_outcome is d48_calls[0][1]
    assert result.final_outcome.kind == (
        "grammar_complete" if is_match else "grammar_no_continuation"
    )
    assert result.output_token_ids == result.final_iteration.output_token_ids
    assert target.create_calls == [target_root]
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls
    assert draft.active_checkpoint_count == 1
    assert target.active_checkpoint_count == 1
    assert draft.cached_token_ids == target.cached_token_ids
    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert constraint.active_state_count == 0
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert not rest[-2].outcomes
    assert not rest[-1].outcomes


@pytest.mark.parametrize(
    ("second_route", "second_match", "expected_kind", "expected_output"),
    [
        ("zero", True, "grammar_complete", (3,)),
        ("zero", False, "grammar_no_continuation", (3,)),
        ("final_empty", True, "grammar_complete", (3, 4, 5)),
        ("final_empty", False, "grammar_no_continuation", (3, 4, 5)),
        ("mismatch", True, "handoff_available", (3, 6)),
        ("bonus", True, "handoff_available", (3, 4, 5, 6)),
    ],
)
def test_first_handoff_runs_exactly_one_second_transaction_for_every_outcome(
    monkeypatch,
    second_route,
    second_match,
    expected_kind,
    expected_output,
):
    d47_calls, d48_calls = _record_composed_calls(monkeypatch)
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint(
        second_route,
        second_match=second_match,
    )
    coordinated = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
    )
    (
        result,
        _,
        draft,
        target,
        draft_root,
        target_root,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
    ) = coordinated

    assert len(d47_calls) == 2
    assert len(d48_calls) == 2
    first_args, first_kwargs, first_iteration = d47_calls[0]
    second_args, second_kwargs, second_iteration = d47_calls[1]
    assert first_args[0] is second_args[0] is draft
    assert first_args[1] is second_args[1] is target
    assert second_args[2] == first_iteration.uncached_next_token_id == 3
    assert second_args[3] is constraint
    assert second_args[4] is first_iteration.committed_state
    assert first_args[5] is second_args[5] is draft_mask
    assert first_args[6] is second_args[6] is target_mask
    assert first_kwargs["proposal_bound"] == second_kwargs["proposal_bound"]
    assert first_kwargs["draft_select_token"] is second_kwargs["draft_select_token"]
    assert first_kwargs["target_select_token"] is second_kwargs["target_select_token"]
    assert first_kwargs["draft_select_token"] is draft_selector
    assert first_kwargs["target_select_token"] is target_selector
    assert second_kwargs["draft_root_checkpoint"].cache_length == (
        first_iteration.final_cache_length
    )
    assert second_kwargs["target_root_checkpoint"].cache_length == (
        first_iteration.final_cache_length
    )
    assert result.final_iteration is second_iteration
    assert result.final_outcome is d48_calls[1][1]
    assert result.final_outcome.kind == expected_kind
    assert result.output_token_ids == expected_output
    assert result.output_token_ids == (
        first_iteration.output_token_ids + second_iteration.output_token_ids
    )
    assert second_iteration.initial_cache_length == first_iteration.final_cache_length
    assert draft.cache_length == target.cache_length == second_iteration.final_cache_length
    expected_cache = (
        PROMPT
        + (CURRENT_TOKEN_ID,)
        + first_iteration.accepted_token_ids
        + (3,)
        + second_iteration.accepted_token_ids
    )
    assert draft.cached_token_ids == target.cached_token_ids == expected_cache
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls
    assert not draft_selector.outcomes
    assert not target_selector.outcomes

    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert constraint.active_state_count == 0
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT


@pytest.mark.parametrize("position", [0, 1, 2])
def test_first_mismatch_at_every_proposal_position_hands_off_exactly_once(position):
    constraint, draft_outcomes, target_outcomes = _first_mismatch_constraint(position)
    result, _, draft, target, *_ = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
        proposal_bound=3,
    )
    accepted_prefix = (1, 2, 4)[:position]
    assert result.output_token_ids == accepted_prefix + (6,)
    assert result.final_outcome.kind == "grammar_complete"
    assert draft.cached_token_ids == target.cached_token_ids == (
        PROMPT + (CURRENT_TOKEN_ID,) + accepted_prefix + (6,)
    )
    assert [token for _, token in constraint.advance_calls].count(6) == 1


def test_first_full_acceptance_bonus_hands_off_without_duplicate_emission():
    constraint, draft_outcomes, target_outcomes = _first_bonus_constraint()
    result, _, draft, target, *_ = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
    )
    assert result.output_token_ids == (1, 2, 3)
    assert result.final_outcome.kind == "grammar_complete"
    assert draft.cached_token_ids == target.cached_token_ids == (7, 0, 1, 2, 3)
    assert [token for _, token in constraint.advance_calls].count(3) == 1


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("is_match", [False, True])
def test_second_d45_no_decision_at_every_position_returns_terminal_evidence(
    monkeypatch,
    position,
    is_match,
):
    _install_second_mock_d44(monkeypatch, (4, 5))
    d47_calls, d48_calls = _record_composed_calls(monkeypatch)
    constraint, draft_outcomes, target_outcomes = _second_no_decision_constraint(
        position,
        is_match=is_match,
    )
    result, _, draft, target, *_ = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
    )
    assert len(d47_calls) == len(d48_calls) == 2
    assert result.final_outcome.kind == (
        "grammar_complete" if is_match else "grammar_no_continuation"
    )
    assert result.output_token_ids == (3,) + (4,)[:position]
    assert draft.cached_token_ids == target.cached_token_ids == (
        PROMPT + (CURRENT_TOKEN_ID, 3) + (4,)[:position]
    )
    assert result.final_iteration.acceptance_no_decision_selection is not None


def test_handoff_token_is_not_advanced_or_emitted_again_by_d49(monkeypatch):
    d47_calls, _ = _record_composed_calls(monkeypatch)
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("final_empty")
    result, *rest = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
    )
    draft, target = rest[1:3]

    assert result.output_token_ids == (3, 4, 5)
    assert [token for _, token in constraint.advance_calls].count(3) == 1
    assert d47_calls[1][0][2] == 3
    assert draft.cached_token_ids == target.cached_token_ids == (7, 0, 3, 4, 5)


def test_naturally_repeated_handoff_token_is_preserved():
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("repeat")
    result, _, draft, target, *_ = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
    )
    assert result.output_token_ids == (3, 3, 5)
    assert draft.cached_token_ids == target.cached_token_ids == (7, 0, 3, 3, 5)


def test_opaque_none_intermediate_and_final_state_use_explicit_ownership():
    constraint = NoneStateConstraint()
    coordinated = _coordinate(
        constraint,
        draft_outcomes=[1],
        target_outcomes=[3],
        proposal_bound=1,
    )
    result, starting_state, draft, target, draft_root, target_root, *_ = coordinated
    assert result.output_token_ids == (3,)
    assert result.final_outcome.kind == "grammar_complete"
    assert result.final_iteration.committed_state is None
    assert constraint.live_start is False
    assert constraint.live_none is True
    assert [token for _, token in constraint.advance_calls].count(3) == 1
    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert not constraint.live_none
    assert starting_state is constraint.starting_state


@pytest.mark.parametrize("bad_length", [0, -1, True, 1.5])
def test_initial_root_preflight_rejects_without_backend_or_state_work(bad_length):
    class Root:
        cache_length = bad_length

    constraint, _, _ = _terminal_constraint("zero", is_match=True)
    draft, target, draft_root, _ = _prefilled_pair()
    starting_state = constraint.init_state()
    initial_rollbacks = (len(draft.rollback_calls), len(target.rollback_calls))
    error = TypeError if bad_length is True or bad_length == 1.5 else ValueError
    with pytest.raises(error):
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector([]),
            target_select_token=RecordingSelector([]),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=Root(),
        )
    assert (len(draft.rollback_calls), len(target.rollback_calls)) == initial_rollbacks
    assert constraint.active_state_count == 1
    constraint.release_state(starting_state)


def test_first_d47_failure_propagates_exactly_without_outer_cleanup(monkeypatch):
    failure = RuntimeError("first D47 failure")
    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )
    constraint, _, _ = _terminal_constraint("zero", is_match=True)
    draft, target, draft_root, target_root = _prefilled_pair()
    starting_state = constraint.init_state()
    with pytest.raises(RuntimeError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector([]),
            target_select_token=RecordingSelector([]),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert draft.rollback_calls == []
    assert target.rollback_calls == []
    assert constraint.active_state_count == 1
    constraint.release_state(starting_state)


def test_d48_failure_enters_outer_cleanup_and_propagates_by_identity(monkeypatch):
    failure = RuntimeError("classification failed")
    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        lambda result: (_ for _ in ()).throw(failure),
    )
    constraint, draft_outcomes, target_outcomes = _terminal_constraint(
        "final_empty",
        is_match=True,
    )
    draft, target, draft_root, target_root = _prefilled_pair()
    starting_state = constraint.init_state()
    with pytest.raises(RuntimeError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert constraint.active_state_count == 0
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls


def test_second_d47_failure_restores_initial_roots_and_settles_state(monkeypatch):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    failure = RuntimeError("second D47 failure")
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return real_d47(*args, **kwargs)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        fail_second,
    )
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
    draft, target, draft_root, target_root = _prefilled_pair()
    starting_state = constraint.init_state()
    with pytest.raises(RuntimeError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert calls == 2
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_second_d47_post_consumption_failure_is_not_retried_or_rewound():
    failure = RuntimeError("second target selector failed")
    constraint, draft_outcomes, _ = _two_stage_constraint("final_empty")
    draft, target, draft_root, target_root = _prefilled_pair()
    draft_selector = RecordingSelector(draft_outcomes)
    target_selector = RecordingSelector([3, failure])
    with pytest.raises(RuntimeError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert len(target_selector.calls) == 2
    assert not target_selector.outcomes
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_wrong_type_first_d48_result_is_cleaned_up(monkeypatch):
    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        lambda result: object(),
    )
    constraint, draft_outcomes, target_outcomes = _terminal_constraint(
        "final_empty",
        is_match=True,
    )
    draft, target, draft_root, target_root = _prefilled_pair()
    with pytest.raises(GrammarMaskedSpeculativeHandoffInvariantError, match="D48"):
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert constraint.active_state_count == 0


def test_wrong_type_second_d48_result_is_cleaned_up(monkeypatch):
    real_classifier = handoff_module.classify_grammar_masked_speculative_outcome
    calls = 0

    def malformed_second(result):
        nonlocal calls
        calls += 1
        if calls == 2:
            return object()
        return real_classifier(result)

    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        malformed_second,
    )
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
    draft, target, draft_root, target_root = _prefilled_pair()
    with pytest.raises(GrammarMaskedSpeculativeHandoffInvariantError, match="D48"):
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert calls == 2
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_changed_vocabulary_metadata_after_first_success_is_detected(monkeypatch):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration

    def mutate_vocab(*args, **kwargs):
        result = real_d47(*args, **kwargs)
        args[1]._vocab_size += 1
        return result

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        mutate_vocab,
    )
    constraint, draft_outcomes, target_outcomes = _terminal_constraint(
        "final_empty",
        is_match=True,
    )
    draft, target, draft_root, target_root = _prefilled_pair()
    with pytest.raises(
        GrammarMaskedSpeculativeHandoffInvariantError,
        match="vocabulary sizes",
    ):
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert constraint.active_state_count == 0


def test_released_committed_state_is_detected_and_cleanup_is_idempotent(monkeypatch):
    real_classifier = handoff_module.classify_grammar_masked_speculative_outcome
    constraint, draft_outcomes, target_outcomes = _terminal_constraint(
        "final_empty",
        is_match=True,
    )

    def release_before_return(result):
        outcome = real_classifier(result)
        constraint.release_state(result.committed_state)
        return outcome

    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        release_before_return,
    )
    draft, target, draft_root, target_root = _prefilled_pair()
    with pytest.raises(
        GrammarMaskedSpeculativeHandoffInvariantError,
        match="status could not be read",
    ):
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert constraint.active_state_count == 0


def test_malformed_target_intermediate_is_owned_before_validation_and_released():
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
    draft, target, draft_root, target_root = _prefilled_pair(
        target_type=MalformedIntermediateTarget
    )
    with pytest.raises(
        GrammarMaskedSpeculativeHandoffInvariantError,
        match="target intermediate checkpoint",
    ):
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert target.malformed_release_calls == [target.malformed_checkpoint]
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_cache_mutation_during_intermediate_creation_is_detected_and_cleaned_up():
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
    draft, target, draft_root, target_root = _prefilled_pair(
        target_type=MutatingIntermediateTarget
    )
    with pytest.raises(
        GrammarMaskedSpeculativeHandoffInvariantError,
        match="target backend reported cache length",
    ):
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_target_intermediate_creation_failure_settles_partial_pair(monkeypatch):
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
    draft, target, draft_root, target_root = _prefilled_pair()
    original_create = target.create_cache_checkpoint
    failure = RuntimeError("target intermediate creation failed")
    calls = 0

    def fail_second_create():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return original_create()

    monkeypatch.setattr(target, "create_cache_checkpoint", fail_second_create)
    with pytest.raises(RuntimeError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert calls == 1
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_result_construction_failure_restores_roots_and_releases_state(monkeypatch):
    failure = RuntimeError("D49 result construction failed")

    def fail_construction(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        handoff_module,
        "GrammarMaskedSpeculativeHandoffResult",
        fail_construction,
    )
    constraint, draft_outcomes, target_outcomes = _terminal_constraint(
        "final_empty",
        is_match=True,
    )
    draft, target, draft_root, target_root = _prefilled_pair()
    with pytest.raises(RuntimeError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert constraint.active_state_count == 0


def test_success_path_intermediate_release_failure_gets_one_cleanup_retry(monkeypatch):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    calls = 0
    draft, target, draft_root, target_root = _prefilled_pair()
    original_release = draft.release_cache_checkpoint
    failed_checkpoint = None
    attempts = []
    failure = RuntimeError("draft intermediate release failed")

    def record_second(*args, **kwargs):
        nonlocal calls, failed_checkpoint
        calls += 1
        result = real_d47(*args, **kwargs)
        if calls == 2:
            failed_checkpoint = kwargs["draft_root_checkpoint"]
        return result

    def fail_once(checkpoint, /):
        if checkpoint is failed_checkpoint:
            attempts.append(checkpoint)
            if len(attempts) == 1:
                raise failure
        return original_release(checkpoint)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        record_second,
    )
    monkeypatch.setattr(draft, "release_cache_checkpoint", fail_once)
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("final_empty")
    with pytest.raises(RuntimeError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert len(attempts) == 2
    assert calls == 2
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_outer_cleanup_aggregates_all_five_operations_in_documented_order(
    monkeypatch,
):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    draft, target, draft_root, target_root = _prefilled_pair()
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
    operation_log = []
    cleanup_enabled = False
    d47_calls = 0
    original_draft_rollback = draft.rollback_cache
    original_target_rollback = target.rollback_cache
    original_draft_release = draft.release_cache_checkpoint
    original_target_release = target.release_cache_checkpoint
    original_state_release = constraint.release_state
    original_failure = RuntimeError("second D47 failed")

    def fail_second(*args, **kwargs):
        nonlocal cleanup_enabled, d47_calls
        d47_calls += 1
        if d47_calls == 2:
            cleanup_enabled = True
            raise original_failure
        return real_d47(*args, **kwargs)

    def draft_rollback(checkpoint, /):
        if cleanup_enabled:
            operation_log.append("draft initial root rollback")
            raise RuntimeError("draft rollback cleanup")
        return original_draft_rollback(checkpoint)

    def target_rollback(checkpoint, /):
        if cleanup_enabled:
            operation_log.append("target initial root rollback")
            raise RuntimeError("target rollback cleanup")
        return original_target_rollback(checkpoint)

    def draft_release(checkpoint, /):
        if cleanup_enabled:
            operation_log.append("draft intermediate root release")
            raise RuntimeError("draft release cleanup")
        return original_draft_release(checkpoint)

    def target_release(checkpoint, /):
        if cleanup_enabled:
            operation_log.append("target intermediate root release")
            raise RuntimeError("target release cleanup")
        return original_target_release(checkpoint)

    def state_release(state, /):
        if cleanup_enabled:
            operation_log.append("committed state release")
            raise RuntimeError("state release cleanup")
        return original_state_release(state)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        fail_second,
    )
    monkeypatch.setattr(draft, "rollback_cache", draft_rollback)
    monkeypatch.setattr(target, "rollback_cache", target_rollback)
    monkeypatch.setattr(draft, "release_cache_checkpoint", draft_release)
    monkeypatch.setattr(target, "release_cache_checkpoint", target_release)
    monkeypatch.setattr(constraint, "release_state", state_release)

    with pytest.raises(GrammarMaskedSpeculativeHandoffCleanupError) as captured:
        coordinate_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    error = captured.value
    assert error.original_failure is original_failure
    assert error.__cause__ is original_failure
    assert [label for label, _ in error.cleanup_failures] == operation_log == [
        "draft initial root rollback",
        "target initial root rollback",
        "draft intermediate root release",
        "target intermediate root release",
        "committed state release",
    ]


def test_one_thousand_alternating_operations_have_bounded_lifecycles():
    draft, target, draft_root, target_root = _prefilled_pair()
    for position in range(1000):
        if position % 2:
            constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
        else:
            constraint, draft_outcomes, target_outcomes = _terminal_constraint(
                "zero",
                is_match=True,
            )
        coordinated = _coordinate(
            constraint,
            draft_outcomes=draft_outcomes,
            target_outcomes=target_outcomes,
            draft=draft,
            target=target,
            draft_root=draft_root,
            target_root=target_root,
        )
        result = coordinated[0]
        constraint.release_state(result.final_iteration.committed_state)
        draft.rollback_cache(draft_root)
        target.rollback_cache(target_root)
        assert constraint.active_state_count == 0
        assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
        assert draft.cached_token_ids == target.cached_token_ids == PROMPT

    draft_allocations = [checkpoint.allocation_id for checkpoint in draft.create_calls]
    target_allocations = [checkpoint.allocation_id for checkpoint in target.create_calls]
    assert draft_allocations == sorted(set(draft_allocations))
    assert target_allocations == sorted(set(target_allocations))


def test_successful_result_retains_no_runtime_or_intermediate_evidence():
    constraint, draft_outcomes, target_outcomes = _two_stage_constraint("zero")
    coordinated = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
    )
    (
        result,
        _,
        draft,
        target,
        draft_root,
        target_root,
        draft_mask,
        target_mask,
        selectors_d,
        selectors_t,
    ) = coordinated
    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    references = [
        weakref.ref(draft),
        weakref.ref(target),
        weakref.ref(constraint),
        weakref.ref(draft_mask),
        weakref.ref(target_mask),
        weakref.ref(selectors_d),
        weakref.ref(selectors_t),
    ]
    del (
        coordinated,
        draft,
        target,
        draft_root,
        target_root,
        constraint,
        draft_mask,
        target_mask,
        selectors_d,
        selectors_t,
    )
    gc.collect()
    assert all(reference() is None for reference in references)
    assert [field.name for field in fields(result)] == [
        "output_token_ids",
        "final_iteration",
        "final_outcome",
    ]


def test_isolated_import_and_execution_are_optional_runtime_free():
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        from onyx_cuda import (
            GrammarMaskedSpeculativeHandoffResult,
            GrammarMaskedSpeculativeIterationResult,
            GrammarMaskedSpeculativeOutcomeResult,
        )

        terminal = GrammarMaskedSpeculativeIterationResult(
            proposal_token_ids=(),
            accepted_count=0,
            replacement_token_id=None,
            initial_cache_length=1,
            final_cache_length=2,
            uncached_next_token_id=None,
            shortening_selection=__import__('onyx_cuda').GrammarMaskedSelectionResult(
                (), True, None
            ),
            acceptance_no_decision_selection=None,
            final_row_no_decision_selection=None,
            committed_state=None,
            committed_state_is_match=True,
        )
        result = GrammarMaskedSpeculativeHandoffResult(
            (), terminal, GrammarMaskedSpeculativeOutcomeResult('grammar_complete')
        )
        assert result.final_iteration is terminal
        forbidden = {{
            'onyx', 'mlx', 'torch', 'transformers', 'tokenizers', 'huggingface_hub',
            'bitsandbytes', 'accelerate', 'onnxruntime', 'psutil', 'onyx_cuda._native'
        }}
        assert not (forbidden & set(sys.modules))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
