import inspect
import subprocess
import sys
import textwrap
from dataclasses import fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_speculative_handoff as handoff_module
from onyx_cuda import (
    GrammarMaskedSpeculativeHandoffCleanupError,
    GrammarMaskedSpeculativeHandoffResult,
    coordinate_grammar_masked_speculative_handoff,
    coordinate_multi_iteration_grammar_masked_speculative_handoff,
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
    def __init__(self, *, model_id, events):
        self.role = model_id
        self.events = events
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
        self.events.append((self.role, "create", checkpoint))
        return checkpoint

    def rollback_cache(self, checkpoint, /):
        self.rollback_calls.append(checkpoint)
        self.events.append((self.role, "rollback", checkpoint))
        return super().rollback_cache(checkpoint)

    def release_cache_checkpoint(self, checkpoint, /):
        self.release_calls.append(checkpoint)
        self.events.append((self.role, "release", checkpoint))
        return super().release_cache_checkpoint(checkpoint)


class MalformedIntermediateTarget(RecordingBackend):
    class MalformedCheckpoint:
        pass

    def __init__(self, *, model_id, events):
        self.malformed_checkpoint = None
        self.malformed_release_calls = []
        super().__init__(model_id=model_id, events=events)

    def create_cache_checkpoint(self):
        if len(self.create_calls) == 1:
            self.malformed_checkpoint = self.MalformedCheckpoint()
            self.events.append((self.role, "create", self.malformed_checkpoint))
            return self.malformed_checkpoint
        return super().create_cache_checkpoint()

    def release_cache_checkpoint(self, checkpoint, /):
        if checkpoint is self.malformed_checkpoint:
            self.malformed_release_calls.append(checkpoint)
            self.events.append((self.role, "release", checkpoint))
            return None
        return super().release_cache_checkpoint(checkpoint)


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


def _chain_constraint(
    handoff_count,
    *,
    proposal_lengths=None,
    final_match=True,
):
    if proposal_lengths is None:
        proposal_lengths = (1,) * handoff_count
    proposal_lengths = tuple(proposal_lengths)
    assert len(proposal_lengths) == handoff_count
    assert all(length in {1, 2} for length in proposal_lengths)

    transitions = []
    supports = []
    match_states = set()
    draft_outcomes = []
    target_outcomes = []
    for position, proposal_length in enumerate(proposal_lengths):
        stage = f"s{position}"
        next_stage = f"s{position + 1}"
        first_draft_state = f"d{position}_1"
        transitions.extend(
            (
                (stage, 1, first_draft_state),
                (stage, 3, next_stage),
            )
        )
        supports.append((stage, (1, 3)))
        draft_outcomes.append(1)
        target_outcomes.append(3)
        if proposal_length == 1:
            supports.append((first_draft_state, ()))
            match_states.add(first_draft_state)
        else:
            final_draft_state = f"d{position}_2"
            transitions.append((first_draft_state, 2, final_draft_state))
            supports.extend(
                (
                    (first_draft_state, (2,)),
                    (final_draft_state, ()),
                )
            )
            draft_outcomes.append(2)
            match_states.add(final_draft_state)

    terminal_state = f"s{handoff_count}"
    supports.append((terminal_state, ()))
    match_states.add(terminal_state)
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=tuple(transitions),
        valid_token_ids=tuple(supports),
        match_states=frozenset(match_states),
    )
    return (
        RecordingConstraint(
            _vocabulary(),
            grammar_type="regex",
            program=program,
            force_nonmatch=not final_match,
        ),
        draft_outcomes,
        target_outcomes,
    )


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
    elif second_route == "final_empty":
        transitions.extend(
            (
                ("handoff", 4, "q1"),
                ("q1", 5, "done"),
            )
        )
        supports.extend(
            (
                ("handoff", (4,)),
                ("q1", (5,)),
                ("done", ()),
            )
        )
        draft_outcomes.extend((4, 5))
        target_outcomes.extend((4, 5))
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
        RecordingConstraint(
            _vocabulary(),
            grammar_type="regex",
            program=program,
            force_nonmatch=force_nonmatch,
        ),
        draft_outcomes,
        target_outcomes,
    )


def _prefilled_pair(*, target_type=RecordingBackend):
    events = []
    draft = RecordingBackend(model_id="draft", events=events)
    target = target_type(model_id="target", events=events)
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    return draft, target, draft_root, target_root, events


def _coordinate(
    constraint,
    *,
    draft_outcomes,
    target_outcomes,
    iteration_bound,
    proposal_bound=PROPOSAL_BOUND,
    draft=None,
    target=None,
    draft_root=None,
    target_root=None,
):
    if draft is None:
        draft, target, draft_root, target_root, events = _prefilled_pair()
    else:
        events = draft.events
    starting_state = constraint.init_state()
    draft_mask = RecordingMask()
    target_mask = RecordingMask()
    draft_selector = RecordingSelector(draft_outcomes)
    target_selector = RecordingSelector(target_outcomes)
    result = coordinate_multi_iteration_grammar_masked_speculative_handoff(
        draft,
        target,
        CURRENT_TOKEN_ID,
        constraint,
        starting_state,
        draft_mask,
        target_mask,
        iteration_bound=iteration_bound,
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
        events,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
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


def test_public_surface_is_additive_and_preserves_d49_contract():
    assert handoff_module.__all__ == [
        "GrammarMaskedSpeculativeHandoffCleanupError",
        "GrammarMaskedSpeculativeHandoffError",
        "GrammarMaskedSpeculativeHandoffInvariantError",
        "GrammarMaskedSpeculativeHandoffResult",
        "coordinate_grammar_masked_speculative_handoff",
        "coordinate_multi_iteration_grammar_masked_speculative_handoff",
    ]
    signature = inspect.signature(
        coordinate_multi_iteration_grammar_masked_speculative_handoff
    )
    assert list(signature.parameters) == [
        "draft_backend",
        "target_backend",
        "current_token_id",
        "constraint",
        "starting_state",
        "draft_logit_mask",
        "target_logit_mask",
        "iteration_bound",
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
    assert str(signature.return_annotation) == (
        "GrammarMaskedSpeculativeHandoffResult[StateT]"
    )

    d49_signature = inspect.signature(coordinate_grammar_masked_speculative_handoff)
    assert "iteration_bound" not in d49_signature.parameters
    assert [field.name for field in fields(GrammarMaskedSpeculativeHandoffResult)] == [
        "output_token_ids",
        "final_iteration",
        "final_outcome",
    ]


@pytest.mark.parametrize(
    ("bad_bound", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (1.0, TypeError),
        ("1", TypeError),
    ],
)
def test_iteration_bound_fails_before_any_other_observation(
    monkeypatch,
    bad_bound,
    error_type,
):
    class Exploding:
        def __getattribute__(self, name):
            raise AssertionError(f"unexpected observation: {name}")

    def unexpected(*args, **kwargs):
        raise AssertionError("D47/D48 must not be called")

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        unexpected,
    )
    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        unexpected,
    )
    resource = Exploding()
    with pytest.raises(error_type, match="iteration_bound"):
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            resource,
            resource,
            0,
            resource,
            resource,
            resource,
            resource,
            iteration_bound=bad_bound,
            proposal_bound=1,
            draft_select_token=resource,
            target_select_token=resource,
            draft_root_checkpoint=resource,
            target_root_checkpoint=resource,
        )


def test_iteration_bound_rejects_integer_subclasses_before_observation():
    class IntegerSubclass(int):
        pass

    with pytest.raises(TypeError, match="exact integer"):
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            iteration_bound=IntegerSubclass(1),
            proposal_bound=1,
            draft_select_token=None,
            target_select_token=None,
            draft_root_checkpoint=None,
            target_root_checkpoint=None,
        )


@pytest.mark.parametrize(
    ("handoff_count", "expected_kind", "expected_output"),
    [
        (0, "grammar_complete", ()),
        (1, "handoff_available", (3,)),
    ],
)
def test_bound_one_returns_terminal_or_handoff_without_intermediate_roots(
    monkeypatch,
    handoff_count,
    expected_kind,
    expected_output,
):
    d47_calls, d48_calls = _record_composed_calls(monkeypatch)
    constraint, draft_outcomes, target_outcomes = _chain_constraint(handoff_count)
    coordinated = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
        iteration_bound=1,
    )
    result, _, draft, target, draft_root, target_root, *_ = coordinated

    assert len(d47_calls) == len(d48_calls) == 1
    assert result.output_token_ids == expected_output
    assert result.final_iteration is d47_calls[0][2]
    assert result.final_outcome is d48_calls[0][1]
    assert result.final_outcome.kind == expected_kind
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert target.create_calls == [target_root]
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls

    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert constraint.active_state_count == 0
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT


@pytest.mark.parametrize("final_match", [False, True])
@pytest.mark.parametrize("terminal_position", [1, 2, 3, 4])
def test_terminal_outcome_stops_at_every_position(
    monkeypatch,
    terminal_position,
    final_match,
):
    d47_calls, d48_calls = _record_composed_calls(monkeypatch)
    handoff_count = terminal_position - 1
    constraint, draft_outcomes, target_outcomes = _chain_constraint(
        handoff_count,
        proposal_lengths=(1, 2, 1)[:handoff_count],
        final_match=final_match,
    )
    coordinated = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
        iteration_bound=6,
    )
    result, _, draft, target, draft_root, target_root, *_ = coordinated

    assert len(d47_calls) == len(d48_calls) == terminal_position
    assert result.final_outcome.kind == (
        "grammar_complete" if final_match else "grammar_no_continuation"
    )
    assert result.output_token_ids == (3,) * handoff_count
    assert result.final_iteration.proposal_token_ids == ()
    assert draft.cached_token_ids == target.cached_token_ids == (
        PROMPT + (CURRENT_TOKEN_ID,) + (3,) * handoff_count
    )
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1

    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)


def test_multi_iteration_continuity_variable_proposals_and_root_rotation(monkeypatch):
    d47_calls, d48_calls = _record_composed_calls(monkeypatch)
    constraint, draft_outcomes, target_outcomes = _chain_constraint(
        3,
        proposal_lengths=(1, 2, 1),
    )
    coordinated = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
        iteration_bound=6,
    )
    (
        result,
        _,
        draft,
        target,
        draft_root,
        target_root,
        events,
        draft_mask,
        target_mask,
        draft_selector,
        target_selector,
    ) = coordinated

    assert len(d47_calls) == len(d48_calls) == 4
    assert [len(call[2].proposal_token_ids) for call in d47_calls] == [1, 2, 1, 0]
    assert [call[2].accepted_count for call in d47_calls] == [0, 0, 0, 0]
    assert result.output_token_ids == (3, 3, 3)
    assert result.final_iteration is d47_calls[-1][2]
    assert result.final_outcome is d48_calls[-1][1]
    assert result.final_outcome.kind == "grammar_complete"
    assert all(
        d48_call[0] is d47_call[2]
        for d47_call, d48_call in zip(d47_calls, d48_calls)
    )

    for previous, following in zip(d47_calls, d47_calls[1:]):
        previous_args, previous_kwargs, previous_result = previous
        following_args, following_kwargs, following_result = following
        assert following_args[0] is previous_args[0] is draft
        assert following_args[1] is previous_args[1] is target
        assert following_args[2] == previous_result.uncached_next_token_id == 3
        assert following_args[3] is constraint
        assert following_args[4] is previous_result.committed_state
        assert following_args[5] is previous_args[5] is draft_mask
        assert following_args[6] is previous_args[6] is target_mask
        assert following_kwargs["proposal_bound"] == PROPOSAL_BOUND
        assert following_kwargs["draft_select_token"] is draft_selector
        assert following_kwargs["target_select_token"] is target_selector
        assert following_result.initial_cache_length == previous_result.final_cache_length

    assert [token for _, token in constraint.advance_calls].count(3) == 3
    assert draft.cached_token_ids == target.cached_token_ids == (7, 0, 3, 3, 3)
    assert draft.cache_length == target.cache_length == 5
    assert not draft_selector.outcomes
    assert not target_selector.outcomes

    intermediate_pairs = [
        (
            call[1]["draft_root_checkpoint"],
            call[1]["target_root_checkpoint"],
        )
        for call in d47_calls[1:]
    ]
    assert len(intermediate_pairs) == 3
    assert all(
        checkpoint in backend.release_calls
        for pair in intermediate_pairs
        for checkpoint, backend in zip(pair, (draft, target))
    )
    assert draft_root not in draft.release_calls
    assert target_root not in target.release_calls
    labels = {
        id(checkpoint): f"{role}{position}"
        for position, pair in enumerate(intermediate_pairs, start=1)
        for checkpoint, role in zip(pair, ("d", "t"))
    }
    rotation_events = [
        f"{operation} {labels[id(checkpoint)]}"
        for _, operation, checkpoint in events
        if operation != "rollback" and id(checkpoint) in labels
    ]
    assert rotation_events == [
        "create d1",
        "create t1",
        "create d2",
        "create t2",
        "release d1",
        "release t1",
        "create d3",
        "create t3",
        "release d2",
        "release t2",
        "release d3",
        "release t3",
    ]
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1

    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert constraint.active_state_count == 0


def test_bound_exhaustion_returns_final_handoff_without_policy_conversion(monkeypatch):
    d47_calls, d48_calls = _record_composed_calls(monkeypatch)
    constraint, draft_outcomes, target_outcomes = _chain_constraint(
        4,
        proposal_lengths=(1, 2, 1, 2),
    )
    coordinated = _coordinate(
        constraint,
        draft_outcomes=draft_outcomes,
        target_outcomes=target_outcomes,
        iteration_bound=3,
    )
    result, _, draft, target, draft_root, target_root, *_ = coordinated

    assert len(d47_calls) == len(d48_calls) == 3
    assert result.final_iteration is d47_calls[-1][2]
    assert result.final_outcome is d48_calls[-1][1]
    assert result.final_outcome.kind == "handoff_available"
    assert result.output_token_ids == (3, 3, 3)
    assert draft.cached_token_ids == target.cached_token_ids == (7, 0, 3, 3)
    assert len(target.create_calls) == 3
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1

    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)


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
def test_bound_two_preserves_every_second_outcome_family(
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
        iteration_bound=2,
    )
    result, _, draft, target, draft_root, target_root, *_ = coordinated

    assert len(d47_calls) == len(d48_calls) == 2
    assert result.output_token_ids == expected_output
    assert result.final_iteration is d47_calls[-1][2]
    assert result.final_outcome is d48_calls[-1][1]
    assert result.final_outcome.kind == expected_kind
    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)


@pytest.mark.parametrize("handoff_count", [0, 1, 2])
@pytest.mark.parametrize("final_match", [False, True])
def test_bound_two_is_observationally_equivalent_to_d49(
    handoff_count,
    final_match,
):
    first = _chain_constraint(handoff_count, final_match=final_match)
    second = _chain_constraint(handoff_count, final_match=final_match)
    first_pair = _prefilled_pair()
    second_pair = _prefilled_pair()
    first_constraint, first_draft_outcomes, first_target_outcomes = first
    second_constraint, second_draft_outcomes, second_target_outcomes = second
    first_draft, first_target, first_draft_root, first_target_root, _ = first_pair
    second_draft, second_target, second_draft_root, second_target_root, _ = second_pair
    first_draft_selector = RecordingSelector(first_draft_outcomes)
    first_target_selector = RecordingSelector(first_target_outcomes)
    second_draft_selector = RecordingSelector(second_draft_outcomes)
    second_target_selector = RecordingSelector(second_target_outcomes)

    d50 = coordinate_multi_iteration_grammar_masked_speculative_handoff(
        first_draft,
        first_target,
        CURRENT_TOKEN_ID,
        first_constraint,
        first_constraint.init_state(),
        RecordingMask(),
        RecordingMask(),
        iteration_bound=2,
        proposal_bound=PROPOSAL_BOUND,
        draft_select_token=first_draft_selector,
        target_select_token=first_target_selector,
        draft_root_checkpoint=first_draft_root,
        target_root_checkpoint=first_target_root,
    )
    d49 = coordinate_grammar_masked_speculative_handoff(
        second_draft,
        second_target,
        CURRENT_TOKEN_ID,
        second_constraint,
        second_constraint.init_state(),
        RecordingMask(),
        RecordingMask(),
        proposal_bound=PROPOSAL_BOUND,
        draft_select_token=second_draft_selector,
        target_select_token=second_target_selector,
        draft_root_checkpoint=second_draft_root,
        target_root_checkpoint=second_target_root,
    )

    assert d50.output_token_ids == d49.output_token_ids
    assert d50.final_outcome.kind == d49.final_outcome.kind
    assert d50.final_iteration.proposal_token_ids == d49.final_iteration.proposal_token_ids
    assert d50.final_iteration.accepted_count == d49.final_iteration.accepted_count
    assert first_draft.cached_token_ids == second_draft.cached_token_ids
    assert first_target.cached_token_ids == second_target.cached_token_ids
    assert len(first_draft_selector.calls) == len(second_draft_selector.calls)
    assert len(first_target_selector.calls) == len(second_target_selector.calls)
    assert first_draft.active_checkpoint_count == second_draft.active_checkpoint_count == 1
    assert first_target.active_checkpoint_count == second_target.active_checkpoint_count == 1

    first_constraint.release_state(d50.final_iteration.committed_state)
    second_constraint.release_state(d49.final_iteration.committed_state)


def test_first_d47_failure_stays_outside_outer_cleanup(monkeypatch):
    failure = RuntimeError("first D47 failed")

    def fail_first(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        fail_first,
    )
    constraint, draft_outcomes, target_outcomes = _chain_constraint(1)
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    starting_state = constraint.init_state()
    with pytest.raises(RuntimeError) as captured:
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            starting_state,
            RecordingMask(),
            RecordingMask(),
            iteration_bound=2,
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert draft.rollback_calls == target.rollback_calls == []
    assert constraint.active_state_count == 1
    constraint.release_state(starting_state)


def test_later_d47_failure_has_no_classification_and_cleans_everything(monkeypatch):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    real_d48 = handoff_module.classify_grammar_masked_speculative_outcome
    failure = RuntimeError("second D47 failed")
    d47_calls = 0
    d48_calls = 0

    def fail_second(*args, **kwargs):
        nonlocal d47_calls
        d47_calls += 1
        if d47_calls == 2:
            raise failure
        return real_d47(*args, **kwargs)

    def record_d48(result):
        nonlocal d48_calls
        d48_calls += 1
        return real_d48(result)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        fail_second,
    )
    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        record_d48,
    )
    constraint, draft_outcomes, target_outcomes = _chain_constraint(2)
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    with pytest.raises(RuntimeError) as captured:
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            iteration_bound=3,
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert d47_calls == 2
    assert d48_calls == 1
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_later_d47_post_consumption_failure_is_not_retried_or_rewound():
    failure = RuntimeError("third target selector failed")
    constraint, draft_outcomes, _ = _chain_constraint(3)
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    draft_selector = RecordingSelector(draft_outcomes)
    target_selector = RecordingSelector([3, 3, failure])

    with pytest.raises(RuntimeError) as captured:
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            iteration_bound=4,
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=draft_selector,
            target_select_token=target_selector,
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert len(draft_selector.calls) == 3
    assert len(target_selector.calls) == 3
    assert not draft_selector.outcomes
    assert not target_selector.outcomes
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


@pytest.mark.parametrize("failure_position", [1, 2, 3])
def test_d48_failure_at_every_reachable_position_cleans_by_identity(
    monkeypatch,
    failure_position,
):
    real_d48 = handoff_module.classify_grammar_masked_speculative_outcome
    failure = RuntimeError(f"D48 failed at {failure_position}")
    calls = 0

    def fail_selected(result):
        nonlocal calls
        calls += 1
        if calls == failure_position:
            raise failure
        return real_d48(result)

    monkeypatch.setattr(
        handoff_module,
        "classify_grammar_masked_speculative_outcome",
        fail_selected,
    )
    constraint, draft_outcomes, target_outcomes = _chain_constraint(3)
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    with pytest.raises(RuntimeError) as captured:
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            iteration_bound=4,
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert calls == failure_position
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_malformed_first_intermediate_target_is_released_with_partial_pair():
    constraint, draft_outcomes, target_outcomes = _chain_constraint(2)
    draft, target, draft_root, target_root, _ = _prefilled_pair(
        target_type=MalformedIntermediateTarget
    )
    with pytest.raises(
        handoff_module.GrammarMaskedSpeculativeHandoffInvariantError,
        match="target next intermediate checkpoint",
    ):
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            iteration_bound=3,
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


def test_final_intermediate_release_failure_gets_one_cleanup_retry(monkeypatch):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    original_release = draft.release_cache_checkpoint
    final_root = None
    attempts = []
    failure = RuntimeError("final intermediate release failed")
    calls = 0

    def record_roots(*args, **kwargs):
        nonlocal calls, final_root
        calls += 1
        if calls == 2:
            final_root = kwargs["draft_root_checkpoint"]
        return real_d47(*args, **kwargs)

    def fail_once(checkpoint, /):
        if checkpoint is final_root:
            attempts.append(checkpoint)
            if len(attempts) == 1:
                raise failure
        return original_release(checkpoint)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        record_roots,
    )
    monkeypatch.setattr(draft, "release_cache_checkpoint", fail_once)
    constraint, draft_outcomes, target_outcomes = _chain_constraint(1)
    with pytest.raises(RuntimeError) as captured:
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            iteration_bound=2,
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert captured.value is failure
    assert len(attempts) == 2
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_cleanup_aggregates_all_seven_operations_in_global_order(monkeypatch):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    constraint, draft_outcomes, target_outcomes = _chain_constraint(3)
    original_failure = RuntimeError("current draft retirement failed")
    cleanup_enabled = False
    d47_calls = 0
    current_draft = None
    original_draft_rollback = draft.rollback_cache
    original_target_rollback = target.rollback_cache
    original_draft_release = draft.release_cache_checkpoint
    original_target_release = target.release_cache_checkpoint
    original_state_release = constraint.release_state

    def record_second_root(*args, **kwargs):
        nonlocal d47_calls, current_draft
        d47_calls += 1
        if d47_calls == 2:
            current_draft = kwargs["draft_root_checkpoint"]
        return real_d47(*args, **kwargs)

    def draft_rollback(checkpoint, /):
        if cleanup_enabled:
            raise RuntimeError("draft rollback cleanup")
        return original_draft_rollback(checkpoint)

    def target_rollback(checkpoint, /):
        if cleanup_enabled:
            raise RuntimeError("target rollback cleanup")
        return original_target_rollback(checkpoint)

    def draft_release(checkpoint, /):
        nonlocal cleanup_enabled
        if checkpoint is current_draft and not cleanup_enabled:
            cleanup_enabled = True
            raise original_failure
        if cleanup_enabled:
            raise RuntimeError("draft release cleanup")
        return original_draft_release(checkpoint)

    def target_release(checkpoint, /):
        if cleanup_enabled:
            raise RuntimeError("target release cleanup")
        return original_target_release(checkpoint)

    def state_release(state, /):
        if cleanup_enabled:
            raise RuntimeError("state release cleanup")
        return original_state_release(state)

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        record_second_root,
    )
    monkeypatch.setattr(draft, "rollback_cache", draft_rollback)
    monkeypatch.setattr(target, "rollback_cache", target_rollback)
    monkeypatch.setattr(draft, "release_cache_checkpoint", draft_release)
    monkeypatch.setattr(target, "release_cache_checkpoint", target_release)
    monkeypatch.setattr(constraint, "release_state", state_release)

    with pytest.raises(GrammarMaskedSpeculativeHandoffCleanupError) as captured:
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            iteration_bound=4,
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    error = captured.value
    assert error.original_failure is original_failure
    assert error.__cause__ is original_failure
    assert [label for label, _ in error.cleanup_failures] == [
        "draft initial root rollback",
        "target initial root rollback",
        "draft intermediate root release",
        "target intermediate root release",
        "draft next intermediate root release",
        "target next intermediate root release",
        "committed state release",
    ]


def test_opaque_none_state_uses_explicit_ownership_across_handoff():
    constraint = NoneStateConstraint()
    coordinated = _coordinate(
        constraint,
        draft_outcomes=[1],
        target_outcomes=[3],
        iteration_bound=2,
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


def test_changed_vocabulary_after_later_iteration_fails_closed(monkeypatch):
    real_d47 = handoff_module.coordinate_grammar_masked_speculative_iteration
    calls = 0

    def mutate_after_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_d47(*args, **kwargs)
        if calls == 2:
            args[1]._vocab_size += 1
        return result

    monkeypatch.setattr(
        handoff_module,
        "coordinate_grammar_masked_speculative_iteration",
        mutate_after_second,
    )
    constraint, draft_outcomes, target_outcomes = _chain_constraint(2)
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    with pytest.raises(
        handoff_module.GrammarMaskedSpeculativeHandoffInvariantError,
        match="vocabulary sizes|vocab_size changed",
    ):
        coordinate_multi_iteration_grammar_masked_speculative_handoff(
            draft,
            target,
            CURRENT_TOKEN_ID,
            constraint,
            constraint.init_state(),
            RecordingMask(),
            RecordingMask(),
            iteration_bound=3,
            proposal_bound=PROPOSAL_BOUND,
            draft_select_token=RecordingSelector(draft_outcomes),
            target_select_token=RecordingSelector(target_outcomes),
            draft_root_checkpoint=draft_root,
            target_root_checkpoint=target_root,
        )
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT
    assert draft.active_checkpoint_count == target.active_checkpoint_count == 1
    assert constraint.active_state_count == 0


def test_one_thousand_varying_operations_keep_lifecycles_bounded():
    draft, target, draft_root, target_root, _ = _prefilled_pair()
    for position in range(1000):
        iteration_bound = position % 4 + 1
        exhaust_bound = position % 2 == 0
        handoff_count = iteration_bound if exhaust_bound else iteration_bound - 1
        constraint, draft_outcomes, target_outcomes = _chain_constraint(
            handoff_count,
            proposal_lengths=tuple(
                1 + ((position + index) % 2) for index in range(handoff_count)
            ),
            final_match=position % 3 != 0,
        )
        coordinated = _coordinate(
            constraint,
            draft_outcomes=draft_outcomes,
            target_outcomes=target_outcomes,
            iteration_bound=iteration_bound,
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


def test_isolated_import_and_bounds_execution_are_optional_runtime_free():
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        from onyx_cuda import (
            coordinate_multi_iteration_grammar_masked_speculative_handoff,
        )
        from onyx_cuda.testing import (
            FakeAutoregressiveBackend,
            FakeGrammarConstraint,
            FakeGrammarProgram,
        )

        vocab = tuple(bytes((token,)) for token in range(8))
        rows = tuple(tuple(float(token) for token in range(8)) for _ in range(32))

        class Mask:
            vocab_size = 8
            def apply(self, logits, valid_token_ids, /):
                return logits

        class Selector:
            def __init__(self, values):
                self.values = list(values)
            def __call__(self, row):
                return self.values.pop(0)

        for bound in (1, 2, 4):
            transitions = []
            supports = []
            for position in range(3):
                transitions.extend(((f's{{position}}', 1, f'd{{position}}'),
                                    (f's{{position}}', 3, f's{{position + 1}}')))
                supports.extend(((f's{{position}}', (1, 3)),
                                 (f'd{{position}}', ())))
            supports.append(('s3', ()))
            constraint = FakeGrammarConstraint(
                vocab,
                grammar_type='regex',
                program=FakeGrammarProgram(
                    initial_state='s0',
                    transitions=tuple(transitions),
                    valid_token_ids=tuple(supports),
                    match_states=frozenset({{'d0', 'd1', 'd2', 's3'}}),
                ),
            )
            draft = FakeAutoregressiveBackend(rows, model_id='draft')
            target = FakeAutoregressiveBackend(rows, model_id='target')
            draft.prefill((7,))
            target.prefill((7,))
            draft_root = draft.create_cache_checkpoint()
            target_root = target.create_cache_checkpoint()
            result = coordinate_multi_iteration_grammar_masked_speculative_handoff(
                draft, target, 0, constraint, constraint.init_state(), Mask(), Mask(),
                iteration_bound=bound,
                proposal_bound=2,
                draft_select_token=Selector([1, 1, 1]),
                target_select_token=Selector([3, 3, 3]),
                draft_root_checkpoint=draft_root,
                target_root_checkpoint=target_root,
            )
            assert result.output_token_ids == (3,) * min(bound, 3)
            constraint.release_state(result.final_iteration.committed_state)

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
