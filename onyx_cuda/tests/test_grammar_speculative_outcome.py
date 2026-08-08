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
import onyx_cuda.grammar_speculative_outcome as outcome_module
from onyx_cuda import (
    BackendError,
    GrammarMaskedDraftProposalResult,
    GrammarMaskedSelectionResult,
    GrammarMaskedSpeculativeIterationError,
    GrammarMaskedSpeculativeIterationResult,
    GrammarMaskedSpeculativeOutcomeError,
    GrammarMaskedSpeculativeOutcomeInvariantError,
    GrammarMaskedSpeculativeOutcomeResult,
    SpeculativeIterationError,
    classify_grammar_masked_speculative_outcome,
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
        self.rollback_calls = []
        self.release_calls = []
        super().__init__(SCRIPT, model_id=model_id)

    def decode(self, token_id, /):
        self.decode_calls.append(token_id)
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
        return self.outcomes.pop(0)


class HostileState:
    def __bool__(self):
        raise AssertionError("state truthiness must not be inspected")

    def __eq__(self, other):
        del other
        raise AssertionError("state equality must not be inspected")

    def __hash__(self):
        raise AssertionError("state hashing must not be inspected")

    def __str__(self):
        raise AssertionError("state string conversion must not be inspected")

    def __repr__(self):
        raise AssertionError("state representation must not be inspected")

    def __call__(self):
        raise AssertionError("state call behavior must not be inspected")

    def __getattribute__(self, name):
        del name
        raise AssertionError("state attributes must not be inspected")


class EmptyNonmatchingConstraint:
    def __init__(self):
        self.state = object()
        self.live = True
        self.release_calls = []

    @property
    def vocab_size(self):
        return VOCAB_SIZE

    @property
    def grammar_type(self):
        return "regex"

    @property
    def active_state_count(self):
        return int(self.live)

    def init_state(self):
        return self.state

    def advance_state(self, state, token_id, /):
        del state, token_id
        raise AssertionError("empty support must not advance")

    def get_valid_token_ids(self, state, /):
        self._require_live(state)
        return ()

    def is_match_state(self, state, /):
        self._require_live(state)
        return False

    def is_dead_state(self, state, /):
        self._require_live(state)
        return False

    def release_state(self, state, /):
        self._require_live(state)
        self.release_calls.append(state)
        self.live = False

    def release_states(self, states, /):
        for state in states:
            self.release_state(state)

    def reset(self):
        self.live = False

    def _require_live(self, state):
        if state is not self.state or not self.live:
            raise RuntimeError("unknown or released state")


def _selection(is_match):
    return GrammarMaskedSelectionResult((), is_match, None)


def _direct_result(
    route,
    *,
    is_match=False,
    accepted_count=1,
    proposal=(1, 2, 3),
    shortening_is_match=None,
    state=None,
):
    initial_cache_length = 2
    shortening = (
        None if shortening_is_match is None else _selection(shortening_is_match)
    )
    if route == "zero":
        return GrammarMaskedSpeculativeIterationResult(
            (),
            0,
            None,
            initial_cache_length,
            initial_cache_length + 1,
            None,
            _selection(is_match),
            None,
            None,
            state,
            is_match,
        )
    proposal = tuple(proposal)
    if route == "acceptance_empty":
        return GrammarMaskedSpeculativeIterationResult(
            proposal,
            accepted_count,
            None,
            initial_cache_length,
            initial_cache_length + 1 + accepted_count,
            None,
            shortening,
            _selection(is_match),
            None,
            state,
            is_match,
        )
    if route == "mismatch":
        replacement = 6 if proposal[accepted_count] != 6 else 5
        return GrammarMaskedSpeculativeIterationResult(
            proposal,
            accepted_count,
            replacement,
            initial_cache_length,
            initial_cache_length + 1 + accepted_count,
            replacement,
            shortening,
            None,
            None,
            state,
            is_match,
        )
    if route == "bonus":
        return GrammarMaskedSpeculativeIterationResult(
            proposal,
            len(proposal),
            None,
            initial_cache_length,
            initial_cache_length + 1 + len(proposal),
            6,
            shortening,
            None,
            None,
            state,
            is_match,
        )
    if route == "final_empty":
        return GrammarMaskedSpeculativeIterationResult(
            proposal,
            len(proposal),
            None,
            initial_cache_length,
            initial_cache_length + 1 + len(proposal),
            None,
            shortening,
            None,
            _selection(is_match),
            state,
            is_match,
        )
    raise AssertionError(f"unknown route: {route}")


def _constraint(*, final_support=(5,)):
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
        grammar_type="regex",
        program=program,
    )


def _empty_constraint(*, is_match):
    program = FakeGrammarProgram(
        initial_state="s0",
        transitions=(),
        valid_token_ids=(("s0", ()),),
        match_states=frozenset({"s0"} if is_match else ()),
    )
    return FakeGrammarConstraint(
        tuple(bytes((token,)) for token in range(VOCAB_SIZE)),
        grammar_type="regex",
        program=program,
    )


def _coordinate(constraint, *, draft_outcomes, target_outcomes):
    draft = RecordingBackend(model_id="draft")
    target = RecordingBackend(model_id="target")
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
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
        proposal_bound=2,
        draft_select_token=draft_selector,
        target_select_token=target_selector,
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    return (
        result,
        constraint,
        draft,
        target,
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
        del constraint, starting_state, logit_mask, proposal_bound, select_token
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


def _mutate(result, field_name, value):
    object.__setattr__(result, field_name, value)
    return result


def test_public_surface_signature_result_and_error_hierarchy():
    assert outcome_module.__all__ == [
        "GrammarMaskedSpeculativeOutcomeError",
        "GrammarMaskedSpeculativeOutcomeInvariantError",
        "GrammarMaskedSpeculativeOutcomeResult",
        "classify_grammar_masked_speculative_outcome",
    ]
    package = sys.modules["onyx_cuda"]
    for symbol in outcome_module.__all__:
        assert getattr(package, symbol) is getattr(outcome_module, symbol)
    assert issubclass(
        GrammarMaskedSpeculativeOutcomeError,
        GrammarMaskedSpeculativeIterationError,
    )
    assert issubclass(GrammarMaskedSpeculativeOutcomeError, SpeculativeIterationError)
    assert issubclass(GrammarMaskedSpeculativeOutcomeError, BackendError)
    assert issubclass(
        GrammarMaskedSpeculativeOutcomeInvariantError,
        GrammarMaskedSpeculativeOutcomeError,
    )

    signature = inspect.signature(classify_grammar_masked_speculative_outcome)
    assert tuple(signature.parameters) == ("iteration_result",)
    parameter = signature.parameters["iteration_result"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty
    assert [field.name for field in fields(GrammarMaskedSpeculativeOutcomeResult)] == [
        "kind"
    ]
    result = GrammarMaskedSpeculativeOutcomeResult("handoff_available")
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.kind = "grammar_complete"


@pytest.mark.parametrize(
    "kind",
    ["handoff_available", "grammar_complete", "grammar_no_continuation"],
)
def test_direct_result_accepts_only_the_three_exact_literals(kind):
    assert GrammarMaskedSpeculativeOutcomeResult(kind).kind == kind


class StringSubclass(str):
    pass


@pytest.mark.parametrize("kind", [None, 1, True, StringSubclass("grammar_complete")])
def test_direct_result_rejects_non_exact_strings(kind):
    with pytest.raises(TypeError):
        GrammarMaskedSpeculativeOutcomeResult(kind)


@pytest.mark.parametrize(
    "kind",
    ["", "complete", "Grammar_Complete", "grammar complete", "length", "stop"],
)
def test_direct_result_rejects_unsupported_strings(kind):
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        GrammarMaskedSpeculativeOutcomeResult(kind)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("committed_match", [False, True])
def test_every_mismatch_position_classifies_as_handoff(position, committed_match):
    result = _direct_result(
        "mismatch",
        accepted_count=position,
        is_match=committed_match,
    )
    outcome = classify_grammar_masked_speculative_outcome(result)
    assert outcome.kind == "handoff_available"


@pytest.mark.parametrize("proposal", [(1,), (1, 2, 3)])
@pytest.mark.parametrize("committed_match", [False, True])
def test_full_acceptance_bonus_classifies_as_handoff(proposal, committed_match):
    result = _direct_result("bonus", proposal=proposal, is_match=committed_match)
    assert classify_grammar_masked_speculative_outcome(result).kind == (
        "handoff_available"
    )


@pytest.mark.parametrize("is_match", [False, True])
def test_zero_token_terminal_classification_uses_d44_evidence(is_match):
    result = _direct_result("zero", is_match=is_match, state=None)
    expected = "grammar_complete" if is_match else "grammar_no_continuation"
    assert classify_grammar_masked_speculative_outcome(result).kind == expected


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("is_match", [False, True])
def test_acceptance_empty_classification_uses_d45_evidence(position, is_match):
    result = _direct_result(
        "acceptance_empty",
        accepted_count=position,
        is_match=is_match,
    )
    expected = "grammar_complete" if is_match else "grammar_no_continuation"
    assert classify_grammar_masked_speculative_outcome(result).kind == expected


@pytest.mark.parametrize("proposal", [(1,), (1, 2, 3)])
@pytest.mark.parametrize("is_match", [False, True])
def test_final_row_empty_classification_uses_d46_evidence(proposal, is_match):
    result = _direct_result("final_empty", proposal=proposal, is_match=is_match)
    expected = "grammar_complete" if is_match else "grammar_no_continuation"
    assert classify_grammar_masked_speculative_outcome(result).kind == expected


@pytest.mark.parametrize("history_match", [False, True])
@pytest.mark.parametrize("terminal_match", [False, True])
def test_nonzero_d44_history_is_nonterminal_for_d45_empty(
    history_match,
    terminal_match,
):
    result = _direct_result(
        "acceptance_empty",
        is_match=terminal_match,
        shortening_is_match=history_match,
    )
    history = result.shortening_selection
    expected = "grammar_complete" if terminal_match else "grammar_no_continuation"
    assert classify_grammar_masked_speculative_outcome(result).kind == expected
    assert result.shortening_selection is history
    assert history.is_match is history_match


@pytest.mark.parametrize("history_match", [False, True])
@pytest.mark.parametrize("terminal_match", [False, True])
def test_nonzero_d44_history_is_nonterminal_for_d46_empty(
    history_match,
    terminal_match,
):
    result = _direct_result(
        "final_empty",
        is_match=terminal_match,
        shortening_is_match=history_match,
    )
    history = result.shortening_selection
    expected = "grammar_complete" if terminal_match else "grammar_no_continuation"
    assert classify_grammar_masked_speculative_outcome(result).kind == expected
    assert result.shortening_selection is history
    assert history.is_match is history_match


@pytest.mark.parametrize("route", ["mismatch", "bonus"])
@pytest.mark.parametrize("history_match", [False, True])
def test_handoff_wins_over_nonzero_d44_history(route, history_match):
    result = _direct_result(
        route,
        is_match=True,
        shortening_is_match=history_match,
    )
    history = result.shortening_selection
    assert classify_grammar_masked_speculative_outcome(result).kind == (
        "handoff_available"
    )
    assert result.shortening_selection is history


def test_opaque_none_and_hostile_states_are_never_inspected():
    none_result = _direct_result("zero", is_match=True, state=None)
    hostile_result = _direct_result("bonus", is_match=True, state=HostileState())
    assert classify_grammar_masked_speculative_outcome(none_result).kind == (
        "grammar_complete"
    )
    assert classify_grammar_masked_speculative_outcome(hostile_result).kind == (
        "handoff_available"
    )


def test_genuine_d47_compositions_cover_all_five_routes_without_runtime_work(
    monkeypatch,
):
    cases = []
    cases.append(
        (
            _coordinate(
                _empty_constraint(is_match=True),
                draft_outcomes=(),
                target_outcomes=(),
            ),
            "grammar_complete",
        )
    )
    cases.append(
        (
            _coordinate(
                _constraint(),
                draft_outcomes=(1, 2),
                target_outcomes=(1, 4),
            ),
            "handoff_available",
        )
    )
    cases.append(
        (
            _coordinate(
                _constraint(),
                draft_outcomes=(1, 2),
                target_outcomes=(1, 2, 5),
            ),
            "handoff_available",
        )
    )
    cases.append(
        (
            _coordinate(
                _constraint(final_support=()),
                draft_outcomes=(1, 2),
                target_outcomes=(1, 2),
            ),
            "grammar_complete",
        )
    )

    _install_mock_d44(monkeypatch, (1, 2))
    cases.append(
        (
            _coordinate(
                EmptyNonmatchingConstraint(),
                draft_outcomes=(),
                target_outcomes=(),
            ),
            "grammar_no_continuation",
        )
    )

    for components, expected_kind in cases:
        (
            result,
            constraint,
            draft,
            target,
            draft_mask,
            target_mask,
            draft_selector,
            target_selector,
        ) = components
        snapshot = (
            draft.cached_token_ids,
            target.cached_token_ids,
            tuple(draft.decode_calls),
            tuple(target.decode_calls),
            tuple(draft.verify_calls),
            tuple(target.verify_calls),
            tuple(draft.rollback_calls),
            tuple(target.rollback_calls),
            tuple(draft.release_calls),
            tuple(target.release_calls),
            tuple(draft_mask.calls),
            tuple(target_mask.calls),
            tuple(draft_selector.calls),
            tuple(target_selector.calls),
            constraint.active_state_count,
            draft.active_checkpoint_count,
            target.active_checkpoint_count,
        )

        outcome = classify_grammar_masked_speculative_outcome(result)

        assert outcome.kind == expected_kind
        assert snapshot == (
            draft.cached_token_ids,
            target.cached_token_ids,
            tuple(draft.decode_calls),
            tuple(target.decode_calls),
            tuple(draft.verify_calls),
            tuple(target.verify_calls),
            tuple(draft.rollback_calls),
            tuple(target.rollback_calls),
            tuple(draft.release_calls),
            tuple(target.release_calls),
            tuple(draft_mask.calls),
            tuple(target_mask.calls),
            tuple(draft_selector.calls),
            tuple(target_selector.calls),
            constraint.active_state_count,
            draft.active_checkpoint_count,
            target.active_checkpoint_count,
        )
        constraint.release_state(result.committed_state)


def test_classifier_does_not_call_d47_derived_properties(monkeypatch):
    result = _direct_result("mismatch")

    def fail(_self):
        raise AssertionError("D47 derived properties must not be read")

    for name in (
        "shortened",
        "acceptance_decision_made",
        "fully_accepted",
        "accepted_token_ids",
        "rejected_proposal_token_id",
        "output_token_ids",
    ):
        monkeypatch.setattr(
            GrammarMaskedSpeculativeIterationResult,
            name,
            property(fail),
        )
    assert classify_grammar_masked_speculative_outcome(result).kind == (
        "handoff_available"
    )


def test_non_d47_input_is_a_type_error_before_field_access():
    field_names = {
        field.name for field in fields(GrammarMaskedSpeculativeIterationResult)
    }

    class Trap:
        def __getattribute__(self, name):
            if name in field_names:
                raise AssertionError("non-D47 stored fields must not be read")
            return super().__getattribute__(name)

    with pytest.raises(TypeError):
        classify_grammar_masked_speculative_outcome(Trap())


def test_all_iteration_fields_are_acquired_once_in_dataclass_order():
    field_names = tuple(field.name for field in fields(GrammarMaskedSpeculativeIterationResult))

    class RecordingIteration(GrammarMaskedSpeculativeIterationResult):
        recording = False
        reads = None

        def __getattribute__(self, name):
            if type(self).recording and name in field_names:
                type(self).reads.append(name)
            return super().__getattribute__(name)

    result = RecordingIteration(
        (1,), 0, 2, 2, 3, 2, None, None, None, None, False
    )
    RecordingIteration.reads = []
    RecordingIteration.recording = True
    try:
        assert classify_grammar_masked_speculative_outcome(result).kind == (
            "handoff_available"
        )
    finally:
        RecordingIteration.recording = False
    assert RecordingIteration.reads == list(field_names)


def test_unreadable_iteration_field_raises_chained_invariant_after_acquisition():
    original = RuntimeError("unreadable")

    class UnreadableIteration(GrammarMaskedSpeculativeIterationResult):
        failing = False

        def __getattribute__(self, name):
            if type(self).failing and name == "committed_state":
                raise original
            return super().__getattribute__(name)

    result = UnreadableIteration(
        (1,), 0, 2, 2, 3, 2, None, None, None, None, False
    )
    object.__setattr__(result, "proposal_token_ids", [])
    UnreadableIteration.failing = True
    try:
        with pytest.raises(
            GrammarMaskedSpeculativeOutcomeInvariantError,
            match="committed_state",
        ) as raised:
            classify_grammar_masked_speculative_outcome(result)
    finally:
        UnreadableIteration.failing = False
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("proposal_token_ids", []),
        ("proposal_token_ids", (True,)),
        ("proposal_token_ids", (-1,)),
        ("accepted_count", True),
        ("accepted_count", -1),
        ("accepted_count", 4),
        ("initial_cache_length", True),
        ("initial_cache_length", 0),
        ("initial_cache_length", -1),
        ("final_cache_length", True),
        ("final_cache_length", -1),
        ("final_cache_length", 99),
        ("replacement_token_id", True),
        ("replacement_token_id", -1),
        ("replacement_token_id", 2),
        ("uncached_next_token_id", True),
        ("uncached_next_token_id", -1),
        ("uncached_next_token_id", 5),
        ("committed_state_is_match", 1),
    ],
)
def test_malformed_scalar_and_tuple_evidence_is_rejected(field_name, value):
    result = _direct_result("mismatch")
    _mutate(result, field_name, value)
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        classify_grammar_masked_speculative_outcome(result)


@pytest.mark.parametrize(
    "route,mutations",
    [
        ("zero", (("shortening_selection", None),)),
        ("zero", (("accepted_count", 1),)),
        ("zero", (("replacement_token_id", 4),)),
        ("zero", (("uncached_next_token_id", 4),)),
        ("zero", (("acceptance_no_decision_selection", _selection(False)),)),
        ("zero", (("final_row_no_decision_selection", _selection(False)),)),
        ("acceptance_empty", (("accepted_count", 3),)),
        ("acceptance_empty", (("replacement_token_id", 6),)),
        ("acceptance_empty", (("uncached_next_token_id", 6),)),
        (
            "acceptance_empty",
            (("final_row_no_decision_selection", _selection(False)),),
        ),
        ("mismatch", (("replacement_token_id", None),)),
        ("mismatch", (("uncached_next_token_id", None),)),
        ("mismatch", (("final_row_no_decision_selection", _selection(False)),)),
        ("bonus", (("uncached_next_token_id", None),)),
        ("bonus", (("replacement_token_id", 6),)),
        ("bonus", (("acceptance_no_decision_selection", _selection(False)),)),
        ("final_empty", (("uncached_next_token_id", 6),)),
        ("final_empty", (("replacement_token_id", 6),)),
    ],
)
def test_malformed_route_relationships_are_rejected(route, mutations):
    result = _direct_result(route)
    for field_name, value in mutations:
        _mutate(result, field_name, value)
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        classify_grammar_masked_speculative_outcome(result)


@pytest.mark.parametrize("route,field_name", [("zero", "shortening_selection"), ("acceptance_empty", "acceptance_no_decision_selection"), ("final_empty", "final_row_no_decision_selection")])
@pytest.mark.parametrize("selection_field,value", [("valid_token_ids", (1,)), ("is_match", 1), ("selected_token_id", 0)])
def test_malformed_terminal_selection_fields_are_rejected(
    route,
    field_name,
    selection_field,
    value,
):
    result = _direct_result(route)
    selection = getattr(result, field_name)
    object.__setattr__(selection, selection_field, value)
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        classify_grammar_masked_speculative_outcome(result)


def test_non_selection_evidence_and_terminal_match_disagreement_are_rejected():
    wrong_type = _mutate(_direct_result("zero"), "shortening_selection", object())
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        classify_grammar_masked_speculative_outcome(wrong_type)

    disagreement = _mutate(
        _direct_result("final_empty", is_match=True),
        "committed_state_is_match",
        False,
    )
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        classify_grammar_masked_speculative_outcome(disagreement)


def test_unreadable_selection_field_raises_chained_invariant():
    original = RuntimeError("unreadable selection")

    class UnreadableSelection(GrammarMaskedSelectionResult):
        failing = False

        def __getattribute__(self, name):
            if type(self).failing and name == "is_match":
                raise original
            return super().__getattribute__(name)

    selection = UnreadableSelection((), True, None)
    result = _direct_result("zero", is_match=True)
    object.__setattr__(result, "shortening_selection", selection)
    UnreadableSelection.failing = True
    try:
        with pytest.raises(
            GrammarMaskedSpeculativeOutcomeInvariantError,
            match="shortening_selection is_match",
        ) as raised:
            classify_grammar_masked_speculative_outcome(result)
    finally:
        UnreadableSelection.failing = False
    assert raised.value.__cause__ is original


def test_malformed_handoff_is_fully_validated_before_dispatch():
    result = _direct_result("bonus")
    object.__setattr__(result, "final_cache_length", 99)
    assert result.uncached_next_token_id is not None
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        classify_grammar_masked_speculative_outcome(result)


def test_result_construction_failure_propagates_without_mutation(monkeypatch):
    result = _direct_result("mismatch", state=HostileState())
    before = tuple(
        object.__getattribute__(result, field.name)
        for field in fields(GrammarMaskedSpeculativeIterationResult)
    )
    failure = RuntimeError("construction failed")

    def fail(*, kind):
        del kind
        raise failure

    monkeypatch.setattr(outcome_module, "GrammarMaskedSpeculativeOutcomeResult", fail)
    with pytest.raises(RuntimeError) as raised:
        classify_grammar_masked_speculative_outcome(result)
    assert raised.value is failure
    after = tuple(
        object.__getattribute__(result, field.name)
        for field in fields(GrammarMaskedSpeculativeIterationResult)
    )
    assert all(left is right for left, right in zip(before, after, strict=True))


def test_malformed_composed_result_is_rejected(monkeypatch):
    monkeypatch.setattr(
        outcome_module,
        "GrammarMaskedSpeculativeOutcomeResult",
        lambda *, kind: {"kind": kind},
    )
    with pytest.raises(GrammarMaskedSpeculativeOutcomeInvariantError):
        classify_grammar_masked_speculative_outcome(_direct_result("mismatch"))


def test_successful_result_retains_none_of_the_borrowed_evidence():
    class WeakIteration(GrammarMaskedSpeculativeIterationResult):
        pass

    class WeakSelection(GrammarMaskedSelectionResult):
        pass

    class WeakState:
        pass

    state = WeakState()
    selection = WeakSelection((), True, None)
    iteration = WeakIteration(
        (), 0, None, 2, 3, None, selection, None, None, state, True
    )
    iteration_ref = weakref.ref(iteration)
    selection_ref = weakref.ref(selection)
    state_ref = weakref.ref(state)

    outcome = classify_grammar_masked_speculative_outcome(iteration)
    del iteration, selection, state
    gc.collect()

    assert iteration_ref() is None
    assert selection_ref() is None
    assert state_ref() is None
    assert outcome.kind == "grammar_complete"
    assert tuple(field.name for field in fields(outcome)) == ("kind",)


def test_one_thousand_alternating_classifications_are_stable_and_bounded():
    inputs = (
        _direct_result("mismatch", accepted_count=0),
        _direct_result("mismatch", accepted_count=2),
        _direct_result("bonus"),
        _direct_result("zero", is_match=True),
        _direct_result("zero", is_match=False),
        _direct_result("acceptance_empty", is_match=True),
        _direct_result("acceptance_empty", is_match=False),
        _direct_result("final_empty", is_match=True),
        _direct_result("final_empty", is_match=False),
    )
    snapshots = tuple(
        tuple(
            object.__getattribute__(result, field.name)
            for field in fields(GrammarMaskedSpeculativeIterationResult)
        )
        for result in inputs
    )
    expected = (
        "handoff_available",
        "handoff_available",
        "handoff_available",
        "grammar_complete",
        "grammar_no_continuation",
        "grammar_complete",
        "grammar_no_continuation",
        "grammar_complete",
        "grammar_no_continuation",
    )

    for iteration in range(1_000):
        position = iteration % len(inputs)
        assert classify_grammar_masked_speculative_outcome(inputs[position]).kind == (
            expected[position]
        )

    for result, snapshot in zip(inputs, snapshots, strict=True):
        current = tuple(
            object.__getattribute__(result, field.name)
            for field in fields(GrammarMaskedSpeculativeIterationResult)
        )
        assert all(left is right for left, right in zip(snapshot, current, strict=True))


def test_isolated_import_and_classification_load_no_optional_runtime():
    code = (
        f"import sys\nsys.path.insert(0, {str(PACKAGE_ROOT / 'src')!r})\n"
        + textwrap.dedent(
            """
            import onyx_cuda.grammar_speculative_outcome as module
            from onyx_cuda import (
                GrammarMaskedSelectionResult,
                GrammarMaskedSpeculativeIterationResult,
                GrammarMaskedSpeculativeOutcomeResult,
                classify_grammar_masked_speculative_outcome,
            )

            empty_true = GrammarMaskedSelectionResult((), True, None)
            empty_false = GrammarMaskedSelectionResult((), False, None)
            state = object()
            inputs = (
                GrammarMaskedSpeculativeIterationResult(
                    (1,), 0, 2, 1, 2, 2, None, None, None, state, False
                ),
                GrammarMaskedSpeculativeIterationResult(
                    (1,), 1, None, 1, 3, 3, None, None, None, state, True
                ),
                GrammarMaskedSpeculativeIterationResult(
                    (), 0, None, 1, 2, None, empty_true, None, None, None, True
                ),
                GrammarMaskedSpeculativeIterationResult(
                    (1,), 0, None, 1, 2, None, None, empty_false, None, state, False
                ),
                GrammarMaskedSpeculativeIterationResult(
                    (1,), 1, None, 1, 3, None, None, None, empty_true, state, True
                ),
            )
            assert [classify_grammar_masked_speculative_outcome(value).kind for value in inputs] == [
                "handoff_available",
                "handoff_available",
                "grammar_complete",
                "grammar_no_continuation",
                "grammar_complete",
            ]
            assert module.GrammarMaskedSpeculativeOutcomeResult is GrammarMaskedSpeculativeOutcomeResult
            forbidden = (
                "onyx", "mlx", "torch", "transformers", "tokenizers",
                "huggingface_hub", "bitsandbytes", "accelerate", "onnxruntime",
                "onyx_cuda._grammar_native",
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
