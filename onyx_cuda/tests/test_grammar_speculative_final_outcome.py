import gc
import inspect
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_speculative_final_outcome as final_outcome_module
from onyx_cuda import (
    GrammarMaskedSelectionResult,
    GrammarMaskedSpeculativeFinalOutcomeError,
    GrammarMaskedSpeculativeFinalOutcomeInvariantError,
    GrammarMaskedSpeculativeFinalOutcomeResult,
    GrammarMaskedSpeculativeHandoffError,
    GrammarMaskedSpeculativeHandoffResult,
    GrammarMaskedSpeculativeIterationResult,
    GrammarMaskedSpeculativeOutcomeResult,
    classify_grammar_masked_speculative_outcome,
    coordinate_multi_iteration_grammar_masked_speculative_handoff,
    decide_grammar_masked_speculative_final_outcome,
)
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    FakeGrammarConstraint,
    FakeGrammarProgram,
)


VOCAB_SIZE = 8
EOS_TOKEN_ID = 6
PROMPT = (7,)
CURRENT_TOKEN_ID = 0
SCRIPT = tuple(
    tuple(float(row * VOCAB_SIZE + token) for token in range(VOCAB_SIZE))
    for row in range(64)
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class StringSubclass(str):
    pass


class IntegerSubclass(int):
    pass


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


class Mask:
    @property
    def vocab_size(self):
        return VOCAB_SIZE

    def apply(self, logits, valid_token_ids, /):
        del valid_token_ids
        return logits


class Selector:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self, row):
        del row
        return self.values.pop(0)


def _selection(is_match):
    return GrammarMaskedSelectionResult((), is_match, None)


def _iteration(
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
    proposal = tuple(proposal)
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


def _handoff_result(
    route,
    *,
    accumulated_prefix=(),
    **iteration_kwargs,
):
    iteration = _iteration(route, **iteration_kwargs)
    outcome = classify_grammar_masked_speculative_outcome(iteration)
    output_token_ids = tuple(accumulated_prefix) + iteration.output_token_ids
    return GrammarMaskedSpeculativeHandoffResult(
        output_token_ids,
        iteration,
        outcome,
    )


def _decide(result):
    return decide_grammar_masked_speculative_final_outcome(
        result,
        vocab_size=VOCAB_SIZE,
        eos_token_id=EOS_TOKEN_ID,
    )


def _vocabulary():
    return tuple(bytes((token,)) for token in range(VOCAB_SIZE))


def _coordinate_d50(*, handoff_count, iteration_bound, final_match=True):
    transitions = []
    supports = []
    match_states = set()
    for position in range(handoff_count):
        stage = f"s{position}"
        draft_child = f"d{position}"
        next_stage = f"s{position + 1}"
        transitions.extend(
            (
                (stage, 1, draft_child),
                (stage, 3, next_stage),
            )
        )
        supports.extend(((stage, (1, 3)), (draft_child, ())))
        match_states.add(draft_child)
    terminal_state = f"s{handoff_count}"
    supports.append((terminal_state, ()))
    match_states.add(terminal_state)
    constraint = RecordingConstraint(
        _vocabulary(),
        grammar_type="regex",
        program=FakeGrammarProgram(
            initial_state="s0",
            transitions=tuple(transitions),
            valid_token_ids=tuple(supports),
            match_states=frozenset(match_states),
        ),
        force_nonmatch=not final_match,
    )
    draft = FakeAutoregressiveBackend(SCRIPT, model_id="draft")
    target = FakeAutoregressiveBackend(SCRIPT, model_id="target")
    draft.prefill(PROMPT)
    target.prefill(PROMPT)
    draft_root = draft.create_cache_checkpoint()
    target_root = target.create_cache_checkpoint()
    result = coordinate_multi_iteration_grammar_masked_speculative_handoff(
        draft,
        target,
        CURRENT_TOKEN_ID,
        constraint,
        constraint.init_state(),
        Mask(),
        Mask(),
        iteration_bound=iteration_bound,
        proposal_bound=1,
        draft_select_token=Selector([1] * handoff_count),
        target_select_token=Selector([3] * handoff_count),
        draft_root_checkpoint=draft_root,
        target_root_checkpoint=target_root,
    )
    return result, constraint, draft, target, draft_root, target_root


def _settle_d50(coordinated):
    result, constraint, draft, target, draft_root, target_root = coordinated
    constraint.release_state(result.final_iteration.committed_state)
    draft.rollback_cache(draft_root)
    target.rollback_cache(target_root)
    assert constraint.active_state_count == 0
    assert draft.cached_token_ids == target.cached_token_ids == PROMPT


def _mutate(value, field_name, replacement):
    object.__setattr__(value, field_name, replacement)
    return value


def test_public_surface_signature_result_and_error_hierarchy():
    assert final_outcome_module.__all__ == [
        "GrammarMaskedSpeculativeFinalOutcomeError",
        "GrammarMaskedSpeculativeFinalOutcomeInvariantError",
        "GrammarMaskedSpeculativeFinalOutcomeResult",
        "decide_grammar_masked_speculative_final_outcome",
    ]
    package = sys.modules["onyx_cuda"]
    for symbol in final_outcome_module.__all__:
        assert getattr(package, symbol) is getattr(final_outcome_module, symbol)
    assert issubclass(
        GrammarMaskedSpeculativeFinalOutcomeError,
        GrammarMaskedSpeculativeHandoffError,
    )
    assert issubclass(
        GrammarMaskedSpeculativeFinalOutcomeInvariantError,
        GrammarMaskedSpeculativeFinalOutcomeError,
    )
    assert not hasattr(final_outcome_module, "GrammarMaskedSpeculativeFinalOutcomeCleanupError")

    signature = inspect.signature(decide_grammar_masked_speculative_final_outcome)
    assert tuple(signature.parameters) == (
        "handoff_result",
        "vocab_size",
        "eos_token_id",
    )
    assert signature.parameters["handoff_result"].kind is (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert signature.parameters["vocab_size"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["eos_token_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert [
        field.name for field in fields(GrammarMaskedSpeculativeFinalOutcomeResult)
    ] == [
        "output_token_ids",
        "final_iteration",
        "final_outcome",
        "disposition",
        "grammar_completion_token_id",
    ]
    result = _decide(_handoff_result("zero", is_match=True))
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "iteration_bound")
    assert not hasattr(result, "executed_iteration_count")
    assert not hasattr(result, "finish_reason")
    with pytest.raises(FrozenInstanceError):
        result.disposition = "grammar_no_continuation"


@pytest.mark.parametrize(
    ("route", "kwargs", "expected_disposition"),
    [
        ("zero", {"is_match": True}, "grammar_complete"),
        ("zero", {"is_match": False}, "grammar_no_continuation"),
        ("acceptance_empty", {"accepted_count": 0, "is_match": True}, "grammar_complete"),
        ("acceptance_empty", {"accepted_count": 1, "is_match": False}, "grammar_no_continuation"),
        ("acceptance_empty", {"accepted_count": 2, "is_match": True}, "grammar_complete"),
        ("final_empty", {"proposal": (1,), "is_match": True}, "grammar_complete"),
        ("final_empty", {"proposal": (1, 2, 3), "is_match": False}, "grammar_no_continuation"),
        ("mismatch", {"accepted_count": 0, "is_match": False}, "iteration_bound_exhausted"),
        ("mismatch", {"accepted_count": 1, "is_match": True}, "iteration_bound_exhausted"),
        ("mismatch", {"accepted_count": 2, "is_match": False}, "iteration_bound_exhausted"),
        ("bonus", {"proposal": (1,), "is_match": False}, "iteration_bound_exhausted"),
        ("bonus", {"proposal": (1, 2, 3), "is_match": True}, "iteration_bound_exhausted"),
    ],
)
def test_every_terminal_route_and_final_kind_map_through_d48(
    route,
    kwargs,
    expected_disposition,
):
    handoff = _handoff_result(route, **kwargs)
    result = _decide(handoff)
    assert result.disposition == expected_disposition
    assert result.output_token_ids is handoff.output_token_ids
    assert result.final_iteration is handoff.final_iteration
    assert result.final_outcome is handoff.final_outcome
    assert result.final_iteration.committed_state is handoff.final_iteration.committed_state


@pytest.mark.parametrize("history_match", [False, True])
@pytest.mark.parametrize("terminal_match", [False, True])
@pytest.mark.parametrize("route", ["acceptance_empty", "final_empty"])
def test_nonzero_d44_history_never_overrides_the_final_terminal_kind(
    route,
    terminal_match,
    history_match,
):
    result = _decide(
        _handoff_result(
            route,
            is_match=terminal_match,
            shortening_is_match=history_match,
        )
    )
    assert result.disposition == (
        "grammar_complete" if terminal_match else "grammar_no_continuation"
    )


@pytest.mark.parametrize("route", ["mismatch", "bonus"])
@pytest.mark.parametrize("committed_match", [False, True])
def test_handoff_precedence_maps_only_to_bound_exhaustion(route, committed_match):
    result = _decide(
        _handoff_result(
            route,
            is_match=committed_match,
            shortening_is_match=True,
        )
    )
    assert result.disposition == "iteration_bound_exhausted"
    assert result.grammar_completion_token_id is None


@pytest.mark.parametrize(
    ("route", "is_match", "expected_disposition"),
    [
        ("zero", True, "grammar_complete"),
        ("zero", False, "grammar_no_continuation"),
        ("mismatch", True, "iteration_bound_exhausted"),
    ],
)
def test_direct_result_contract_and_token_views(route, is_match, expected_disposition):
    handoff = _handoff_result(route, is_match=is_match, accumulated_prefix=(4,))
    completion_token = EOS_TOKEN_ID if expected_disposition == "grammar_complete" else None
    result = GrammarMaskedSpeculativeFinalOutcomeResult(
        handoff.output_token_ids,
        handoff.final_iteration,
        handoff.final_outcome,
        expected_disposition,
        completion_token,
    )
    assert result.visible_token_ids is handoff.output_token_ids
    if expected_disposition == "grammar_complete":
        assert result.sampled_token_ids == handoff.output_token_ids + (EOS_TOKEN_ID,)
    else:
        assert result.sampled_token_ids is handoff.output_token_ids


def test_completion_eos_is_one_hidden_occurrence_even_when_numeric_id_repeats():
    handoff = _handoff_result(
        "final_empty",
        proposal=(EOS_TOKEN_ID,),
        is_match=True,
    )
    result = _decide(handoff)
    assert handoff.output_token_ids == (EOS_TOKEN_ID,)
    assert result.visible_token_ids == (EOS_TOKEN_ID,)
    assert result.sampled_token_ids == (EOS_TOKEN_ID, EOS_TOKEN_ID)
    assert result.grammar_completion_token_id == EOS_TOKEN_ID
    assert result.final_iteration.uncached_next_token_id is None


@pytest.mark.parametrize(
    "disposition",
    ["grammar_complete", "grammar_no_continuation", "iteration_bound_exhausted"],
)
def test_direct_result_rejects_string_subclass_disposition(disposition):
    handoff = _handoff_result("zero", is_match=True)
    with pytest.raises(TypeError):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            handoff.output_token_ids,
            handoff.final_iteration,
            handoff.final_outcome,
            StringSubclass(disposition),
            EOS_TOKEN_ID,
        )


@pytest.mark.parametrize("disposition", ["", "handoff_available", "stop", "length"])
def test_direct_result_rejects_unsupported_dispositions(disposition):
    handoff = _handoff_result("zero", is_match=True)
    with pytest.raises(GrammarMaskedSpeculativeFinalOutcomeInvariantError):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            handoff.output_token_ids,
            handoff.final_iteration,
            handoff.final_outcome,
            disposition,
            EOS_TOKEN_ID,
        )


@pytest.mark.parametrize(
    ("route", "is_match", "disposition", "completion_token"),
    [
        ("zero", True, "grammar_complete", None),
        ("zero", True, "grammar_no_continuation", None),
        ("zero", False, "grammar_no_continuation", EOS_TOKEN_ID),
        ("mismatch", True, "iteration_bound_exhausted", EOS_TOKEN_ID),
        ("mismatch", True, "grammar_complete", EOS_TOKEN_ID),
    ],
)
def test_direct_result_rejects_disposition_eos_and_evidence_disagreement(
    route,
    is_match,
    disposition,
    completion_token,
):
    handoff = _handoff_result(route, is_match=is_match)
    with pytest.raises(GrammarMaskedSpeculativeFinalOutcomeInvariantError):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            handoff.output_token_ids,
            handoff.final_iteration,
            handoff.final_outcome,
            disposition,
            completion_token,
        )


@pytest.mark.parametrize(
    ("completion_token", "error_type"),
    [
        (True, TypeError),
        (IntegerSubclass(EOS_TOKEN_ID), TypeError),
        (1.0, TypeError),
        ("6", TypeError),
        (-1, ValueError),
    ],
)
def test_direct_result_rejects_malformed_completion_tokens(
    completion_token,
    error_type,
):
    handoff = _handoff_result("zero", is_match=True)
    with pytest.raises(error_type):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            handoff.output_token_ids,
            handoff.final_iteration,
            handoff.final_outcome,
            "grammar_complete",
            completion_token,
        )


def test_direct_result_rejects_terminal_match_fact_disagreement():
    handoff = _handoff_result("zero", is_match=True)
    _mutate(handoff.final_iteration, "committed_state_is_match", False)
    with pytest.raises(
        GrammarMaskedSpeculativeFinalOutcomeInvariantError,
        match="matching terminal evidence",
    ):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            handoff.output_token_ids,
            handoff.final_iteration,
            handoff.final_outcome,
            "grammar_complete",
            EOS_TOKEN_ID,
        )


@pytest.mark.parametrize(
    ("output", "error_type"),
    [
        ([], TypeError),
        ((True,), TypeError),
        ((1.0,), TypeError),
        ((-1,), ValueError),
    ],
)
def test_direct_result_rejects_malformed_accumulated_output(output, error_type):
    handoff = _handoff_result("zero", is_match=True)
    with pytest.raises(error_type):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            output,
            handoff.final_iteration,
            handoff.final_outcome,
            "grammar_complete",
            EOS_TOKEN_ID,
        )


def test_direct_result_requires_genuine_final_evidence_and_exact_suffix():
    handoff = _handoff_result("mismatch", accepted_count=1)
    with pytest.raises(TypeError):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            handoff.output_token_ids,
            object(),
            handoff.final_outcome,
            "iteration_bound_exhausted",
            None,
        )
    with pytest.raises(TypeError):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            handoff.output_token_ids,
            handoff.final_iteration,
            object(),
            "iteration_bound_exhausted",
            None,
        )
    with pytest.raises(
        GrammarMaskedSpeculativeFinalOutcomeInvariantError,
        match="suffix",
    ):
        GrammarMaskedSpeculativeFinalOutcomeResult(
            (7, handoff.output_token_ids[-1]),
            handoff.final_iteration,
            handoff.final_outcome,
            "iteration_bound_exhausted",
            None,
        )


@pytest.mark.parametrize(
    ("vocab_size", "error_type"),
    [
        (True, TypeError),
        (IntegerSubclass(8), TypeError),
        (8.0, TypeError),
        ("8", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_vocab_size_policy_fails_before_d50_observation(
    monkeypatch,
    vocab_size,
    error_type,
):
    monkeypatch.setattr(
        final_outcome_module,
        "_read_attribute",
        lambda *args, **kwargs: pytest.fail("D50 evidence was observed"),
    )
    monkeypatch.setattr(
        final_outcome_module,
        "classify_grammar_masked_speculative_outcome",
        lambda value: pytest.fail(f"D48 observed {value!r}"),
    )
    with pytest.raises(error_type):
        decide_grammar_masked_speculative_final_outcome(
            object(),
            vocab_size=vocab_size,
            eos_token_id=0,
        )


@pytest.mark.parametrize(
    ("eos_token_id", "error_type"),
    [
        (True, TypeError),
        (IntegerSubclass(1), TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (-1, ValueError),
        (VOCAB_SIZE, ValueError),
        (VOCAB_SIZE + 1, ValueError),
    ],
)
def test_eos_policy_fails_before_d50_observation(
    monkeypatch,
    eos_token_id,
    error_type,
):
    monkeypatch.setattr(
        final_outcome_module,
        "_read_attribute",
        lambda *args, **kwargs: pytest.fail("D50 evidence was observed"),
    )
    with pytest.raises(error_type):
        decide_grammar_masked_speculative_final_outcome(
            object(),
            vocab_size=VOCAB_SIZE,
            eos_token_id=eos_token_id,
        )


def test_non_d50_input_is_a_type_error_before_field_access():
    class HostileInput:
        def __getattribute__(self, name):
            del name
            raise AssertionError("non-D50 fields must not be read")

    with pytest.raises(TypeError, match="type could not be determined"):
        _decide(HostileInput())


def test_d50_fields_are_acquired_once_in_dataclass_order(monkeypatch):
    handoff = _handoff_result("zero", is_match=True)
    real_read = final_outcome_module._read_attribute
    reads = []

    def record(value, name, *, label):
        if value is handoff:
            reads.append(name)
        return real_read(value, name, label=label)

    monkeypatch.setattr(final_outcome_module, "_read_attribute", record)
    _decide(handoff)
    assert reads == ["output_token_ids", "final_iteration", "final_outcome"]


def test_unreadable_d50_field_preserves_the_original_cause():
    failure = RuntimeError("unreadable final iteration")

    class UnreadableHandoffResult(GrammarMaskedSpeculativeHandoffResult):
        blocked = False

        def __getattribute__(self, name):
            if name == "final_iteration" and object.__getattribute__(self, "blocked"):
                raise failure
            return super().__getattribute__(name)

    valid = _handoff_result("zero", is_match=True)
    handoff = UnreadableHandoffResult(
        valid.output_token_ids,
        valid.final_iteration,
        valid.final_outcome,
    )
    object.__setattr__(handoff, "blocked", True)
    with pytest.raises(
        GrammarMaskedSpeculativeFinalOutcomeInvariantError,
        match="could not be read",
    ) as caught:
        _decide(handoff)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("malformed_output", [[], (True,), (-1,), (VOCAB_SIZE,)])
def test_malformed_d50_output_fails_closed(malformed_output):
    handoff = _mutate(
        _handoff_result("zero", is_match=True),
        "output_token_ids",
        malformed_output,
    )
    with pytest.raises(GrammarMaskedSpeculativeFinalOutcomeInvariantError):
        _decide(handoff)


def test_wrong_final_result_types_fail_closed():
    for field_name in ("final_iteration", "final_outcome"):
        handoff = _mutate(
            _handoff_result("zero", is_match=True),
            field_name,
            object(),
        )
        with pytest.raises(GrammarMaskedSpeculativeFinalOutcomeInvariantError):
            _decide(handoff)


def test_tampered_d47_evidence_is_wrapped_with_d48_failure_as_cause():
    handoff = _handoff_result("mismatch", accepted_count=1)
    _mutate(handoff.final_iteration, "accepted_count", True)
    with pytest.raises(
        GrammarMaskedSpeculativeFinalOutcomeInvariantError,
        match="reclassified",
    ) as caught:
        _decide(handoff)
    assert caught.value.__cause__ is not None
    assert caught.value.__cause__.__class__.__module__ == (
        "onyx_cuda.grammar_speculative_outcome"
    )


def test_stored_and_recomputed_d48_disagreement_fails_closed():
    handoff = _handoff_result("zero", is_match=True)
    _mutate(handoff.final_outcome, "kind", "grammar_no_continuation")
    with pytest.raises(
        GrammarMaskedSpeculativeFinalOutcomeInvariantError,
        match="disagrees",
    ):
        _decide(handoff)


def test_final_suffix_terminal_match_and_handoff_relationships_fail_closed():
    mismatch = _handoff_result("mismatch", accepted_count=1, accumulated_prefix=(4,))
    _mutate(mismatch, "output_token_ids", (4, mismatch.output_token_ids[-1]))
    with pytest.raises(GrammarMaskedSpeculativeFinalOutcomeInvariantError, match="suffix"):
        _decide(mismatch)

    terminal = _handoff_result("zero", is_match=True)
    _mutate(terminal.final_iteration, "committed_state_is_match", False)
    with pytest.raises(GrammarMaskedSpeculativeFinalOutcomeInvariantError):
        _decide(terminal)

    ordinary_iteration = _iteration("mismatch", accepted_count=0)

    class MisreportedOutputIteration(GrammarMaskedSpeculativeIterationResult):
        @property
        def output_token_ids(self):
            return (4,)

    iteration = MisreportedOutputIteration(
        *(getattr(ordinary_iteration, field.name) for field in fields(ordinary_iteration))
    )
    mismatch = GrammarMaskedSpeculativeHandoffResult(
        (4,),
        iteration,
        GrammarMaskedSpeculativeOutcomeResult("handoff_available"),
    )
    with pytest.raises(
        GrammarMaskedSpeculativeFinalOutcomeInvariantError,
        match="handoff token",
    ):
        _decide(mismatch)


def test_d48_is_called_exactly_once_and_original_outcome_is_retained(monkeypatch):
    handoff = _handoff_result("final_empty", is_match=True)
    stored_outcome = handoff.final_outcome
    real_classifier = final_outcome_module.classify_grammar_masked_speculative_outcome
    calls = []

    def record(iteration):
        calls.append(iteration)
        return real_classifier(iteration)

    monkeypatch.setattr(
        final_outcome_module,
        "classify_grammar_masked_speculative_outcome",
        record,
    )
    result = _decide(handoff)
    assert calls == [handoff.final_iteration]
    assert result.final_outcome is stored_outcome


def test_opaque_none_and_hostile_states_are_read_but_never_inspected():
    for state in (None, HostileState()):
        handoff = _handoff_result("zero", is_match=True, state=state)
        result = _decide(handoff)
        assert result.final_iteration.committed_state is state


@pytest.mark.parametrize(
    ("handoff_count", "iteration_bound", "final_match", "disposition"),
    [
        (0, 1, True, "grammar_complete"),
        (0, 1, False, "grammar_no_continuation"),
        (1, 1, True, "iteration_bound_exhausted"),
        (1, 3, True, "grammar_complete"),
        (1, 3, False, "grammar_no_continuation"),
        (2, 2, True, "iteration_bound_exhausted"),
    ],
)
def test_genuine_d50_results_preserve_output_cache_state_and_final_evidence(
    handoff_count,
    iteration_bound,
    final_match,
    disposition,
):
    coordinated = _coordinate_d50(
        handoff_count=handoff_count,
        iteration_bound=iteration_bound,
        final_match=final_match,
    )
    handoff, constraint, draft, target, *_ = coordinated
    output = handoff.output_token_ids
    final_iteration = handoff.final_iteration
    final_outcome = handoff.final_outcome
    state = final_iteration.committed_state
    advance_calls = tuple(constraint.advance_calls)
    release_calls = tuple(constraint.release_calls)
    draft_cache = draft.cached_token_ids
    target_cache = target.cached_token_ids

    result = _decide(handoff)

    assert result.disposition == disposition
    assert result.output_token_ids is output
    assert result.final_iteration is final_iteration
    assert result.final_outcome is final_outcome
    assert result.final_iteration.committed_state is state
    assert tuple(constraint.advance_calls) == advance_calls
    assert tuple(constraint.release_calls) == release_calls
    assert draft.cached_token_ids == draft_cache
    assert target.cached_token_ids == target_cache
    if disposition == "grammar_complete":
        assert result.sampled_token_ids == output + (EOS_TOKEN_ID,)
        assert result.visible_token_ids is output
        assert EOS_TOKEN_ID not in draft.cached_token_ids
        assert EOS_TOKEN_ID not in target.cached_token_ids
    elif disposition == "iteration_bound_exhausted":
        assert result.sampled_token_ids is output
        assert output[-1] == final_iteration.uncached_next_token_id == 3
        assert output.count(3) == handoff_count
        assert draft.cached_token_ids.count(3) == max(handoff_count - 1, 0)
        assert target.cached_token_ids.count(3) == max(handoff_count - 1, 0)
    else:
        assert result.sampled_token_ids is output
        assert result.grammar_completion_token_id is None

    _settle_d50(coordinated)


def test_result_construction_failure_leaves_input_state_caller_owned(monkeypatch):
    coordinated = _coordinate_d50(
        handoff_count=0,
        iteration_bound=1,
        final_match=True,
    )
    handoff, constraint, *_ = coordinated
    failure = RuntimeError("construction failed")
    release_calls = tuple(constraint.release_calls)

    def fail_construction(**kwargs):
        del kwargs
        raise failure

    monkeypatch.setattr(
        final_outcome_module,
        "GrammarMaskedSpeculativeFinalOutcomeResult",
        fail_construction,
    )
    with pytest.raises(RuntimeError) as caught:
        _decide(handoff)
    assert caught.value is failure
    assert tuple(constraint.release_calls) == release_calls
    assert constraint.active_state_count == 1
    _settle_d50(coordinated)


def test_injected_d48_failure_leaves_input_state_caller_owned(monkeypatch):
    coordinated = _coordinate_d50(
        handoff_count=0,
        iteration_bound=1,
        final_match=True,
    )
    handoff, constraint, *_ = coordinated
    failure = RuntimeError("D48 failed")
    release_calls = tuple(constraint.release_calls)

    def fail_classification(iteration):
        del iteration
        raise failure

    monkeypatch.setattr(
        final_outcome_module,
        "classify_grammar_masked_speculative_outcome",
        fail_classification,
    )
    with pytest.raises(GrammarMaskedSpeculativeFinalOutcomeInvariantError) as caught:
        _decide(handoff)
    assert caught.value.__cause__ is failure
    assert tuple(constraint.release_calls) == release_calls
    assert constraint.active_state_count == 1
    _settle_d50(coordinated)


def test_result_does_not_retain_the_d50_wrapper():
    class WeakHandoffResult(GrammarMaskedSpeculativeHandoffResult):
        pass

    valid = _handoff_result("final_empty", is_match=True)
    handoff = WeakHandoffResult(
        valid.output_token_ids,
        valid.final_iteration,
        valid.final_outcome,
    )
    reference = weakref.ref(handoff)
    result = _decide(handoff)
    del handoff
    gc.collect()
    assert reference() is None
    assert result.output_token_ids is valid.output_token_ids
    assert result.final_iteration is valid.final_iteration
    assert result.final_outcome is valid.final_outcome


def test_one_thousand_alternating_decisions_are_deterministic_and_stateless():
    wrappers = (
        _handoff_result("zero", is_match=True, state=None),
        _handoff_result("zero", is_match=False, state=HostileState()),
        _handoff_result("mismatch", accepted_count=1, is_match=True),
    )
    expected = (
        "grammar_complete",
        "grammar_no_continuation",
        "iteration_bound_exhausted",
    )
    for position in range(1000):
        index = position % len(wrappers)
        result = _decide(wrappers[index])
        assert result.disposition == expected[index]
        assert result.output_token_ids is wrappers[index].output_token_ids
    mutable_module_values = [
        value
        for name, value in vars(final_outcome_module).items()
        if not name.startswith("__") and isinstance(value, (dict, list, set))
    ]
    assert mutable_module_values == []


def test_isolated_import_and_decision_are_optional_runtime_free():
    source_root = str(PACKAGE_ROOT / "src")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {source_root!r})
        from onyx_cuda import (
            GrammarMaskedSelectionResult,
            GrammarMaskedSpeculativeHandoffResult,
            GrammarMaskedSpeculativeIterationResult,
            GrammarMaskedSpeculativeOutcomeResult,
            decide_grammar_masked_speculative_final_outcome,
        )

        selection = GrammarMaskedSelectionResult((), True, None)
        iteration = GrammarMaskedSpeculativeIterationResult(
            (), 0, None, 1, 2, None, selection, None, None, None, True
        )
        handoff = GrammarMaskedSpeculativeHandoffResult(
            (), iteration, GrammarMaskedSpeculativeOutcomeResult('grammar_complete')
        )
        result = decide_grammar_masked_speculative_final_outcome(
            handoff, vocab_size=8, eos_token_id=6
        )
        assert result.disposition == 'grammar_complete'
        assert result.sampled_token_ids == (6,)
        assert result.visible_token_ids == ()

        forbidden = (
            'onyx', 'mlx', 'torch', 'transformers', 'tokenizers',
            'huggingface_hub', 'bitsandbytes', 'accelerate', 'onnxruntime',
            'psutil', 'onyx_cuda._native',
        )
        assert not any(
            name == prefix or name.startswith(prefix + '.')
            for name in sys.modules
            for prefix in forbidden
        )
        print('isolated-d51-ok')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "isolated-d51-ok"
