import gc
import inspect
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import onyx_cuda.grammar_speculative_request_policy as request_policy_module
from onyx_cuda import (
    GrammarMaskedSelectionResult,
    GrammarMaskedSpeculativeFinalOutcomeError,
    GrammarMaskedSpeculativeFinalOutcomeResult,
    GrammarMaskedSpeculativeIterationResult,
    GrammarMaskedSpeculativeOutcomeResult,
    GrammarMaskedSpeculativeRequestPolicyError,
    GrammarMaskedSpeculativeRequestPolicyInvariantError,
    GrammarMaskedSpeculativeRequestPolicyResult,
    coordinate_multi_iteration_grammar_masked_speculative_handoff,
    decide_grammar_masked_speculative_final_outcome,
    decide_grammar_masked_speculative_request_policy,
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


def _terminal_iteration(*, is_match, state=None):
    selection = GrammarMaskedSelectionResult((), is_match, None)
    return GrammarMaskedSpeculativeIterationResult(
        (),
        0,
        None,
        2,
        3,
        None,
        selection,
        None,
        None,
        state,
        is_match,
    )


def _handoff_iteration(*, token_id=3, is_match=False, state=None):
    return GrammarMaskedSpeculativeIterationResult(
        (1,),
        0,
        token_id,
        2,
        3,
        token_id,
        None,
        None,
        None,
        state,
        is_match,
    )


def _d51(disposition, *, output=None, state=None, committed_match=None):
    if disposition == "grammar_complete":
        output = (2,) if output is None else output
        iteration = _terminal_iteration(is_match=True, state=state)
        outcome = GrammarMaskedSpeculativeOutcomeResult("grammar_complete")
        completion_token = EOS_TOKEN_ID
    elif disposition == "grammar_no_continuation":
        output = (2,) if output is None else output
        iteration = _terminal_iteration(is_match=False, state=state)
        outcome = GrammarMaskedSpeculativeOutcomeResult("grammar_no_continuation")
        completion_token = None
    else:
        output = (2, 3) if output is None else output
        is_match = False if committed_match is None else committed_match
        iteration = _handoff_iteration(
            token_id=output[-1],
            is_match=is_match,
            state=state,
        )
        outcome = GrammarMaskedSpeculativeOutcomeResult("handoff_available")
        completion_token = None
    return GrammarMaskedSpeculativeFinalOutcomeResult(
        output,
        iteration,
        outcome,
        disposition,
        completion_token,
    )


def _decide(
    result,
    *,
    stop=None,
    eligible=False,
    budget=None,
    vocab_size=VOCAB_SIZE,
):
    if budget is None:
        budget = len(result.sampled_token_ids) + 1
    return decide_grammar_masked_speculative_request_policy(
        result,
        vocab_size=vocab_size,
        matched_stop_token_ids=stop,
        matched_stop_is_eligible=eligible,
        available_output_token_budget=budget,
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
        transitions.extend(((stage, 1, draft_child), (stage, 3, next_stage)))
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


def _expected_request_disposition(d51_disposition, stop_kind, remaining):
    if stop_kind == "eligible":
        return "stop"
    if d51_disposition == "grammar_complete":
        return "grammar_complete"
    if remaining == 0:
        return "output_budget_exhausted"
    if d51_disposition == "grammar_no_continuation":
        return "grammar_no_continuation"
    return "continuation_permitted"


def test_public_surface_signature_result_and_error_hierarchy():
    assert request_policy_module.__all__ == [
        "GrammarMaskedSpeculativeRequestPolicyError",
        "GrammarMaskedSpeculativeRequestPolicyInvariantError",
        "GrammarMaskedSpeculativeRequestPolicyResult",
        "decide_grammar_masked_speculative_request_policy",
    ]
    package = sys.modules["onyx_cuda"]
    for symbol in request_policy_module.__all__:
        assert getattr(package, symbol) is getattr(request_policy_module, symbol)
    assert issubclass(
        GrammarMaskedSpeculativeRequestPolicyError,
        GrammarMaskedSpeculativeFinalOutcomeError,
    )
    assert issubclass(
        GrammarMaskedSpeculativeRequestPolicyInvariantError,
        GrammarMaskedSpeculativeRequestPolicyError,
    )
    assert not hasattr(
        request_policy_module,
        "GrammarMaskedSpeculativeRequestPolicyCleanupError",
    )

    signature = inspect.signature(decide_grammar_masked_speculative_request_policy)
    assert tuple(signature.parameters) == (
        "final_outcome_result",
        "vocab_size",
        "matched_stop_token_ids",
        "matched_stop_is_eligible",
        "available_output_token_budget",
    )
    assert signature.parameters["final_outcome_result"].kind is (
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    for parameter_name in tuple(signature.parameters)[1:]:
        assert signature.parameters[parameter_name].kind is (
            inspect.Parameter.KEYWORD_ONLY
        )
        assert signature.parameters[parameter_name].default is inspect.Parameter.empty

    assert [
        field.name for field in fields(GrammarMaskedSpeculativeRequestPolicyResult)
    ] == [
        "final_outcome_result",
        "request_disposition",
        "matched_stop_token_ids",
        "remaining_output_token_budget",
    ]
    result = _decide(_d51("grammar_complete"))
    assert not hasattr(result, "__dict__")
    for omitted_name in (
        "finish_reason",
        "output_token_ids",
        "final_iteration",
        "final_outcome",
        "grammar_completion_token_id",
        "uncached_next_token_id",
        "committed_state",
        "iteration_bound",
        "executed_iteration_count",
        "backend",
        "history",
    ):
        assert not hasattr(result, omitted_name)
    with pytest.raises(FrozenInstanceError):
        result.request_disposition = "stop"


@pytest.mark.parametrize(
    (
        "d51_disposition",
        "request_disposition",
        "matched_stop",
        "remaining_budget",
    ),
    [
        ("grammar_complete", "stop", (2,), 0),
        ("grammar_complete", "grammar_complete", None, 0),
        ("grammar_no_continuation", "output_budget_exhausted", None, 0),
        ("grammar_no_continuation", "grammar_no_continuation", None, 1),
        ("iteration_bound_exhausted", "output_budget_exhausted", None, 0),
        ("iteration_bound_exhausted", "continuation_permitted", None, 1),
    ],
)
def test_direct_result_contract_for_all_valid_dispositions(
    d51_disposition,
    request_disposition,
    matched_stop,
    remaining_budget,
):
    final = _d51(d51_disposition)
    result = GrammarMaskedSpeculativeRequestPolicyResult(
        final,
        request_disposition,
        matched_stop,
        remaining_budget,
    )
    assert result.final_outcome_result is final
    assert result.matched_stop_token_ids is matched_stop
    assert result.sampled_token_ids == final.sampled_token_ids
    assert result.visible_token_ids is final.output_token_ids
    expected_terminal = request_disposition != "continuation_permitted"
    assert result.request_is_terminal is expected_terminal
    assert result.further_generation_permitted is not expected_terminal


@pytest.mark.parametrize(
    "disposition",
    [
        "stop",
        "grammar_complete",
        "output_budget_exhausted",
        "grammar_no_continuation",
        "continuation_permitted",
    ],
)
def test_direct_result_rejects_string_subclass_disposition(disposition):
    with pytest.raises(TypeError):
        GrammarMaskedSpeculativeRequestPolicyResult(
            _d51("grammar_complete"),
            StringSubclass(disposition),
            None,
            0,
        )


@pytest.mark.parametrize(
    "disposition",
    ["", "length", "iteration_bound_exhausted", "handoff_available"],
)
def test_direct_result_rejects_unsupported_disposition(disposition):
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        GrammarMaskedSpeculativeRequestPolicyResult(
            _d51("grammar_complete"),
            disposition,
            None,
            0,
        )


@pytest.mark.parametrize(
    ("stop", "error_type"),
    [
        ([], TypeError),
        ("2", TypeError),
        ((), ValueError),
        ((True,), TypeError),
        ((IntegerSubclass(2),), TypeError),
        ((2.0,), TypeError),
        ((-1,), ValueError),
    ],
)
def test_direct_result_rejects_malformed_effective_stop(stop, error_type):
    with pytest.raises(error_type):
        GrammarMaskedSpeculativeRequestPolicyResult(
            _d51("grammar_complete"),
            "stop",
            stop,
            0,
        )


@pytest.mark.parametrize(
    ("remaining_budget", "error_type"),
    [
        (True, TypeError),
        (IntegerSubclass(1), TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (-1, ValueError),
    ],
)
def test_direct_result_rejects_malformed_remaining_budget(
    remaining_budget,
    error_type,
):
    with pytest.raises(error_type):
        GrammarMaskedSpeculativeRequestPolicyResult(
            _d51("grammar_complete"),
            "grammar_complete",
            None,
            remaining_budget,
        )


@pytest.mark.parametrize(
    (
        "d51_disposition",
        "request_disposition",
        "matched_stop",
        "remaining_budget",
    ),
    [
        ("grammar_complete", "stop", None, 0),
        ("grammar_complete", "grammar_complete", (2,), 0),
        ("grammar_no_continuation", "grammar_complete", None, 1),
        ("grammar_complete", "output_budget_exhausted", None, 0),
        ("grammar_no_continuation", "output_budget_exhausted", None, 1),
        ("grammar_complete", "grammar_no_continuation", None, 1),
        ("grammar_no_continuation", "grammar_no_continuation", None, 0),
        ("grammar_no_continuation", "continuation_permitted", None, 1),
        ("iteration_bound_exhausted", "continuation_permitted", None, 0),
    ],
)
def test_direct_result_rejects_cross_inconsistent_policy(
    d51_disposition,
    request_disposition,
    matched_stop,
    remaining_budget,
):
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        GrammarMaskedSpeculativeRequestPolicyResult(
            _d51(d51_disposition),
            request_disposition,
            matched_stop,
            remaining_budget,
        )


def test_direct_result_requires_a_genuine_d51_result():
    with pytest.raises(TypeError):
        GrammarMaskedSpeculativeRequestPolicyResult(
            object(),
            "grammar_complete",
            None,
            0,
        )


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"vocab_size": True}, TypeError),
        ({"vocab_size": IntegerSubclass(8)}, TypeError),
        ({"vocab_size": 8.0}, TypeError),
        ({"vocab_size": "8"}, TypeError),
        ({"vocab_size": 0}, ValueError),
        ({"vocab_size": -1}, ValueError),
        ({"available_output_token_budget": True}, TypeError),
        ({"available_output_token_budget": IntegerSubclass(1)}, TypeError),
        ({"available_output_token_budget": 1.0}, TypeError),
        ({"available_output_token_budget": "1"}, TypeError),
        ({"available_output_token_budget": 0}, ValueError),
        ({"available_output_token_budget": -1}, ValueError),
        ({"matched_stop_is_eligible": 1}, TypeError),
        ({"matched_stop_is_eligible": None}, TypeError),
        ({"matched_stop_token_ids": []}, TypeError),
        ({"matched_stop_token_ids": "2"}, TypeError),
        ({"matched_stop_token_ids": b"2"}, TypeError),
        ({"matched_stop_token_ids": ()}, ValueError),
        ({"matched_stop_token_ids": (True,)}, TypeError),
        ({"matched_stop_token_ids": (IntegerSubclass(2),)}, TypeError),
        ({"matched_stop_token_ids": (2.0,)}, TypeError),
        ({"matched_stop_token_ids": (-1,)}, ValueError),
        ({"matched_stop_token_ids": (VOCAB_SIZE,)}, ValueError),
        ({"matched_stop_token_ids": None, "matched_stop_is_eligible": True}, ValueError),
    ],
)
def test_invalid_request_context_fails_before_d51_observation(
    monkeypatch,
    overrides,
    error_type,
):
    monkeypatch.setattr(
        request_policy_module,
        "_require_final_outcome_result",
        lambda *args, **kwargs: pytest.fail("D51 type was observed"),
    )
    monkeypatch.setattr(
        request_policy_module,
        "_read_attribute",
        lambda *args, **kwargs: pytest.fail("D51 evidence was observed"),
    )
    kwargs = {
        "vocab_size": VOCAB_SIZE,
        "matched_stop_token_ids": None,
        "matched_stop_is_eligible": False,
        "available_output_token_budget": 1,
    }
    kwargs.update(overrides)
    with pytest.raises(error_type):
        decide_grammar_masked_speculative_request_policy(object(), **kwargs)


def test_non_d51_input_is_a_type_error_before_field_access():
    class HostileInput:
        def __getattribute__(self, name):
            del name
            raise AssertionError("non-D51 fields must not be read")

    with pytest.raises(TypeError, match="type could not be determined"):
        _decide(HostileInput(), budget=1)


def test_d51_stored_fields_are_snapshotted_in_dataclass_order(monkeypatch):
    final = _d51("grammar_complete")
    real_read = request_policy_module._read_attribute
    reads = []
    stored_fields = {
        "output_token_ids",
        "final_iteration",
        "final_outcome",
        "disposition",
        "grammar_completion_token_id",
    }

    def record(value, name, *, label):
        if value is final and name in stored_fields:
            reads.append(name)
        return real_read(value, name, label=label)

    monkeypatch.setattr(request_policy_module, "_read_attribute", record)
    _decide(final)
    expected = [
        "output_token_ids",
        "final_iteration",
        "final_outcome",
        "disposition",
        "grammar_completion_token_id",
    ]
    assert reads
    assert len(reads) % len(expected) == 0
    for start in range(0, len(reads), len(expected)):
        assert reads[start : start + len(expected)] == expected


@pytest.mark.parametrize(
    ("d51_disposition", "stop_kind", "remaining", "expected"),
    [
        (d51_disposition, stop_kind, remaining, expected)
        for d51_disposition in (
            "grammar_complete",
            "grammar_no_continuation",
            "iteration_bound_exhausted",
        )
        for stop_kind in ("none", "eligible", "ineligible")
        for remaining in (0, 1)
        for expected in (
            _expected_request_disposition(
                d51_disposition,
                stop_kind,
                remaining,
            ),
        )
    ],
)
def test_complete_d51_stop_budget_precedence_matrix(
    d51_disposition,
    stop_kind,
    remaining,
    expected,
):
    final = _d51(d51_disposition)
    stop = None if stop_kind == "none" else tuple([2, 2])
    sampled_count = len(final.sampled_token_ids)
    result = _decide(
        final,
        stop=stop,
        eligible=stop_kind == "eligible",
        budget=sampled_count + remaining,
    )
    assert result.request_disposition == expected
    assert result.remaining_output_token_budget == remaining
    if stop_kind == "eligible":
        assert result.matched_stop_token_ids is stop
    else:
        assert result.matched_stop_token_ids is None
    assert result.final_outcome_result is final
    assert result.sampled_token_ids == final.sampled_token_ids
    assert result.visible_token_ids is final.output_token_ids


def test_eligible_cross_slice_stop_evidence_is_accepted_by_identity():
    final = _d51("iteration_bound_exhausted", output=(3,))
    stop = (5, 3)
    result = _decide(final, stop=stop, eligible=True)
    assert result.request_disposition == "stop"
    assert result.matched_stop_token_ids is stop


def test_empty_completion_output_consumes_one_hidden_eos_occurrence():
    final = _d51("grammar_complete", output=())
    result = _decide(final, budget=1)
    assert final.visible_token_ids == ()
    assert final.sampled_token_ids == (EOS_TOKEN_ID,)
    assert result.request_disposition == "grammar_complete"
    assert result.remaining_output_token_budget == 0


def test_nonempty_completion_counts_visible_tokens_and_hidden_eos():
    final = _d51("grammar_complete", output=(2, 4))
    exact = _decide(final, budget=3)
    extra = _decide(final, budget=4)
    assert exact.request_disposition == "grammar_complete"
    assert exact.remaining_output_token_budget == 0
    assert extra.remaining_output_token_budget == 1
    with pytest.raises(
        GrammarMaskedSpeculativeRequestPolicyInvariantError,
        match="exceeds",
    ):
        _decide(final, budget=len(final.visible_token_ids))


def test_stop_tokens_and_repeated_ids_count_as_sampled_occurrences():
    final = _d51("iteration_bound_exhausted", output=(3, 3, 3))
    stop = (3, 3)
    result = _decide(final, stop=stop, eligible=True, budget=3)
    assert result.request_disposition == "stop"
    assert result.remaining_output_token_budget == 0
    assert result.sampled_token_ids is final.output_token_ids
    assert result.visible_token_ids is final.output_token_ids
    assert result.visible_token_ids == (3, 3, 3)


def test_stop_equal_to_hidden_eos_wins_without_changing_token_views():
    final = _d51("grammar_complete", output=(EOS_TOKEN_ID, EOS_TOKEN_ID))
    stop = (EOS_TOKEN_ID,)
    result = _decide(final, stop=stop, eligible=True, budget=3)
    assert result.request_disposition == "stop"
    assert result.remaining_output_token_budget == 0
    assert result.visible_token_ids == (EOS_TOKEN_ID, EOS_TOKEN_ID)
    assert result.sampled_token_ids == (
        EOS_TOKEN_ID,
        EOS_TOKEN_ID,
        EOS_TOKEN_ID,
    )


@pytest.mark.parametrize("committed_match", [False, True])
def test_bound_exhaustion_with_budget_is_the_only_continuation_route(
    committed_match,
):
    state = HostileState()
    final = _d51(
        "iteration_bound_exhausted",
        state=state,
        committed_match=committed_match,
    )
    result = _decide(final)
    assert result.request_disposition == "continuation_permitted"
    assert result.request_is_terminal is False
    assert result.further_generation_permitted is True
    assert result.final_outcome_result.final_iteration.uncached_next_token_id == 3
    assert result.visible_token_ids[-1] == 3
    assert result.final_outcome_result.final_iteration.committed_state is state


def test_stop_and_budget_can_make_the_same_handoff_terminal_without_settlement():
    state = object()
    final = _d51("iteration_bound_exhausted", state=state)
    stopped = _decide(final, stop=(3,), eligible=True)
    exhausted = _decide(final, budget=len(final.sampled_token_ids))
    assert stopped.request_disposition == "stop"
    assert exhausted.request_disposition == "output_budget_exhausted"
    for result in (stopped, exhausted):
        iteration = result.final_outcome_result.final_iteration
        assert iteration is final.final_iteration
        assert iteration.committed_state is state
        assert iteration.uncached_next_token_id == final.output_token_ids[-1]


def test_d52_imports_no_d47_through_d51_operations():
    for operation_name in (
        "coordinate_grammar_masked_speculative_iteration",
        "classify_grammar_masked_speculative_outcome",
        "coordinate_grammar_masked_speculative_handoff",
        "coordinate_multi_iteration_grammar_masked_speculative_handoff",
        "decide_grammar_masked_speculative_final_outcome",
    ):
        assert not hasattr(request_policy_module, operation_name)


@pytest.mark.parametrize(
    ("handoff_count", "iteration_bound", "final_match", "expected"),
    [
        (0, 1, True, "grammar_complete"),
        (0, 1, False, "grammar_no_continuation"),
        (1, 1, True, "continuation_permitted"),
        (1, 3, True, "grammar_complete"),
        (2, 2, True, "continuation_permitted"),
    ],
)
def test_genuine_d50_d51_composition_preserves_runtime_and_evidence(
    handoff_count,
    iteration_bound,
    final_match,
    expected,
):
    coordinated = _coordinate_d50(
        handoff_count=handoff_count,
        iteration_bound=iteration_bound,
        final_match=final_match,
    )
    handoff, constraint, draft, target, *_ = coordinated
    final = decide_grammar_masked_speculative_final_outcome(
        handoff,
        vocab_size=VOCAB_SIZE,
        eos_token_id=EOS_TOKEN_ID,
    )
    output = final.output_token_ids
    iteration = final.final_iteration
    outcome = final.final_outcome
    state = iteration.committed_state
    advance_calls = tuple(constraint.advance_calls)
    release_calls = tuple(constraint.release_calls)
    draft_cache = draft.cached_token_ids
    target_cache = target.cached_token_ids

    result = _decide(final)

    assert result.request_disposition == expected
    assert result.final_outcome_result is final
    assert result.visible_token_ids is output
    assert result.final_outcome_result.final_iteration is iteration
    assert result.final_outcome_result.final_outcome is outcome
    assert result.final_outcome_result.final_iteration.committed_state is state
    assert tuple(constraint.advance_calls) == advance_calls
    assert tuple(constraint.release_calls) == release_calls
    assert draft.cached_token_ids == draft_cache
    assert target.cached_token_ids == target_cache
    _settle_d50(coordinated)


def test_opaque_none_and_hostile_states_are_retained_but_never_inspected():
    for state in (None, HostileState()):
        final = _d51("grammar_complete", state=state)
        result = _decide(final)
        assert result.final_outcome_result.final_iteration.committed_state is state


def test_unreadable_d51_field_preserves_original_cause():
    failure = RuntimeError("unreadable final outcome")

    class UnreadableFinalResult(GrammarMaskedSpeculativeFinalOutcomeResult):
        blocked = False

        def __getattribute__(self, name):
            if name == "final_outcome" and object.__getattribute__(self, "blocked"):
                raise failure
            return super().__getattribute__(name)

    valid = _d51("grammar_complete")
    final = UnreadableFinalResult(
        valid.output_token_ids,
        valid.final_iteration,
        valid.final_outcome,
        valid.disposition,
        valid.grammar_completion_token_id,
    )
    object.__setattr__(final, "blocked", True)
    with pytest.raises(
        GrammarMaskedSpeculativeRequestPolicyInvariantError,
        match="could not be read",
    ) as caught:
        _decide(final)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("malformed_output", [[], (True,), (-1,), (VOCAB_SIZE,)])
def test_malformed_d51_output_fails_closed(malformed_output):
    final = _mutate(
        _d51("grammar_complete"),
        "output_token_ids",
        malformed_output,
    )
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        _decide(final, budget=3)


def test_wrong_final_evidence_types_and_unsupported_disposition_fail_closed():
    for field_name in ("final_iteration", "final_outcome"):
        final = _mutate(_d51("grammar_complete"), field_name, object())
        with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
            _decide(final)
    final = _mutate(_d51("grammar_complete"), "disposition", "stop")
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        _decide(final)


def test_completion_token_and_d48_disagreement_fail_closed():
    final = _mutate(
        _d51("grammar_complete"),
        "grammar_completion_token_id",
        VOCAB_SIZE,
    )
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        _decide(final)
    final = _d51("grammar_complete")
    _mutate(final.final_outcome, "kind", "grammar_no_continuation")
    with pytest.raises(
        GrammarMaskedSpeculativeRequestPolicyInvariantError,
        match="disagrees",
    ):
        _decide(final)


def test_terminal_match_handoff_and_final_output_relationships_fail_closed():
    final = _d51("grammar_complete")
    _mutate(final.final_iteration, "committed_state_is_match", False)
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        _decide(final)

    final = _d51("iteration_bound_exhausted")
    _mutate(final, "output_token_ids", (2, 4))
    with pytest.raises(
        GrammarMaskedSpeculativeRequestPolicyInvariantError,
        match="suffix|handoff token",
    ):
        _decide(final)

    final = _d51("grammar_no_continuation")
    _mutate(final.final_iteration, "uncached_next_token_id", 3)
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        _decide(final)


def test_misreported_d51_token_views_fail_closed():
    class MisreportedVisibleResult(GrammarMaskedSpeculativeFinalOutcomeResult):
        @property
        def visible_token_ids(self):
            return tuple(list(self.output_token_ids))

    class MisreportedSampledResult(GrammarMaskedSpeculativeFinalOutcomeResult):
        @property
        def sampled_token_ids(self):
            return tuple(list(self.output_token_ids))

    valid = _d51("grammar_no_continuation")
    for result_type in (MisreportedVisibleResult, MisreportedSampledResult):
        final = result_type(
            valid.output_token_ids,
            valid.final_iteration,
            valid.final_outcome,
            valid.disposition,
            valid.grammar_completion_token_id,
        )
        with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
            _decide(final)


def test_sampled_budget_overrun_does_not_mutate_or_truncate_d51():
    final = _d51("grammar_complete", output=(2, 4))
    output = final.output_token_ids
    iteration = final.final_iteration
    state = iteration.committed_state
    with pytest.raises(GrammarMaskedSpeculativeRequestPolicyInvariantError):
        _decide(final, budget=2)
    assert final.output_token_ids is output
    assert final.visible_token_ids is output
    assert final.sampled_token_ids == (2, 4, EOS_TOKEN_ID)
    assert final.final_iteration is iteration
    assert final.final_iteration.committed_state is state


def test_result_construction_failure_leaves_genuine_state_caller_owned(monkeypatch):
    coordinated = _coordinate_d50(
        handoff_count=0,
        iteration_bound=1,
        final_match=True,
    )
    handoff, constraint, *_ = coordinated
    final = decide_grammar_masked_speculative_final_outcome(
        handoff,
        vocab_size=VOCAB_SIZE,
        eos_token_id=EOS_TOKEN_ID,
    )
    failure = RuntimeError("construction failed")
    release_calls = tuple(constraint.release_calls)

    def fail_construction(**kwargs):
        del kwargs
        raise failure

    monkeypatch.setattr(
        request_policy_module,
        "GrammarMaskedSpeculativeRequestPolicyResult",
        fail_construction,
    )
    with pytest.raises(RuntimeError) as caught:
        _decide(final)
    assert caught.value is failure
    assert tuple(constraint.release_calls) == release_calls
    assert constraint.active_state_count == 1
    _settle_d50(coordinated)


def test_result_retains_exact_d51_wrapper_and_no_independent_runtime_fields():
    class WeakFinalResult(GrammarMaskedSpeculativeFinalOutcomeResult):
        pass

    valid = _d51("iteration_bound_exhausted", state=object())
    final = WeakFinalResult(
        valid.output_token_ids,
        valid.final_iteration,
        valid.final_outcome,
        valid.disposition,
        valid.grammar_completion_token_id,
    )
    reference = weakref.ref(final)
    result = _decide(final, stop=(5,), eligible=False)
    del final
    gc.collect()
    assert reference() is result.final_outcome_result
    assert result.matched_stop_token_ids is None
    assert [field.name for field in fields(result)] == [
        "final_outcome_result",
        "request_disposition",
        "matched_stop_token_ids",
        "remaining_output_token_budget",
    ]


def test_one_thousand_alternating_decisions_are_deterministic_and_stateless():
    finals = (
        _d51("grammar_complete", state=None),
        _d51("grammar_no_continuation", state=HostileState()),
        _d51("iteration_bound_exhausted", committed_match=True),
    )
    expected = (
        "grammar_complete",
        "grammar_no_continuation",
        "continuation_permitted",
    )
    for position in range(1000):
        index = position % len(finals)
        result = _decide(finals[index])
        assert result.request_disposition == expected[index]
        assert result.final_outcome_result is finals[index]
    mutable_module_values = [
        value
        for name, value in vars(request_policy_module).items()
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
            GrammarMaskedSpeculativeFinalOutcomeResult,
            GrammarMaskedSpeculativeIterationResult,
            GrammarMaskedSpeculativeOutcomeResult,
            decide_grammar_masked_speculative_request_policy,
        )

        selection = GrammarMaskedSelectionResult((), True, None)
        iteration = GrammarMaskedSpeculativeIterationResult(
            (), 0, None, 1, 2, None, selection, None, None, None, True
        )
        final = GrammarMaskedSpeculativeFinalOutcomeResult(
            (),
            iteration,
            GrammarMaskedSpeculativeOutcomeResult('grammar_complete'),
            'grammar_complete',
            6,
        )
        result = decide_grammar_masked_speculative_request_policy(
            final,
            vocab_size=8,
            matched_stop_token_ids=None,
            matched_stop_is_eligible=False,
            available_output_token_budget=1,
        )
        assert result.request_disposition == 'grammar_complete'
        assert result.remaining_output_token_budget == 0
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
        print('isolated-d52-ok')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "isolated-d52-ok"
