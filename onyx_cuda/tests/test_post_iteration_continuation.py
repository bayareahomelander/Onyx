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

import onyx_cuda.continuation as continuation_module
from onyx_cuda import (
    MatchReplaceAcceptanceResult,
    PostIterationContinuationInvariantError,
    PostIterationContinuationResult,
    TemperatureTopPSelection,
    create_reference_sampler,
    decide_match_replace_acceptance,
    decide_post_iteration_continuation,
    select_highest_logit,
)
from onyx_cuda.testing import FakeAutoregressiveBackend
from onyx_cuda.verification import BatchedTargetVerificationResult


PROPOSAL = (2, 3, 4, 5)
VOCAB_SIZE = 16


class TupleSubclass(tuple):
    pass


class HostileRow:
    """Opaque row that fails if continuation code tries to interpret it."""

    __slots__ = ("label", "__weakref__")

    def __init__(self, label):
        self.label = label

    def __len__(self):
        raise AssertionError("continuation inspected row length")

    def __iter__(self):
        raise AssertionError("continuation iterated a row")

    def __getitem__(self, key):
        raise AssertionError(f"continuation indexed a row with {key!r}")

    def __bool__(self):
        raise AssertionError("continuation tested row truthiness")

    def __eq__(self, other):
        raise AssertionError(f"continuation compared a row with {other!r}")

    def __copy__(self):
        raise AssertionError("continuation copied a row")

    def __deepcopy__(self, memo):
        raise AssertionError(f"continuation deep-copied a row with {memo!r}")


class RecordingSelector:
    def __init__(self, token_id=0):
        self.token_id = token_id
        self.calls = []

    def __call__(self, row):
        self.calls.append(row)
        return self.token_id


class FailingSelector:
    def __init__(self):
        self.calls = []

    def __call__(self, row):
        self.calls.append(row)
        raise AssertionError("selector must not be called")


class NonCallableSelector:
    def __init__(self):
        self.calls = []


def _rows(proposal=PROPOSAL):
    return tuple(HostileRow(f"row-{position}") for position in range(len(proposal) + 1))


def _full_acceptance(proposal=PROPOSAL):
    return MatchReplaceAcceptanceResult(proposal, len(proposal), None)


def _mismatch(position, replacement_token_id=9, proposal=PROPOSAL):
    return MatchReplaceAcceptanceResult(proposal, position, replacement_token_id)


def _tampered_acceptance(**changes):
    result = _full_acceptance()
    for name, value in changes.items():
        object.__setattr__(result, name, value)
    return result


def _fake_state(backend):
    return (
        backend.cache_length,
        backend.cached_token_ids,
        backend._next_row,
        backend._epoch,
        backend.active_checkpoint_count,
        backend._next_checkpoint_id,
        tuple(backend._cache_checkpoints.items()),
    )


def _logit_row(selected_token_id, vocab_size=VOCAB_SIZE):
    return tuple(
        20.0 if token_id == selected_token_id else float(-token_id)
        for token_id in range(vocab_size)
    )


def _scripted_rows(selected_token_ids):
    return tuple(_logit_row(token_id) for token_id in selected_token_ids)


def test_public_contract_module_exports_signature_and_error_hierarchy():
    import onyx_cuda

    current_module = importlib.import_module("onyx_cuda.continuation")
    symbol_names = (
        "PostIterationContinuationError",
        "PostIterationContinuationInvariantError",
        "PostIterationContinuationResult",
        "decide_post_iteration_continuation",
    )
    for symbol_name in symbol_names:
        symbol = getattr(current_module, symbol_name)
        assert symbol.__module__ == "onyx_cuda.continuation"
        assert getattr(onyx_cuda, symbol_name) is symbol
        assert symbol_name in onyx_cuda.__all__

    assert current_module.__all__ == [
        "PostIterationContinuationError",
        "PostIterationContinuationInvariantError",
        "PostIterationContinuationResult",
        "decide_post_iteration_continuation",
    ]
    assert issubclass(current_module.PostIterationContinuationError, RuntimeError)
    assert issubclass(
        current_module.PostIterationContinuationInvariantError,
        current_module.PostIterationContinuationError,
    )

    parameters = inspect.signature(current_module.decide_post_iteration_continuation).parameters
    assert tuple(parameters) == (
        "proposal_token_ids",
        "target_logit_rows",
        "acceptance_result",
        "vocab_size",
        "select_token",
    )
    assert parameters["vocab_size"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["select_token"].kind is inspect.Parameter.KEYWORD_ONLY


def test_result_is_frozen_slotted_minimal_equal_and_retains_only_token_output():
    output = (2, 3, 9)
    result = PostIterationContinuationResult(output, 9)

    assert result.output_token_ids is output
    assert result == PostIterationContinuationResult((2, 3, 9), 9)
    assert [field.name for field in fields(result)] == [
        "output_token_ids",
        "uncached_next_token_id",
    ]
    assert not hasattr(result, "__dict__")
    for forbidden_attribute in (
        "proposal_token_ids",
        "acceptance_result",
        "accepted_count",
        "replacement_token_id",
        "bonus_token_id",
        "next_current_token_id",
        "target_logit_rows",
        "select_token",
        "backend",
        "cache_length",
        "checkpoint",
        "metrics",
    ):
        assert not hasattr(result, forbidden_attribute)
    with pytest.raises(FrozenInstanceError):
        result.uncached_next_token_id = 8


@pytest.mark.parametrize(
    ("output_token_ids", "error", "message"),
    [
        ([1], TypeError, "output_token_ids must be a tuple"),
        (TupleSubclass((1,)), TypeError, "output_token_ids must be a tuple"),
        ((), ValueError, "output_token_ids cannot be empty"),
        ((True,), TypeError, "output token at position 0 must be an integer"),
        ((1.5,), TypeError, "output token at position 0 must be an integer"),
        (("1",), TypeError, "output token at position 0 must be an integer"),
        ((-1,), ValueError, "output token at position 0 cannot be negative"),
    ],
)
def test_result_rejects_invalid_output_token_ids(output_token_ids, error, message):
    with pytest.raises(error, match=message):
        PostIterationContinuationResult(output_token_ids, 1)


@pytest.mark.parametrize(
    ("uncached_next_token_id", "error", "message"),
    [
        (True, TypeError, "uncached_next_token_id must be an integer"),
        (1.5, TypeError, "uncached_next_token_id must be an integer"),
        ("1", TypeError, "uncached_next_token_id must be an integer"),
        (-1, ValueError, "uncached_next_token_id cannot be negative"),
    ],
)
def test_result_rejects_invalid_uncached_next_token(
    uncached_next_token_id,
    error,
    message,
):
    with pytest.raises(error, match=message):
        PostIterationContinuationResult((1,), uncached_next_token_id)


def test_result_requires_the_uncached_token_to_be_the_final_output_token():
    with pytest.raises(
        PostIterationContinuationInvariantError,
        match="must equal the final output token",
    ):
        PostIterationContinuationResult((1, 2), 1)


def test_direct_result_construction_has_no_vocabulary_upper_bound():
    large_token_id = 10**12
    assert PostIterationContinuationResult((large_token_id,), large_token_id) == (
        PostIterationContinuationResult((large_token_id,), large_token_id)
    )


def test_one_token_mismatch_reuses_d33_output_without_selecting_any_row():
    proposal = (2,)
    target_rows = _rows(proposal)
    acceptance = _mismatch(0, 7, proposal)
    selector = FailingSelector()

    result = decide_post_iteration_continuation(
        proposal,
        target_rows,
        acceptance,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )

    assert selector.calls == []
    assert result.output_token_ids == acceptance.output_token_ids == (7,)
    assert result.uncached_next_token_id == 7


def test_one_token_full_acceptance_selects_only_the_final_row_by_identity():
    proposal = (2,)
    target_rows = _rows(proposal)
    selector = RecordingSelector(7)

    result = decide_post_iteration_continuation(
        proposal,
        target_rows,
        _full_acceptance(proposal),
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )

    assert len(selector.calls) == 1
    assert selector.calls[0] is target_rows[1]
    assert result == PostIterationContinuationResult((2, 7), 7)


@pytest.mark.parametrize("mismatch_position", range(len(PROPOSAL)))
def test_every_multi_token_mismatch_reuses_the_exact_accepted_prefix_and_replacement(
    mismatch_position,
):
    target_rows = _rows()
    replacement = 8 + mismatch_position
    acceptance = _mismatch(mismatch_position, replacement)
    selector = FailingSelector()

    result = decide_post_iteration_continuation(
        PROPOSAL,
        target_rows,
        acceptance,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )

    expected = PROPOSAL[:mismatch_position] + (replacement,)
    assert selector.calls == []
    assert result.output_token_ids == acceptance.output_token_ids == expected
    assert result.uncached_next_token_id == replacement


def test_multi_token_full_acceptance_selects_only_the_final_row():
    target_rows = _rows()
    selector = RecordingSelector(12)

    result = decide_post_iteration_continuation(
        PROPOSAL,
        target_rows,
        _full_acceptance(),
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )

    assert len(selector.calls) == 1
    assert selector.calls[0] is target_rows[-1]
    assert all(selector.calls[0] is not row for row in target_rows[:-1])
    assert result.output_token_ids == PROPOSAL + (12,)
    assert result.uncached_next_token_id == 12


def test_bonus_equal_to_the_final_proposal_token_is_valid():
    target_rows = _rows()
    selector = RecordingSelector(PROPOSAL[-1])

    result = decide_post_iteration_continuation(
        PROPOSAL,
        target_rows,
        _full_acceptance(),
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )

    assert result.output_token_ids == PROPOSAL + (PROPOSAL[-1],)
    assert result.uncached_next_token_id == PROPOSAL[-1]


@pytest.mark.parametrize(
    ("proposal_token_ids", "vocab_size", "error", "message"),
    [
        ([2, 3], VOCAB_SIZE, TypeError, "proposal_token_ids must be a tuple"),
        (
            TupleSubclass((2, 3)),
            VOCAB_SIZE,
            TypeError,
            "proposal_token_ids must be a tuple",
        ),
        ((), VOCAB_SIZE, ValueError, "proposal_token_ids cannot be empty"),
        ((2, True), VOCAB_SIZE, TypeError, "proposal token at position 1 must be an integer"),
        ((2, 3.0), VOCAB_SIZE, TypeError, "proposal token at position 1 must be an integer"),
        ((2, "3"), VOCAB_SIZE, TypeError, "proposal token at position 1 must be an integer"),
        ((2, -1), VOCAB_SIZE, ValueError, "proposal token at position 1 cannot be negative"),
        (
            (2, VOCAB_SIZE),
            VOCAB_SIZE,
            ValueError,
            r"proposal token at position 1 must be within \[0, 16\)",
        ),
        ((2, 3), True, TypeError, "vocab_size must be an integer"),
        ((2, 3), 16.0, TypeError, "vocab_size must be an integer"),
        ((2, 3), "16", TypeError, "vocab_size must be an integer"),
        ((2, 3), 0, ValueError, "vocab_size must be greater than zero"),
        ((2, 3), -1, ValueError, "vocab_size must be greater than zero"),
    ],
)
def test_primitive_proposal_and_vocabulary_failures_happen_before_selection(
    proposal_token_ids,
    vocab_size,
    error,
    message,
):
    selector = RecordingSelector()
    acceptance = MatchReplaceAcceptanceResult((2, 3), 2, None)

    with pytest.raises(error, match=message):
        decide_post_iteration_continuation(
            proposal_token_ids,
            (object(), object(), object()),
            acceptance,
            vocab_size=vocab_size,
            select_token=selector,
        )

    assert selector.calls == []


@pytest.mark.parametrize(
    ("target_logit_rows", "error", "message"),
    [
        ([object()] * 3, TypeError, "target_logit_rows must be a tuple"),
        (
            TupleSubclass((object(), object(), object())),
            TypeError,
            "target_logit_rows must be a tuple",
        ),
        (
            (),
            PostIterationContinuationInvariantError,
            "contains 0 rows; expected 3",
        ),
        (
            (object(),),
            PostIterationContinuationInvariantError,
            "contains 1 rows; expected 3",
        ),
        (
            (object(), object()),
            PostIterationContinuationInvariantError,
            "contains 2 rows; expected 3",
        ),
        (
            (object(),) * 4,
            PostIterationContinuationInvariantError,
            "contains 4 rows; expected 3",
        ),
        (
            (object(),) * 8,
            PostIterationContinuationInvariantError,
            "contains 8 rows; expected 3",
        ),
    ],
)
def test_row_container_and_count_failures_happen_before_selection(
    target_logit_rows,
    error,
    message,
):
    proposal = (2, 3)
    selector = RecordingSelector()

    with pytest.raises(error, match=message):
        decide_post_iteration_continuation(
            proposal,
            target_logit_rows,
            _full_acceptance(proposal),
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    assert selector.calls == []


@pytest.mark.parametrize("acceptance_result", [None, object(), (PROPOSAL, 4, None)])
def test_non_d33_acceptance_evidence_is_rejected_before_selection(acceptance_result):
    selector = RecordingSelector()

    with pytest.raises(
        TypeError,
        match="acceptance_result must be a MatchReplaceAcceptanceResult",
    ):
        decide_post_iteration_continuation(
            PROPOSAL,
            _rows(),
            acceptance_result,
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    assert selector.calls == []


def _acceptance_evidence_cases():
    different_proposal = (2, 3, 4, 6)
    different_acceptance = MatchReplaceAcceptanceResult(
        different_proposal,
        len(different_proposal),
        None,
    )
    return (
        (
            _tampered_acceptance(proposal_token_ids=list(PROPOSAL)),
            "proposal_token_ids must be a tuple",
        ),
        (
            _tampered_acceptance(proposal_token_ids=TupleSubclass(PROPOSAL)),
            "proposal_token_ids must be a tuple",
        ),
        (
            _tampered_acceptance(proposal_token_ids=(True, *PROPOSAL[1:])),
            "proposal token at position 0 must be an integer",
        ),
        (
            _tampered_acceptance(proposal_token_ids=(2.0, *PROPOSAL[1:])),
            "proposal token at position 0 must be an integer",
        ),
        (different_acceptance, "proposal does not match"),
        (_tampered_acceptance(accepted_count=True), "accepted_count must be an integer"),
        (_tampered_acceptance(accepted_count=1.0), "accepted_count must be an integer"),
        (_tampered_acceptance(accepted_count=-1), "accepted_count must be within"),
        (_tampered_acceptance(accepted_count=len(PROPOSAL) + 1), "accepted_count must be within"),
        (_tampered_acceptance(accepted_count=1), "mismatch evidence must contain"),
        (_tampered_acceptance(replacement_token_id=9), "fully accepted evidence cannot contain"),
        (
            _tampered_acceptance(accepted_count=1, replacement_token_id=True),
            "replacement_token_id must be an integer",
        ),
        (
            _tampered_acceptance(accepted_count=1, replacement_token_id=1.5),
            "replacement_token_id must be an integer",
        ),
        (
            _tampered_acceptance(accepted_count=1, replacement_token_id="9"),
            "replacement_token_id must be an integer",
        ),
        (
            _tampered_acceptance(accepted_count=1, replacement_token_id=-1),
            r"replacement_token_id must be within \[0, 16\)",
        ),
        (
            _tampered_acceptance(accepted_count=1, replacement_token_id=VOCAB_SIZE),
            r"replacement_token_id must be within \[0, 16\)",
        ),
        (
            _tampered_acceptance(
                accepted_count=1,
                replacement_token_id=PROPOSAL[1],
            ),
            "must differ from the rejected proposal token",
        ),
    )


@pytest.mark.parametrize(("acceptance_result", "message"), _acceptance_evidence_cases())
def test_malformed_or_mixed_acceptance_evidence_fails_before_selection(
    acceptance_result,
    message,
):
    selector = RecordingSelector()

    with pytest.raises(PostIterationContinuationInvariantError, match=message):
        decide_post_iteration_continuation(
            PROPOSAL,
            _rows(),
            acceptance_result,
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    assert selector.calls == []


@pytest.mark.parametrize(
    "acceptance_result",
    [_mismatch(1), _full_acceptance()],
    ids=["mismatch", "full-acceptance"],
)
def test_noncallable_selector_is_rejected_on_both_outcomes(acceptance_result):
    selector = NonCallableSelector()

    with pytest.raises(TypeError, match="select_token must be callable"):
        decide_post_iteration_continuation(
            PROPOSAL,
            _rows(),
            acceptance_result,
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    assert selector.calls == []


def test_selection_policy_cannot_replace_the_borrowed_selector_callable():
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=7)

    with pytest.raises(TypeError, match="select_token must be callable"):
        decide_post_iteration_continuation(
            PROPOSAL,
            _rows(),
            _mismatch(0),
            vocab_size=VOCAB_SIZE,
            select_token=policy,
        )


def test_full_acceptance_selector_exception_propagates_unchanged_after_one_call():
    target_rows = _rows()
    calls = []
    failure = LookupError("final-row selection failed")

    def selector(row):
        calls.append(row)
        raise failure

    with pytest.raises(LookupError, match=str(failure)) as raised:
        decide_post_iteration_continuation(
            PROPOSAL,
            target_rows,
            _full_acceptance(),
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    assert raised.value is failure
    assert len(calls) == 1
    assert calls[0] is target_rows[-1]


@pytest.mark.parametrize(
    ("invalid_token_id", "error", "message"),
    [
        (True, TypeError, "selected bonus token must be an integer"),
        (1.5, TypeError, "selected bonus token must be an integer"),
        ("1", TypeError, "selected bonus token must be an integer"),
        (-1, ValueError, "selected bonus token cannot be negative"),
        (
            VOCAB_SIZE,
            ValueError,
            r"selected bonus token must be within \[0, 16\)",
        ),
    ],
)
def test_invalid_bonus_return_fails_after_exactly_one_final_row_call(
    invalid_token_id,
    error,
    message,
):
    target_rows = _rows()
    selector = RecordingSelector(invalid_token_id)

    with pytest.raises(error, match=message):
        decide_post_iteration_continuation(
            PROPOSAL,
            target_rows,
            _full_acceptance(),
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    assert len(selector.calls) == 1
    assert selector.calls[0] is target_rows[-1]


def test_result_construction_failure_propagates_after_the_required_final_row_call(
    monkeypatch,
):
    target_rows = _rows()
    selector = RecordingSelector(12)
    failure = RuntimeError("injected continuation result failure")

    def fail_result_construction(**kwargs):
        raise failure

    monkeypatch.setattr(
        continuation_module,
        "PostIterationContinuationResult",
        fail_result_construction,
    )

    with pytest.raises(RuntimeError, match=str(failure)) as raised:
        decide_post_iteration_continuation(
            PROPOSAL,
            target_rows,
            _full_acceptance(),
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    assert raised.value is failure
    assert len(selector.calls) == 1
    assert selector.calls[0] is target_rows[-1]


@pytest.mark.parametrize("first_outcome", ["exception", "invalid"])
def test_failed_full_acceptance_selection_does_not_rewind_caller_state(first_outcome):
    target_rows = _rows()
    failure = RuntimeError("consumed failure")
    outcomes = iter((failure if first_outcome == "exception" else True, 11))
    calls = []

    def selector(row):
        calls.append(row)
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    expected_error = RuntimeError if first_outcome == "exception" else TypeError
    with pytest.raises(expected_error):
        decide_post_iteration_continuation(
            PROPOSAL,
            target_rows,
            _full_acceptance(),
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )

    probe_row = object()
    assert selector(probe_row) == 11
    assert calls == [target_rows[-1], probe_row]


def test_one_selector_session_receives_d33_decision_rows_then_d37_final_row_in_order():
    proposal = (2, 3, 4)
    target_rows = _rows(proposal)
    selected_tokens = iter((*proposal, 11))
    calls = []

    def selector(row):
        calls.append(row)
        return next(selected_tokens)

    acceptance = decide_match_replace_acceptance(
        proposal,
        target_rows,
        select_token=selector,
    )
    result = decide_post_iteration_continuation(
        proposal,
        target_rows,
        acceptance,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )

    assert acceptance.fully_accepted
    assert len(calls) == len(target_rows)
    assert all(actual is expected for actual, expected in zip(calls, target_rows))
    assert result == PostIterationContinuationResult(proposal + (11,), 11)


@pytest.mark.parametrize("mismatch_position", [0, 1, 2])
def test_d37_adds_no_selector_consumption_after_a_d33_mismatch(mismatch_position):
    proposal = (2, 3, 4)
    target_rows = _rows(proposal)
    selected = (*proposal[:mismatch_position], 8 + mismatch_position)
    calls = []

    def selector(row):
        calls.append(row)
        return selected[len(calls) - 1]

    acceptance = decide_match_replace_acceptance(
        proposal,
        target_rows,
        select_token=selector,
    )
    calls_after_acceptance = tuple(calls)
    result = decide_post_iteration_continuation(
        proposal,
        target_rows,
        acceptance,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )

    assert acceptance.accepted_count == mismatch_position
    assert tuple(calls) == calls_after_acceptance
    assert len(calls) == mismatch_position + 1
    assert all(actual is expected for actual, expected in zip(calls, target_rows))
    assert result.output_token_ids == acceptance.output_token_ids


def _seeded_full_composition(policy, logits, proposal):
    selector = create_reference_sampler(policy)
    rows = (logits,) * (len(proposal) + 1)
    acceptance = decide_match_replace_acceptance(
        proposal,
        rows,
        select_token=selector,
    )
    return decide_post_iteration_continuation(
        proposal,
        rows,
        acceptance,
        vocab_size=len(logits),
        select_token=selector,
    )


def test_fresh_same_seed_sessions_replay_the_complete_d33_plus_d37_outcome():
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=37)
    logits = (0.0, 0.0, 0.0, 0.0)
    control = create_reference_sampler(policy)
    expected_draws = tuple(control(logits) for _ in range(4))
    proposal = expected_draws[:3]

    first = _seeded_full_composition(policy, logits, proposal)
    second = _seeded_full_composition(policy, logits, proposal)

    expected = PostIterationContinuationResult(expected_draws, expected_draws[-1])
    assert first == second == expected


@pytest.mark.parametrize("mismatch_position", [0, 1, 2])
def test_seeded_session_after_mismatch_still_has_the_next_caller_owned_draw(
    mismatch_position,
):
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=91)
    logits = (0.0, 0.0, 0.0, 0.0)
    control = create_reference_sampler(policy)
    expected_draws = tuple(control(logits) for _ in range(mismatch_position + 2))
    rejected_token = (expected_draws[mismatch_position] + 1) % len(logits)
    proposal = (*expected_draws[:mismatch_position], rejected_token)
    rows = (logits,) * (len(proposal) + 1)
    selector = create_reference_sampler(policy)

    acceptance = decide_match_replace_acceptance(
        proposal,
        rows,
        select_token=selector,
    )
    result = decide_post_iteration_continuation(
        proposal,
        rows,
        acceptance,
        vocab_size=len(logits),
        select_token=selector,
    )
    next_draw = selector(logits)

    assert acceptance.accepted_count == mismatch_position
    assert result.uncached_next_token_id == expected_draws[mismatch_position]
    assert next_draw == expected_draws[mismatch_position + 1]


@pytest.mark.parametrize(
    ("target_decisions", "accepted_count", "replacement_token_id", "expected_output"),
    [
        ((8, 9, 10, 11), 0, 8, (8,)),
        ((2, 8, 10, 11), 1, 8, (2, 8)),
        ((2, 3, 8, 11), 2, 8, (2, 3, 8)),
        ((2, 3, 4, 8), 3, None, (2, 3, 4, 8)),
    ],
)
def test_real_d30_d33_evidence_composes_without_backend_or_checkpoint_mutation(
    target_decisions,
    accepted_count,
    replacement_token_id,
    expected_output,
):
    proposal = (2, 3, 4)
    target_backend = FakeAutoregressiveBackend(_scripted_rows((0, *target_decisions)))
    target_backend.prefill((6, 7))
    root = target_backend.create_cache_checkpoint()
    verification = target_backend.verify_proposal(1, proposal)
    assert isinstance(verification, BatchedTargetVerificationResult)
    exact_rows = verification.logit_rows
    selector_calls = []

    def selector(row):
        selector_calls.append(row)
        return select_highest_logit(row)

    acceptance = decide_match_replace_acceptance(
        proposal,
        exact_rows,
        select_token=selector,
    )
    target_before = _fake_state(target_backend)

    try:
        result = decide_post_iteration_continuation(
            proposal,
            exact_rows,
            acceptance,
            vocab_size=target_backend.vocab_size,
            select_token=selector,
        )

        assert acceptance.accepted_count == accepted_count
        assert acceptance.replacement_token_id == replacement_token_id
        assert result.output_token_ids == expected_output
        assert result.uncached_next_token_id == expected_output[-1]
        assert _fake_state(target_backend) == target_before
        assert target_backend.cached_token_ids == (6, 7, 1, *proposal)
        expected_calls = (
            len(proposal) + 1 if replacement_token_id is None else accepted_count + 1
        )
        assert len(selector_calls) == expected_calls
        assert all(
            actual is expected
            for actual, expected in zip(selector_calls, exact_rows[:expected_calls])
        )
        if replacement_token_id is None:
            assert selector_calls[-1] is exact_rows[-1]
        else:
            assert all(row is not exact_rows[-1] for row in selector_calls)
        assert [field.name for field in fields(result)] == [
            "output_token_ids",
            "uncached_next_token_id",
        ]
        assert not hasattr(result, "acceptance_result")
        assert not hasattr(result, "target_logit_rows")
    finally:
        target_backend.release_cache_checkpoint(root)


class WeakRow:
    __slots__ = ("selected_token_id", "__weakref__")

    def __init__(self, selected_token_id):
        self.selected_token_id = selected_token_id


class WeakSelector:
    __slots__ = ("calls", "__weakref__")

    def __init__(self):
        self.calls = 0

    def __call__(self, row):
        self.calls += 1
        return row.selected_token_id


@pytest.mark.parametrize("fully_accepted", [False, True])
def test_retained_result_does_not_retain_rows_selector_or_acceptance_evidence(
    fully_accepted,
):
    proposal = (2,)
    decision_row = WeakRow(7)
    final_row = WeakRow(8)
    target_rows = (decision_row, final_row)
    selector = WeakSelector()
    acceptance = _full_acceptance(proposal) if fully_accepted else _mismatch(0, 7, proposal)
    decision_ref = weakref.ref(decision_row)
    final_ref = weakref.ref(final_row)
    selector_ref = weakref.ref(selector)

    result = decide_post_iteration_continuation(
        proposal,
        target_rows,
        acceptance,
        vocab_size=VOCAB_SIZE,
        select_token=selector,
    )
    expected = (2, 8) if fully_accepted else (7,)
    assert result.output_token_ids == expected

    del target_rows
    del decision_row
    del final_row
    del selector
    del acceptance
    gc.collect()

    assert decision_ref() is None
    assert final_ref() is None
    assert selector_ref() is None
    assert result == PostIterationContinuationResult(expected, expected[-1])


def test_one_thousand_alternating_decisions_have_exact_calls_and_no_module_state_growth():
    selector = RecordingSelector(11)
    mismatch = _mismatch(2, 9)
    full = _full_acceptance()

    for iteration in range(1_000):
        acceptance = mismatch if iteration % 2 == 0 else full
        result = decide_post_iteration_continuation(
            PROPOSAL,
            tuple(object() for _ in range(len(PROPOSAL) + 1)),
            acceptance,
            vocab_size=VOCAB_SIZE,
            select_token=selector,
        )
        expected = (2, 3, 9) if iteration % 2 == 0 else (*PROPOSAL, 11)
        assert result.output_token_ids == expected

    assert len(selector.calls) == 500
    assert not {
        name
        for name, value in vars(continuation_module).items()
        if not name.startswith("__") and isinstance(value, (dict, list, set))
    }


def test_isolated_source_import_and_both_outcomes_load_no_optional_runtime():
    package_root = Path(__file__).resolve().parents[1]
    source_root = package_root / "src"
    script = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {str(source_root)!r})
        import onyx_cuda

        mismatch_evidence = onyx_cuda.MatchReplaceAcceptanceResult((1, 2), 1, 7)
        mismatch = onyx_cuda.decide_post_iteration_continuation(
            (1, 2),
            (object(), object(), object()),
            mismatch_evidence,
            vocab_size=10,
            select_token=lambda row: (_ for _ in ()).throw(
                AssertionError("mismatch selected a row")
            ),
        )
        assert mismatch.output_token_ids == (1, 7)
        assert mismatch.uncached_next_token_id == 7

        final_row = object()
        full_evidence = onyx_cuda.MatchReplaceAcceptanceResult((1, 2), 2, None)
        full = onyx_cuda.decide_post_iteration_continuation(
            (1, 2),
            (object(), object(), final_row),
            full_evidence,
            vocab_size=10,
            select_token=lambda row: 8 if row is final_row else -1,
        )
        assert full.output_token_ids == (1, 2, 8)
        assert full.uncached_next_token_id == 8

        forbidden = (
            "onyx",
            "mlx",
            "torch",
            "transformers",
            "tokenizers",
            "huggingface_hub",
            "bitsandbytes",
            "accelerate",
            "onnxruntime",
            "psutil",
        )
        loaded = tuple(sys.modules)
        assert "onyx_cuda._grammar_native" not in loaded
        assert not any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for module_name in loaded
            for prefix in forbidden
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
