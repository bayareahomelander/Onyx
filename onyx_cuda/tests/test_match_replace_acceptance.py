import gc
import inspect
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import onyx_cuda.acceptance as acceptance_module
from onyx_cuda import (
    MatchReplaceAcceptanceError,
    MatchReplaceAcceptanceInvariantError,
    MatchReplaceAcceptanceResult,
    TemperatureTopPSelection,
    create_reference_sampler,
    decide_match_replace_acceptance,
    generate_draft_proposal,
    select_highest_logit,
)
from onyx_cuda.testing import FakeAutoregressiveBackend


PROPOSAL = (10, 11, 12, 13)


class TupleSubclass(tuple):
    pass


class HostileRow:
    """Opaque row that fails if acceptance tries to interpret it directly."""

    __slots__ = ("label", "__weakref__")

    def __init__(self, label):
        self.label = label

    def __len__(self):
        raise AssertionError("acceptance inspected row length")

    def __iter__(self):
        raise AssertionError("acceptance iterated a row")

    def __getitem__(self, key):
        raise AssertionError(f"acceptance indexed a row with {key!r}")

    def __bool__(self):
        raise AssertionError("acceptance tested row truthiness")

    def __eq__(self, other):
        raise AssertionError(f"acceptance compared a row with {other!r}")

    def __copy__(self):
        raise AssertionError("acceptance copied a row")

    def __deepcopy__(self, memo):
        raise AssertionError(f"acceptance deep-copied a row with {memo!r}")


class ExpectedPrefixSelector:
    def __init__(self, expected_rows, selected_token_ids):
        self.expected_rows = tuple(expected_rows)
        self.selected_token_ids = tuple(selected_token_ids)
        self.calls = []

    def __call__(self, row):
        position = len(self.calls)
        if position >= len(self.expected_rows):
            raise AssertionError("selector received an unused decision or final row")
        if row is not self.expected_rows[position]:
            raise AssertionError(f"selector received the wrong row at position {position}")
        self.calls.append(row)
        return self.selected_token_ids[position]


class RecordingSelector:
    def __init__(self, token_id=0):
        self.token_id = token_id
        self.calls = []

    def __call__(self, row):
        self.calls.append(row)
        return self.token_id


def _fake_state(backend):
    return (
        backend.cache_length,
        backend.cached_token_ids,
        backend._next_row,
        backend.active_checkpoint_count,
        backend._next_checkpoint_id,
        backend._epoch,
        tuple(backend._cache_checkpoints.items()),
    )


def _logit_row(selected_token_id, vocab_size=8):
    return tuple(
        20.0 if token_id == selected_token_id else float(-token_id)
        for token_id in range(vocab_size)
    )


def _scripted_rows(selected_token_ids):
    return tuple(_logit_row(token_id) for token_id in selected_token_ids)


def test_public_contract_module_ownership_exports_signature_and_error_hierarchy():
    import onyx_cuda

    symbols = (
        MatchReplaceAcceptanceError,
        MatchReplaceAcceptanceInvariantError,
        MatchReplaceAcceptanceResult,
        decide_match_replace_acceptance,
    )
    for symbol in symbols:
        assert symbol.__module__ == "onyx_cuda.acceptance"
        assert getattr(onyx_cuda, symbol.__name__) is symbol
        assert symbol.__name__ in onyx_cuda.__all__

    assert issubclass(MatchReplaceAcceptanceError, RuntimeError)
    assert issubclass(MatchReplaceAcceptanceInvariantError, MatchReplaceAcceptanceError)

    parameters = inspect.signature(decide_match_replace_acceptance).parameters
    assert tuple(parameters) == ("proposal_token_ids", "target_logit_rows", "select_token")
    assert parameters["select_token"].kind is inspect.Parameter.KEYWORD_ONLY


def test_result_is_frozen_slotted_minimal_equal_and_retains_only_the_proposal():
    proposal = (2, 3, 4)
    result = MatchReplaceAcceptanceResult(
        proposal_token_ids=proposal,
        accepted_count=1,
        replacement_token_id=7,
    )

    assert result.proposal_token_ids is proposal
    assert result == MatchReplaceAcceptanceResult(proposal, 1, 7)
    assert [field.name for field in fields(result)] == [
        "proposal_token_ids",
        "accepted_count",
        "replacement_token_id",
    ]
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "target_logit_rows")
    assert not hasattr(result, "select_token")
    with pytest.raises(FrozenInstanceError):
        result.accepted_count = 2


@pytest.mark.parametrize(
    (
        "accepted_count",
        "replacement_token_id",
        "accepted_token_ids",
        "rejected_proposal_token_id",
        "output_token_ids",
    ),
    [
        (0, 20, (), 10, (20,)),
        (1, 21, (10,), 11, (10, 21)),
        (2, 22, (10, 11), 12, (10, 11, 22)),
        (3, 23, (10, 11, 12), 13, (10, 11, 12, 23)),
        (4, None, PROPOSAL, None, PROPOSAL),
    ],
)
def test_result_properties_cover_every_mismatch_position_and_full_acceptance(
    accepted_count,
    replacement_token_id,
    accepted_token_ids,
    rejected_proposal_token_id,
    output_token_ids,
):
    result = MatchReplaceAcceptanceResult(
        proposal_token_ids=PROPOSAL,
        accepted_count=accepted_count,
        replacement_token_id=replacement_token_id,
    )

    assert result.fully_accepted is (accepted_count == len(PROPOSAL))
    assert result.accepted_token_ids == accepted_token_ids
    assert result.rejected_proposal_token_id == rejected_proposal_token_id
    assert result.output_token_ids == output_token_ids
    assert result.output_token_ids
    assert result.next_current_token_id == output_token_ids[-1]


@pytest.mark.parametrize(
    ("proposal_token_ids", "error", "message"),
    [
        ([1, 2], TypeError, "proposal_token_ids must be a tuple"),
        (TupleSubclass((1, 2)), TypeError, "proposal_token_ids must be a tuple"),
        ((), ValueError, "proposal_token_ids cannot be empty"),
        ((1, True), TypeError, "proposal token at position 1 must be an integer"),
        ((1, 2.0), TypeError, "proposal token at position 1 must be an integer"),
        ((1, "2"), TypeError, "proposal token at position 1 must be an integer"),
        ((1, -2), ValueError, "proposal token at position 1 cannot be negative"),
    ],
)
def test_result_rejects_malformed_proposals(proposal_token_ids, error, message):
    with pytest.raises(error, match=message):
        MatchReplaceAcceptanceResult(proposal_token_ids, 0, 3)


@pytest.mark.parametrize(
    ("accepted_count", "error", "message"),
    [
        (True, TypeError, "accepted_count must be an integer"),
        (1.0, TypeError, "accepted_count must be an integer"),
        ("1", TypeError, "accepted_count must be an integer"),
        (-1, MatchReplaceAcceptanceInvariantError, "accepted_count must be within"),
        (3, MatchReplaceAcceptanceInvariantError, "accepted_count must be within"),
    ],
)
def test_result_rejects_invalid_accepted_counts(accepted_count, error, message):
    with pytest.raises(error, match=message):
        MatchReplaceAcceptanceResult((1, 2), accepted_count, 3)


@pytest.mark.parametrize(
    ("accepted_count", "replacement_token_id", "error", "message"),
    [
        (
            2,
            3,
            MatchReplaceAcceptanceInvariantError,
            "fully accepted result cannot contain a replacement token",
        ),
        (
            1,
            None,
            MatchReplaceAcceptanceInvariantError,
            "partially accepted result must contain a replacement token",
        ),
        (1, True, TypeError, "replacement_token_id must be an integer"),
        (1, 3.0, TypeError, "replacement_token_id must be an integer"),
        (1, "3", TypeError, "replacement_token_id must be an integer"),
        (1, -3, ValueError, "replacement_token_id cannot be negative"),
        (
            1,
            2,
            MatchReplaceAcceptanceInvariantError,
            "replacement_token_id must differ from the rejected proposal token",
        ),
    ],
)
def test_result_rejects_invalid_replacement_relationships(
    accepted_count,
    replacement_token_id,
    error,
    message,
):
    with pytest.raises(error, match=message):
        MatchReplaceAcceptanceResult((1, 2), accepted_count, replacement_token_id)


def test_result_allows_replacement_without_a_vocabulary_upper_bound():
    result = MatchReplaceAcceptanceResult((1,), 0, 10**12)
    assert result.output_token_ids == (10**12,)


@pytest.mark.parametrize(
    ("proposal_token_id", "selected_token_id", "accepted_count", "replacement_token_id"),
    [
        (4, 4, 1, None),
        (4, 7, 0, 7),
    ],
)
def test_one_token_decision_never_selects_the_required_final_row(
    proposal_token_id,
    selected_token_id,
    accepted_count,
    replacement_token_id,
):
    decision_row = HostileRow("decision")
    final_row = HostileRow("final")
    selector = ExpectedPrefixSelector((decision_row,), (selected_token_id,))

    result = decide_match_replace_acceptance(
        (proposal_token_id,),
        (decision_row, final_row),
        select_token=selector,
    )

    assert selector.calls == [decision_row]
    assert result.accepted_count == accepted_count
    assert result.replacement_token_id == replacement_token_id
    assert result.output_token_ids == (
        (proposal_token_id,) if replacement_token_id is None else (replacement_token_id,)
    )


@pytest.mark.parametrize("mismatch_position", [0, 1, 2, 3, None])
def test_multi_token_decision_uses_exact_row_prefix_and_stops_at_first_mismatch(
    mismatch_position,
):
    decision_rows = tuple(HostileRow(f"decision-{position}") for position in range(len(PROPOSAL)))
    final_row = HostileRow("final")
    selected_token_ids = list(PROPOSAL)
    if mismatch_position is not None:
        selected_token_ids[mismatch_position] = 90 + mismatch_position
    call_count = len(PROPOSAL) if mismatch_position is None else mismatch_position + 1
    selector = ExpectedPrefixSelector(
        decision_rows[:call_count],
        selected_token_ids[:call_count],
    )

    result = decide_match_replace_acceptance(
        PROPOSAL,
        (*decision_rows, final_row),
        select_token=selector,
    )

    accepted_count = len(PROPOSAL) if mismatch_position is None else mismatch_position
    replacement_token_id = (
        None if mismatch_position is None else selected_token_ids[mismatch_position]
    )
    expected_output = (
        PROPOSAL
        if replacement_token_id is None
        else PROPOSAL[:accepted_count] + (replacement_token_id,)
    )
    assert len(selector.calls) == call_count
    assert all(
        actual is expected
        for actual, expected in zip(selector.calls, decision_rows[:call_count])
    )
    assert all(row is not final_row for row in selector.calls)
    assert result.accepted_count == accepted_count
    assert result.accepted_token_ids == PROPOSAL[:accepted_count]
    assert result.rejected_proposal_token_id == (
        None if mismatch_position is None else PROPOSAL[mismatch_position]
    )
    assert result.replacement_token_id == replacement_token_id
    assert result.output_token_ids == expected_output
    assert result.next_current_token_id == expected_output[-1]


@pytest.mark.parametrize(
    ("proposal_token_ids", "target_logit_rows", "error", "message"),
    [
        ([1, 2], (object(), object(), object()), TypeError, "proposal_token_ids must be a tuple"),
        (
            TupleSubclass((1, 2)),
            (object(), object(), object()),
            TypeError,
            "proposal_token_ids must be a tuple",
        ),
        ((), (object(),), ValueError, "proposal_token_ids cannot be empty"),
        (
            (1, True),
            (object(), object(), object()),
            TypeError,
            "proposal token at position 1 must be an integer",
        ),
        (
            (1, -2),
            (object(), object(), object()),
            ValueError,
            "proposal token at position 1 cannot be negative",
        ),
        ((1, 2), [object()] * 3, TypeError, "target_logit_rows must be a tuple"),
        (
            (1, 2),
            TupleSubclass((object(), object(), object())),
            TypeError,
            "target_logit_rows must be a tuple",
        ),
        (
            (1, 2),
            (),
            MatchReplaceAcceptanceInvariantError,
            "contains 0 rows; expected 3 for proposal length 2",
        ),
        (
            (1, 2),
            (object(),),
            MatchReplaceAcceptanceInvariantError,
            "contains 1 rows; expected 3 for proposal length 2",
        ),
        (
            (1, 2),
            (object(), object()),
            MatchReplaceAcceptanceInvariantError,
            "contains 2 rows; expected 3 for proposal length 2",
        ),
        (
            (1, 2),
            (object(),) * 4,
            MatchReplaceAcceptanceInvariantError,
            "contains 4 rows; expected 3 for proposal length 2",
        ),
        (
            (1, 2),
            (object(),) * 8,
            MatchReplaceAcceptanceInvariantError,
            "contains 8 rows; expected 3 for proposal length 2",
        ),
    ],
)
def test_structural_input_failures_happen_before_selector_state_changes(
    proposal_token_ids,
    target_logit_rows,
    error,
    message,
):
    selector = RecordingSelector()

    with pytest.raises(error, match=message):
        decide_match_replace_acceptance(
            proposal_token_ids,
            target_logit_rows,
            select_token=selector,
        )

    assert selector.calls == []


@pytest.mark.parametrize("selector", [None, 1, "selector"])
def test_noncallable_selector_is_rejected_after_structural_validation_without_selection(selector):
    with pytest.raises(TypeError, match="select_token must be callable"):
        decide_match_replace_acceptance(
            (1,),
            (object(), object()),
            select_token=selector,
        )


@pytest.mark.parametrize("failure_position", [0, 1, 2])
@pytest.mark.parametrize(
    ("invalid_token_id", "error", "message"),
    [
        (True, TypeError, "must be an integer"),
        (1.5, TypeError, "must be an integer"),
        ("2", TypeError, "must be an integer"),
        (-1, ValueError, "cannot be negative"),
    ],
)
def test_invalid_selector_returns_stop_at_the_exact_position(
    failure_position,
    invalid_token_id,
    error,
    message,
):
    proposal = (3, 4, 5)
    rows = tuple(HostileRow(f"decision-{position}") for position in range(3))
    final_row = HostileRow("final")
    selected = (*proposal[:failure_position], invalid_token_id)
    selector = ExpectedPrefixSelector(rows[: failure_position + 1], selected)

    with pytest.raises(error, match=rf"proposal position {failure_position}.*{message}"):
        decide_match_replace_acceptance(
            proposal,
            (*rows, final_row),
            select_token=selector,
        )

    assert len(selector.calls) == failure_position + 1
    assert all(row is not final_row for row in selector.calls)


@pytest.mark.parametrize("failure_position", [0, 1, 2])
def test_selector_exception_propagates_unchanged_without_rewinding_or_later_calls(
    failure_position,
):
    proposal = (3, 4, 5)
    rows = tuple(HostileRow(f"decision-{position}") for position in range(3))
    final_row = HostileRow("final")
    calls = []
    failure = RuntimeError(f"selector failure at {failure_position}")

    def selector(row):
        position = len(calls)
        calls.append(row)
        if position == failure_position:
            raise failure
        return proposal[position]

    with pytest.raises(RuntimeError, match=str(failure)) as raised:
        decide_match_replace_acceptance(
            proposal,
            (*rows, final_row),
            select_token=selector,
        )

    assert raised.value is failure
    assert len(calls) == failure_position + 1
    assert all(actual is expected for actual, expected in zip(calls, rows))
    assert all(row is not final_row for row in calls)


def test_selector_owned_row_fault_is_not_translated_to_an_acceptance_error():
    row = HostileRow("faulting")
    final_row = HostileRow("final")
    failure = LookupError("row interpretation failed")

    def selector(actual):
        assert actual is row
        raise failure

    with pytest.raises(LookupError, match="row interpretation failed") as raised:
        decide_match_replace_acceptance((1,), (row, final_row), select_token=selector)

    assert raised.value is failure


@pytest.mark.parametrize(
    ("selected_token_ids", "expected_calls"),
    [
        ((9,), 1),
        (PROPOSAL, len(PROPOSAL)),
    ],
)
def test_result_construction_failure_propagates_after_required_selector_calls(
    monkeypatch,
    selected_token_ids,
    expected_calls,
):
    proposal = PROPOSAL if expected_calls > 1 else (1,)
    rows = tuple(HostileRow(f"decision-{position}") for position in range(len(proposal)))
    final_row = HostileRow("final")
    selector = ExpectedPrefixSelector(rows[:expected_calls], selected_token_ids)
    failure = RuntimeError("injected result construction failure")

    def fail_result_construction(**kwargs):
        raise failure

    monkeypatch.setattr(
        acceptance_module,
        "MatchReplaceAcceptanceResult",
        fail_result_construction,
    )

    with pytest.raises(RuntimeError, match=str(failure)) as raised:
        decide_match_replace_acceptance(
            proposal,
            (*rows, final_row),
            select_token=selector,
        )

    assert raised.value is failure
    assert len(selector.calls) == expected_calls
    assert all(row is not final_row for row in selector.calls)


def test_one_stateful_selector_session_continues_across_decisions():
    values = iter((20, 21))
    calls = []

    def selector(row):
        calls.append(row)
        return next(values)

    first_rows = (HostileRow("first"), HostileRow("first-final"))
    second_rows = (HostileRow("second"), HostileRow("second-final"))
    first = decide_match_replace_acceptance((1,), first_rows, select_token=selector)
    second = decide_match_replace_acceptance((1,), second_rows, select_token=selector)

    assert first.output_token_ids == (20,)
    assert second.output_token_ids == (21,)
    assert calls == [first_rows[0], second_rows[0]]


def test_fresh_same_seed_reference_sessions_replay_the_same_acceptance_outcome():
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=37)
    logits = (0.0, 0.0, 0.0, 0.0)
    control = create_reference_sampler(policy)
    expected_draws = tuple(control(logits) for _ in range(3))
    proposal = (*expected_draws[:2], 99)
    rows = (logits, logits, logits, logits)

    first = decide_match_replace_acceptance(
        proposal,
        rows,
        select_token=create_reference_sampler(policy),
    )
    second = decide_match_replace_acceptance(
        proposal,
        rows,
        select_token=create_reference_sampler(policy),
    )

    assert first == second == MatchReplaceAcceptanceResult(proposal, 2, expected_draws[2])


@pytest.mark.parametrize("mismatch_position", [0, 1, 2])
def test_reused_seeded_session_continues_after_exact_mismatch_consumption(mismatch_position):
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=91)
    logits = (0.0, 0.0, 0.0, 0.0)
    control = create_reference_sampler(policy)
    expected_draws = tuple(control(logits) for _ in range(mismatch_position + 2))
    proposal = (*expected_draws[:mismatch_position], 99)
    sampler = create_reference_sampler(policy)

    first = decide_match_replace_acceptance(
        proposal,
        (logits,) * (len(proposal) + 1),
        select_token=sampler,
    )
    second = decide_match_replace_acceptance(
        (99,),
        (logits, logits),
        select_token=sampler,
    )

    assert first.accepted_count == mismatch_position
    assert first.replacement_token_id == expected_draws[mismatch_position]
    assert second.replacement_token_id == expected_draws[mismatch_position + 1]


@pytest.mark.parametrize(
    ("target_decisions", "accepted_count", "replacement_token_id"),
    [
        ((1, 5, 6, 7), 1, 5),
        ((1, 2, 3, 7), 3, None),
    ],
)
def test_actual_d30_d32_evidence_composes_without_backend_or_checkpoint_mutation(
    target_decisions,
    accepted_count,
    replacement_token_id,
):
    draft_backend = FakeAutoregressiveBackend(_scripted_rows((0, 1, 2, 3, 4)))
    target_backend = FakeAutoregressiveBackend(_scripted_rows((0, *target_decisions)))
    draft_backend.prefill((6, 7))
    proposal = generate_draft_proposal(
        draft_backend,
        0,
        proposal_length=3,
        select_token=select_highest_logit,
    )
    target_backend.prefill((6, 7))
    verification = target_backend.verify_proposal(0, proposal.proposal_token_ids)
    draft_before = _fake_state(draft_backend)
    target_before = _fake_state(target_backend)

    try:
        result = decide_match_replace_acceptance(
            proposal.proposal_token_ids,
            verification.logit_rows,
            select_token=select_highest_logit,
        )

        assert result.accepted_count == accepted_count
        assert result.replacement_token_id == replacement_token_id
        assert _fake_state(draft_backend) == draft_before
        assert _fake_state(target_backend) == target_before
        assert draft_backend.active_checkpoint_count == len(proposal.rollback_checkpoints)
        assert all(
            checkpoint.allocation_id in draft_backend._cache_checkpoints
            for checkpoint in proposal.rollback_checkpoints
        )
        if not result.fully_accepted:
            rejection_handle = proposal.rollback_checkpoints[result.accepted_count]
            assert rejection_handle.cache_length == proposal.initial_cache_length + 1 + accepted_count
    finally:
        for checkpoint in proposal.rollback_checkpoints:
            draft_backend.release_cache_checkpoint(checkpoint)

    assert draft_backend.active_checkpoint_count == 0


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


def test_retained_result_does_not_retain_rows_or_selector():
    decision_row = WeakRow(7)
    final_row = WeakRow(8)
    selector = WeakSelector()
    target_rows = (decision_row, final_row)
    decision_ref = weakref.ref(decision_row)
    final_ref = weakref.ref(final_row)
    selector_ref = weakref.ref(selector)

    result = decide_match_replace_acceptance((1,), target_rows, select_token=selector)
    assert result.output_token_ids == (7,)

    del target_rows
    del decision_row
    del final_row
    del selector
    gc.collect()

    assert decision_ref() is None
    assert final_ref() is None
    assert selector_ref() is None
    assert result == MatchReplaceAcceptanceResult((1,), 0, 7)


class DecisionRow:
    __slots__ = ("selected_token_id", "is_final")

    def __init__(self, selected_token_id, *, is_final=False):
        self.selected_token_id = selected_token_id
        self.is_final = is_final


class CountingSelector:
    def __init__(self):
        self.calls = 0

    def __call__(self, row):
        self.calls += 1
        if row.is_final:
            raise AssertionError("bounded reuse selected the final row")
        return row.selected_token_id


def test_one_thousand_bounded_decisions_have_exact_calls_and_no_module_state_growth():
    selector = CountingSelector()
    outcomes = (
        ((90, 11, 12, 13), 0, (90,)),
        ((10, 11, 92, 13), 2, (10, 11, 92)),
        ((10, 11, 12, 93), 3, (10, 11, 12, 93)),
        (PROPOSAL, 4, PROPOSAL),
    )
    expected_calls = 0

    for iteration in range(1_000):
        selected, accepted_count, output = outcomes[iteration % len(outcomes)]
        rows = tuple(DecisionRow(token_id) for token_id in selected)
        final_row = DecisionRow(0, is_final=True)
        result = decide_match_replace_acceptance(
            PROPOSAL,
            (*rows, final_row),
            select_token=selector,
        )
        expected_calls += len(PROPOSAL) if accepted_count >= 3 else accepted_count + 1

        assert result.accepted_count == accepted_count
        assert result.output_token_ids == output

    assert selector.calls == expected_calls == 3_000
    assert not {
        name
        for name, value in vars(acceptance_module).items()
        if not name.startswith("__") and isinstance(value, (dict, list, set))
    }


def test_isolated_source_import_and_decisions_load_no_optional_runtime():
    package_root = Path(__file__).resolve().parents[1]
    source_root = package_root / "src"
    script = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {str(source_root)!r})
        import onyx_cuda

        mismatch = onyx_cuda.decide_match_replace_acceptance(
            (1, 2),
            (1, 9, 99),
            select_token=lambda row: row,
        )
        assert mismatch.accepted_token_ids == (1,)
        assert mismatch.rejected_proposal_token_id == 2
        assert mismatch.output_token_ids == (1, 9)
        assert mismatch.next_current_token_id == 9

        full = onyx_cuda.decide_match_replace_acceptance(
            (1, 2),
            (1, 2, 99),
            select_token=lambda row: row,
        )
        assert full.fully_accepted
        assert full.output_token_ids == (1, 2)
        assert full.next_current_token_id == 2

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
