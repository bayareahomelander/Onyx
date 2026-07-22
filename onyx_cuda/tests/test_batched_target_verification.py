from dataclasses import FrozenInstanceError, fields

import pytest

from onyx_cuda import (
    AutoregressiveBackend,
    BackendStateError,
    BatchedTargetVerificationBackend,
    BatchedTargetVerificationResult,
    CacheCheckpointStateError,
    ModelStep,
    TorchCUDATargetBackend,
)
from onyx_cuda.testing import FakeAutoregressiveBackend, ScriptExhaustedError


VOCAB_SIZE = 5
SCRIPT = tuple(
    tuple(float(row_number * 10 + token_id) for token_id in range(VOCAB_SIZE))
    for row_number in range(16)
)


class MinimumBackend:
    model_id = "minimum"
    vocab_size = VOCAB_SIZE
    cache_length = 0

    def prefill(self, prompt_token_ids, /):
        return ModelStep(logits=SCRIPT[0], cache_length=len(prompt_token_ids))

    def decode(self, token_id, /):
        return ModelStep(logits=SCRIPT[1], cache_length=self.cache_length + 1)

    def reset(self):
        return None


class TupleSubclass(tuple):
    pass


class OnePassProposal:
    def __init__(self, token_ids):
        self._token_ids = token_ids
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("proposal was materialized more than once")
        return iter(self._token_ids)


class ShortStagingBackend(FakeAutoregressiveBackend):
    def _stage_verification_rows(self, row_count):
        return super()._stage_verification_rows(row_count)[:-1]


class NonTupleStagingBackend(FakeAutoregressiveBackend):
    def _stage_verification_rows(self, row_count):
        return list(super()._stage_verification_rows(row_count))


class FailingStagingBackend(FakeAutoregressiveBackend):
    def _stage_verification_rows(self, row_count):
        raise RuntimeError("injected staging failure")


class FailingResultBackend(FakeAutoregressiveBackend):
    def _build_verification_result(self, *, logit_rows, cache_length):
        raise RuntimeError("injected result failure")


class MalformedResultBackend(FakeAutoregressiveBackend):
    def _build_verification_result(self, *, logit_rows, cache_length):
        return BatchedTargetVerificationResult(logit_rows=(), cache_length=cache_length)


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


def test_capability_is_runtime_checkable_and_separate_from_minimum_backend():
    fake = FakeAutoregressiveBackend(SCRIPT)
    minimum = MinimumBackend()
    uninitialized_production_backend = object.__new__(TorchCUDATargetBackend)

    assert isinstance(fake, AutoregressiveBackend)
    assert isinstance(fake, BatchedTargetVerificationBackend)
    assert isinstance(minimum, AutoregressiveBackend)
    assert not isinstance(minimum, BatchedTargetVerificationBackend)
    assert isinstance(uninitialized_production_backend, BatchedTargetVerificationBackend)
    assert hasattr(TorchCUDATargetBackend, "verify_proposal")


def test_result_is_frozen_slotted_minimal_and_retains_native_rows():
    first_row = object()
    final_row = object()
    result = BatchedTargetVerificationResult(
        logit_rows=(first_row, final_row),
        cache_length=7,
    )

    assert result.logit_rows[0] is first_row
    assert result.logit_rows[1] is final_row
    assert result == BatchedTargetVerificationResult(
        logit_rows=(first_row, final_row),
        cache_length=7,
    )
    assert [field.name for field in fields(result)] == ["logit_rows", "cache_length"]
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.cache_length = 8


@pytest.mark.parametrize(
    ("logit_rows", "error", "message"),
    [
        ([], TypeError, "logit_rows must be a tuple"),
        (TupleSubclass(((1.0,),)), TypeError, "logit_rows must be a tuple"),
        ((), ValueError, "logit_rows cannot be empty"),
    ],
)
def test_result_requires_an_exact_nonempty_tuple(logit_rows, error, message):
    with pytest.raises(error, match=message):
        BatchedTargetVerificationResult(logit_rows=logit_rows, cache_length=0)


@pytest.mark.parametrize(
    ("cache_length", "error", "message"),
    [
        (True, TypeError, "cache_length must be an integer"),
        (1.0, TypeError, "cache_length must be an integer"),
        (-1, ValueError, "cache_length cannot be negative"),
    ],
)
def test_result_rejects_invalid_cache_lengths(cache_length, error, message):
    with pytest.raises(error, match=message):
        BatchedTargetVerificationResult(logit_rows=((1.0,),), cache_length=cache_length)


def test_public_contract_module_ownership_and_exports():
    import onyx_cuda

    assert BatchedTargetVerificationResult.__module__ == "onyx_cuda.verification"
    assert BatchedTargetVerificationBackend.__module__ == "onyx_cuda.verification"
    assert onyx_cuda.BatchedTargetVerificationResult is BatchedTargetVerificationResult
    assert onyx_cuda.BatchedTargetVerificationBackend is BatchedTargetVerificationBackend
    assert "BatchedTargetVerificationResult" in onyx_cuda.__all__
    assert "BatchedTargetVerificationBackend" in onyx_cuda.__all__


def test_one_token_proposal_returns_two_rows_and_appends_the_exact_suffix():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))

    result = backend.verify_proposal(2, (3,))

    assert result == BatchedTargetVerificationResult(
        logit_rows=(SCRIPT[1], SCRIPT[2]),
        cache_length=4,
    )
    assert result.logit_rows[0] is backend._scripted_logits[1]
    assert result.logit_rows[1] is backend._scripted_logits[2]
    assert backend.cache_length == result.cache_length == 4
    assert backend.cached_token_ids == (0, 1, 2, 3)
    assert backend._next_row == 3


def test_multi_token_row_alignment_exposes_the_unused_final_row_in_native_order():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))
    proposal = (3, 1, 4)

    result = backend.verify_proposal(2, proposal)

    # Rows 0..n-1 judge proposal tokens 0..n-1. Row n is the post-proposal row;
    # D30 exposes it without applying acceptance or bonus-token policy.
    decision_rows = result.logit_rows[: len(proposal)]
    post_proposal_row = result.logit_rows[len(proposal)]
    assert decision_rows == SCRIPT[1:4]
    assert post_proposal_row == SCRIPT[4]
    assert result.logit_rows == SCRIPT[1:5]
    assert len(result.logit_rows) == len(proposal) + 1
    assert all(len(row) == backend.vocab_size for row in result.logit_rows)
    assert backend.cached_token_ids == (0, 1, 2, *proposal)
    assert backend.cache_length == result.cache_length == 6


def test_batch_is_observationally_equivalent_to_ordered_decode_calls():
    batched = FakeAutoregressiveBackend(SCRIPT)
    sequential = FakeAutoregressiveBackend(SCRIPT)
    batched.prefill((0, 1))
    sequential.prefill((0, 1))
    input_suffix = (4, 3, 2, 1)

    result = batched.verify_proposal(input_suffix[0], input_suffix[1:])
    sequential_rows = tuple(sequential.decode(token_id).logits for token_id in input_suffix)

    assert result.logit_rows == sequential_rows
    assert result.cache_length == sequential.cache_length == batched.cache_length
    assert batched.cached_token_ids == sequential.cached_token_ids == (0, 1, *input_suffix)
    assert batched._next_row == sequential._next_row


def test_proposal_is_materialized_once_and_rows_are_returned_without_conversion():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    proposal = OnePassProposal((2, 3))

    result = backend.verify_proposal(1, proposal)

    assert proposal.iterations == 1
    assert result.logit_rows == backend._scripted_logits[1:4]
    assert all(
        returned is scripted
        for returned, scripted in zip(result.logit_rows, backend._scripted_logits[1:4])
    )


def test_consecutive_batches_have_no_hidden_proposal_length_limit():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))

    first = backend.verify_proposal(1, (2,))
    second = backend.verify_proposal(3, (4, 0, 1, 2, 3))

    assert first.logit_rows == SCRIPT[1:3]
    assert second.logit_rows == SCRIPT[3:9]
    assert backend.cached_token_ids == (0, 1, 2, 3, 4, 0, 1, 2, 3)
    assert backend.cache_length == second.cache_length == 9
    assert backend._next_row == 9


def test_verification_requires_prefill_without_mutation():
    backend = FakeAutoregressiveBackend(SCRIPT)
    before = _fake_state(backend)

    with pytest.raises(BackendStateError, match="prefill"):
        backend.verify_proposal(0, (1,))

    assert _fake_state(backend) == before


@pytest.mark.parametrize(
    ("current_token", "proposal", "error", "message"),
    [
        (True, (1,), TypeError, "current token must be an integer"),
        ("1", (1,), TypeError, "current token must be an integer"),
        (-1, (1,), ValueError, "current token -1"),
        (VOCAB_SIZE, (1,), ValueError, f"current token {VOCAB_SIZE}"),
        (1, None, TypeError, "proposal_token_ids must be a sequence"),
        (1, (), ValueError, "proposal_token_ids cannot be empty"),
        (1, (0, True), TypeError, "proposal token at position 1 must be an integer"),
        (1, (0, "2"), TypeError, "proposal token at position 1 must be an integer"),
        (1, (0, -1), ValueError, "proposal token at position 1 -1"),
        (1, (0, VOCAB_SIZE), ValueError, f"proposal token at position 1 {VOCAB_SIZE}"),
    ],
)
def test_invalid_inputs_preserve_active_sequence_and_checkpoint_registry(
    current_token,
    proposal,
    error,
    message,
):
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    checkpoint = backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises(error, match=message):
        backend.verify_proposal(current_token, proposal)

    assert _fake_state(backend) == before
    backend.rollback_cache(checkpoint)
    assert _fake_state(backend) == before


@pytest.mark.parametrize("available_rows", (0, 1))
def test_zero_or_partial_script_capacity_fails_atomically(available_rows):
    backend = FakeAutoregressiveBackend(SCRIPT[: available_rows + 1])
    backend.prefill((0,))
    checkpoint = backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises(ScriptExhaustedError, match="scripted logits"):
        backend.verify_proposal(1, (2,))

    assert _fake_state(backend) == before
    backend.rollback_cache(checkpoint)
    assert _fake_state(backend) == before


@pytest.mark.parametrize(
    ("backend_type", "error", "message"),
    [
        (ShortStagingBackend, ValueError, "returned 1 rows; expected 2"),
        (NonTupleStagingBackend, TypeError, "rows must be a tuple"),
    ],
)
def test_malformed_staged_row_container_or_count_is_atomic(backend_type, error, message):
    backend = backend_type(SCRIPT)
    backend.prefill((0,))
    backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises(error, match=message):
        backend.verify_proposal(1, (2,))

    assert _fake_state(backend) == before


@pytest.mark.parametrize(
    ("malformed_row", "error", "message"),
    [
        ([1.0] * VOCAB_SIZE, TypeError, "row at position 1 must be a tuple"),
        ((1.0,), ValueError, "position 1 has vocabulary size 1"),
    ],
)
def test_malformed_staged_row_container_or_width_is_atomic(
    malformed_row,
    error,
    message,
):
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    backend.create_cache_checkpoint()
    rows = list(backend._scripted_logits)
    rows[2] = malformed_row
    backend._scripted_logits = tuple(rows)
    before = _fake_state(backend)

    with pytest.raises(error, match=message):
        backend.verify_proposal(1, (2,))

    assert _fake_state(backend) == before


@pytest.mark.parametrize(
    ("backend_type", "message"),
    [
        (FailingStagingBackend, "injected staging failure"),
        (FailingResultBackend, "injected result failure"),
    ],
)
def test_injected_staging_or_result_failure_is_atomic(backend_type, message):
    backend = backend_type(SCRIPT)
    backend.prefill((0,))
    backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises(RuntimeError, match=message):
        backend.verify_proposal(1, (2,))

    assert _fake_state(backend) == before


def test_result_constructor_rejection_is_atomic():
    backend = MalformedResultBackend(SCRIPT)
    backend.prefill((0,))
    backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises(ValueError, match="logit_rows cannot be empty"):
        backend.verify_proposal(1, (2,))

    assert _fake_state(backend) == before


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("_cached_token_ids", [0], "exact cached token prefix"),
        ("_cache_length", 2, "exact cached token prefix"),
        ("_next_row", True, "scripted-logits position"),
        ("_next_row", len(SCRIPT) + 1, "scripted-logits position"),
    ],
)
def test_corrupt_active_state_is_rejected_before_mutation(attribute, value, message):
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    setattr(backend, attribute, value)
    before = _fake_state(backend)

    with pytest.raises(BackendStateError, match=message):
        backend.verify_proposal(1, (2,))

    assert _fake_state(backend) == before


def test_checkpoint_rollback_restores_batch_and_exact_result_replay():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))
    checkpoint = backend.create_cache_checkpoint()
    snapshot = backend._cache_checkpoints[checkpoint.allocation_id]

    first = backend.verify_proposal(2, (3, 4))
    post_batch = backend.create_cache_checkpoint()

    assert backend._next_row - snapshot.next_row == (
        backend.cache_length - checkpoint.cache_length
    )
    assert backend.active_checkpoint_count == 2

    backend.rollback_cache(checkpoint)

    assert backend.cached_token_ids == snapshot.cached_token_ids == (0, 1)
    assert backend.cache_length == checkpoint.cache_length == 2
    assert backend._next_row == snapshot.next_row == 1
    assert backend.active_checkpoint_count == 1
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(post_batch)

    replay = backend.verify_proposal(2, (3, 4))
    assert replay == first
    assert replay.logit_rows == SCRIPT[1:4]


def test_alternative_suffix_after_rollback_reuses_rows_but_retains_exact_tokens():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    checkpoint = backend.create_cache_checkpoint()
    original = backend.verify_proposal(1, (2, 3))

    backend.rollback_cache(checkpoint)
    alternative = backend.verify_proposal(4, (3, 2))

    assert alternative == original
    assert alternative.logit_rows == SCRIPT[1:4]
    assert backend.cached_token_ids == (0, 4, 3, 2)


def test_nested_and_same_position_handles_retain_d28_lifetimes_around_a_batch():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    earliest = backend.create_cache_checkpoint()
    backend.decode(1)
    before_batch = backend.create_cache_checkpoint()
    same_position_peer = backend.create_cache_checkpoint()
    backend.verify_proposal(2, (3, 4))
    after_batch = backend.create_cache_checkpoint()

    backend.rollback_cache(before_batch)

    assert backend.cached_token_ids == (0, 1)
    assert backend.active_checkpoint_count == 3
    backend.rollback_cache(same_position_peer)
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(after_batch)

    backend.rollback_cache(earliest)
    assert backend.cached_token_ids == (0,)
    assert backend.active_checkpoint_count == 1
    for discarded in (before_batch, same_position_peer):
        with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
            backend.rollback_cache(discarded)


def test_checkpoint_release_is_cache_neutral_and_idempotent_after_verification():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    released = backend.create_cache_checkpoint()
    retained = backend.create_cache_checkpoint()
    backend.verify_proposal(1, (2,))
    cache_state = (backend.cache_length, backend.cached_token_ids, backend._next_row)

    backend.release_cache_checkpoint(released)
    backend.release_cache_checkpoint(released)

    assert (backend.cache_length, backend.cached_token_ids, backend._next_row) == cache_state
    assert backend.active_checkpoint_count == 1
    backend.rollback_cache(retained)
    assert backend.cached_token_ids == (0,)


@pytest.mark.parametrize("replacement", (None, (4,)))
def test_reset_or_replacement_prefill_invalidates_verification_era_handles(replacement):
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    before_batch = backend.create_cache_checkpoint()
    backend.verify_proposal(1, (2,))
    after_batch = backend.create_cache_checkpoint()

    if replacement is None:
        backend.reset()
    else:
        backend.prefill(replacement)

    assert backend.active_checkpoint_count == 0
    for checkpoint in (before_batch, after_batch):
        before = _fake_state(backend)
        with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
            backend.rollback_cache(checkpoint)
        assert _fake_state(backend) == before
        backend.release_cache_checkpoint(checkpoint)
        assert _fake_state(backend) == before


def test_bounded_batch_rollback_release_has_no_registry_or_cursor_drift():
    backend = FakeAutoregressiveBackend(SCRIPT[:3])
    backend.prefill((0,))
    base_state = (backend.cache_length, backend.cached_token_ids, backend._next_row)

    for iteration in range(1_000):
        checkpoint = backend.create_cache_checkpoint()
        result = backend.verify_proposal(
            iteration % VOCAB_SIZE,
            ((iteration + 1) % VOCAB_SIZE,),
        )
        assert result.logit_rows == SCRIPT[1:3]
        backend.rollback_cache(checkpoint)
        backend.release_cache_checkpoint(checkpoint)
        assert (backend.cache_length, backend.cached_token_ids, backend._next_row) == base_state
        assert backend.active_checkpoint_count == 0


def test_verification_across_epochs_cannot_alias_recycled_checkpoint_allocations():
    backend = FakeAutoregressiveBackend(SCRIPT[:3])
    previous = None

    for epoch_number in range(100):
        backend.prefill((epoch_number % VOCAB_SIZE,))
        current = backend.create_cache_checkpoint()
        backend.verify_proposal(1, (2,))
        if previous is not None:
            assert current.allocation_id == previous.allocation_id == 1
            assert current.epoch != previous.epoch
            before = _fake_state(backend)
            backend.release_cache_checkpoint(previous)
            assert _fake_state(backend) == before
            with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
                backend.rollback_cache(previous)
            assert _fake_state(backend) == before
        previous = current

    assert backend.active_checkpoint_count == 1
    backend.release_cache_checkpoint(previous)
    assert backend.active_checkpoint_count == 0
