from dataclasses import FrozenInstanceError, replace

import pytest

from onyx_cuda import (
    AutoregressiveBackend,
    BackendError,
    BackendStateError,
    CacheCheckpoint,
    CacheCheckpointStateError,
    CheckpointableAutoregressiveBackend,
    ModelStep,
)
from onyx_cuda.testing import (
    FakeAutoregressiveBackend,
    FakeCacheCheckpoint,
    ScriptExhaustedError,
)


SCRIPT = (
    (0.7, 0.2, 0.1, 0.0),
    (0.1, 0.6, 0.2, 0.1),
    (0.2, 0.1, 0.6, 0.1),
    (0.1, 0.2, 0.1, 0.6),
    (0.4, 0.3, 0.2, 0.1),
)


class MinimumBackend:
    """Existing minimum contract without the optional checkpoint capability."""

    model_id = "minimum"
    vocab_size = 4
    cache_length = 0

    def prefill(self, prompt_token_ids, /):
        return ModelStep(logits=SCRIPT[0], cache_length=len(prompt_token_ids))

    def decode(self, token_id, /):
        return ModelStep(logits=SCRIPT[1], cache_length=self.cache_length + 1)

    def reset(self):
        return None


def _fake_state(backend):
    return (
        backend.cache_length,
        backend.cached_token_ids,
        backend.active_checkpoint_count,
        backend._next_row,
        backend._next_checkpoint_id,
        tuple(backend._cache_checkpoints),
    )


def test_checkpoint_capability_is_structurally_separate_from_minimum_backend():
    fake = FakeAutoregressiveBackend(SCRIPT)
    minimum = MinimumBackend()

    assert isinstance(fake, AutoregressiveBackend)
    assert isinstance(fake, CheckpointableAutoregressiveBackend)
    assert isinstance(minimum, AutoregressiveBackend)
    assert not isinstance(minimum, CheckpointableAutoregressiveBackend)


def test_fake_checkpoint_is_immutable_slotted_and_satisfies_public_contract():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))

    checkpoint = backend.create_cache_checkpoint()
    copied_checkpoint = replace(checkpoint)

    assert isinstance(checkpoint, CacheCheckpoint)
    assert checkpoint.cache_length == backend.cache_length == 2
    assert copied_checkpoint == checkpoint
    assert copied_checkpoint is not checkpoint
    assert not hasattr(checkpoint, "__dict__")
    with pytest.raises(FrozenInstanceError):
        checkpoint.cache_length = 3

    backend.rollback_cache(copied_checkpoint)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"owner_id": True}, TypeError, "owner_id must be an integer"),
        ({"owner_id": 0}, ValueError, "owner_id must be greater than zero"),
        ({"epoch": 0}, ValueError, "epoch must be greater than zero"),
        ({"allocation_id": 0}, ValueError, "allocation_id must be greater than zero"),
        ({"cache_length": True}, TypeError, "cache_length must be an integer"),
        ({"cache_length": -1}, ValueError, "cache_length cannot be negative"),
    ],
)
def test_fake_checkpoint_constructor_rejects_invalid_metadata(changes, error, message):
    values = {"owner_id": 1, "epoch": 1, "allocation_id": 1, "cache_length": 0}
    values.update(changes)

    with pytest.raises(error, match=message):
        FakeCacheCheckpoint(**values)


def test_checkpoint_state_error_belongs_to_backend_error_hierarchy():
    assert issubclass(CacheCheckpointStateError, BackendStateError)
    assert issubclass(CacheCheckpointStateError, BackendError)


def test_checkpoint_requires_an_active_prefilled_sequence_without_mutation():
    backend = FakeAutoregressiveBackend(SCRIPT)
    before = _fake_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="prefill"):
        backend.create_cache_checkpoint()

    assert _fake_state(backend) == before


def test_rollback_restores_exact_prefix_script_position_and_reusable_checkpoint():
    backend = FakeAutoregressiveBackend(SCRIPT)
    prefill = backend.prefill((0, 1))
    checkpoint = backend.create_cache_checkpoint()

    first_advance = backend.decode(2)
    second_advance = backend.decode(3)

    assert prefill == ModelStep(logits=SCRIPT[0], cache_length=2)
    assert first_advance == ModelStep(logits=SCRIPT[1], cache_length=3)
    assert second_advance == ModelStep(logits=SCRIPT[2], cache_length=4)
    assert backend.cached_token_ids == (0, 1, 2, 3)

    backend.rollback_cache(checkpoint)

    assert backend.cache_length == checkpoint.cache_length == 2
    assert backend.cached_token_ids == (0, 1)
    assert backend.active_checkpoint_count == 1
    assert backend.decode(1) == ModelStep(logits=SCRIPT[1], cache_length=3)
    assert backend.cached_token_ids == (0, 1, 1)

    backend.rollback_cache(checkpoint)
    assert backend.decode(3) == ModelStep(logits=SCRIPT[1], cache_length=3)
    assert backend.cached_token_ids == (0, 1, 3)


def test_nested_rollback_retains_prefix_handles_and_discards_suffix_handles():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    earliest = backend.create_cache_checkpoint()
    backend.decode(1)
    middle = backend.create_cache_checkpoint()
    backend.decode(2)
    deepest = backend.create_cache_checkpoint()
    backend.decode(3)

    backend.rollback_cache(middle)

    assert backend.cache_length == 2
    assert backend.cached_token_ids == (0, 1)
    assert backend.active_checkpoint_count == 2
    before_rejected_rollback = _fake_state(backend)
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(deepest)
    assert _fake_state(backend) == before_rejected_rollback
    backend.rollback_cache(middle)

    backend.rollback_cache(earliest)

    assert backend.cache_length == 1
    assert backend.cached_token_ids == (0,)
    assert backend.active_checkpoint_count == 1
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(middle)
    assert backend.decode(3).logits == SCRIPT[1]


def test_repeated_same_position_checkpoints_have_independent_lifetimes():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))
    first = backend.create_cache_checkpoint()
    second = backend.create_cache_checkpoint()

    assert first != second
    assert first.cache_length == second.cache_length == 2
    assert first.allocation_id != second.allocation_id
    assert backend.active_checkpoint_count == 2

    backend.release_cache_checkpoint(first)
    after_release = _fake_state(backend)
    backend.release_cache_checkpoint(first)

    assert _fake_state(backend) == after_release
    assert backend.active_checkpoint_count == 1
    with pytest.raises(CacheCheckpointStateError, match="released"):
        backend.rollback_cache(first)

    before_same_position_rollback = _fake_state(backend)
    backend.rollback_cache(second)
    assert _fake_state(backend) == before_same_position_rollback

    backend.release_cache_checkpoint(second)
    backend.release_cache_checkpoint(second)
    assert backend.active_checkpoint_count == 0


def test_wrong_checkpoint_type_is_rejected_atomically_for_rollback_and_release():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    checkpoint = backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises(TypeError, match="FakeCacheCheckpoint"):
        backend.rollback_cache(object())
    assert _fake_state(backend) == before
    with pytest.raises(TypeError, match="FakeCacheCheckpoint"):
        backend.release_cache_checkpoint(object())
    assert _fake_state(backend) == before

    backend.rollback_cache(checkpoint)


def test_foreign_checkpoint_is_rejected_atomically_for_rollback_and_release():
    owner = FakeAutoregressiveBackend(SCRIPT)
    owner.prefill((0,))
    foreign = owner.create_cache_checkpoint()

    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((1,))
    local = backend.create_cache_checkpoint()
    backend.decode(2)
    before = _fake_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="another backend"):
        backend.rollback_cache(foreign)
    assert _fake_state(backend) == before
    with pytest.raises(CacheCheckpointStateError, match="another backend"):
        backend.release_cache_checkpoint(foreign)
    assert _fake_state(backend) == before

    backend.rollback_cache(local)
    owner.rollback_cache(foreign)


def test_reset_invalidates_all_handles_and_clears_exact_execution_state():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    first = backend.create_cache_checkpoint()
    backend.decode(1)
    second = backend.create_cache_checkpoint()

    backend.reset()

    assert backend.cache_length == 0
    assert backend.cached_token_ids == ()
    assert backend.active_checkpoint_count == 0
    assert backend._next_row == 0
    for checkpoint in (first, second):
        before = _fake_state(backend)
        with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
            backend.rollback_cache(checkpoint)
        assert _fake_state(backend) == before
        backend.release_cache_checkpoint(checkpoint)
        assert _fake_state(backend) == before


def test_identical_prefill_replacement_cannot_alias_an_old_checkpoint():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))
    old = backend.create_cache_checkpoint()

    backend.prefill((0, 1))
    current = backend.create_cache_checkpoint()

    assert old.allocation_id == current.allocation_id
    assert old.cache_length == current.cache_length
    assert old.epoch != current.epoch
    before = _fake_state(backend)
    backend.release_cache_checkpoint(old)
    assert _fake_state(backend) == before
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(old)
    assert _fake_state(backend) == before

    backend.rollback_cache(current)


def test_unknown_and_mismatched_metadata_are_rejected_atomically():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))
    valid = backend.create_cache_checkpoint()
    invalid_checkpoints = (
        replace(valid, allocation_id=valid.allocation_id + 100),
        replace(valid, cache_length=valid.cache_length + 1),
    )

    for invalid in invalid_checkpoints:
        before = _fake_state(backend)
        with pytest.raises(CacheCheckpointStateError):
            backend.rollback_cache(invalid)
        assert _fake_state(backend) == before
        backend.rollback_cache(valid)
        assert _fake_state(backend) == before


def test_canonical_forward_checkpoint_is_rejected_atomically():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0, 1))
    checkpoint = backend.create_cache_checkpoint()

    # Model an invalid adapter state in which a canonical handle is ahead of the active cache.
    backend._cached_token_ids = (0,)
    backend._cache_length = 1
    before = _fake_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="ahead"):
        backend.rollback_cache(checkpoint)

    assert _fake_state(backend) == before


def test_metadata_mismatch_cannot_release_an_active_allocation():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    checkpoint = backend.create_cache_checkpoint()
    mismatched = replace(checkpoint, cache_length=checkpoint.cache_length + 1)
    before = _fake_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="metadata"):
        backend.release_cache_checkpoint(mismatched)

    assert _fake_state(backend) == before
    backend.rollback_cache(checkpoint)


def test_released_checkpoint_cannot_rollback_or_affect_same_position_peer():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    released = backend.create_cache_checkpoint()
    retained = backend.create_cache_checkpoint()
    backend.release_cache_checkpoint(released)
    before = _fake_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="released"):
        backend.rollback_cache(released)

    assert _fake_state(backend) == before
    backend.rollback_cache(retained)


@pytest.mark.parametrize("token_id", (-1, 4, True, "1"))
def test_invalid_decode_preserves_checkpoint_and_complete_fake_state(token_id):
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    checkpoint = backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises((TypeError, ValueError)):
        backend.decode(token_id)

    assert _fake_state(backend) == before
    assert backend.decode(1).logits == SCRIPT[1]
    backend.rollback_cache(checkpoint)
    assert _fake_state(backend) == before


def test_script_exhaustion_preserves_cache_and_checkpoint_registry():
    backend = FakeAutoregressiveBackend(SCRIPT[:2])
    backend.prefill((0,))
    earliest = backend.create_cache_checkpoint()
    backend.decode(1)
    later = backend.create_cache_checkpoint()
    before = _fake_state(backend)

    with pytest.raises(ScriptExhaustedError, match="no scripted logits"):
        backend.decode(2)

    assert _fake_state(backend) == before
    backend.rollback_cache(later)
    assert _fake_state(backend) == before
    backend.rollback_cache(earliest)
    assert backend.decode(3).logits == SCRIPT[1]


@pytest.mark.parametrize("prompt", ((), (True,), (4,)))
def test_invalid_prefill_preserves_active_sequence_and_checkpoint(prompt):
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill((0,))
    checkpoint = backend.create_cache_checkpoint()
    backend.decode(1)
    before = _fake_state(backend)

    with pytest.raises((TypeError, ValueError)):
        backend.prefill(prompt)

    assert _fake_state(backend) == before
    backend.rollback_cache(checkpoint)
    assert backend.cached_token_ids == (0,)
    assert backend.decode(2).logits == SCRIPT[1]


def test_bounded_reuse_does_not_grow_active_checkpoint_registry():
    backend = FakeAutoregressiveBackend(SCRIPT[:2])
    backend.prefill((0,))

    for iteration in range(1_000):
        checkpoint = backend.create_cache_checkpoint()
        step = backend.decode(iteration % backend.vocab_size)
        assert step == ModelStep(logits=SCRIPT[1], cache_length=2)
        backend.rollback_cache(checkpoint)
        backend.release_cache_checkpoint(checkpoint)
        assert backend.cache_length == 1
        assert backend.cached_token_ids == (0,)
        assert backend.active_checkpoint_count == 0


def test_repeated_prefill_epochs_never_alias_recycled_allocations():
    backend = FakeAutoregressiveBackend(SCRIPT)
    previous = None

    for epoch_number in range(100):
        backend.prefill((epoch_number % backend.vocab_size,))
        current = backend.create_cache_checkpoint()
        if previous is not None:
            assert current.allocation_id == previous.allocation_id == 1
            assert current.epoch != previous.epoch
            before = _fake_state(backend)
            backend.release_cache_checkpoint(previous)
            assert _fake_state(backend) == before
            with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
                backend.rollback_cache(previous)
            assert _fake_state(backend) == before
        backend.release_cache_checkpoint(current)
        previous = current

    assert backend.active_checkpoint_count == 0
