from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import AutoregressiveBackend, BackendStateError, ModelStep
from onyx_cuda.testing import FakeAutoregressiveBackend, ScriptExhaustedError


SCRIPT = (
    (0.1, 0.2, 0.7),
    (0.8, 0.1, 0.1),
    (0.2, 0.6, 0.2),
)


def test_fake_backend_satisfies_the_runtime_contract():
    backend = FakeAutoregressiveBackend(SCRIPT, model_id="deterministic-target")

    assert isinstance(backend, AutoregressiveBackend)
    assert backend.model_id == "deterministic-target"
    assert backend.vocab_size == 3
    assert backend.cache_length == 0


def test_prefill_and_decode_advance_logical_cache_length():
    backend = FakeAutoregressiveBackend(SCRIPT)

    prefill = backend.prefill([2, 1])
    first_decode = backend.decode(2)
    second_decode = backend.decode(0)

    assert prefill == ModelStep(logits=SCRIPT[0], cache_length=2)
    assert first_decode == ModelStep(logits=SCRIPT[1], cache_length=3)
    assert second_decode == ModelStep(logits=SCRIPT[2], cache_length=4)
    assert backend.cache_length == 4


def test_prefill_starts_a_fresh_deterministic_sequence():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill([0, 1])
    backend.decode(2)

    restarted = backend.prefill([1])

    assert restarted == ModelStep(logits=SCRIPT[0], cache_length=1)
    assert backend.cache_length == 1


def test_reset_clears_cache_and_rewinds_script():
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill([0])
    backend.decode(1)

    backend.reset()

    assert backend.cache_length == 0
    assert backend.prefill([2]).logits == SCRIPT[0]


def test_decode_requires_prefill():
    backend = FakeAutoregressiveBackend(SCRIPT)

    with pytest.raises(BackendStateError, match="prefill"):
        backend.decode(0)


def test_script_exhaustion_does_not_advance_cache():
    backend = FakeAutoregressiveBackend([SCRIPT[0]])
    backend.prefill([0, 1])

    with pytest.raises(ScriptExhaustedError, match="no scripted logits"):
        backend.decode(2)

    assert backend.cache_length == 2


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ([], "at least one row"),
        ([[]], "row 1 cannot be empty"),
        ([[0.1, 0.2], [0.3]], "same vocabulary size"),
        ([[0.1, "not-a-number"]], "row 1 must be numeric"),
    ],
)
def test_rejects_invalid_scripts(script, message):
    with pytest.raises(ValueError, match=message):
        FakeAutoregressiveBackend(script)


def test_rejects_empty_model_identifier():
    with pytest.raises(ValueError, match="model_id"):
        FakeAutoregressiveBackend(SCRIPT, model_id="  ")


def test_rejects_non_string_model_identifier():
    with pytest.raises(TypeError, match="model_id must be a string"):
        FakeAutoregressiveBackend(SCRIPT, model_id=None)


@pytest.mark.parametrize("prompt", [[], [-1], [3], [True], ["1"]])
def test_rejects_invalid_prompts(prompt):
    backend = FakeAutoregressiveBackend(SCRIPT)

    with pytest.raises((TypeError, ValueError)):
        backend.prefill(prompt)


@pytest.mark.parametrize("token_id", [-1, 3, True, "1"])
def test_rejects_invalid_decode_tokens_without_advancing_cache(token_id):
    backend = FakeAutoregressiveBackend(SCRIPT)
    backend.prefill([0])

    with pytest.raises((TypeError, ValueError)):
        backend.decode(token_id)

    assert backend.cache_length == 1
    assert backend.decode(0).logits == SCRIPT[1]


def test_model_step_is_immutable_and_rejects_negative_cache_length():
    step = ModelStep(logits=SCRIPT[0], cache_length=1)

    with pytest.raises(FrozenInstanceError):
        step.cache_length = 2
    with pytest.raises(TypeError, match="must be an integer"):
        ModelStep(logits=SCRIPT[0], cache_length=1.5)
    with pytest.raises(ValueError, match="cannot be negative"):
        ModelStep(logits=SCRIPT[0], cache_length=-1)
