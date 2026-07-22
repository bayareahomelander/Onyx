import random
from dataclasses import FrozenInstanceError

import pytest

from onyx_cuda import (
    GREEDY_SELECTION,
    MAX_SAMPLING_SEED,
    GreedySelection,
    SamplingDistributionError,
    TemperatureTopPSelection,
    create_reference_sampler,
    validate_selection_policy,
)


def test_selection_policies_are_explicit_immutable_and_validated():
    greedy = GreedySelection()
    sampling = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=7)

    assert greedy == GREEDY_SELECTION
    assert validate_selection_policy(greedy) is greedy
    assert validate_selection_policy(sampling) is sampling

    with pytest.raises(FrozenInstanceError):
        sampling.seed = 8


@pytest.mark.parametrize("value", [0, -1, True, "1", float("inf"), float("nan")])
def test_rejects_invalid_temperature(value):
    with pytest.raises((TypeError, ValueError), match="temperature"):
        TemperatureTopPSelection(temperature=value, top_p=1.0, seed=0)


@pytest.mark.parametrize(
    "value",
    [0, -0.1, 1.1, True, "1", float("inf"), float("nan")],
)
def test_rejects_invalid_top_p(value):
    with pytest.raises((TypeError, ValueError), match="top_p"):
        TemperatureTopPSelection(temperature=1.0, top_p=value, seed=0)


@pytest.mark.parametrize("value", [-1, MAX_SAMPLING_SEED + 1, True, 1.5, "1"])
def test_rejects_invalid_seed(value):
    with pytest.raises((TypeError, ValueError), match="seed"):
        TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=value)


def test_accepts_portable_seed_boundaries_and_numeric_policy_values():
    assert TemperatureTopPSelection(1, 1, 0).seed == 0
    maximum = TemperatureTopPSelection(1.0, 1.0, MAX_SAMPLING_SEED)
    assert maximum.seed == MAX_SAMPLING_SEED
    assert create_reference_sampler(maximum)((0.0,)) == 0


def test_rejects_unknown_policy_and_non_sampling_reference_policy():
    with pytest.raises(TypeError, match="selection must be"):
        validate_selection_policy(None)
    with pytest.raises(TypeError, match="TemperatureTopPSelection"):
        create_reference_sampler(GREEDY_SELECTION)


def test_reference_sampler_replays_an_exact_seeded_sequence():
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=7)
    logits = (0.0, 0.0, 0.0)

    first = create_reference_sampler(policy)
    second = create_reference_sampler(policy)

    assert tuple(first(logits) for _ in range(6)) == (0, 0, 1, 0, 1, 1)
    assert tuple(second(logits) for _ in range(6)) == (0, 0, 1, 0, 1, 1)


def test_reference_sampler_does_not_advance_global_python_rng():
    original = random.getstate()
    try:
        random.seed(1234)
        before = random.getstate()
        sampler = create_reference_sampler(
            TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=7)
        )

        sampler((0.0, 1.0, 2.0))

        assert random.getstate() == before
    finally:
        random.setstate(original)


def test_reference_temperature_changes_the_seeded_distribution():
    logits = (0.0, 1.0, 2.0)
    cold = create_reference_sampler(
        TemperatureTopPSelection(temperature=0.1, top_p=1.0, seed=0)
    )
    hot = create_reference_sampler(
        TemperatureTopPSelection(temperature=10.0, top_p=1.0, seed=0)
    )

    assert cold(logits) == 2
    assert hot(logits) == 0


def test_reference_top_p_uses_stable_token_id_ties_and_keeps_one_token():
    sampler = create_reference_sampler(
        TemperatureTopPSelection(temperature=1.0, top_p=0.5, seed=99)
    )

    assert tuple(sampler((0.0, 0.0)) for _ in range(10)) == (0,) * 10


def test_reference_sampler_allows_negative_infinity_masking():
    sampler = create_reference_sampler(
        TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=0)
    )

    assert sampler((float("-inf"), 0.0, float("-inf"))) == 1


def test_reference_sampler_treats_positive_infinities_as_equal_support():
    sampler = create_reference_sampler(
        TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=0)
    )

    assert tuple(
        sampler((float("inf"), 0.0, float("inf"))) for _ in range(6)
    ) == (2, 2, 0, 0, 2, 0)


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        ((), "cannot be empty"),
        ((0.0, float("nan")), "cannot contain NaN"),
        ((float("-inf"), float("-inf")), "every value is -inf"),
        ((0.0, "invalid"), "numeric"),
    ],
)
def test_reference_sampler_rejects_invalid_or_degenerate_logits(logits, message):
    sampler = create_reference_sampler(
        TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=0)
    )

    with pytest.raises(SamplingDistributionError, match=message):
        sampler(logits)
