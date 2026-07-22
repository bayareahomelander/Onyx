"""Framework-neutral target-token selection policies and reference sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


MAX_SAMPLING_SEED = (1 << 63) - 1


class SelectionError(ValueError):
    """Base error raised by framework-neutral token selection."""


class SamplingDistributionError(SelectionError):
    """Raised when logits cannot form a valid sampling distribution."""


@dataclass(frozen=True, slots=True)
class GreedySelection:
    """Explicit deterministic argmax selection policy."""


@dataclass(frozen=True, slots=True)
class TemperatureTopPSelection:
    """Explicit seeded temperature and nucleus-sampling policy.

    Temperature must be finite and positive, top-p must be within ``(0, 1]``, and seed must be a
    portable non-negative 63-bit integer. Sampling accepts negative infinity as zero-probability
    mask values and treats positive-infinity logits as equal-probability support.
    """

    temperature: float
    top_p: float
    seed: int

    def __post_init__(self) -> None:
        _validate_probability_parameter(
            self.temperature,
            label="temperature",
            minimum_exclusive=0.0,
            maximum_inclusive=None,
        )
        _validate_probability_parameter(
            self.top_p,
            label="top_p",
            minimum_exclusive=0.0,
            maximum_inclusive=1.0,
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0 or self.seed > MAX_SAMPLING_SEED:
            raise ValueError(
                f"seed must be within [0, {MAX_SAMPLING_SEED}]"
            )


SelectionPolicy = GreedySelection | TemperatureTopPSelection
GREEDY_SELECTION = GreedySelection()


def validate_selection_policy(policy: SelectionPolicy) -> SelectionPolicy:
    """Return a supported immutable policy or reject an unknown policy type."""

    if not isinstance(policy, (GreedySelection, TemperatureTopPSelection)):
        raise TypeError(
            "selection must be a GreedySelection or TemperatureTopPSelection"
        )
    return policy


def create_reference_sampler(
    policy: TemperatureTopPSelection,
) -> Callable[[Sequence[float]], int]:
    """Create one deterministic CPU sampling session for fake-backend tests.

    This reference path deliberately uses a private Python RNG. Its exact sequence is stable for
    repeated reference sessions with the same seed, but is not required to match PyTorch's CUDA
    generator sequence.
    """

    if not isinstance(policy, TemperatureTopPSelection):
        raise TypeError("policy must be a TemperatureTopPSelection")
    generator = random.Random(policy.seed)

    def select_token(logits: Sequence[float]) -> int:
        weighted_tokens = _filtered_reference_weights(logits, policy)
        total_weight = sum(weight for _, weight in weighted_tokens)
        threshold = generator.random() * total_weight
        cumulative = 0.0
        for token_id, weight in weighted_tokens:
            cumulative += weight
            if threshold < cumulative:
                return token_id
        return weighted_tokens[-1][0]

    return select_token


def _filtered_reference_weights(
    logits: Sequence[float],
    policy: TemperatureTopPSelection,
) -> tuple[tuple[int, float], ...]:
    try:
        values = tuple(float(value) for value in logits)
    except (TypeError, ValueError) as exc:
        raise SamplingDistributionError("logits must contain numeric values") from exc
    if not values:
        raise SamplingDistributionError("logits cannot be empty")
    if any(math.isnan(value) for value in values):
        raise SamplingDistributionError("logits cannot contain NaN")

    positive_infinity_ids = tuple(
        token_id for token_id, value in enumerate(values) if value == math.inf
    )
    if positive_infinity_ids:
        equal_weight = 1.0 / len(positive_infinity_ids)
        probabilities = tuple(
            (token_id, equal_weight) for token_id in positive_infinity_ids
        )
    else:
        finite_values = tuple(value for value in values if value != -math.inf)
        if not finite_values:
            raise SamplingDistributionError(
                "logits cannot form a distribution because every value is -inf"
            )
        maximum = max(finite_values)
        unnormalized = tuple(
            0.0
            if value == -math.inf
            else math.exp((value - maximum) / float(policy.temperature))
            for value in values
        )
        total = sum(unnormalized)
        if not math.isfinite(total) or total <= 0.0:
            raise SamplingDistributionError("logits produced invalid probability mass")
        probabilities = tuple(
            (token_id, weight / total)
            for token_id, weight in enumerate(unnormalized)
            if weight > 0.0
        )

    ranked = sorted(probabilities, key=lambda item: (-item[1], item[0]))
    kept = []
    cumulative = 0.0
    for token_id, probability in ranked:
        kept.append((token_id, probability))
        cumulative += probability
        if cumulative >= float(policy.top_p):
            break
    if not kept:
        raise SamplingDistributionError("top-p filtering retained no probability mass")
    return tuple(kept)


def _validate_probability_parameter(
    value: float,
    *,
    label: str,
    minimum_exclusive: float,
    maximum_inclusive: float | None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    if numeric <= minimum_exclusive:
        raise ValueError(f"{label} must be greater than {minimum_exclusive}")
    if maximum_inclusive is not None and numeric > maximum_inclusive:
        raise ValueError(f"{label} must be at most {maximum_inclusive}")
