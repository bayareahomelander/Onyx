"""Framework-neutral masked selection from one borrowed grammar state."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .constrained_generation import GrammarLogitMask
from .grammar import GrammarConstraint, GrammarError

__all__ = [
    "GrammarMaskedSelectionError",
    "GrammarMaskedSelectionInvariantError",
    "GrammarMaskedSelectionResult",
    "select_grammar_masked_token",
]


class GrammarMaskedSelectionError(GrammarError):
    """Base error raised by one grammar-supported masked selection."""


class GrammarMaskedSelectionInvariantError(GrammarMaskedSelectionError):
    """Raised when component metadata or one-row evidence violates the D42 contract."""


@dataclass(frozen=True, slots=True)
class GrammarMaskedSelectionResult:
    """Immutable native grammar support and its optional selected token."""

    valid_token_ids: tuple[int, ...]
    is_match: bool
    selected_token_id: int | None

    def __post_init__(self) -> None:
        _validate_result_support(self.valid_token_ids)
        if type(self.is_match) is not bool:
            raise TypeError("is_match must be a boolean")
        if not self.valid_token_ids:
            if self.selected_token_id is not None:
                raise ValueError("empty valid_token_ids require selected_token_id to be None")
            return
        if type(self.selected_token_id) is not int:
            raise TypeError(
                "nonempty valid_token_ids require selected_token_id to be an integer"
            )
        if self.selected_token_id < 0:
            raise ValueError("selected_token_id cannot be negative")
        if not _contains_token(self.valid_token_ids, self.selected_token_id):
            raise ValueError("selected_token_id must belong to valid_token_ids")


LogitsT = TypeVar("LogitsT")
StateT = TypeVar("StateT")


def select_grammar_masked_token(
    constraint: GrammarConstraint[StateT],
    state: StateT,
    logits: LogitsT,
    logit_mask: GrammarLogitMask[LogitsT],
    *,
    vocab_size: int,
    select_token: Callable[[LogitsT], int],
) -> GrammarMaskedSelectionResult:
    """Select at most one token from one row using a borrowed grammar state's support."""

    validated_vocab_size = _validate_vocab_size(vocab_size)
    _require_protocol_conformance(
        constraint,
        GrammarConstraint,
        label="constraint",
    )
    _require_protocol_conformance(
        logit_mask,
        GrammarLogitMask,
        label="logit_mask",
    )
    if not callable(select_token):
        raise TypeError("select_token must be callable")

    constraint_vocab_size = _read_positive_component_vocab_size(
        constraint,
        label="constraint",
    )
    grammar_type = _read_attribute(constraint, "grammar_type", label="constraint")
    if type(grammar_type) is not str or grammar_type not in {"regex", "json_schema"}:
        raise GrammarMaskedSelectionInvariantError(
            "constraint grammar_type must be 'regex' or 'json_schema'"
        )
    mask_vocab_size = _read_positive_component_vocab_size(
        logit_mask,
        label="logit_mask",
    )
    if (
        constraint_vocab_size != validated_vocab_size
        or mask_vocab_size != validated_vocab_size
    ):
        raise GrammarMaskedSelectionInvariantError(
            "explicit, constraint, and mask vocabulary sizes must match exactly: "
            f"explicit={validated_vocab_size}, constraint={constraint_vocab_size}, "
            f"mask={mask_vocab_size}"
        )

    is_dead = _require_state_boolean(
        constraint.is_dead_state(state),
        operation="is_dead_state",
    )
    if is_dead:
        raise GrammarMaskedSelectionInvariantError("borrowed grammar state must not be dead")
    is_match = _require_state_boolean(
        constraint.is_match_state(state),
        operation="is_match_state",
    )
    valid_token_ids = _validate_native_valid_token_ids(
        constraint.get_valid_token_ids(state),
        vocab_size=validated_vocab_size,
    )

    if not valid_token_ids:
        return GrammarMaskedSelectionResult(
            valid_token_ids=valid_token_ids,
            is_match=is_match,
            selected_token_id=None,
        )

    masked_logits = logit_mask.apply(logits, valid_token_ids)
    selected_token_id = _validate_selected_token(
        select_token(masked_logits),
        vocab_size=validated_vocab_size,
        valid_token_ids=valid_token_ids,
    )
    return GrammarMaskedSelectionResult(
        valid_token_ids=valid_token_ids,
        is_match=is_match,
        selected_token_id=selected_token_id,
    )


def _validate_vocab_size(value: object) -> int:
    if type(value) is not int:
        raise TypeError("vocab_size must be an integer")
    if value <= 0:
        raise ValueError("vocab_size must be greater than zero")
    return value


def _require_protocol_conformance(
    component: object,
    protocol: object,
    *,
    label: str,
) -> None:
    try:
        conforms = isinstance(component, protocol)
    except Exception as exc:
        raise GrammarMaskedSelectionInvariantError(
            f"{label} runtime conformance could not be determined"
        ) from exc
    if not conforms:
        raise TypeError(f"{label} must satisfy {protocol.__name__}")


def _read_attribute(component: object, name: str, *, label: str) -> object:
    try:
        return getattr(component, name)
    except Exception as exc:
        raise GrammarMaskedSelectionInvariantError(
            f"{label} {name} could not be read"
        ) from exc


def _read_positive_component_vocab_size(component: object, *, label: str) -> int:
    value = _read_attribute(component, "vocab_size", label=label)
    if type(value) is not int or value <= 0:
        raise GrammarMaskedSelectionInvariantError(
            f"{label} vocab_size must be a positive integer"
        )
    return value


def _require_state_boolean(value: object, *, operation: str) -> bool:
    if type(value) is not bool:
        raise GrammarMaskedSelectionInvariantError(
            f"constraint {operation}() must return a boolean"
        )
    return value


def _validate_native_valid_token_ids(
    token_ids: object,
    *,
    vocab_size: int,
) -> tuple[int, ...]:
    if type(token_ids) is not tuple:
        raise GrammarMaskedSelectionInvariantError(
            "constraint valid-token output must be an exact tuple"
        )
    previous = -1
    for token_id in token_ids:
        if type(token_id) is not int:
            raise GrammarMaskedSelectionInvariantError(
                "constraint valid-token output must contain Python integers"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise GrammarMaskedSelectionInvariantError(
                f"constraint returned token ID {token_id} outside vocabulary range "
                f"[0, {vocab_size})"
            )
        if token_id <= previous:
            raise GrammarMaskedSelectionInvariantError(
                "constraint valid-token output must be strictly increasing and unique"
            )
        previous = token_id
    return token_ids


def _validate_result_support(token_ids: object) -> tuple[int, ...]:
    if type(token_ids) is not tuple:
        raise TypeError("valid_token_ids must be an exact tuple")
    previous = -1
    for token_id in token_ids:
        if type(token_id) is not int:
            raise TypeError("valid_token_ids must contain Python integers")
        if token_id < 0:
            raise ValueError("valid_token_ids cannot contain negative token IDs")
        if token_id <= previous:
            raise ValueError("valid_token_ids must be strictly increasing and unique")
        previous = token_id
    return token_ids


def _validate_selected_token(
    token_id: object,
    *,
    vocab_size: int,
    valid_token_ids: tuple[int, ...],
) -> int:
    if type(token_id) is not int:
        raise GrammarMaskedSelectionInvariantError(
            "selected token ID must be a Python integer"
        )
    if token_id < 0 or token_id >= vocab_size:
        raise GrammarMaskedSelectionInvariantError(
            f"selected token ID {token_id} is outside vocabulary range [0, {vocab_size})"
        )
    if not _contains_token(valid_token_ids, token_id):
        raise GrammarMaskedSelectionInvariantError(
            f"selected token ID {token_id} is outside the exact grammar support"
        )
    return token_id


def _contains_token(token_ids: tuple[int, ...], token_id: int) -> bool:
    position = bisect_left(token_ids, token_id)
    return position < len(token_ids) and token_ids[position] == token_id
