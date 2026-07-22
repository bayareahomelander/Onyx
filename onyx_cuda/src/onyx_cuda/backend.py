"""Framework-neutral contracts for autoregressive model execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable


class BackendError(RuntimeError):
    """Base error raised by an Onyx model backend."""


class BackendStateError(BackendError):
    """Raised when a backend operation is invalid for its current state."""


LogitsT_co = TypeVar("LogitsT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class ModelStep(Generic[LogitsT_co]):
    """Next-token logits and the logical cache length that produced them."""

    logits: LogitsT_co
    cache_length: int

    def __post_init__(self) -> None:
        if isinstance(self.cache_length, bool) or not isinstance(self.cache_length, int):
            raise TypeError("cache_length must be an integer")
        if self.cache_length < 0:
            raise ValueError("cache_length cannot be negative")


@runtime_checkable
class AutoregressiveBackend(Protocol[LogitsT_co]):
    """Minimum backend behavior required by target-only generation.

    ``prefill`` starts a fresh sequence and returns logits for the first generated token.
    ``decode`` consumes one selected token and returns logits for the following token.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def cache_length(self) -> int: ...

    def prefill(self, prompt_token_ids: Sequence[int], /) -> ModelStep[LogitsT_co]: ...

    def decode(self, token_id: int, /) -> ModelStep[LogitsT_co]: ...

    def reset(self) -> None: ...
