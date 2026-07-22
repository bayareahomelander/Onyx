"""Framework-neutral contracts for batched target verification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .backend import AutoregressiveBackend


LogitsT_co = TypeVar("LogitsT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class BatchedTargetVerificationResult(Generic[LogitsT_co]):
    """Ordered native logit rows and cache length from one verification batch."""

    logit_rows: tuple[LogitsT_co, ...]
    cache_length: int

    def __post_init__(self) -> None:
        if type(self.logit_rows) is not tuple:
            raise TypeError("logit_rows must be a tuple")
        if not self.logit_rows:
            raise ValueError("logit_rows cannot be empty")
        if isinstance(self.cache_length, bool) or not isinstance(self.cache_length, int):
            raise TypeError("cache_length must be an integer")
        if self.cache_length < 0:
            raise ValueError("cache_length cannot be negative")


@runtime_checkable
class BatchedTargetVerificationBackend(
    AutoregressiveBackend[LogitsT_co],
    Protocol[LogitsT_co],
):
    """Optional backend capability for one ordered target-verification batch."""

    def verify_proposal(
        self,
        current_token_id: int,
        proposal_token_ids: Sequence[int],
        /,
    ) -> BatchedTargetVerificationResult[LogitsT_co]: ...


__all__ = [
    "BatchedTargetVerificationBackend",
    "BatchedTargetVerificationResult",
]
