"""Framework-neutral cache checkpoint contracts for autoregressive backends."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from .backend import AutoregressiveBackend, BackendStateError


class CacheCheckpointStateError(BackendStateError):
    """Raised when a cache checkpoint operation violates ownership or lifecycle rules."""


@runtime_checkable
class CacheCheckpoint(Protocol):
    """Opaque handle for one exact logical autoregressive-cache prefix."""

    @property
    def cache_length(self) -> int: ...


LogitsT_co = TypeVar("LogitsT_co", covariant=True)
CheckpointT = TypeVar("CheckpointT", bound=CacheCheckpoint)


@runtime_checkable
class CheckpointableAutoregressiveBackend(
    AutoregressiveBackend[LogitsT_co],
    Protocol[LogitsT_co, CheckpointT],
):
    """Optional backend capability for exact cache checkpoint and rollback."""

    def create_cache_checkpoint(self) -> CheckpointT: ...

    def rollback_cache(self, checkpoint: CheckpointT, /) -> None: ...

    def release_cache_checkpoint(self, checkpoint: CheckpointT, /) -> None: ...
