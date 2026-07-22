"""Framework-neutral tokenizer contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


class TokenizerError(ValueError):
    """Base error raised when text cannot be represented by a tokenizer."""


class UnknownTextTokenError(TokenizerError):
    """Raised when input text contains a unit absent from the tokenizer vocabulary."""


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Minimum tokenizer behavior required by a text generation engine."""

    @property
    def tokenizer_id(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, /) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int], /) -> str: ...
