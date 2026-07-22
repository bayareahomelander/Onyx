"""Framework-neutral incremental text-generation contracts and helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from .generation import GenerationResult
from .stop_sequences import pending_stop_prefix_length
from .tokenizer import TokenizerAdapter


class StreamingError(ValueError):
    """Base error raised when incremental text cannot be represented safely."""


class StreamingInvariantError(StreamingError):
    """Raised when incremental state disagrees with completed generation state."""


class StreamingCleanupError(StreamingError):
    """Raised when an incomplete stream cannot release backend state safely."""


@dataclass(frozen=True, slots=True)
class TextGenerationResult:
    """Decoded text plus both user-visible and sampled token metadata."""

    tokenizer_id: str
    text: str
    output_token_ids: tuple[int, ...]
    generation: GenerationResult

    def __post_init__(self) -> None:
        if self.output_token_ids != self.generation.visible_token_ids:
            raise ValueError("output_token_ids do not match the generation finish reason")

    @property
    def sampled_token_ids(self) -> tuple[int, ...]:
        return self.generation.token_ids

    @property
    def generated_tokens(self) -> int:
        return self.generation.generated_tokens


@dataclass(frozen=True, slots=True)
class TextGenerationDelta:
    """One nonempty, stable text fragment from an incremental generation."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text generation delta text must be a string")
        if not self.text:
            raise ValueError("text generation delta text cannot be empty")


@dataclass(frozen=True, slots=True)
class TextGenerationComplete:
    """Terminal event containing the same result returned by non-streaming generation."""

    result: TextGenerationResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, TextGenerationResult):
            raise TypeError("text generation completion result must be TextGenerationResult")


TextGenerationEvent: TypeAlias = TextGenerationDelta | TextGenerationComplete


class _StopTokenBuffer:
    """Withhold possible stop prefixes until they are safe to expose."""

    def __init__(self, stop_token_sequences: Sequence[tuple[int, ...]]) -> None:
        self._stops = tuple(stop_token_sequences)
        self._pending: list[int] = []

    def push(self, token_id: int) -> tuple[int, ...]:
        self._pending.append(token_id)
        held_length = pending_stop_prefix_length(self._pending, self._stops)
        released_length = len(self._pending) - held_length
        return self._release(released_length)

    def finish(self, token_id: int, result: GenerationResult) -> tuple[int, ...]:
        self._pending.append(token_id)
        sampled_token_ids = result.token_ids
        visible_token_ids = result.visible_token_ids
        if (
            len(visible_token_ids) > len(sampled_token_ids)
            or sampled_token_ids[: len(visible_token_ids)] != visible_token_ids
        ):
            raise StreamingInvariantError(
                "visible generation tokens must be a prefix of sampled tokens"
            )
        hidden_suffix = sampled_token_ids[len(visible_token_ids) :]
        if not hidden_suffix:
            return self._release(len(self._pending))
        if len(hidden_suffix) > len(self._pending):
            raise StreamingInvariantError(
                "terminal hidden tokens were released before generation completed"
            )
        if tuple(self._pending[-len(hidden_suffix) :]) != hidden_suffix:
            raise StreamingInvariantError(
                "terminal hidden tokens are inconsistent with the pending stream suffix"
            )
        return self._release(len(self._pending) - len(hidden_suffix), clear=True)

    def _release(self, length: int, *, clear: bool = False) -> tuple[int, ...]:
        released = tuple(self._pending[:length])
        if clear:
            self._pending.clear()
        else:
            del self._pending[:length]
        return released


class _StableTextDecoder:
    """Decode cumulative token IDs while emitting only monotonic stable text."""

    def __init__(self, tokenizer: TokenizerAdapter) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._text = ""

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self._token_ids)

    @property
    def text(self) -> str:
        return self._text

    def append(self, token_ids: Sequence[int], *, final: bool) -> str:
        self._token_ids.extend(token_ids)
        decoded = self._tokenizer.decode(self._token_ids)
        if not isinstance(decoded, str):
            raise StreamingInvariantError("tokenizer decode must return a string")
        stable_text = decoded if final else decoded.rstrip("\ufffd")
        if not stable_text.startswith(self._text):
            raise StreamingInvariantError(
                "tokenizer decoding changed text that was already emitted"
            )
        delta = stable_text[len(self._text) :]
        self._text = stable_text
        return delta
