"""Framework-neutral contracts for token-level grammar constraints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Protocol, TypeVar, runtime_checkable


class GrammarError(RuntimeError):
    """Base error raised by grammar compilation or state operations."""


class GrammarCompilationError(GrammarError):
    """Raised when a regex or JSON Schema constraint cannot be compiled."""


class GrammarStateError(GrammarError):
    """Raised when a grammar state handle or transition is invalid."""


class GrammarRuntimeUnavailableError(GrammarError):
    """Raised when the optional native grammar runtime cannot be loaded."""


class GrammarRuntimeCompatibilityError(GrammarError):
    """Raised when a native grammar runtime violates the expected ABI contract."""


GrammarType = Literal["regex", "json_schema"]
GrammarStateT = TypeVar("GrammarStateT")


@runtime_checkable
class GrammarConstraint(Protocol[GrammarStateT]):
    """One compiled grammar with explicit, independently branchable states."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def grammar_type(self) -> GrammarType: ...

    def init_state(self) -> GrammarStateT: ...

    def advance_state(self, state: GrammarStateT, token_id: int, /) -> GrammarStateT: ...

    def get_valid_token_ids(self, state: GrammarStateT, /) -> tuple[int, ...]: ...

    def is_match_state(self, state: GrammarStateT, /) -> bool: ...

    def is_dead_state(self, state: GrammarStateT, /) -> bool: ...

    def release_state(self, state: GrammarStateT, /) -> None: ...

    def release_states(self, states: Sequence[GrammarStateT], /) -> None: ...

    def reset(self) -> None: ...


@runtime_checkable
class GrammarCompiler(Protocol[GrammarStateT]):
    """Compile regex or JSON Schema source into a fresh grammar constraint."""

    def compile_regex(
        self,
        vocabulary: Sequence[bytes],
        pattern: str,
        /,
    ) -> GrammarConstraint[GrammarStateT]: ...

    def compile_json_schema(
        self,
        vocabulary: Sequence[bytes],
        schema: str,
        /,
    ) -> GrammarConstraint[GrammarStateT]: ...


def _normalize_grammar_vocabulary(vocabulary: Sequence[bytes]) -> tuple[bytes, ...]:
    try:
        tokens = tuple(vocabulary)
    except TypeError as exc:
        raise TypeError("grammar vocabulary must be a sequence of bytes") from exc
    if not tokens:
        raise ValueError("grammar vocabulary cannot be empty")
    for token_id, token_bytes in enumerate(tokens):
        if not isinstance(token_bytes, bytes):
            raise TypeError(f"grammar vocabulary entry {token_id} must be bytes")
    return tokens


def _validate_regex_pattern(pattern: str) -> str:
    if not isinstance(pattern, str):
        raise TypeError("regex pattern must be a string")
    if not pattern:
        raise ValueError("regex pattern cannot be empty")
    return pattern


def _validate_json_schema(schema: str) -> str:
    if not isinstance(schema, str):
        raise TypeError("JSON Schema must be a string")
    if not schema.strip():
        raise ValueError("JSON Schema cannot be empty")
    try:
        parsed = json.loads(schema, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise ValueError(f"JSON Schema must contain valid JSON: {detail}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON Schema root must be an object")
    return schema


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite constant {value!r} is not valid JSON")


def _validate_grammar_token_id(token_id: int, vocab_size: int) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError("grammar token ID must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"grammar token ID {token_id} is outside vocabulary range [0, {vocab_size})"
        )
    return token_id
