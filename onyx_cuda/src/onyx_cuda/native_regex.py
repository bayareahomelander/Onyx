"""Lazy typed adapter for the independent Windows native regex runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .grammar import (
    GrammarCompilationError,
    GrammarRuntimeCompatibilityError,
    GrammarStateError,
    GrammarType,
    _normalize_grammar_vocabulary,
    _validate_grammar_token_id,
    _validate_regex_pattern,
)
from .native_grammar import (
    _NativeGrammarRuntime,
    _get_native_attribute,
    _load_native_grammar_runtime,
)


_STATE_CONSTRUCTION_TOKEN = object()
_CONSTRAINT_CONSTRUCTION_TOKEN = object()
_NATIVE_CONSTRAINT_METHODS = (
    "init_state",
    "advance_state",
    "get_valid_token_ids",
    "is_match_state",
    "is_dead_state",
    "release_state",
    "release_states",
    "reset",
)


@dataclass(frozen=True, slots=True, init=False)
class NativeRegexState:
    """Opaque state owned by one native regex constraint."""

    _owner: object = field(repr=False)
    _native_state: Any = field(repr=False)

    def __init__(
        self,
        owner: object,
        native_state: Any,
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _STATE_CONSTRUCTION_TOKEN:
            raise TypeError("NativeRegexState values are created by a native regex constraint")
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_native_state", native_state)

    @classmethod
    def _create(cls, owner: object, native_state: Any) -> NativeRegexState:
        return cls(
            owner,
            native_state,
            _construction_token=_STATE_CONSTRUCTION_TOKEN,
        )


class NativeRegexConstraint:
    """Typed Python boundary for one compiled native regex and its explicit states."""

    def __init__(
        self,
        runtime: _NativeGrammarRuntime,
        native_constraint: Any,
        vocab_size: int,
        *,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _CONSTRAINT_CONSTRUCTION_TOKEN:
            raise TypeError("NativeRegexConstraint values are created by compile_native_regex()")
        self._runtime = runtime
        self._native_constraint = native_constraint
        self._vocab_size = vocab_size
        self._owner = object()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def grammar_type(self) -> GrammarType:
        return "regex"

    def init_state(self) -> NativeRegexState:
        native_state = self._call_state_method("init_state")
        return self._wrap_state(native_state)

    def advance_state(
        self,
        state: NativeRegexState,
        token_id: int,
        /,
    ) -> NativeRegexState:
        native_state = self._native_state(state)
        token_id = _validate_grammar_token_id(token_id, self.vocab_size)
        child = self._call_state_method("advance_state", native_state, token_id)
        return self._wrap_state(child)

    def get_valid_token_ids(self, state: NativeRegexState, /) -> tuple[int, ...]:
        native_state = self._native_state(state)
        raw_token_ids = self._call_state_method("get_valid_token_ids", native_state)
        try:
            token_ids = tuple(raw_token_ids)
        except Exception as exc:
            raise GrammarRuntimeCompatibilityError(
                "native regex get_valid_token_ids() must return an iterable of token IDs"
            ) from exc
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise GrammarRuntimeCompatibilityError(
                    "native regex valid-token output must contain integers"
                )
            if token_id < 0 or token_id >= self.vocab_size:
                raise GrammarRuntimeCompatibilityError(
                    f"native regex returned token ID {token_id} outside vocabulary range "
                    f"[0, {self.vocab_size})"
                )
        if token_ids != tuple(sorted(set(token_ids))):
            raise GrammarRuntimeCompatibilityError(
                "native regex valid-token output must be unique and ascending"
            )
        return token_ids

    def is_match_state(self, state: NativeRegexState, /) -> bool:
        result = self._call_state_method("is_match_state", self._native_state(state))
        if not isinstance(result, bool):
            raise GrammarRuntimeCompatibilityError(
                "native regex is_match_state() must return a boolean"
            )
        return result

    def is_dead_state(self, state: NativeRegexState, /) -> bool:
        result = self._call_state_method("is_dead_state", self._native_state(state))
        if not isinstance(result, bool):
            raise GrammarRuntimeCompatibilityError(
                "native regex is_dead_state() must return a boolean"
            )
        return result

    def release_state(self, state: NativeRegexState, /) -> None:
        result = self._call_state_method("release_state", self._native_state(state))
        self._require_none(result, operation="release_state")

    def release_states(self, states: Sequence[NativeRegexState], /) -> None:
        try:
            normalized = tuple(states)
        except TypeError as exc:
            raise TypeError("native regex states must be a sequence") from exc
        native_states = [self._native_state(state) for state in normalized]
        result = self._call_state_method("release_states", native_states)
        self._require_none(result, operation="release_states")

    def reset(self) -> None:
        result = self._call_state_method("reset")
        self._require_none(result, operation="reset")

    def _native_state(self, state: NativeRegexState) -> Any:
        if not isinstance(state, NativeRegexState):
            raise TypeError("native regex state must be a NativeRegexState")
        if state._owner is not self._owner:
            raise GrammarStateError("native regex state belongs to another constraint")
        if not _is_native_instance(
            state._native_state,
            self._runtime.regex_state_type,
            operation="validating a native regex state",
        ):
            raise GrammarStateError("native regex state contains an invalid native handle")
        return state._native_state

    def _wrap_state(self, native_state: Any) -> NativeRegexState:
        if not _is_native_instance(
            native_state,
            self._runtime.regex_state_type,
            operation="validating a returned native regex state",
        ):
            raise GrammarRuntimeCompatibilityError(
                "native regex state operation returned an incompatible state value"
            )
        return NativeRegexState._create(self._owner, native_state)

    def _call_state_method(self, name: str, *args: Any) -> Any:
        method = _get_native_attribute(
            self._native_constraint,
            name,
            owner="native regex constraint",
        )
        if not callable(method):
            raise GrammarRuntimeCompatibilityError(
                f"native regex constraint does not provide callable {name}()"
            )
        try:
            return method(*args)
        except self._runtime.regex_state_error_type as exc:
            raise GrammarStateError(f"native regex {name}() failed: {exc}") from exc
        except Exception as exc:
            raise GrammarRuntimeCompatibilityError(
                f"native regex {name}() failed unexpectedly: {exc}"
            ) from exc

    @staticmethod
    def _require_none(result: Any, *, operation: str) -> None:
        if result is not None:
            raise GrammarRuntimeCompatibilityError(
                f"native regex {operation}() must return None"
            )


def compile_native_regex(
    vocabulary: Sequence[bytes],
    pattern: str,
    /,
) -> NativeRegexConstraint:
    """Compile one regex lazily through the independently packaged native runtime."""

    normalized_vocabulary = _normalize_grammar_vocabulary(vocabulary)
    pattern = _validate_regex_pattern(pattern)
    runtime = _load_native_grammar_runtime()
    try:
        native_constraint = runtime.compile_regex(normalized_vocabulary, pattern)
    except runtime.regex_compilation_error_type as exc:
        raise GrammarCompilationError(f"native regex compilation failed: {exc}") from exc
    except Exception as exc:
        raise GrammarRuntimeCompatibilityError(
            f"native regex compile_regex() failed unexpectedly: {exc}"
        ) from exc

    if not _is_native_instance(
        native_constraint,
        runtime.regex_constraint_type,
        operation="validating the compiled native regex constraint",
    ):
        raise GrammarRuntimeCompatibilityError(
            "native regex compile_regex() returned an incompatible constraint"
        )
    for method_name in _NATIVE_CONSTRAINT_METHODS:
        method = _get_native_attribute(
            native_constraint,
            method_name,
            owner="native regex constraint",
        )
        if not callable(method):
            raise GrammarRuntimeCompatibilityError(
                f"native regex constraint does not provide callable {method_name}()"
            )
    native_vocab_size = _get_native_attribute(
        native_constraint,
        "vocab_size",
        owner="native regex constraint",
    )
    if isinstance(native_vocab_size, bool) or not isinstance(native_vocab_size, int):
        raise GrammarRuntimeCompatibilityError(
            "native regex constraint vocab_size must be an integer"
        )
    if native_vocab_size != len(normalized_vocabulary):
        raise GrammarRuntimeCompatibilityError(
            f"native regex constraint vocabulary size {native_vocab_size} does not match "
            f"caller vocabulary size {len(normalized_vocabulary)}"
        )
    return NativeRegexConstraint(
        runtime,
        native_constraint,
        len(normalized_vocabulary),
        _construction_token=_CONSTRAINT_CONSTRUCTION_TOKEN,
    )


def _is_native_instance(value: Any, expected_type: type[Any], *, operation: str) -> bool:
    try:
        return isinstance(value, expected_type)
    except Exception as exc:
        raise GrammarRuntimeCompatibilityError(
            f"native regex failed while {operation}: {exc}"
        ) from exc
