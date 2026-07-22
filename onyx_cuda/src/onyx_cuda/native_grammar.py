"""Lazy loader for the independently packaged Windows grammar native module."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .grammar import (
    GrammarRuntimeCompatibilityError,
    GrammarRuntimeUnavailableError,
)


NATIVE_GRAMMAR_ABI_VERSION = 3
NATIVE_GRAMMAR_MODULE = "onyx_cuda._grammar_native"


@dataclass(frozen=True, slots=True)
class NativeGrammarRuntimeInfo:
    """Identity returned by the qualified native grammar runtime."""

    module_name: str
    runtime_version: str
    abi_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.module_name, str):
            raise TypeError("native grammar module name must be a string")
        if not self.module_name.strip():
            raise ValueError("native grammar module name cannot be empty")
        if not isinstance(self.runtime_version, str):
            raise TypeError("native grammar runtime version must be a string")
        if not self.runtime_version.strip():
            raise ValueError("native grammar runtime version cannot be empty")
        if isinstance(self.abi_version, bool) or not isinstance(self.abi_version, int):
            raise TypeError("native grammar ABI version must be an integer")
        if self.abi_version <= 0:
            raise ValueError("native grammar ABI version must be greater than zero")


@dataclass(frozen=True, slots=True)
class _NativeGrammarRuntime:
    module: Any
    info: NativeGrammarRuntimeInfo
    compile_regex: Any
    regex_constraint_type: type[Any]
    regex_state_type: type[Any]
    regex_compilation_error_type: type[Exception]
    regex_state_error_type: type[Exception]
    compile_json_schema: Any
    json_constraint_type: type[Any]
    json_state_type: type[Any]
    json_compilation_error_type: type[Exception]
    json_state_error_type: type[Exception]


def load_native_grammar_runtime() -> NativeGrammarRuntimeInfo:
    """Load and validate the optional ABI-3 native runtime on explicit request."""

    return _load_native_grammar_runtime().info


def _load_native_grammar_runtime() -> _NativeGrammarRuntime:
    try:
        module = importlib.import_module(NATIVE_GRAMMAR_MODULE)
    except (ImportError, OSError) as exc:
        raise GrammarRuntimeUnavailableError(
            f"native grammar runtime {NATIVE_GRAMMAR_MODULE!r} is unavailable: {exc}"
        ) from exc
    except Exception as exc:
        raise GrammarRuntimeUnavailableError(
            f"native grammar runtime {NATIVE_GRAMMAR_MODULE!r} failed during import: {exc}"
        ) from exc

    runtime_version = _call_runtime_field(module, "runtime_version")
    abi_version = _call_runtime_field(module, "grammar_abi_version")
    if not isinstance(runtime_version, str) or not runtime_version.strip():
        raise GrammarRuntimeCompatibilityError(
            "native grammar runtime_version() must return a nonempty string"
        )
    if isinstance(abi_version, bool) or not isinstance(abi_version, int):
        raise GrammarRuntimeCompatibilityError(
            "native grammar grammar_abi_version() must return an integer"
        )
    if abi_version != NATIVE_GRAMMAR_ABI_VERSION:
        raise GrammarRuntimeCompatibilityError(
            f"native grammar ABI {abi_version} does not match required ABI "
            f"{NATIVE_GRAMMAR_ABI_VERSION}"
        )

    compile_regex = _get_native_attribute(
        module,
        "compile_regex",
        owner="native grammar runtime",
    )
    if not callable(compile_regex):
        raise GrammarRuntimeCompatibilityError(
            "native grammar ABI 3 does not provide callable compile_regex()"
        )

    constraint_type = _require_native_type(module, "_NativeRegexConstraint")
    state_type = _require_native_type(module, "_NativeRegexState")
    compilation_error_type = _require_native_exception_type(
        module, "NativeRegexCompilationError"
    )
    state_error_type = _require_native_exception_type(module, "NativeRegexStateError")
    compile_json_schema = _get_native_attribute(
        module,
        "compile_json_schema",
        owner="native grammar runtime",
    )
    if not callable(compile_json_schema):
        raise GrammarRuntimeCompatibilityError(
            "native grammar ABI 3 does not provide callable compile_json_schema()"
        )
    json_constraint_type = _require_native_type(module, "_NativeJsonConstraint")
    json_state_type = _require_native_type(module, "_NativeJsonState")
    json_compilation_error_type = _require_native_exception_type(
        module, "NativeJsonCompilationError"
    )
    json_state_error_type = _require_native_exception_type(module, "NativeJsonStateError")
    info = NativeGrammarRuntimeInfo(
        module_name=NATIVE_GRAMMAR_MODULE,
        runtime_version=runtime_version,
        abi_version=abi_version,
    )
    return _NativeGrammarRuntime(
        module=module,
        info=info,
        compile_regex=compile_regex,
        regex_constraint_type=constraint_type,
        regex_state_type=state_type,
        regex_compilation_error_type=compilation_error_type,
        regex_state_error_type=state_error_type,
        compile_json_schema=compile_json_schema,
        json_constraint_type=json_constraint_type,
        json_state_type=json_state_type,
        json_compilation_error_type=json_compilation_error_type,
        json_state_error_type=json_state_error_type,
    )


def _call_runtime_field(module: Any, name: str) -> Any:
    value = _get_native_attribute(module, name, owner="native grammar runtime")
    if not callable(value):
        raise GrammarRuntimeCompatibilityError(
            f"native grammar runtime does not provide callable {name}()"
        )
    try:
        return value()
    except Exception as exc:
        raise GrammarRuntimeCompatibilityError(
            f"native grammar {name}() failed: {exc}"
        ) from exc


def _require_native_type(module: Any, name: str) -> type[Any]:
    value = _get_native_attribute(module, name, owner="native grammar runtime")
    if not isinstance(value, type):
        raise GrammarRuntimeCompatibilityError(
            f"native grammar runtime does not provide type {name}"
        )
    return value


def _require_native_exception_type(module: Any, name: str) -> type[Exception]:
    value = _get_native_attribute(module, name, owner="native grammar runtime")
    if not isinstance(value, type) or not issubclass(value, Exception):
        raise GrammarRuntimeCompatibilityError(
            f"native grammar runtime does not provide exception type {name}"
        )
    return value


def _get_native_attribute(value: Any, name: str, *, owner: str) -> Any:
    """Read one ABI attribute without leaking failures from malformed native objects."""

    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise GrammarRuntimeCompatibilityError(
            f"{owner} does not provide attribute {name}"
        ) from exc
    except Exception as exc:
        raise GrammarRuntimeCompatibilityError(
            f"{owner} attribute {name} access failed: {exc}"
        ) from exc
