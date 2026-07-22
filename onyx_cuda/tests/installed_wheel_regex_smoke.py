"""Installed-wheel regression qualification for the native regex adapter under ABI 3.

This file is intentionally not named ``test_*.py``. The source-tree pytest suite must remain
native-optional; qualification invokes this script with the Python executable from a clean
environment containing only the built wheel.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import onyx_cuda
from onyx_cuda import (
    GrammarCompilationError,
    GrammarRuntimeCompatibilityError,
    GrammarStateError,
    compile_native_regex,
    load_native_grammar_runtime,
)

from regex_parity_fixtures import INVALID_REGEX_PATTERNS, REGEX_PARITY_CASES


FORBIDDEN_RUNTIME_PREFIXES = (
    "onyx",
    "mlx",
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "onnxruntime",
    "psutil",
)
LIFECYCLE_ITERATIONS = 1_000


def main() -> None:
    _require_installed_package()
    if "onyx_cuda._grammar_native" in sys.modules:
        raise AssertionError("normal onyx_cuda import loaded the native grammar extension")
    _require_forbidden_runtimes_absent()

    info = load_native_grammar_runtime()
    if (info.runtime_version, info.abi_version) != ("0.1.0", 3):
        raise AssertionError(f"unexpected native runtime identity: {info!r}")
    _require_json_surface_present()

    for case in REGEX_PARITY_CASES:
        _check_parity_case(case)
    _check_invalid_patterns()
    _check_state_lifecycle()
    _check_repeated_lifecycle()
    _require_forbidden_runtimes_absent()

    print(
        "installed-wheel regex qualification passed: "
        f"parity_cases={len(REGEX_PARITY_CASES)} "
        f"invalid_patterns={len(INVALID_REGEX_PATTERNS)} "
        f"lifecycle_iterations={LIFECYCLE_ITERATIONS} "
        f"runtime={info.runtime_version} abi={info.abi_version}"
    )


def _require_installed_package() -> None:
    package_path = Path(onyx_cuda.__file__).resolve()
    environment = Path(sys.prefix).resolve()
    if not package_path.is_relative_to(environment):
        raise AssertionError(
            f"qualification imported onyx_cuda outside the clean environment: {package_path}"
        )


def _require_json_surface_present() -> None:
    native = importlib.import_module("onyx_cuda._grammar_native")
    public_symbols = (
        "compile_native_json_schema",
        "NativeJsonConstraint",
        "NativeJsonState",
    )
    native_symbols = (
        "compile_json_schema",
        "_NativeJsonConstraint",
        "_NativeJsonState",
        "NativeJsonCompilationError",
        "NativeJsonStateError",
    )
    missing_public = [name for name in public_symbols if not hasattr(onyx_cuda, name)]
    missing_native = [name for name in native_symbols if not hasattr(native, name)]
    if missing_public or missing_native:
        raise AssertionError(
            "ABI-3 runtime is missing JSON symbols: "
            f"public={missing_public!r} native={missing_native!r}"
        )
    if hasattr(onyx_cuda, "NativeGrammarCompiler"):
        raise AssertionError("ABI-3 package unexpectedly exposes a complete native compiler")


def _check_parity_case(case) -> None:
    constraint = compile_native_regex(case.vocabulary, case.pattern)
    initial = constraint.init_state()
    initial_snapshot = _snapshot(constraint, initial)

    for expectation in case.states:
        state = initial
        children = []
        for token_id in expectation.token_ids:
            state = constraint.advance_state(state, token_id)
            children.append(state)
        actual = _snapshot(constraint, state)
        expected = (
            expectation.valid_token_ids,
            expectation.is_match,
            expectation.is_dead,
        )
        if actual != expected:
            raise AssertionError(
                f"parity mismatch for {case.name} tokens={expectation.token_ids}: "
                f"actual={actual!r} expected={expected!r}"
            )
        constraint.release_states(children)

    if _snapshot(constraint, initial) != initial_snapshot:
        raise AssertionError(f"parity traversal mutated the parent for {case.name}")
    constraint.release_state(initial)
    constraint.release_state(initial)


def _check_invalid_patterns() -> None:
    for pattern in INVALID_REGEX_PATTERNS:
        try:
            compile_native_regex((b"a",), pattern)
        except GrammarCompilationError:
            continue
        raise AssertionError(f"invalid regex pattern compiled successfully: {pattern!r}")


def _check_state_lifecycle() -> None:
    first = compile_native_regex((b"a", b"x", b""), "a")
    second = compile_native_regex((b"a", b"x", b""), "a")
    initial = first.init_state()
    foreign = second.init_state()
    dead = first.advance_state(initial, 1)
    dead_child = first.advance_state(dead, 0)
    empty = first.advance_state(initial, 2)

    if _snapshot(first, dead) != ((), False, True):
        raise AssertionError("invalid token did not create the expected dead child")
    if _snapshot(first, dead_child) != ((), False, True):
        raise AssertionError("advancing a dead state did not create another dead child")
    if _snapshot(first, empty) != _snapshot(first, initial) or empty == initial:
        raise AssertionError("empty token did not create an independent equivalent child")

    try:
        first.release_states((initial, foreign))
    except GrammarStateError:
        pass
    else:
        raise AssertionError("bulk release accepted a foreign state")
    if first.get_valid_token_ids(initial) != (0,):
        raise AssertionError("failed bulk release removed a previously valid state")

    first.release_state(initial)
    first.release_state(initial)
    try:
        first.get_valid_token_ids(initial)
    except GrammarStateError:
        pass
    else:
        raise AssertionError("released state remained queryable")

    stale = first.init_state()
    first.reset()
    try:
        first.is_match_state(stale)
    except GrammarStateError:
        pass
    else:
        raise AssertionError("reset did not invalidate the earlier state epoch")

    fresh = first.init_state()
    try:
        first.advance_state(fresh, len((b"a", b"x", b"")))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range token ID did not raise ValueError")
    try:
        first.advance_state(fresh, True)
    except TypeError:
        pass
    else:
        raise AssertionError("boolean token ID did not raise TypeError")


def _check_repeated_lifecycle() -> None:
    constraint = compile_native_regex((b"a", b"x"), "a")
    for _ in range(LIFECYCLE_ITERATIONS):
        initial = constraint.init_state()
        matched = constraint.advance_state(initial, 0)
        dead = constraint.advance_state(initial, 1)
        constraint.release_states((initial, matched, dead))
        constraint.reset()


def _snapshot(constraint, state) -> tuple[tuple[int, ...], bool, bool]:
    return (
        constraint.get_valid_token_ids(state),
        constraint.is_match_state(state),
        constraint.is_dead_state(state),
    )


def _require_forbidden_runtimes_absent() -> None:
    loaded = tuple(sys.modules)
    for prefix in FORBIDDEN_RUNTIME_PREFIXES:
        if any(name == prefix or name.startswith(f"{prefix}.") for name in loaded):
            raise AssertionError(f"installed-wheel qualification imported forbidden {prefix!r}")


if __name__ == "__main__":
    try:
        main()
    except (GrammarRuntimeCompatibilityError, GrammarStateError) as exc:
        raise AssertionError(f"native regex qualification failed: {exc}") from exc
