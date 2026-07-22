"""Installed-wheel qualification for the native JSON adapter.

This file is intentionally not named ``test_*.py``. The source-tree pytest suite remains
native-optional; qualification invokes this script with a clean environment containing only the
built wheel and supplies the committed D19 fixture directory on ``sys.path``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import onyx_cuda
from json_parity_fixtures import JSON_PARITY_CASES, JSON_SCHEMA_OBSERVATIONS
from onyx_cuda import (
    GrammarCompilationError,
    GrammarRuntimeCompatibilityError,
    GrammarStateError,
    compile_native_json_schema,
    load_native_grammar_runtime,
)


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
_WINDOWS_VALID_TOKEN_OVERRIDES = {
    # RFC 8259 does not permit an array to close immediately after a comma.
    ("root_array_typed_items_and_bounds", (0, 2, 4)): (2, 3, 5),
    # A sign and decimal point require the next strict number phase, not any numeric fragment.
    ("root_number_fraction_and_exponent", (0,)): (1, 3, 7),
    ("root_number_fraction_and_exponent", (1, 2)): (1, 3, 7),
    # maxLength has been reached, so another escaped code point cannot begin.
    ("valid_escaped_quote", (0, 1)): (0,),
}


def main() -> None:
    _require_installed_package()
    if "onyx_cuda._grammar_native" in sys.modules:
        raise AssertionError("normal onyx_cuda import loaded the native grammar extension")
    _require_forbidden_runtimes_absent()

    info = load_native_grammar_runtime()
    if (info.runtime_version, info.abi_version) != ("0.1.0", 3):
        raise AssertionError(f"unexpected native runtime identity: {info!r}")
    _require_json_surface()

    supported_cases = tuple(
        case for case in JSON_PARITY_CASES if case.disposition == "windows_parity"
    )
    for case in supported_cases:
        _check_parity_case(case)
    _check_schema_observations()
    _check_windows_corrections()
    _check_state_lifecycle()
    _check_repeated_lifecycle()
    _require_forbidden_runtimes_absent()

    print(
        "installed-wheel JSON qualification passed: "
        f"parity_cases={len(supported_cases)} "
        f"schema_observations={len(JSON_SCHEMA_OBSERVATIONS)} "
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


def _require_json_surface() -> None:
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
            "ABI-3 JSON surface is incomplete: "
            f"public={missing_public!r} native={missing_native!r}"
        )
    if hasattr(onyx_cuda, "NativeGrammarCompiler"):
        raise AssertionError("ABI-3 package unexpectedly exposes a complete native compiler")


def _check_parity_case(case) -> None:
    constraint = compile_native_json_schema(case.vocabulary, case.schema)
    initial = constraint.init_state()
    initial_snapshot = _snapshot(constraint, initial)

    for expectation in case.states:
        state, children = _advance_path(constraint, initial, expectation.token_ids)
        actual = _snapshot(constraint, state)
        expected = _windows_expectation(case, expectation)
        if actual != expected:
            raise AssertionError(
                f"JSON parity mismatch for {case.name} tokens={expectation.token_ids}: "
                f"actual={actual!r} expected={expected!r}"
            )
        constraint.release_states(children)

    for rejected in case.rejected_transitions:
        parent, children = _advance_path(
            constraint,
            initial,
            rejected.prefix_token_ids,
        )
        before = _snapshot(constraint, parent)
        if rejected.token_id in before[0]:
            raise AssertionError(
                f"rejected JSON token is advertised as valid for {case.name}: "
                f"prefix={rejected.prefix_token_ids!r} token={rejected.token_id}"
            )
        try:
            constraint.advance_state(parent, rejected.token_id)
        except GrammarStateError:
            pass
        else:
            raise AssertionError(
                f"rejected JSON transition returned a child for {case.name}: "
                f"prefix={rejected.prefix_token_ids!r} token={rejected.token_id}"
            )
        after = _snapshot(constraint, parent)
        if after != before:
            raise AssertionError(
                f"rejected JSON transition mutated its parent for {case.name}: "
                f"before={before!r} after={after!r}"
            )
        constraint.release_states(children)

    if _snapshot(constraint, initial) != initial_snapshot:
        raise AssertionError(f"JSON parity traversal mutated the parent for {case.name}")
    constraint.release_state(initial)
    constraint.release_state(initial)


def _check_schema_observations() -> None:
    basic_value_errors = {"malformed_json", "non_object_schema"}
    for observation in JSON_SCHEMA_OBSERVATIONS:
        if observation.selected_windows_policy == "support":
            constraint = compile_native_json_schema((b"{", b"}"), observation.schema)
            state = constraint.init_state()
            constraint.release_state(state)
            continue
        expected_error = ValueError if observation.name in basic_value_errors else GrammarCompilationError
        try:
            compile_native_json_schema((b"{", b"}"), observation.schema)
        except expected_error:
            continue
        except Exception as exc:
            raise AssertionError(
                f"schema observation {observation.name} raised {type(exc).__name__}; "
                f"expected {expected_error.__name__}"
            ) from exc
        raise AssertionError(
            f"schema observation {observation.name} compiled but should raise "
            f"{expected_error.__name__}"
        )


def _check_windows_corrections() -> None:
    invalid_escape = compile_native_json_schema(
        (b'"', b"\\x", b"a", b""),
        '{"type":"string","minLength":1,"maxLength":1}',
    )
    quote = invalid_escape.advance_state(invalid_escape.init_state(), 0)
    _require_rejected(invalid_escape, quote, 1, label="invalid JSON escape")

    strict_number = compile_native_json_schema(
        (b"01", b"-01", b"1", b""),
        '{"type":"number"}',
    )
    number_root = strict_number.init_state()
    _require_rejected(strict_number, number_root, 0, label="leading-zero number")
    _require_rejected(strict_number, number_root, 1, label="negative leading-zero number")

    unicode_length = compile_native_json_schema(
        ('"é"'.encode(), '"éé"'.encode(), b""),
        '{"type":"string","minLength":2,"maxLength":2}',
    )
    unicode_root = unicode_length.init_state()
    _require_rejected(unicode_length, unicode_root, 0, label="Unicode code-point length")
    two_code_points = unicode_length.advance_state(unicode_root, 1)
    if not unicode_length.is_match_state(two_code_points):
        raise AssertionError("two Unicode code points did not satisfy length two")

    numeric_enum = compile_native_json_schema(
        (b"2", b"3", b"null", b""),
        '{"enum":[2,3,null]}',
    )
    enum_value = numeric_enum.advance_state(numeric_enum.init_state(), 0)
    if not numeric_enum.is_match_state(enum_value):
        raise AssertionError("single-byte numeric enum did not complete")

    decoded_pattern = compile_native_json_schema(
        (br'"\n"', b""),
        r'{"type":"string","pattern":"^\\n$","minLength":1,"maxLength":1}',
    )
    decoded_value = decoded_pattern.advance_state(decoded_pattern.init_state(), 0)
    if not decoded_pattern.is_match_state(decoded_value):
        raise AssertionError("decoded newline did not satisfy the decoded-value pattern")
    designator_pattern = compile_native_json_schema(
        (br'"\n"', b""),
        r'{"type":"string","pattern":"^n$","minLength":1,"maxLength":1}',
    )
    _require_rejected(
        designator_pattern,
        designator_pattern.init_state(),
        0,
        label="escape-designator pattern",
    )

    trailing_whitespace = compile_native_json_schema(
        (b"true \n\t", b""),
        '{"type":"boolean"}',
    )
    completed = trailing_whitespace.advance_state(trailing_whitespace.init_state(), 0)
    if not trailing_whitespace.is_match_state(completed):
        raise AssertionError("trailing structural whitespace did not preserve a match")


def _check_state_lifecycle() -> None:
    vocabulary = (b"null", b"x", b"")
    schema = '{"type":"null"}'
    first = compile_native_json_schema(vocabulary, schema)
    second = compile_native_json_schema(vocabulary, schema)
    initial = first.init_state()
    foreign = second.init_state()
    matched = first.advance_state(initial, 0)
    empty = first.advance_state(initial, 2)

    if _snapshot(first, matched) != ((), True, False):
        raise AssertionError("JSON null token did not create the expected match state")
    if _snapshot(first, empty) != _snapshot(first, initial) or empty == initial:
        raise AssertionError("empty JSON token did not create an independent equivalent child")
    _require_rejected(first, initial, 1, label="invalid JSON token")

    try:
        first.release_states((initial, foreign))
    except GrammarStateError:
        pass
    else:
        raise AssertionError("bulk JSON release accepted a foreign state")
    if first.get_valid_token_ids(initial) != (0,):
        raise AssertionError("failed bulk JSON release removed a valid state")

    first.release_state(initial)
    first.release_state(initial)
    try:
        first.get_valid_token_ids(initial)
    except GrammarStateError:
        pass
    else:
        raise AssertionError("released JSON state remained queryable")

    stale = first.init_state()
    first.reset()
    try:
        first.is_match_state(stale)
    except GrammarStateError:
        pass
    else:
        raise AssertionError("JSON reset did not invalidate the earlier state epoch")

    fresh = first.init_state()
    try:
        first.advance_state(fresh, len(vocabulary))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range JSON token ID did not raise ValueError")
    try:
        first.advance_state(fresh, True)
    except TypeError:
        pass
    else:
        raise AssertionError("boolean JSON token ID did not raise TypeError")


def _check_repeated_lifecycle() -> None:
    constraint = compile_native_json_schema((b"null", b""), '{"type":"null"}')
    for _ in range(LIFECYCLE_ITERATIONS):
        initial = constraint.init_state()
        matched = constraint.advance_state(initial, 0)
        empty = constraint.advance_state(initial, 1)
        constraint.release_states((initial, matched, empty))
        constraint.reset()


def _advance_path(constraint, initial, token_ids):
    state = initial
    children = []
    for token_id in token_ids:
        state = constraint.advance_state(state, token_id)
        children.append(state)
    return state, tuple(children)


def _windows_expectation(case, expectation):
    valid_token_ids = _WINDOWS_VALID_TOKEN_OVERRIDES.get(
        (case.name, expectation.token_ids),
        expectation.valid_token_ids,
    )
    if expectation.is_match:
        trailing_whitespace_ids = tuple(
            token_id
            for token_id, token_bytes in enumerate(case.vocabulary)
            if token_bytes and all(byte in b" \t\n\r" for byte in token_bytes)
        )
        valid_token_ids = tuple(sorted(set(valid_token_ids + trailing_whitespace_ids)))
    return valid_token_ids, expectation.is_match, expectation.is_dead


def _require_rejected(constraint, parent, token_id, *, label) -> None:
    before = _snapshot(constraint, parent)
    try:
        constraint.advance_state(parent, token_id)
    except GrammarStateError:
        pass
    else:
        raise AssertionError(f"{label} returned a child instead of raising GrammarStateError")
    after = _snapshot(constraint, parent)
    if after != before:
        raise AssertionError(f"{label} mutated its parent: before={before!r} after={after!r}")


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
        raise AssertionError(f"native JSON qualification failed: {exc}") from exc
