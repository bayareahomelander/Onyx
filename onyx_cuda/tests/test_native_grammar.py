import importlib
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import onyx_cuda.native_grammar as native_grammar_module
from onyx_cuda import (
    NATIVE_GRAMMAR_ABI_VERSION,
    NATIVE_GRAMMAR_MODULE,
    GrammarRuntimeCompatibilityError,
    GrammarRuntimeUnavailableError,
    NativeGrammarRuntimeInfo,
    load_native_grammar_runtime,
)


class FakeNativeRegexCompilationError(Exception):
    pass


class FakeNativeRegexStateError(Exception):
    pass


class FakeNativeJsonCompilationError(Exception):
    pass


class FakeNativeJsonStateError(Exception):
    pass


class FakeNativeRegexState:
    pass


class FakeNativeRegexConstraint:
    pass


class FakeNativeJsonState:
    pass


class FakeNativeJsonConstraint:
    pass


class ExplodingAttributeProxy:
    def __init__(self, target, attribute):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_attribute", attribute)

    def __getattribute__(self, name):
        if name == object.__getattribute__(self, "_attribute"):
            raise RuntimeError(f"{name} attribute exploded")
        target = object.__getattribute__(self, "_target")
        return getattr(target, name)


def fake_native_module(*, runtime_version="0.1.0", abi_version=3):
    return SimpleNamespace(
        runtime_version=lambda: runtime_version,
        grammar_abi_version=lambda: abi_version,
        compile_regex=lambda vocabulary, pattern: FakeNativeRegexConstraint(),
        compile_json_schema=lambda vocabulary, schema: FakeNativeJsonConstraint(),
        _NativeRegexConstraint=FakeNativeRegexConstraint,
        _NativeRegexState=FakeNativeRegexState,
        NativeRegexCompilationError=FakeNativeRegexCompilationError,
        NativeRegexStateError=FakeNativeRegexStateError,
        _NativeJsonConstraint=FakeNativeJsonConstraint,
        _NativeJsonState=FakeNativeJsonState,
        NativeJsonCompilationError=FakeNativeJsonCompilationError,
        NativeJsonStateError=FakeNativeJsonStateError,
    )


def test_native_runtime_load_is_explicit_and_returns_immutable_identity(monkeypatch):
    requested = []

    def load(name):
        requested.append(name)
        return fake_native_module()

    monkeypatch.setattr(native_grammar_module.importlib, "import_module", load)

    info = load_native_grammar_runtime()

    assert requested == [NATIVE_GRAMMAR_MODULE]
    assert info == NativeGrammarRuntimeInfo(
        module_name="onyx_cuda._grammar_native",
        runtime_version="0.1.0",
        abi_version=NATIVE_GRAMMAR_ABI_VERSION,
    )
    with pytest.raises(FrozenInstanceError):
        info.abi_version = 3


@pytest.mark.parametrize("failure", [ImportError("missing"), OSError("load failed")])
def test_native_runtime_reports_typed_unavailable_error(monkeypatch, failure):
    monkeypatch.setattr(
        native_grammar_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(GrammarRuntimeUnavailableError, match="is unavailable"):
        load_native_grammar_runtime()


def test_native_runtime_wraps_unexpected_import_failure(monkeypatch):
    monkeypatch.setattr(
        native_grammar_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(RuntimeError("initialization failed")),
    )

    with pytest.raises(GrammarRuntimeUnavailableError, match="failed during import"):
        load_native_grammar_runtime()


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (SimpleNamespace(grammar_abi_version=lambda: 3), "runtime_version"),
        (SimpleNamespace(runtime_version=lambda: "0.1.0"), "grammar_abi_version"),
        (fake_native_module(runtime_version=""), "nonempty string"),
        (fake_native_module(abi_version=True), "must return an integer"),
        (fake_native_module(abi_version=2), "does not match required ABI"),
    ],
)
def test_native_runtime_rejects_malformed_or_incompatible_module(
    monkeypatch,
    module,
    message,
):
    monkeypatch.setattr(native_grammar_module.importlib, "import_module", lambda name: module)

    with pytest.raises(GrammarRuntimeCompatibilityError, match=message):
        load_native_grammar_runtime()


def test_native_runtime_wraps_identity_function_failure(monkeypatch):
    module = fake_native_module()
    module.runtime_version = lambda: (_ for _ in ()).throw(RuntimeError("identity failed"))
    monkeypatch.setattr(native_grammar_module.importlib, "import_module", lambda name: module)

    with pytest.raises(GrammarRuntimeCompatibilityError, match="identity failed"):
        load_native_grammar_runtime()


@pytest.mark.parametrize(
    "attribute",
    [
        "runtime_version",
        "compile_regex",
        "_NativeRegexState",
        "compile_json_schema",
        "_NativeJsonState",
    ],
)
def test_native_runtime_wraps_attribute_access_failure(monkeypatch, attribute):
    module = ExplodingAttributeProxy(fake_native_module(), attribute)
    monkeypatch.setattr(native_grammar_module.importlib, "import_module", lambda name: module)

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match=rf"{attribute} access failed",
    ) as captured:
        load_native_grammar_runtime()

    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("compile_regex", None, "compile_regex"),
        ("_NativeRegexConstraint", None, "_NativeRegexConstraint"),
        ("_NativeRegexState", None, "_NativeRegexState"),
        ("NativeRegexCompilationError", None, "NativeRegexCompilationError"),
        ("NativeRegexStateError", None, "NativeRegexStateError"),
        ("compile_json_schema", None, "compile_json_schema"),
        ("_NativeJsonConstraint", None, "_NativeJsonConstraint"),
        ("_NativeJsonState", None, "_NativeJsonState"),
        ("NativeJsonCompilationError", None, "NativeJsonCompilationError"),
        ("NativeJsonStateError", None, "NativeJsonStateError"),
    ],
)
def test_native_runtime_rejects_missing_abi_surface(
    monkeypatch,
    attribute,
    value,
    message,
):
    module = fake_native_module()
    setattr(module, attribute, value)
    monkeypatch.setattr(native_grammar_module.importlib, "import_module", lambda name: module)

    with pytest.raises(GrammarRuntimeCompatibilityError, match=message):
        load_native_grammar_runtime()


@pytest.mark.parametrize(
    "attribute",
    ["NativeRegexStateError", "NativeJsonStateError"],
)
def test_native_runtime_rejects_control_flow_exception_types(monkeypatch, attribute):
    module = fake_native_module()
    setattr(module, attribute, KeyboardInterrupt)
    monkeypatch.setattr(native_grammar_module.importlib, "import_module", lambda name: module)

    with pytest.raises(GrammarRuntimeCompatibilityError, match=attribute):
        load_native_grammar_runtime()


def test_importing_native_loader_module_does_not_request_extension(monkeypatch):
    requested = []
    original = importlib.import_module

    def record(name, *args, **kwargs):
        requested.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", record)
    sys.modules.pop("onyx_cuda.native_grammar", None)
    importlib.import_module("onyx_cuda.native_grammar")

    assert NATIVE_GRAMMAR_MODULE not in requested


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({"module_name": "", "runtime_version": "0.1.0", "abi_version": 2}, ValueError, "module"),
        ({"module_name": None, "runtime_version": "0.1.0", "abi_version": 2}, TypeError, "string"),
        ({"module_name": "x", "runtime_version": "", "abi_version": 2}, ValueError, "version"),
        ({"module_name": "x", "runtime_version": None, "abi_version": 2}, TypeError, "string"),
        ({"module_name": "x", "runtime_version": "1", "abi_version": True}, TypeError, "integer"),
        ({"module_name": "x", "runtime_version": "1", "abi_version": 0}, ValueError, "greater"),
    ],
)
def test_runtime_info_rejects_invalid_values(values, error, message):
    with pytest.raises(error, match=message):
        NativeGrammarRuntimeInfo(**values)
