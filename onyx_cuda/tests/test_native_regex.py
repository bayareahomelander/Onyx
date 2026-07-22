from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import onyx_cuda.native_grammar as native_grammar_module
from onyx_cuda import (
    GrammarCompilationError,
    GrammarConstraint,
    GrammarRuntimeCompatibilityError,
    GrammarStateError,
    NativeRegexConstraint,
    NativeRegexState,
    compile_native_regex,
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
    def __init__(self, owner, epoch, value):
        self.owner = owner
        self.epoch = epoch
        self.value = value


class ExplodingIterable:
    def __iter__(self):
        raise RuntimeError("native result iteration exploded")


_DEFAULT_VALID_OUTPUT = object()


class FakeNativeRegexConstraint:
    def __init__(self, vocabulary, *, valid_output=_DEFAULT_VALID_OUTPUT):
        self.vocabulary = tuple(vocabulary)
        self.vocab_size = len(self.vocabulary)
        self.epoch = 1
        self.next_value = 1
        self.states = {}
        self.known = set()
        self.valid_output = valid_output

    def init_state(self):
        return self._insert("initial")

    def advance_state(self, state, token_id):
        logical = self._logical(state)
        token_bytes = self.vocabulary[token_id]
        if not token_bytes:
            target = logical
        elif logical == "dead":
            target = "dead"
        elif logical == "initial" and token_bytes == b"a":
            target = "matched"
        else:
            target = "dead"
        return self._insert(target)

    def get_valid_token_ids(self, state):
        if self.valid_output is not _DEFAULT_VALID_OUTPUT:
            return self.valid_output
        logical = self._logical(state)
        if logical == "initial":
            return [
                token_id
                for token_id, token_bytes in enumerate(self.vocabulary)
                if token_bytes == b"a"
            ]
        return []

    def is_match_state(self, state):
        return self._logical(state) == "matched"

    def is_dead_state(self, state):
        return self._logical(state) == "dead"

    def release_state(self, state):
        self._validate_known(state)
        self.states.pop(state.value, None)

    def release_states(self, states):
        for state in states:
            self._validate_known(state)
        for state in states:
            self.states.pop(state.value, None)

    def reset(self):
        self.epoch += 1
        self.next_value = 1
        self.states.clear()
        self.known.clear()

    def _insert(self, logical):
        value = self.next_value
        self.next_value += 1
        self.states[value] = logical
        self.known.add(value)
        return FakeNativeRegexState(self, self.epoch, value)

    def _logical(self, state):
        self._validate_known(state)
        try:
            return self.states[state.value]
        except KeyError as exc:
            raise FakeNativeRegexStateError("state is released") from exc

    def _validate_known(self, state):
        if state.owner is not self:
            raise FakeNativeRegexStateError("foreign state")
        if state.epoch != self.epoch:
            raise FakeNativeRegexStateError("stale state")
        if state.value not in self.known:
            raise FakeNativeRegexStateError("unknown state")


class FakeNativeJsonState:
    pass


class FakeNativeJsonConstraint:
    pass


class ExplodingAttributeConstraint(FakeNativeRegexConstraint):
    def __init__(self, vocabulary):
        self.exploding_attribute = None
        super().__init__(vocabulary)

    def __getattribute__(self, name):
        if name not in {"exploding_attribute", "__dict__"}:
            exploding_attribute = object.__getattribute__(self, "exploding_attribute")
            if name == exploding_attribute:
                raise RuntimeError(f"{name} attribute exploded")
        return super().__getattribute__(name)


class ExplodingInstanceCheck(type):
    def __instancecheck__(cls, instance):
        raise RuntimeError("native state instance validation exploded")


class ExplodingNativeStateType(metaclass=ExplodingInstanceCheck):
    pass


def fake_native_module(*, compile_factory=None, abi_version=3):
    compile_calls = []

    def compile_regex(vocabulary, pattern):
        compile_calls.append((tuple(vocabulary), pattern))
        if pattern == "(":
            raise FakeNativeRegexCompilationError("invalid pattern")
        if compile_factory is None:
            return FakeNativeRegexConstraint(vocabulary)
        return compile_factory(vocabulary, pattern)

    return SimpleNamespace(
        runtime_version=lambda: "0.1.0",
        grammar_abi_version=lambda: abi_version,
        compile_regex=compile_regex,
        _NativeRegexConstraint=FakeNativeRegexConstraint,
        _NativeRegexState=FakeNativeRegexState,
        NativeRegexCompilationError=FakeNativeRegexCompilationError,
        NativeRegexStateError=FakeNativeRegexStateError,
        compile_json_schema=lambda vocabulary, schema: FakeNativeJsonConstraint(),
        _NativeJsonConstraint=FakeNativeJsonConstraint,
        _NativeJsonState=FakeNativeJsonState,
        NativeJsonCompilationError=FakeNativeJsonCompilationError,
        NativeJsonStateError=FakeNativeJsonStateError,
        compile_calls=compile_calls,
    )


def install_fake_runtime(monkeypatch, module=None):
    module = module or fake_native_module()
    monkeypatch.setattr(native_grammar_module.importlib, "import_module", lambda name: module)
    return module


def test_native_regex_satisfies_protocol_and_preserves_parent(monkeypatch):
    module = install_fake_runtime(monkeypatch)
    constraint = compile_native_regex((b"a", b"x", b""), "a")
    initial = constraint.init_state()
    matched = constraint.advance_state(initial, 0)
    dead = constraint.advance_state(initial, 1)
    empty = constraint.advance_state(initial, 2)

    assert isinstance(constraint, GrammarConstraint)
    assert isinstance(constraint, NativeRegexConstraint)
    assert constraint.vocab_size == 3
    assert constraint.grammar_type == "regex"
    assert constraint.get_valid_token_ids(initial) == (0,)
    assert constraint.is_match_state(matched)
    assert constraint.is_dead_state(dead)
    assert constraint.get_valid_token_ids(empty) == constraint.get_valid_token_ids(initial)
    assert initial != empty
    assert module.compile_calls == [((b"a", b"x", b""), "a")]


def test_factory_validates_inputs_before_requesting_native_runtime(monkeypatch):
    requested = []
    monkeypatch.setattr(
        native_grammar_module.importlib,
        "import_module",
        lambda name: requested.append(name),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        compile_native_regex((), "a")
    with pytest.raises(TypeError, match="entry 1 must be bytes"):
        compile_native_regex((b"a", "b"), "a")
    with pytest.raises(ValueError, match="pattern cannot be empty"):
        compile_native_regex((b"a",), "")

    assert requested == []


def test_native_compilation_and_state_errors_are_translated(monkeypatch):
    install_fake_runtime(monkeypatch)
    with pytest.raises(GrammarCompilationError, match="invalid pattern"):
        compile_native_regex((b"a",), "(")

    constraint = compile_native_regex((b"a",), "a")
    state = constraint.init_state()
    constraint.release_state(state)
    constraint.release_state(state)
    with pytest.raises(GrammarStateError, match="released"):
        constraint.get_valid_token_ids(state)


def test_abi_two_is_rejected_before_native_compilation(monkeypatch):
    module = fake_native_module(abi_version=2)
    install_fake_runtime(monkeypatch, module)

    with pytest.raises(GrammarRuntimeCompatibilityError, match="does not match required ABI"):
        compile_native_regex((b"a",), "a")

    assert module.compile_calls == []


def test_malformed_native_constraint_is_rejected(monkeypatch):
    def missing_method(vocabulary, pattern):
        constraint = FakeNativeRegexConstraint(vocabulary)
        constraint.init_state = None
        return constraint

    module = fake_native_module(compile_factory=missing_method)
    install_fake_runtime(monkeypatch, module)
    with pytest.raises(GrammarRuntimeCompatibilityError, match="init_state"):
        compile_native_regex((b"a",), "a")

    def wrong_vocabulary(vocabulary, pattern):
        constraint = FakeNativeRegexConstraint(vocabulary)
        constraint.vocab_size += 1
        return constraint

    module = fake_native_module(compile_factory=wrong_vocabulary)
    install_fake_runtime(monkeypatch, module)
    with pytest.raises(GrammarRuntimeCompatibilityError, match="does not match"):
        compile_native_regex((b"a",), "a")


def test_native_constraint_attribute_failures_are_wrapped(monkeypatch):
    holder = []

    def build_constraint(vocabulary, pattern):
        constraint = ExplodingAttributeConstraint(vocabulary)
        holder.append(constraint)
        return constraint

    module = fake_native_module(compile_factory=build_constraint)
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_regex((b"a",), "a")
    state = constraint.init_state()
    holder[0].exploding_attribute = "get_valid_token_ids"

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="get_valid_token_ids access failed",
    ) as captured:
        constraint.get_valid_token_ids(state)

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_native_constraint_initialization_failures_are_wrapped(monkeypatch):
    def build_constraint(vocabulary, pattern):
        constraint = ExplodingAttributeConstraint(vocabulary)
        constraint.exploding_attribute = "vocab_size"
        return constraint

    module = fake_native_module(compile_factory=build_constraint)
    install_fake_runtime(monkeypatch, module)

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="vocab_size access failed",
    ) as captured:
        compile_native_regex((b"a",), "a")

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_native_result_iteration_failures_are_wrapped(monkeypatch):
    module = fake_native_module(
        compile_factory=lambda vocabulary, pattern: FakeNativeRegexConstraint(
            vocabulary,
            valid_output=ExplodingIterable(),
        )
    )
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_regex((b"a",), "a")

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="iterable of token IDs",
    ) as captured:
        constraint.get_valid_token_ids(constraint.init_state())

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_native_instance_validation_failures_are_wrapped(monkeypatch):
    module = fake_native_module()
    module._NativeRegexState = ExplodingNativeStateType
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_regex((b"a",), "a")

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="validating a returned native regex state",
    ) as captured:
        constraint.init_state()

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_foreign_stale_and_invalid_states_are_rejected(monkeypatch):
    install_fake_runtime(monkeypatch)
    first = compile_native_regex((b"a",), "a")
    second = compile_native_regex((b"a",), "a")
    first_state = first.init_state()
    second_state = second.init_state()

    with pytest.raises(GrammarStateError, match="another constraint"):
        first.get_valid_token_ids(second_state)
    with pytest.raises(TypeError, match="NativeRegexState"):
        first.get_valid_token_ids(object())

    first.reset()
    with pytest.raises(GrammarStateError, match="stale"):
        first.is_match_state(first_state)


def test_bulk_release_validation_is_atomic(monkeypatch):
    install_fake_runtime(monkeypatch)
    constraint = compile_native_regex((b"a",), "a")
    first = constraint.init_state()
    second = constraint.init_state()
    native_second = second._native_state
    native_second.owner = object()

    with pytest.raises(GrammarStateError, match="foreign"):
        constraint.release_states((first, second))

    assert constraint.get_valid_token_ids(first) == (0,)


@pytest.mark.parametrize(
    ("valid_output", "message"),
    [
        ([1, 0], "unique and ascending"),
        ([0, 0], "unique and ascending"),
        ([True], "integers"),
        ([2], "outside vocabulary"),
        (None, "iterable"),
    ],
)
def test_malformed_native_valid_token_output_is_rejected(
    monkeypatch,
    valid_output,
    message,
):
    module = fake_native_module(
        compile_factory=lambda vocabulary, pattern: FakeNativeRegexConstraint(
            vocabulary,
            valid_output=valid_output,
        )
    )
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_regex((b"a", b"b"), "a")

    with pytest.raises(GrammarRuntimeCompatibilityError, match=message):
        constraint.get_valid_token_ids(constraint.init_state())


def test_native_state_is_immutable_and_not_publicly_constructible(monkeypatch):
    install_fake_runtime(monkeypatch)
    state = compile_native_regex((b"a",), "a").init_state()

    with pytest.raises(FrozenInstanceError):
        state._owner = object()
    with pytest.raises(TypeError):
        NativeRegexState(object(), object(), _construction_token=object())


@pytest.mark.parametrize("token_id", [True, 1.5, "0"])
def test_token_type_is_validated_before_native_call(monkeypatch, token_id):
    install_fake_runtime(monkeypatch)
    constraint = compile_native_regex((b"a",), "a")
    state = constraint.init_state()

    with pytest.raises(TypeError, match="token ID must be an integer"):
        constraint.advance_state(state, token_id)


def test_token_range_is_validated_before_native_call(monkeypatch):
    install_fake_runtime(monkeypatch)
    constraint = compile_native_regex((b"a",), "a")
    state = constraint.init_state()

    with pytest.raises(ValueError, match="outside vocabulary"):
        constraint.advance_state(state, 1)
