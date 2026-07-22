import inspect
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import onyx_cuda.native_grammar as native_grammar_module
from onyx_cuda import (
    GrammarCompilationError,
    GrammarConstraint,
    GrammarRuntimeCompatibilityError,
    GrammarStateError,
    NativeJsonConstraint,
    NativeJsonState,
    compile_native_json_schema,
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
    def __init__(self, owner, epoch, value):
        self.owner = owner
        self.epoch = epoch
        self.value = value


class ExplodingIterable:
    def __iter__(self):
        raise RuntimeError("native result iteration exploded")


_DEFAULT_VALID_OUTPUT = object()


class FakeNativeJsonConstraint:
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
        elif logical == "initial" and token_bytes == b"{":
            target = "open"
        elif logical == "open" and token_bytes == b"}":
            target = "matched"
        else:
            raise FakeNativeJsonStateError(
                f"JSON token ID {token_id} is invalid without allocating a child"
            )
        return self._insert(target)

    def get_valid_token_ids(self, state):
        if self.valid_output is not _DEFAULT_VALID_OUTPUT:
            return self.valid_output
        logical = self._logical(state)
        expected = b"{" if logical == "initial" else b"}" if logical == "open" else None
        if expected is None:
            return []
        return [
            token_id
            for token_id, token_bytes in enumerate(self.vocabulary)
            if token_bytes == expected
        ]

    def is_match_state(self, state):
        return self._logical(state) == "matched"

    def is_dead_state(self, state):
        self._logical(state)
        return False

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
        return FakeNativeJsonState(self, self.epoch, value)

    def _logical(self, state):
        self._validate_known(state)
        try:
            return self.states[state.value]
        except KeyError as exc:
            raise FakeNativeJsonStateError("state is released") from exc

    def _validate_known(self, state):
        if state.owner is not self:
            raise FakeNativeJsonStateError("foreign state")
        if state.epoch != self.epoch:
            raise FakeNativeJsonStateError("stale state")
        if state.value not in self.known:
            raise FakeNativeJsonStateError("unknown state")


class ExplodingAttributeConstraint(FakeNativeJsonConstraint):
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

    def compile_json_schema(vocabulary, schema):
        compile_calls.append((tuple(vocabulary), schema))
        if schema == '{"type":"mystery"}':
            raise FakeNativeJsonCompilationError("unsupported type mystery")
        if compile_factory is None:
            return FakeNativeJsonConstraint(vocabulary)
        return compile_factory(vocabulary, schema)

    return SimpleNamespace(
        runtime_version=lambda: "0.1.0",
        grammar_abi_version=lambda: abi_version,
        compile_regex=lambda vocabulary, pattern: FakeNativeRegexConstraint(),
        _NativeRegexConstraint=FakeNativeRegexConstraint,
        _NativeRegexState=FakeNativeRegexState,
        NativeRegexCompilationError=FakeNativeRegexCompilationError,
        NativeRegexStateError=FakeNativeRegexStateError,
        compile_json_schema=compile_json_schema,
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


def test_native_json_satisfies_protocol_and_preserves_parent(monkeypatch):
    module = install_fake_runtime(monkeypatch)
    schema = '{"type":"object","properties":{}}'
    constraint = compile_native_json_schema((b"{", b"}", b"x", b""), schema)
    initial = constraint.init_state()
    opened = constraint.advance_state(initial, 0)
    matched = constraint.advance_state(opened, 1)
    empty = constraint.advance_state(initial, 3)

    assert isinstance(constraint, GrammarConstraint)
    assert isinstance(constraint, NativeJsonConstraint)
    assert constraint.vocab_size == 4
    assert constraint.grammar_type == "json_schema"
    assert constraint.get_valid_token_ids(initial) == (0,)
    assert constraint.get_valid_token_ids(opened) == (1,)
    assert constraint.is_match_state(matched)
    assert not constraint.is_dead_state(initial)
    assert constraint.get_valid_token_ids(empty) == constraint.get_valid_token_ids(initial)
    assert initial != empty
    assert module.compile_calls == [((b"{", b"}", b"x", b""), schema)]


def test_factory_is_positional_only():
    parameters = inspect.signature(compile_native_json_schema).parameters

    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in parameters.values()
    )
    with pytest.raises(TypeError, match="keyword"):
        compile_native_json_schema(
            vocabulary=(b"{",),
            schema='{"type":"object","properties":{}}',
        )


def test_factory_validates_basic_inputs_before_requesting_native_runtime(monkeypatch):
    requested = []
    monkeypatch.setattr(
        native_grammar_module.importlib,
        "import_module",
        lambda name: requested.append(name),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        compile_native_json_schema((), '{"type":"object"}')
    with pytest.raises(TypeError, match="entry 1 must be bytes"):
        compile_native_json_schema((b"{", "}"), '{"type":"object"}')
    with pytest.raises(TypeError, match="must be a string"):
        compile_native_json_schema((b"{",), None)
    with pytest.raises(ValueError, match="cannot be empty"):
        compile_native_json_schema((b"{",), "")
    with pytest.raises(ValueError, match="valid JSON"):
        compile_native_json_schema((b"{",), "not json")
    with pytest.raises(ValueError, match="valid JSON"):
        compile_native_json_schema((b"{",), '{"type":NaN}')
    with pytest.raises(ValueError, match="root must be an object"):
        compile_native_json_schema((b"{",), "[]")

    assert requested == []


def test_native_compilation_and_state_errors_are_translated(monkeypatch):
    install_fake_runtime(monkeypatch)
    with pytest.raises(GrammarCompilationError, match="unsupported type mystery"):
        compile_native_json_schema((b"{",), '{"type":"mystery"}')

    constraint = compile_native_json_schema(
        (b"{", b"}", b"x"),
        '{"type":"object","properties":{}}',
    )
    initial = constraint.init_state()
    before = (
        constraint.get_valid_token_ids(initial),
        constraint.is_match_state(initial),
        constraint.is_dead_state(initial),
    )
    with pytest.raises(GrammarStateError, match="without allocating a child"):
        constraint.advance_state(initial, 2)
    after = (
        constraint.get_valid_token_ids(initial),
        constraint.is_match_state(initial),
        constraint.is_dead_state(initial),
    )
    assert after == before

    constraint.release_state(initial)
    constraint.release_state(initial)
    with pytest.raises(GrammarStateError, match="released"):
        constraint.get_valid_token_ids(initial)


def test_abi_two_is_rejected_before_native_compilation(monkeypatch):
    module = fake_native_module(abi_version=2)
    install_fake_runtime(monkeypatch, module)

    with pytest.raises(GrammarRuntimeCompatibilityError, match="does not match required ABI"):
        compile_native_json_schema((b"{",), '{"type":"object"}')

    assert module.compile_calls == []


def test_malformed_native_constraint_is_rejected(monkeypatch):
    def missing_method(vocabulary, schema):
        constraint = FakeNativeJsonConstraint(vocabulary)
        constraint.init_state = None
        return constraint

    module = fake_native_module(compile_factory=missing_method)
    install_fake_runtime(monkeypatch, module)
    with pytest.raises(GrammarRuntimeCompatibilityError, match="init_state"):
        compile_native_json_schema((b"{",), '{"type":"object"}')

    def wrong_vocabulary(vocabulary, schema):
        constraint = FakeNativeJsonConstraint(vocabulary)
        constraint.vocab_size += 1
        return constraint

    module = fake_native_module(compile_factory=wrong_vocabulary)
    install_fake_runtime(monkeypatch, module)
    with pytest.raises(GrammarRuntimeCompatibilityError, match="does not match"):
        compile_native_json_schema((b"{",), '{"type":"object"}')


def test_native_constraint_attribute_failures_are_wrapped(monkeypatch):
    holder = []

    def build_constraint(vocabulary, schema):
        constraint = ExplodingAttributeConstraint(vocabulary)
        holder.append(constraint)
        return constraint

    module = fake_native_module(compile_factory=build_constraint)
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')
    state = constraint.init_state()
    holder[0].exploding_attribute = "get_valid_token_ids"

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="get_valid_token_ids access failed",
    ) as captured:
        constraint.get_valid_token_ids(state)

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_native_constraint_initialization_failures_are_wrapped(monkeypatch):
    def build_constraint(vocabulary, schema):
        constraint = ExplodingAttributeConstraint(vocabulary)
        constraint.exploding_attribute = "vocab_size"
        return constraint

    module = fake_native_module(compile_factory=build_constraint)
    install_fake_runtime(monkeypatch, module)

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="vocab_size access failed",
    ) as captured:
        compile_native_json_schema((b"{",), '{"type":"object"}')

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_unexpected_native_compilation_failure_is_wrapped(monkeypatch):
    def explode(vocabulary, schema):
        raise RuntimeError("JSON compilation exploded")

    module = fake_native_module(compile_factory=explode)
    install_fake_runtime(monkeypatch, module)

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="compile_json_schema.*failed unexpectedly",
    ) as captured:
        compile_native_json_schema((b"{",), '{"type":"object"}')

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_unexpected_native_state_failure_is_wrapped(monkeypatch):
    holder = []

    def build_constraint(vocabulary, schema):
        constraint = FakeNativeJsonConstraint(vocabulary)
        holder.append(constraint)
        return constraint

    module = fake_native_module(compile_factory=build_constraint)
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')
    holder[0].init_state = lambda: (_ for _ in ()).throw(RuntimeError("state exploded"))

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="init_state.*failed unexpectedly",
    ) as captured:
        constraint.init_state()

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_malformed_native_boolean_and_release_results_are_rejected(monkeypatch):
    holder = []

    def build_constraint(vocabulary, schema):
        constraint = FakeNativeJsonConstraint(vocabulary)
        holder.append(constraint)
        return constraint

    module = fake_native_module(compile_factory=build_constraint)
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')
    state = constraint.init_state()

    holder[0].is_match_state = lambda native_state: 1
    with pytest.raises(GrammarRuntimeCompatibilityError, match="must return a boolean"):
        constraint.is_match_state(state)

    holder[0].release_state = lambda native_state: "unexpected"
    with pytest.raises(GrammarRuntimeCompatibilityError, match="must return None"):
        constraint.release_state(state)


def test_native_result_iteration_failures_are_wrapped(monkeypatch):
    module = fake_native_module(
        compile_factory=lambda vocabulary, schema: FakeNativeJsonConstraint(
            vocabulary,
            valid_output=ExplodingIterable(),
        )
    )
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="iterable of token IDs",
    ) as captured:
        constraint.get_valid_token_ids(constraint.init_state())

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_native_instance_validation_failures_are_wrapped(monkeypatch):
    module = fake_native_module()
    module._NativeJsonState = ExplodingNativeStateType
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')

    with pytest.raises(
        GrammarRuntimeCompatibilityError,
        match="validating a returned native JSON state",
    ) as captured:
        constraint.init_state()

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_foreign_stale_and_invalid_states_are_rejected(monkeypatch):
    install_fake_runtime(monkeypatch)
    first = compile_native_json_schema((b"{",), '{"type":"object"}')
    second = compile_native_json_schema((b"{",), '{"type":"object"}')
    first_state = first.init_state()
    second_state = second.init_state()

    with pytest.raises(GrammarStateError, match="another constraint"):
        first.get_valid_token_ids(second_state)
    with pytest.raises(TypeError, match="NativeJsonState"):
        first.get_valid_token_ids(object())

    first.reset()
    with pytest.raises(GrammarStateError, match="stale"):
        first.is_match_state(first_state)


def test_bulk_release_validation_is_atomic(monkeypatch):
    install_fake_runtime(monkeypatch)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')
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
        compile_factory=lambda vocabulary, schema: FakeNativeJsonConstraint(
            vocabulary,
            valid_output=valid_output,
        )
    )
    install_fake_runtime(monkeypatch, module)
    constraint = compile_native_json_schema((b"{", b"}"), '{"type":"object"}')

    with pytest.raises(GrammarRuntimeCompatibilityError, match=message):
        constraint.get_valid_token_ids(constraint.init_state())


def test_native_state_and_constraint_are_not_publicly_constructible(monkeypatch):
    install_fake_runtime(monkeypatch)
    state = compile_native_json_schema((b"{",), '{"type":"object"}').init_state()

    with pytest.raises(FrozenInstanceError):
        state._owner = object()
    with pytest.raises(TypeError):
        NativeJsonState(object(), object(), _construction_token=object())
    with pytest.raises(TypeError):
        NativeJsonConstraint(object(), object(), 1, _construction_token=object())


@pytest.mark.parametrize("token_id", [True, 1.5, "0"])
def test_token_type_is_validated_before_native_call(monkeypatch, token_id):
    install_fake_runtime(monkeypatch)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')
    state = constraint.init_state()

    with pytest.raises(TypeError, match="token ID must be an integer"):
        constraint.advance_state(state, token_id)


def test_token_range_is_validated_before_native_call(monkeypatch):
    install_fake_runtime(monkeypatch)
    constraint = compile_native_json_schema((b"{",), '{"type":"object"}')
    state = constraint.init_state()

    with pytest.raises(ValueError, match="outside vocabulary"):
        constraint.advance_state(state, 1)
