import math
from dataclasses import FrozenInstanceError

import pytest

import onyx_cuda.production_engine as production_engine_module
from onyx_cuda import (
    BackendStateError,
    DEFAULT_TARGET_PROFILE,
    GrammarTimingMetrics,
    JsonSchemaGrammar,
    ModelStep,
    ProductionEngineCleanupError,
    SelectionError,
    RegexGrammar,
    TemperatureTopPSelection,
    TextGenerationComplete,
    TextGenerationDelta,
)
from onyx_cuda.production_engine import (
    ProductionEngineLoadError,
    ProductionTargetTextEngine,
    load_production_target_engine,
)
from onyx_cuda.testing import (
    FakeCharacterTokenizer,
    FakeGrammarConstraint,
    FakeGrammarProgram,
    create_deterministic_grammar_timing_session,
    create_deterministic_metrics_session,
)
from onyx_cuda.torch_backend import (
    TorchBackendExecutionError,
    TorchBackendInvariantError,
    select_cuda_argmax,
)


class FakeDevice:
    def __init__(self, name="cuda:0"):
        self.name = name

    def __str__(self):
        return self.name


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBoolTensor:
    def __init__(self, value, error=None):
        self.value = value
        self.error = error

    def any(self):
        if self.error is not None:
            raise self.error
        return self

    def item(self):
        if self.error is not None:
            raise self.error
        return self.value


class FakeCudaLogits:
    def __init__(
        self,
        values,
        *,
        shape=None,
        is_cuda=True,
        device="cuda:0",
        validation_error=None,
        argmax_error=None,
        selected_override=None,
    ):
        self.values = tuple(values)
        self.shape = shape if shape is not None else (len(self.values),)
        self.is_cuda = is_cuda
        self.device = FakeDevice(device)
        self.validation_error = validation_error
        self.argmax_error = argmax_error
        self.selected_override = selected_override
        self.argmax_dims = []

    def isnan(self):
        contains_nan = any(math.isnan(float(value)) for value in self.values)
        return FakeBoolTensor(contains_nan, self.validation_error)

    def argmax(self, *, dim):
        self.argmax_dims.append(dim)
        if self.argmax_error is not None:
            raise self.argmax_error
        selected = (
            self.selected_override
            if self.selected_override is not None
            else max(range(len(self.values)), key=self.values.__getitem__)
        )
        return FakeScalar(selected)


class FakeProductionBackend:
    model_id = "fake-production-target"
    cache_mode = "fake"

    def __init__(
        self,
        script=None,
        *,
        tokenizer=None,
        reported_vocab_size=None,
        close_error=None,
        reset_error=None,
    ):
        self.script = script or (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        self._tokenizer = tokenizer or FakeCharacterTokenizer(
            ("P", "a", "b", "!"), tokenizer_id="fake-production-tokenizer"
        )
        self.reported_vocab_size = reported_vocab_size
        self.close_error = close_error
        self.reset_error = reset_error
        self.close_calls = 0
        self.reset_calls = 0
        self._cache_length = 0
        self._next_row = 0
        self.closed = False

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.reported_vocab_size or len(self.script[0])

    @property
    def cache_length(self):
        return self._cache_length

    def prefill(self, prompt_token_ids, /):
        self._next_row = 0
        self._cache_length = len(tuple(prompt_token_ids))
        return ModelStep(FakeCudaLogits(self._take_row()), self._cache_length)

    def decode(self, token_id, /):
        if self._cache_length == 0:
            raise BackendStateError("prefill required")
        self._cache_length += 1
        return ModelStep(FakeCudaLogits(self._take_row()), self._cache_length)

    def reset(self):
        self.reset_calls += 1
        self._cache_length = 0
        self._next_row = 0
        if self.reset_error is not None:
            raise self.reset_error

    def close(self):
        self.close_calls += 1
        self.closed = True
        self._cache_length = 0
        if self.close_error is not None:
            raise self.close_error

    def _take_row(self):
        row = self.script[self._next_row]
        self._next_row += 1
        return row


def test_cuda_selector_runs_validation_and_argmax_on_device():
    logits = FakeCudaLogits((-3.0, 2.0, 2.0))

    assert select_cuda_argmax(logits) == 1
    assert logits.argmax_dims == [-1]


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (FakeCudaLogits((), shape=(0,)), "nonempty"),
        (FakeCudaLogits((1.0, 2.0), shape=(1, 2)), "one nonempty"),
        (FakeCudaLogits((1.0,), is_cuda=False), "non-CUDA"),
        (FakeCudaLogits((1.0,), device="cpu"), "expected a CUDA device"),
        (FakeCudaLogits((1.0, float("nan"))), "cannot contain NaN"),
        (FakeCudaLogits((1.0,), selected_override=-1), "outside logits range"),
        (FakeCudaLogits((1.0,), selected_override=True), "integer token ID"),
        (FakeCudaLogits((1.0,), selected_override=1.5), "integer token ID"),
    ],
)
def test_cuda_selector_rejects_invalid_logits_or_results(logits, message):
    with pytest.raises(TorchBackendInvariantError, match=message):
        select_cuda_argmax(logits)


def test_cuda_selector_wraps_validation_and_argmax_failures():
    with pytest.raises(TorchBackendExecutionError, match="validation failed"):
        select_cuda_argmax(
            FakeCudaLogits((1.0,), validation_error=RuntimeError("validation failed"))
        )
    with pytest.raises(TorchBackendExecutionError, match="argmax failed"):
        select_cuda_argmax(FakeCudaLogits((1.0,), argmax_error=RuntimeError("argmax failed")))


def test_production_engine_generates_text_and_trims_multi_token_stop():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(backend)

    result = engine.generate("P", max_new_tokens=3, stop_token_sequences=((2, 3),))

    assert result.text == "a"
    assert result.output_token_ids == (1,)
    assert result.sampled_token_ids == (1, 2, 3)
    assert result.generation.finish_reason == "stop"
    assert result.generation.matched_stop_token_ids == (2, 3)
    assert result.generation.final_cache_length == 3
    assert engine.cache_length == 3
    assert engine.model_id == "fake-production-target"
    assert engine.tokenizer_id == "fake-production-tokenizer"
    assert engine.vocab_size == 4


def test_production_engine_length_finish_and_repeated_restart():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(
        backend,
        create_metrics_session=create_deterministic_metrics_session,
    )

    first = engine.generate("P", max_new_tokens=2)
    second = engine.generate("P", max_new_tokens=2)

    assert first == second
    assert first.text == "ab"
    assert first.generation.finish_reason == "length"
    assert first.generation.final_cache_length == 2


def test_production_engine_creates_a_fresh_metrics_session_per_generation():
    backend = FakeProductionBackend()
    sessions = []

    def create_metrics():
        session = create_deterministic_metrics_session()
        sessions.append(session)
        return session

    engine = ProductionTargetTextEngine(
        backend,
        create_metrics_session=create_metrics,
    )

    first = engine.generate("P", max_new_tokens=1)
    second = engine.generate("P", max_new_tokens=1)

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert first == second


def test_production_stream_matches_generate_and_releases_active_slot():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(
        backend,
        create_metrics_session=create_deterministic_metrics_session,
    )
    expected = engine.generate("P", max_new_tokens=3)

    stream = engine.stream("P", max_new_tokens=3)
    events = list(stream)

    assert "".join(
        event.text for event in events if isinstance(event, TextGenerationDelta)
    ) == expected.text
    assert events[-1] == TextGenerationComplete(expected)
    assert stream.is_closed
    assert engine.cache_length == expected.generation.final_cache_length
    assert engine.generate("P", max_new_tokens=1).text == "a"


def test_production_stream_buffers_multi_token_stop_text():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(backend)

    events = list(
        engine.stream(
            "P",
            max_new_tokens=3,
            stop_token_sequences=((2, 3),),
        )
    )

    assert [event.text for event in events if isinstance(event, TextGenerationDelta)] == [
        "a"
    ]
    result = events[-1].result
    assert result.text == "a"
    assert result.output_token_ids == (1,)
    assert result.sampled_token_ids == (1, 2, 3)


def test_production_stream_prevents_overlapping_stream_or_generate():
    engine = ProductionTargetTextEngine(FakeProductionBackend())
    stream = engine.stream("P", max_new_tokens=3)

    with pytest.raises(BackendStateError, match="active stream"):
        engine.stream("P", max_new_tokens=1)
    with pytest.raises(BackendStateError, match="active stream"):
        engine.generate("P", max_new_tokens=1)

    stream.close()
    assert engine.generate("P", max_new_tokens=1).text == "a"


def test_production_stream_cancellation_resets_and_close_is_idempotent():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(backend)
    stream = engine.stream("P", max_new_tokens=3)

    assert next(stream) == TextGenerationDelta("a")
    stream.close()
    stream.close()

    assert stream.is_closed
    assert backend.cache_length == 0
    assert backend.reset_calls == 1
    assert engine.generate("P", max_new_tokens=1).text == "a"


def test_production_stream_failure_resets_and_releases_active_slot():
    backend = FakeProductionBackend(script=((0.0, 1.0, 0.0, 0.0),))
    engine = ProductionTargetTextEngine(backend)
    stream = engine.stream("P", max_new_tokens=2)

    assert next(stream) == TextGenerationDelta("a")
    with pytest.raises(IndexError):
        next(stream)

    assert stream.is_closed
    assert backend.cache_length == 0
    assert engine.generate("P", max_new_tokens=1).text == "a"


def test_invalid_production_stream_does_not_reserve_active_slot():
    engine = ProductionTargetTextEngine(FakeProductionBackend())

    with pytest.raises(ValueError, match="greater than zero"):
        engine.stream("P", max_new_tokens=0)

    assert engine.generate("P", max_new_tokens=1).text == "a"


def test_production_stream_context_manager_cancels_on_early_exit():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(backend)

    with engine.stream("P", max_new_tokens=3) as stream:
        assert next(stream) == TextGenerationDelta("a")

    assert stream.is_closed
    assert backend.cache_length == 0


def test_engine_close_cancels_active_stream_before_closing_backend():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(backend)
    stream = engine.stream("P", max_new_tokens=3)
    assert next(stream) == TextGenerationDelta("a")

    engine.close()

    assert engine.is_closed
    assert stream.is_closed
    assert backend.reset_calls == 1
    assert backend.close_calls == 1
    assert backend.cache_length == 0


def test_engine_close_reports_active_stream_and_backend_cleanup_failures():
    backend = FakeProductionBackend(
        reset_error=RuntimeError("reset failed"),
        close_error=RuntimeError("close failed"),
    )
    engine = ProductionTargetTextEngine(backend)
    stream = engine.stream("P", max_new_tokens=3)
    assert next(stream) == TextGenerationDelta("a")

    with pytest.raises(
        ProductionEngineCleanupError,
        match="reset failed.*backend cleanup also failed.*close failed",
    ):
        engine.close()

    assert engine.is_closed
    assert stream.is_closed
    assert backend.close_calls == 1


def test_production_seeded_sampling_uses_a_fresh_session_per_generation():
    backend = FakeProductionBackend()
    policies = []

    def create_sampling_selector(policy):
        policies.append(policy)
        selected = iter((2, 1))
        return lambda logits: next(selected)

    engine = ProductionTargetTextEngine(
        backend,
        create_sampling_selector=create_sampling_selector,
        create_metrics_session=create_deterministic_metrics_session,
    )
    policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=7)

    first = engine.generate("P", max_new_tokens=2, selection=policy)
    second = engine.generate("P", max_new_tokens=2, selection=policy)

    assert first == second
    assert first.text == "ba"
    assert first.sampled_token_ids == (2, 1)
    assert first.generation.final_cache_length == 2
    assert policies == [policy, policy]


def test_production_seeded_stream_uses_fresh_session_and_matches_generate():
    backend = FakeProductionBackend()
    policies = []

    def create_sampling_selector(policy):
        policies.append(policy)
        selected = iter((2, 1))
        return lambda logits: next(selected)

    engine = ProductionTargetTextEngine(
        backend,
        create_sampling_selector=create_sampling_selector,
        create_metrics_session=create_deterministic_metrics_session,
    )
    policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=7)

    expected = engine.generate("P", max_new_tokens=2, selection=policy)
    events = list(engine.stream("P", max_new_tokens=2, selection=policy))

    assert events[-1] == TextGenerationComplete(expected)
    assert policies == [policy, policy]


def test_production_sampling_requires_a_configured_factory_before_prefill():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(backend)

    with pytest.raises(SelectionError, match="not configured"):
        engine.generate(
            "P",
            max_new_tokens=1,
            selection=TemperatureTopPSelection(1.0, 1.0, 0),
        )

    assert backend.cache_length == 0


def test_close_is_idempotent_and_blocks_generation():
    backend = FakeProductionBackend()
    engine = ProductionTargetTextEngine(backend)

    engine.close()
    engine.close()

    assert engine.is_closed
    assert engine.cache_length == 0
    assert backend.close_calls == 1
    with pytest.raises(BackendStateError, match="closed"):
        engine.generate("P", max_new_tokens=1)


def test_context_manager_closes_engine():
    backend = FakeProductionBackend()

    with ProductionTargetTextEngine(backend) as engine:
        result = engine.generate("P", max_new_tokens=1)
        assert result.text == "a"

    assert engine.is_closed
    assert backend.closed


def test_close_failure_propagates_after_wrapper_is_marked_closed():
    backend = FakeProductionBackend(close_error=RuntimeError("cleanup failed"))
    engine = ProductionTargetTextEngine(backend)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        engine.close()

    assert engine.is_closed
    assert backend.closed


def test_factory_passes_profile_device_and_offline_options(monkeypatch):
    backend = FakeProductionBackend()
    calls = []

    def load_backend(profile, *, device_index, local_files_only):
        calls.append((profile, device_index, local_files_only))
        return backend

    monkeypatch.setattr(production_engine_module, "load_torch_cuda_target", load_backend)

    engine = load_production_target_engine(device_index=1, local_files_only=True)

    assert calls == [(DEFAULT_TARGET_PROFILE, 1, True)]
    assert engine.model_id == backend.model_id
    engine.close()


def test_typed_torch_backend_uses_cuda_metrics_by_default(monkeypatch):
    class FakeTypedTorchBackend(FakeProductionBackend):
        device_index = 3

    backend = FakeTypedTorchBackend()
    metrics_calls = []
    monkeypatch.setattr(
        production_engine_module,
        "TorchCUDATargetBackend",
        FakeTypedTorchBackend,
    )

    def create_metrics(*, device_index):
        metrics_calls.append(device_index)
        return create_deterministic_metrics_session(cache_mode="transformers_dynamic")

    monkeypatch.setattr(
        production_engine_module,
        "create_torch_metrics_session",
        create_metrics,
    )

    engine = ProductionTargetTextEngine(backend)
    result = engine.generate("P", max_new_tokens=1)

    assert metrics_calls == [3]
    assert result.generation.metrics.cache_mode == "transformers_dynamic"


def test_factory_binds_cuda_sampling_to_the_requested_device(monkeypatch):
    backend = FakeProductionBackend()
    sampler_calls = []
    metrics_calls = []
    policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=7)

    monkeypatch.setattr(
        production_engine_module,
        "load_torch_cuda_target",
        lambda *args, **kwargs: backend,
    )

    def create_sampler(received_policy, *, device_index):
        sampler_calls.append((received_policy, device_index))
        return lambda logits: 1

    monkeypatch.setattr(production_engine_module, "create_cuda_sampler", create_sampler)

    def create_metrics(*, device_index):
        metrics_calls.append(device_index)
        return create_deterministic_metrics_session(cache_mode="transformers_dynamic")

    monkeypatch.setattr(
        production_engine_module,
        "create_torch_metrics_session",
        create_metrics,
    )

    engine = load_production_target_engine(device_index=2, local_files_only=True)
    result = engine.generate("P", max_new_tokens=1, selection=policy)

    assert result.sampled_token_ids == (1,)
    assert sampler_calls == [(policy, 2)]
    assert metrics_calls == [2]
    engine.close()


def test_factory_composition_failure_closes_backend(monkeypatch):
    backend = FakeProductionBackend(reported_vocab_size=3)
    monkeypatch.setattr(
        production_engine_module,
        "load_torch_cuda_target",
        lambda *args, **kwargs: backend,
    )

    with pytest.raises(ProductionEngineLoadError, match="does not match"):
        load_production_target_engine()

    assert backend.closed


def test_factory_reports_composition_and_cleanup_failures(monkeypatch):
    backend = FakeProductionBackend(
        reported_vocab_size=3,
        close_error=RuntimeError("cleanup failed"),
    )
    monkeypatch.setattr(
        production_engine_module,
        "load_torch_cuda_target",
        lambda *args, **kwargs: backend,
    )

    with pytest.raises(
        ProductionEngineLoadError,
        match="does not match.*cleanup also failed.*cleanup failed",
    ):
        load_production_target_engine()


def test_result_remains_immutable_through_production_wrapper():
    engine = ProductionTargetTextEngine(FakeProductionBackend())
    result = engine.generate("P", max_new_tokens=1)

    with pytest.raises(FrozenInstanceError):
        result.text = "changed"
    engine.close()


class FakeProductionGrammarMask:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.calls = []

    def apply(self, logits, valid_token_ids, /):
        self.calls.append(valid_token_ids)
        return FakeCudaLogits(
            tuple(
                value if token_id in valid_token_ids else float("-inf")
                for token_id, value in enumerate(logits.values)
            )
        )

    def apply_with_timing(self, logits, valid_token_ids, timing_session, /):
        result = self.apply(logits, valid_token_ids)
        timing_session.record_mask_timing(0.25, 0.5)
        return result


def _production_grammar_program():
    return FakeGrammarProgram(
        initial_state="s0",
        transitions=(("s0", 1, "match"),),
        valid_token_ids=(("s0", (1,)), ("match", ())),
        match_states=frozenset({"match"}),
    )


def _configure_lazy_grammar_mocks(monkeypatch, backend, *, build_vocabulary=None):
    vocabulary = (b"P", b"a", b"b", b"")
    backend.tokenizer.eos_token_id = 3
    calls = {"build": 0, "mask": 0, "regex": [], "json": []}
    mask = FakeProductionGrammarMask(4)
    constraints = []

    monkeypatch.setattr(
        production_engine_module,
        "load_torch_cuda_target",
        lambda *args, **kwargs: backend,
    )

    def build(tokenizer):
        calls["build"] += 1
        if build_vocabulary is not None:
            return build_vocabulary(calls["build"])
        return vocabulary

    def create_mask(vocab_size, *, device_index):
        calls["mask"] += 1
        assert device_index == 2
        assert vocab_size == len(vocabulary)
        return mask

    def compile_constraint(grammar_type, source):
        calls[grammar_type].append(source)
        constraint = FakeGrammarConstraint(
            vocabulary,
            grammar_type="regex" if grammar_type == "regex" else "json_schema",
            program=_production_grammar_program(),
        )
        constraints.append(constraint)
        return constraint

    monkeypatch.setattr(production_engine_module, "build_qwen_grammar_vocabulary", build)
    monkeypatch.setattr(production_engine_module, "create_cuda_grammar_logit_mask", create_mask)
    monkeypatch.setattr(
        production_engine_module,
        "compile_native_regex",
        lambda received_vocabulary, pattern: compile_constraint("regex", pattern),
    )
    monkeypatch.setattr(
        production_engine_module,
        "compile_native_json_schema",
        lambda received_vocabulary, schema: compile_constraint("json", schema),
    )
    monkeypatch.setattr(
        production_engine_module,
        "create_torch_metrics_session",
        lambda *, device_index: create_deterministic_metrics_session(
            cache_mode="transformers_dynamic"
        ),
    )
    monkeypatch.setattr(
        production_engine_module,
        "create_grammar_timing_session",
        create_deterministic_grammar_timing_session,
    )
    return calls, mask, constraints


def test_loaded_production_engine_is_grammar_lazy_and_reuses_only_vocabulary_and_mask(
    monkeypatch,
):
    backend = FakeProductionBackend()
    calls, mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    unconstrained = engine.generate("P", max_new_tokens=1)
    assert unconstrained.text == "a"
    assert unconstrained.generation.metrics.grammar_timing is None
    unconstrained_streamed = list(engine.stream("P", max_new_tokens=1))[-1].result
    assert unconstrained_streamed.text == "a"
    assert unconstrained_streamed.generation.metrics.grammar_timing is None
    assert calls == {"build": 0, "mask": 0, "regex": [], "json": []}

    regex = engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    )
    schema = '{"const":"a"}'
    json_result = engine.generate_constrained(
        "P",
        grammar=JsonSchemaGrammar(schema),
        max_new_tokens=2,
    )

    assert regex.text == json_result.text == "a"
    assert regex.generation.finish_reason == "grammar_complete"
    assert regex.generation.metrics.grammar_timing == GrammarTimingMetrics(
        compilation_time=1.0,
        state_scan_time=2.0,
        valid_index_transfer_time=0.5,
        mask_application_time=1.0,
    )
    assert json_result.generation.metrics.grammar_timing == (
        regex.generation.metrics.grammar_timing
    )
    assert calls == {"build": 1, "mask": 1, "regex": ["a"], "json": [schema]}
    assert mask.calls == [(1,), (3,), (1,), (3,)]
    assert len(constraints) == 2
    assert constraints[0] is not constraints[1]
    assert all(constraint.active_state_count == 0 for constraint in constraints)
    engine.close()
    assert engine._engine is None
    assert engine._create_grammar_context is None


def test_production_factory_rejects_sparse_mask_without_completed_timing(
    monkeypatch,
):
    class UntimedProductionMask:
        vocab_size = 4
        transport_name = "sparse_valid_indices"

        def apply(self, logits, valid_token_ids, /):
            return logits

    backend = FakeProductionBackend()
    calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    monkeypatch.setattr(
        production_engine_module,
        "create_cuda_grammar_logit_mask",
        lambda vocab_size, *, device_index: UntimedProductionMask(),
    )
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    with pytest.raises(RuntimeError, match="must provide completed.*timing"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=2,
        )

    assert calls["build"] == 1
    assert calls["regex"] == []
    assert constraints == []
    assert backend.cache_length == 0
    engine.close()


def test_partial_lazy_setup_is_not_cached_and_can_retry(monkeypatch):
    backend = FakeProductionBackend()

    def build_vocabulary(call_number):
        if call_number == 1:
            raise RuntimeError("vocabulary build failed")
        return (b"P", b"a", b"b", b"")

    calls, _mask, constraints = _configure_lazy_grammar_mocks(
        monkeypatch,
        backend,
        build_vocabulary=build_vocabulary,
    )
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    with pytest.raises(RuntimeError, match="vocabulary build failed"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=2,
        )

    assert backend.cache_length == 0
    result = engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    )

    assert result.text == "a"
    assert calls["build"] == 2
    assert calls["mask"] == 1
    assert len(constraints) == 1
    engine.close()


def test_production_compilation_failure_keeps_backend_reusable_and_support_cached(
    monkeypatch,
):
    backend = FakeProductionBackend()
    calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    compile_calls = 0
    original_compile = production_engine_module.compile_native_regex

    def fail_once(vocabulary, pattern):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 1:
            raise RuntimeError("compile failed")
        return original_compile(vocabulary, pattern)

    monkeypatch.setattr(production_engine_module, "compile_native_regex", fail_once)
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    with pytest.raises(RuntimeError, match="compile failed"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=2,
        )

    assert backend.cache_length == 0
    assert engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    ).text == "a"
    assert calls["build"] == 1
    assert calls["mask"] == 1
    assert len(constraints) == 1
    engine.close()


def test_production_vocabulary_mismatch_fails_before_mask_or_prefill(monkeypatch):
    backend = FakeProductionBackend()
    calls, _mask, _constraints = _configure_lazy_grammar_mocks(
        monkeypatch,
        backend,
        build_vocabulary=lambda _call: (b"P", b"a", b""),
    )
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    with pytest.raises(RuntimeError, match="must match tokenizer and backend"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=2,
        )

    assert calls["build"] == 1
    assert calls["mask"] == 0
    assert backend.cache_length == 0
    engine.close()


def test_production_eos_must_be_in_range_and_map_to_empty_bytes(monkeypatch):
    backend = FakeProductionBackend()
    calls, _mask, _constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)
    backend.tokenizer.eos_token_id = 2

    with pytest.raises(RuntimeError, match="must map.*empty-byte"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=2,
        )

    assert calls["build"] == 1
    assert calls["mask"] == 0
    assert calls["regex"] == []
    assert backend.cache_length == 0

    backend.tokenizer.eos_token_id = 3
    assert engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    ).text == "a"
    assert calls["build"] == 2
    assert calls["mask"] == 1
    engine.close()


def test_active_production_stream_blocks_constrained_generation_before_lazy_setup(
    monkeypatch,
):
    backend = FakeProductionBackend()
    calls, _mask, _constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)
    stream = engine.stream("P", max_new_tokens=1)

    with pytest.raises(BackendStateError, match="active stream"):
        engine.generate_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=2,
        )

    assert calls["build"] == 0
    stream.close()
    assert engine.generate_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    ).text == "a"
    engine.close()


@pytest.mark.parametrize(
    ("grammar", "call_key", "source"),
    [
        (RegexGrammar("a"), "regex", "a"),
        (JsonSchemaGrammar('{"const":"a"}'), "json", '{"const":"a"}'),
    ],
)
def test_production_constrained_stream_equals_non_streaming_with_fresh_constraints(
    monkeypatch,
    grammar,
    call_key,
    source,
):
    backend = FakeProductionBackend()
    calls, mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    expected = engine.generate_constrained(
        "P",
        grammar=grammar,
        max_new_tokens=2,
    )
    events = list(
        engine.stream_constrained(
            "P",
            grammar=grammar,
            max_new_tokens=2,
        )
    )

    assert [event.text for event in events if isinstance(event, TextGenerationDelta)] == ["a"]
    assert isinstance(events[-1], TextGenerationComplete)
    assert events[-1].result == expected
    assert events[-1].result.sampled_token_ids == (1, 3)
    assert calls[call_key] == [source, source]
    assert calls["build"] == 1
    assert calls["mask"] == 1
    assert mask.calls == [(1,), (3,), (1,), (3,)]
    assert len(constraints) == 2
    assert constraints[0] is not constraints[1]
    assert all(constraint.active_state_count == 0 for constraint in constraints)
    assert backend.reset_calls == 0
    assert engine._active_stream is None
    engine.close()


def test_never_started_production_constrained_stream_closes_without_lazy_setup(
    monkeypatch,
):
    backend = FakeProductionBackend()
    calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    stream = engine.stream_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    )
    assert calls == {"build": 0, "mask": 0, "regex": [], "json": []}
    assert constraints == []
    assert engine._active_stream is stream

    stream.close()
    stream.close()

    assert stream.is_closed
    assert engine._active_stream is None
    assert calls == {"build": 0, "mask": 0, "regex": [], "json": []}
    assert constraints == []
    assert backend.reset_calls == 0
    assert engine.generate("P", max_new_tokens=1).text == "a"
    engine.close()


def test_constrained_and_unconstrained_streams_share_one_active_slot(monkeypatch):
    backend = FakeProductionBackend()
    calls, _mask, _constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)
    grammar = RegexGrammar("a")
    constrained = engine.stream_constrained(
        "P",
        grammar=grammar,
        max_new_tokens=2,
    )

    with pytest.raises(BackendStateError, match="active stream"):
        engine.generate("P", max_new_tokens=1)
    with pytest.raises(BackendStateError, match="active stream"):
        engine.stream("P", max_new_tokens=1)
    with pytest.raises(BackendStateError, match="active stream"):
        engine.generate_constrained("P", grammar=grammar, max_new_tokens=2)
    with pytest.raises(BackendStateError, match="active stream"):
        engine.stream_constrained("P", grammar=grammar, max_new_tokens=2)

    assert calls["build"] == 0
    constrained.close()

    unconstrained = engine.stream("P", max_new_tokens=1)
    with pytest.raises(BackendStateError, match="active stream"):
        engine.stream_constrained("P", grammar=grammar, max_new_tokens=2)
    unconstrained.close()

    assert list(
        engine.stream_constrained("P", grammar=grammar, max_new_tokens=2)
    )[-1].result.text == "a"
    engine.close()


def test_first_iteration_compile_failure_releases_slot_and_keeps_support_retryable(
    monkeypatch,
):
    backend = FakeProductionBackend()
    calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    original_compile = production_engine_module.compile_native_regex
    compile_calls = 0

    def fail_once(vocabulary, pattern):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 1:
            raise RuntimeError("compile failed")
        return original_compile(vocabulary, pattern)

    monkeypatch.setattr(production_engine_module, "compile_native_regex", fail_once)
    engine = load_production_target_engine(device_index=2, local_files_only=True)
    grammar = RegexGrammar("a")
    stream = engine.stream_constrained("P", grammar=grammar, max_new_tokens=2)

    assert calls["build"] == 0
    with pytest.raises(RuntimeError, match="compile failed"):
        next(stream)

    assert stream.is_closed
    assert engine._active_stream is None
    assert backend.cache_length == 0
    assert calls["build"] == 1
    assert calls["mask"] == 1
    assert constraints == []

    completed = list(engine.stream_constrained("P", grammar=grammar, max_new_tokens=2))
    assert completed[-1].result.text == "a"
    assert calls["build"] == 1
    assert calls["mask"] == 1
    assert len(constraints) == 1
    engine.close()


def test_partial_production_constrained_stream_cancels_and_engine_reuses(
    monkeypatch,
):
    backend = FakeProductionBackend()
    _calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)
    grammar = RegexGrammar("a")
    stream = engine.stream_constrained("P", grammar=grammar, max_new_tokens=2)

    assert next(stream) == TextGenerationDelta("a")
    assert constraints[0].active_state_count == 1

    stream.close()
    stream.close()

    assert stream.is_closed
    assert constraints[0].active_state_count == 0
    assert backend.reset_calls == 1
    assert backend.cache_length == 0
    assert engine._active_stream is None
    assert engine.generate("P", max_new_tokens=1).text == "a"
    assert engine.generate_constrained("P", grammar=grammar, max_new_tokens=2).text == "a"
    assert len(constraints) == 2
    engine.close()


def test_production_constrained_context_manager_cancels_on_early_exit(monkeypatch):
    backend = FakeProductionBackend()
    _calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    with engine.stream_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    ) as stream:
        assert next(stream) == TextGenerationDelta("a")

    assert stream.is_closed
    assert constraints[0].active_state_count == 0
    assert backend.reset_calls == 1
    assert engine._active_stream is None
    engine.close()


def test_engine_close_cancels_constrained_stream_before_backend_close(monkeypatch):
    backend = FakeProductionBackend()
    _calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    operations = []
    original_reset = backend.reset
    original_close = backend.close

    def reset():
        operations.append("reset")
        original_reset()

    def close():
        operations.append("close")
        original_close()

    backend.reset = reset
    backend.close = close
    engine = load_production_target_engine(device_index=2, local_files_only=True)
    stream = engine.stream_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    )
    assert next(stream) == TextGenerationDelta("a")

    engine.close()

    assert operations == ["reset", "close"]
    assert stream.is_closed
    assert engine.is_closed
    assert engine._active_stream is None
    assert constraints[0].active_state_count == 0


def test_engine_close_combines_constrained_stream_and_backend_cleanup_failures(
    monkeypatch,
):
    backend = FakeProductionBackend(
        reset_error=RuntimeError("reset failed"),
        close_error=RuntimeError("close failed"),
    )
    _calls, _mask, constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)
    stream = engine.stream_constrained(
        "P",
        grammar=RegexGrammar("a"),
        max_new_tokens=2,
    )
    assert next(stream) == TextGenerationDelta("a")

    with pytest.raises(ProductionEngineCleanupError) as captured:
        engine.close()

    assert "reset failed" in str(captured.value)
    assert "close failed" in str(captured.value)
    assert stream.is_closed
    assert engine.is_closed
    assert constraints[0].active_state_count == 0
    assert backend.reset_calls == 1
    assert backend.close_calls == 1


def test_invalid_production_constrained_stream_does_not_reserve_active_slot(
    monkeypatch,
):
    backend = FakeProductionBackend()
    calls, _mask, _constraints = _configure_lazy_grammar_mocks(monkeypatch, backend)
    engine = load_production_target_engine(device_index=2, local_files_only=True)

    with pytest.raises(ValueError, match="greater than zero"):
        engine.stream_constrained(
            "P",
            grammar=RegexGrammar("a"),
            max_new_tokens=0,
        )

    assert engine._active_stream is None
    assert calls["build"] == 0
    assert engine.generate("P", max_new_tokens=1).text == "a"
    engine.close()
