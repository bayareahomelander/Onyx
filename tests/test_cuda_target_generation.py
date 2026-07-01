import json
import os
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import probe_cuda_target_generation
import onyx_cuda.target_generation as target_generation
from onyx_cuda.target_generation import run_target_only_generation


class FakeDevice:
    def __init__(self, name="cuda:0"):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return str(self) == str(other)


class FakeTensor:
    def __init__(self, shape, *, dtype="float16", device="cuda:0", element_size=2):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = FakeDevice(device)
        self.is_cuda = device.startswith("cuda")
        self._element_size = element_size

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def element_size(self):
        return self._element_size

    def contiguous(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple) and self.ndim == 3:
            return FakeTensor(
                (self.shape[0], self.shape[-1]),
                dtype=self.dtype,
                device=str(self.device),
                element_size=self._element_size,
            )
        raise TypeError(f"unsupported fake tensor index: {key!r}")


class FakeSelectedToken:
    def __init__(self, token_id):
        self.token_id = int(token_id)

    def numel(self):
        return 1

    def item(self):
        return self.token_id


class FakeValidIds:
    def __init__(self, values):
        self.values = tuple(values)

    def numel(self):
        return len(self.values)


class FakeValidIdCache:
    def __init__(self, grammar):
        self.grammar = grammar
        self.cleared = False
        self._states = set()

    def get(self, state, _device):
        return FakeValidIds(self.grammar.get_valid_token_ids(state))

    def get_with_diagnostics(self, state, _device):
        values = self.grammar.get_valid_token_ids(state)
        cache_hit = state in self._states
        self._states.add(state)
        return SimpleNamespace(
            valid_ids=FakeValidIds(values),
            valid_token_count=len(values),
            tensor_bytes=len(values) * 8,
            cache_hit=cache_hit,
            grammar_traversal_s=0.001,
            fingerprint_s=0.002,
            cache_lookup_s=0.003,
            cuda_upload_s=0.0 if cache_hit else 0.004,
        )

    def discard(self, state):
        self._states.discard(state)

    def clear(self):
        self.cleared = True
        self._states.clear()

    @property
    def num_entries(self):
        return len(self._states)


class FakeCache:
    def __init__(self, *, layers=2, sequence_length=2, device="cuda:0", dtype="float16"):
        self._device = device
        self._dtype = dtype
        self.layers = []
        self.grow(sequence_length, layer_count=layers)

    def grow(self, sequence_length, *, layer_count=None):
        layer_count = len(self.layers) if layer_count is None else layer_count
        self.layers = [
            (
                FakeTensor((1, 2, sequence_length, 4), dtype=self._dtype, device=self._device),
                FakeTensor((1, 2, sequence_length, 4), dtype=self._dtype, device=self._device),
            )
            for _ in range(layer_count)
        ]

    def __iter__(self):
        return iter(self.layers)

    def get_seq_length(self):
        return self.layers[0][0].shape[-2]


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, *_args, **_kwargs):
        return FakeEncoded()

    def decode(self, token_ids):
        return b"".join(self.vocabulary[token_id] for token_id in token_ids).decode("ascii")


class FakeEncoded(dict):
    def __init__(self):
        super().__init__(
            input_ids=FakeTensor((1, 2), dtype="int64", element_size=8),
            attention_mask=FakeTensor((1, 2), dtype="int64", element_size=8),
        )

    def to(self, _device):
        return self


class FakeConfig:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.num_hidden_layers = 2


class FakeGrammar:
    target_length = 3

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.states = {1: b""}
        self.next_state = 2
        self.released_states = []
        self.compiled_constraint = None

    def compile_regex(self, regex):
        self.compiled_constraint = ("regex", regex)

    def compile_json_schema(self, schema):
        self.compiled_constraint = ("json_schema", schema)

    def init_state(self):
        return 1

    def get_valid_token_ids(self, state):
        self.states[state]
        return list(range(len(self.vocabulary)))

    def advance_state(self, state, token_id):
        next_state = self.next_state
        self.next_state += 1
        self.states[next_state] = self.states[state] + self.vocabulary[token_id]
        return next_state

    def is_match_state(self, state):
        return len(self.states[state]) >= self.target_length

    def release_states(self, states):
        for state in states:
            self.released_states.append(state)
            self.states.pop(state, None)


def _install_fake_runtime(
    monkeypatch,
    *,
    selected_token_ids=(0, 1, 2),
    target_length=3,
    vocabulary=None,
    fail_decode_at=None,
    grammar_compile_fails=False,
    cleanup_fails=False,
    grammar_factory=None,
):
    references = {}
    cleanup_calls = []
    grammars = []
    call_counts = {"model_load": 0, "prefill": 0, "decode": 0}
    selected_tokens = iter(selected_token_ids)
    decode_failure = RuntimeError("synthetic cached decode failure")
    grammar_compile_failure = ValueError("synthetic grammar compile failure")
    cleanup_failure = RuntimeError("synthetic empty-cache failure")
    vocabulary = vocabulary or [b"1", b"2", b"3", b"4"]

    class FakeCuda:
        @staticmethod
        def device(_device):
            return nullcontext()

        @staticmethod
        def memory_allocated(_device):
            return 0

        @staticmethod
        def memory_reserved(_device):
            return 0

        @staticmethod
        def max_memory_allocated(_device):
            return 8192

        @staticmethod
        def reset_peak_memory_stats():
            cleanup_calls.append("reset_peak_memory_stats")

        @staticmethod
        def empty_cache():
            alive = {name for name, reference in references.items() if reference() is not None}
            assert not alive, f"resources still alive before empty_cache: {sorted(alive)}"
            cleanup_calls.append("empty_cache")
            if cleanup_fails:
                raise cleanup_failure

        @staticmethod
        def synchronize(_device):
            cleanup_calls.append("synchronize")

    class FakeTorch:
        cuda = FakeCuda()
        float16 = "float16"
        float32 = "float32"
        int64 = "int64"

        @staticmethod
        def device(_kind, index):
            return FakeDevice(f"cuda:{index}")

        @staticmethod
        def no_grad():
            return nullcontext()

        @staticmethod
        def is_tensor(value):
            return isinstance(value, FakeTensor)

        @staticmethod
        def tensor(values, *, dtype, device):
            return FakeTensor(
                (len(values), len(values[0])),
                dtype=dtype,
                device=str(device),
                element_size=8,
            )

        @staticmethod
        def ones(shape, *, dtype, device):
            return FakeTensor(shape, dtype=dtype, device=str(device), element_size=8)

        @staticmethod
        def cat(tensors, dim):
            assert dim == -1
            first, second = tensors
            return FakeTensor(
                (first.shape[0], first.shape[1] + second.shape[1]),
                dtype=first.dtype,
                device=str(first.device),
                element_size=8,
            )

    class FakeOutputs:
        def __init__(self, cache, label):
            self.logits = FakeTensor((1, 1, len(vocabulary)))
            self.past_key_values = cache
            references[f"{label}_outputs"] = weakref.ref(self)
            references[f"{label}_logits"] = weakref.ref(self.logits)

    class FakeModel:
        def __init__(self):
            self.cache = None

        def eval(self):
            return self

        def __call__(self, **kwargs):
            if "past_key_values" not in kwargs:
                assert kwargs["use_cache"] is True
                assert kwargs["logits_to_keep"] == 1
                call_counts["prefill"] += 1
                self.cache = FakeCache(sequence_length=2)
                references["cache"] = weakref.ref(self.cache)
                return FakeOutputs(self.cache, "prefill")

            call_counts["decode"] += 1
            assert kwargs["past_key_values"] is self.cache
            assert kwargs["input_ids"].shape == (1, 1)
            if fail_decode_at == call_counts["decode"]:
                raise decode_failure
            self.cache.grow(self.cache.get_seq_length() + 1)
            return FakeOutputs(self.cache, "decode")

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            call_counts["model_load"] += 1
            model = FakeModel()
            references["model"] = weakref.ref(model)
            return model

    if grammar_factory is None:

        class RuntimeGrammar(FakeGrammar):
            def __init__(self, grammar_vocabulary):
                super().__init__(grammar_vocabulary)
                grammars.append(self)

            def compile_json_schema(self, schema):
                if grammar_compile_fails:
                    raise grammar_compile_failure
                super().compile_json_schema(schema)

        RuntimeGrammar.target_length = target_length
        runtime_grammar_factory = RuntimeGrammar
    else:

        def runtime_grammar_factory(grammar_vocabulary):
            grammar = grammar_factory(grammar_vocabulary)
            grammars.append(grammar)
            return grammar

    def load_metadata(*_args, **_kwargs):
        config = FakeConfig(len(vocabulary))
        tokenizer = FakeTokenizer(vocabulary)
        references["config"] = weakref.ref(config)
        references["tokenizer"] = weakref.ref(tokenizer)
        return config, tokenizer, "resolved-revision"

    def inspect_tokenizer(*_args, **_kwargs):
        return SimpleNamespace(compatible=True, errors=[]), vocabulary

    def selector(_logits, valid_ids, *, check_inputs):
        assert check_inputs is True
        token_id = next(selected_tokens)
        assert token_id in valid_ids.values
        selected = FakeSelectedToken(token_id)
        references[f"selected_{token_id}"] = weakref.ref(selected)
        return selected, SimpleNamespace(
            input_preparation_s=0.005,
            input_validation_s=0.006,
            extension_load_s=0.007,
            selector_launch_s=0.008,
        )

    original_collect = target_generation.gc.collect

    class TrackingValidIdCache(FakeValidIdCache):
        def get_with_diagnostics(self, state, device):
            lookup = super().get_with_diagnostics(state, device)
            references[f"valid_ids_{state}"] = weakref.ref(lookup.valid_ids)
            return lookup

    def recording_collect():
        cleanup_calls.append("gc.collect")
        return original_collect()

    monkeypatch.setattr(
        target_generation,
        "_require_runtime",
        lambda: (
            FakeTorch,
            FakeModelLoader,
            lambda **_kwargs: object(),
            runtime_grammar_factory,
        ),
    )
    monkeypatch.setattr(target_generation, "_load_tokenizer_metadata", load_metadata)
    monkeypatch.setattr(target_generation, "inspect_loaded_tokenizer", inspect_tokenizer)
    monkeypatch.setattr(target_generation, "CudaValidIdCache", TrackingValidIdCache)
    monkeypatch.setattr(target_generation, "masked_argmax_tensor_with_diagnostics", selector)
    monkeypatch.setattr(target_generation.gc, "collect", recording_collect)

    return SimpleNamespace(
        references=references,
        grammars=grammars,
        cleanup_calls=cleanup_calls,
        call_counts=call_counts,
        decode_failure=decode_failure,
        grammar_compile_failure=grammar_compile_failure,
        cleanup_failure=cleanup_failure,
    )


def test_target_only_generation_reuses_cache_for_repeated_decode(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch, selected_token_ids=(0, 1, 2), target_length=3)

    report = run_target_only_generation(max_new_tokens=4)

    assert report.passed is True
    assert report.output_text == "123"
    assert report.generated_token_ids == (0, 1, 2)
    assert report.finish_reason == "grammar_complete"
    assert report.cached_decode_steps == 2
    assert report.prefill_cache.sequence_length == 2
    assert report.final_cache.sequence_length == 4
    assert len(report.selection_diagnostics) == 3
    assert report.valid_id_cache_hits == 0
    assert report.valid_id_cache_misses == 3
    assert report.unique_grammar_states == 3
    assert report.peak_valid_id_cache_entries == 1
    assert report.peak_valid_id_tensor_bytes == 32
    assert report.timings.grammar_traversal_s == pytest.approx(0.003)
    assert report.timings.valid_id_fingerprint_s == pytest.approx(0.006)
    assert report.timings.valid_id_cache_lookup_s == pytest.approx(0.009)
    assert report.timings.valid_id_upload_s == pytest.approx(0.012)
    assert report.timings.selector_input_preparation_s == pytest.approx(0.015)
    assert report.timings.selector_input_validation_s == pytest.approx(0.018)
    assert report.timings.selector_extension_load_s == pytest.approx(0.021)
    assert report.timings.selector_launch_s == pytest.approx(0.024)
    formatted = target_generation.format_target_generation_report(report)
    assert "Valid-ID cache: hits=0, misses=3, unique_states=3, peak_entries=1" in formatted
    assert "Rust grammar traversal:" in formatted
    assert "Selector launch:" in formatted
    assert (
        report.final_cache.sequence_length == report.input_token_count + report.cached_decode_steps
    )
    assert runtime.call_counts == {"model_load": 1, "prefill": 1, "decode": 2}
    assert all(reference() is None for reference in runtime.references.values())
    assert runtime.cleanup_calls[-3:] == ["gc.collect", "empty_cache", "synchronize"]


def test_target_only_generation_runs_json_schema_through_repeated_decode(monkeypatch):
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
            "required": ["status"],
        },
        separators=(",", ":"),
    )
    vocabulary = [b"{", b'"status"', b":", b'"ok"', b"}"]
    runtime = _install_fake_runtime(
        monkeypatch,
        selected_token_ids=(0, 1, 2, 3, 4),
        target_length=sum(len(token) for token in vocabulary),
        vocabulary=vocabulary,
    )

    report = run_target_only_generation(
        regex=None,
        json_schema=schema,
    )

    assert report.passed is True
    assert report.constraint_type == "json_schema"
    assert report.regex is None
    assert report.json_schema == schema
    assert report.prompt == target_generation.DEFAULT_JSON_PROMPT
    assert report.max_new_tokens == target_generation.DEFAULT_JSON_MAX_NEW_TOKENS
    assert report.output_text == '{"status":"ok"}'
    assert report.output_json_valid is True
    assert report.finish_reason == "grammar_complete"
    assert report.cached_decode_steps == 4
    serialized = report.to_dict()
    assert serialized["constraint_type"] == "json_schema"
    assert serialized["output_json_valid"] is True
    json.dumps(serialized)
    assert runtime.call_counts == {"model_load": 1, "prefill": 1, "decode": 4}
    assert runtime.grammars[0].compiled_constraint == ("json_schema", schema)
    assert runtime.grammars[0].states == {}


def _rust_grammar_or_skip():
    try:
        from onyx._rust import GrammarConstraint
    except ImportError:
        pytest.skip("onyx Rust extension is not available")
    return GrammarConstraint


def test_target_only_generation_enforces_nested_object_with_real_rust_grammar(monkeypatch):
    grammar_factory = _rust_grammar_or_skip()
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["ok"]},
                    },
                    "required": ["status"],
                }
            },
            "required": ["profile"],
        },
        separators=(",", ":"),
    )
    vocabulary = [
        b"{",
        b'"profile"',
        b":",
        b"{",
        b'"status"',
        b":",
        b'"ok"',
        b"}",
        b"}",
    ]
    runtime = _install_fake_runtime(
        monkeypatch,
        selected_token_ids=tuple(range(len(vocabulary))),
        vocabulary=vocabulary,
        grammar_factory=grammar_factory,
    )

    report = run_target_only_generation(
        regex=None,
        json_schema=schema,
        max_new_tokens=len(vocabulary),
    )

    assert report.passed is True
    assert json.loads(report.output_text) == {"profile": {"status": "ok"}}
    assert report.finish_reason == "grammar_complete"
    assert report.generated_tokens == len(vocabulary)
    assert report.cached_decode_steps == len(vocabulary) - 1
    assert report.valid_id_cache_hits == 0
    assert report.valid_id_cache_misses == len(vocabulary)
    assert runtime.call_counts == {
        "model_load": 1,
        "prefill": 1,
        "decode": len(vocabulary) - 1,
    }


def test_target_only_generation_enforces_bounded_typed_array_with_real_rust_grammar(
    monkeypatch,
):
    grammar_factory = _rust_grammar_or_skip()
    schema = json.dumps(
        {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        separators=(",", ":"),
    )
    vocabulary = [b"[", b"1", b",", b"2", b"]"]
    runtime = _install_fake_runtime(
        monkeypatch,
        selected_token_ids=tuple(range(len(vocabulary))),
        vocabulary=vocabulary,
        grammar_factory=grammar_factory,
    )

    report = run_target_only_generation(
        regex=None,
        json_schema=schema,
        max_new_tokens=len(vocabulary),
    )

    assert report.passed is True
    assert json.loads(report.output_text) == [1, 2]
    assert report.finish_reason == "grammar_complete"
    assert report.generated_tokens == len(vocabulary)
    assert report.cached_decode_steps == len(vocabulary) - 1
    assert report.valid_id_cache_hits == 0
    assert report.valid_id_cache_misses == len(vocabulary)
    assert runtime.call_counts == {
        "model_load": 1,
        "prefill": 1,
        "decode": len(vocabulary) - 1,
    }


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            '{"type":"object","properties":{},"additionalProperties":false}',
            "additionalProperties",
        ),
        ('{"type":"mystery"}', "unsupported JSON Schema type"),
        (
            '{"type":"object","properties":{},"required":["missing"]}',
            "not declared in properties",
        ),
        (
            '{"type":"string","enum":["ok"],"pattern":"^ok$"}',
            "cannot be combined",
        ),
    ],
)
def test_target_only_generation_rejects_unsupported_schema_before_runtime(
    monkeypatch, schema, message
):
    monkeypatch.setattr(
        target_generation,
        "_require_runtime",
        lambda: pytest.fail("runtime loading must not start for an unsupported schema"),
    )

    with pytest.raises(ValueError, match=message):
        run_target_only_generation(regex=None, json_schema=schema)


def test_target_only_generation_requires_one_constraint_before_runtime(monkeypatch):
    monkeypatch.setattr(
        target_generation,
        "_require_runtime",
        lambda: pytest.fail("runtime loading must not start for conflicting constraints"),
    )

    with pytest.raises(ValueError, match="exactly one"):
        run_target_only_generation(
            regex="x",
            json_schema='{"type":"string"}',
        )


def test_target_only_generation_compiles_schema_before_model_loading(monkeypatch):
    schema = '{"type":"object","properties":{}}'
    runtime = _install_fake_runtime(monkeypatch, grammar_compile_fails=True)

    with pytest.raises(ValueError, match="synthetic grammar compile failure") as raised:
        run_target_only_generation(regex=None, json_schema=schema)

    assert raised.value is runtime.grammar_compile_failure
    assert runtime.call_counts == {"model_load": 0, "prefill": 0, "decode": 0}
    assert all(reference() is None for reference in runtime.references.values())
    assert runtime.cleanup_calls[-3:] == ["gc.collect", "empty_cache", "synchronize"]


def test_target_only_generation_supports_stop_strings(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch, selected_token_ids=(0, 1, 2), target_length=4)

    report = run_target_only_generation(max_new_tokens=4, stop_strings=("2",))

    assert report.passed is False
    assert report.finish_reason == "stop"
    assert report.output_text == "1"
    assert report.generated_token_ids == (0, 1)
    assert report.cached_decode_steps == 1
    assert report.final_cache.sequence_length == 3
    assert runtime.call_counts == {"model_load": 1, "prefill": 1, "decode": 1}


def test_target_only_generation_preserves_decode_failure_and_releases_resources(monkeypatch):
    runtime = _install_fake_runtime(
        monkeypatch,
        selected_token_ids=(0, 1),
        target_length=3,
        fail_decode_at=1,
        cleanup_fails=True,
    )

    with pytest.raises(RuntimeError, match="synthetic cached decode failure") as raised:
        run_target_only_generation(max_new_tokens=3)

    assert raised.value is runtime.decode_failure
    assert all(reference() is None for reference in runtime.references.values())
    assert getattr(raised.value, "_onyx_cleanup_failures") == (
        "torch.cuda.empty_cache: RuntimeError: synthetic empty-cache failure",
    )
    assert runtime.cleanup_calls[-3:] == ["gc.collect", "empty_cache", "synchronize"]


def test_target_only_generation_rejects_custom_model_before_runtime_loading(monkeypatch):
    monkeypatch.setattr(
        target_generation,
        "_require_runtime",
        lambda: pytest.fail("runtime loading must not start for an unsupported model"),
    )

    with pytest.raises(ValueError, match="only supports the validated model"):
        run_target_only_generation("other/model")


def test_target_generation_cli_uses_fixed_qwen_model_and_revision(monkeypatch):
    calls = []

    def record(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(probe_cuda_target_generation, "run_target_only_generation", record)
    monkeypatch.setattr(
        probe_cuda_target_generation, "format_target_generation_report", lambda _report: "ok"
    )

    assert probe_cuda_target_generation.main([]) == 0
    assert calls == [
        (
            target_generation.DEFAULT_MODEL_ID,
            {
                "revision": target_generation.DEFAULT_MODEL_REVISION,
                "prompt": target_generation.DEFAULT_PROMPT,
                "regex": target_generation.DEFAULT_REGEX,
                "json_schema": None,
                "max_new_tokens": target_generation.DEFAULT_MAX_NEW_TOKENS,
                "stop_strings": None,
                "local_files_only": False,
                "device_index": 0,
            },
        )
    ]
    assert not hasattr(probe_cuda_target_generation.build_parser().parse_args([]), "model")


def test_target_generation_cli_reads_json_schema_file(monkeypatch, tmp_path):
    schema = '{"type":"object","properties":{}}'
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(schema, encoding="utf-8")
    calls = []

    def record(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(probe_cuda_target_generation, "run_target_only_generation", record)
    monkeypatch.setattr(
        probe_cuda_target_generation, "format_target_generation_report", lambda _report: "ok"
    )

    assert (
        probe_cuda_target_generation.main(
            ["--json-schema-file", str(schema_path), "--prompt", "Return {}"]
        )
        == 0
    )
    assert calls == [
        (
            target_generation.DEFAULT_MODEL_ID,
            {
                "revision": target_generation.DEFAULT_MODEL_REVISION,
                "prompt": "Return {}",
                "regex": None,
                "json_schema": schema,
                "max_new_tokens": target_generation.DEFAULT_JSON_MAX_NEW_TOKENS,
                "stop_strings": None,
                "local_files_only": False,
                "device_index": 0,
            },
        )
    ]


def test_target_generation_cli_returns_dependency_error(monkeypatch):
    monkeypatch.setattr(
        probe_cuda_target_generation,
        "run_target_only_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic load failure")),
    )

    assert probe_cuda_target_generation.main([]) == 2


@pytest.mark.skipif(
    os.environ.get("ONYX_RUN_REAL_MODEL_TEST") != "1",
    reason="set ONYX_RUN_REAL_MODEL_TEST=1 to load the pinned quantized model",
)
def test_live_quantized_target_only_generation():
    report = run_target_only_generation(local_files_only=True)

    assert report.passed is True
    assert report.expected_logits_width == 151_936
    assert report.prefill_cache.layer_count == 24
    assert report.final_cache.layer_count == 24
    assert report.generated_tokens == 4
    assert report.cached_decode_steps == 3
    assert report.memory_snapshots[-1].phase == "after_cleanup"


@pytest.mark.skipif(
    os.environ.get("ONYX_RUN_REAL_MODEL_TEST") != "1",
    reason="set ONYX_RUN_REAL_MODEL_TEST=1 to load the pinned quantized model",
)
def test_live_quantized_target_only_json_generation():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
            "required": ["status"],
        },
        separators=(",", ":"),
    )

    report = run_target_only_generation(
        prompt='Return exactly this compact JSON object and nothing else: {"status":"ok"}',
        regex=None,
        json_schema=schema,
        max_new_tokens=16,
        local_files_only=True,
    )

    assert report.passed is True
    assert report.constraint_type == "json_schema"
    assert report.output_json_valid is True
    assert json.loads(report.output_text) == {"status": "ok"}
    assert report.finish_reason == "grammar_complete"
    assert report.final_cache.layer_count == 24
    assert (
        report.final_cache.sequence_length == report.input_token_count + report.cached_decode_steps
    )
    assert report.memory_snapshots[-1].phase == "after_cleanup"


@pytest.mark.skipif(
    os.environ.get("ONYX_RUN_REAL_MODEL_TEST") != "1",
    reason="set ONYX_RUN_REAL_MODEL_TEST=1 to load the pinned quantized model",
)
@pytest.mark.parametrize(
    ("schema", "prompt", "max_new_tokens", "expected"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["ok"]},
                        },
                        "required": ["status"],
                    }
                },
                "required": ["profile"],
            },
            'Return exactly this compact JSON object: {"profile":{"status":"ok"}}',
            32,
            {"profile": {"status": "ok"}},
        ),
        (
            {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "Return one compact JSON array containing exactly two integers.",
            16,
            None,
        ),
    ],
    ids=("nested-object", "bounded-integer-array"),
)
def test_live_quantized_hardened_json_generation(
    schema,
    prompt,
    max_new_tokens,
    expected,
):
    report = run_target_only_generation(
        prompt=prompt,
        regex=None,
        json_schema=json.dumps(schema, separators=(",", ":")),
        max_new_tokens=max_new_tokens,
        local_files_only=True,
    )

    assert report.passed is True
    parsed = json.loads(report.output_text)
    if expected is None:
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert all(isinstance(item, int) and not isinstance(item, bool) for item in parsed)
    else:
        assert parsed == expected
    assert report.finish_reason == "grammar_complete"
    assert report.valid_id_cache_hits == 0
    assert report.valid_id_cache_misses == report.generated_tokens
    assert report.unique_grammar_states == report.generated_tokens
    assert report.peak_valid_id_cache_entries == 1
    assert report.final_cache.sequence_length == (
        report.input_token_count + report.cached_decode_steps
    )
    assert report.memory_snapshots[-1].phase == "after_cleanup"
