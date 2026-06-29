import os
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import onyx_cuda.kv_cache_probe as kv_cache_probe
import probe_cuda_kv_cache
from onyx_cuda.kv_cache_probe import inspect_kv_cache, run_kv_cache_probe
from onyx_cuda.real_logits_handoff import OneStepSelection, SelectionTimings


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


class FakeCache:
    def __init__(self, *, layers=2, sequence_length=2, device="cuda:0", dtype="float16"):
        self.layers = []
        self._device = device
        self._dtype = dtype
        self.grow(sequence_length, layer_count=layers)

    def grow(self, sequence_length, *, layer_count=None):
        layer_count = len(self.layers) if layer_count is None else layer_count
        self.layers = [
            (
                FakeTensor(
                    (1, 2, sequence_length, 4),
                    dtype=self._dtype,
                    device=self._device,
                ),
                FakeTensor(
                    (1, 2, sequence_length, 4),
                    dtype=self._dtype,
                    device=self._device,
                ),
            )
            for _ in range(layer_count)
        ]

    def __iter__(self):
        return iter(self.layers)

    def get_seq_length(self):
        return self.layers[0][0].shape[-2]


class FakeTorchForInspection:
    float16 = "float16"
    float32 = "float32"

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)


def test_inspect_kv_cache_records_complete_layer_contract():
    cache = FakeCache(layers=2, sequence_length=3)

    snapshot = inspect_kv_cache(
        cache,
        phase="decode",
        expected_layer_count=2,
        expected_batch_size=1,
        expected_sequence_length=3,
        expected_device="cuda:0",
        expected_dtype=FakeTorchForInspection.float16,
        torch=FakeTorchForInspection,
    )

    assert snapshot.cache_type == "FakeCache"
    assert snapshot.layer_count == 2
    assert snapshot.sequence_length == 3
    assert snapshot.storage_bytes == 2 * 2 * (1 * 2 * 3 * 4) * 2
    assert snapshot.layers[0].key_shape == (1, 2, 3, 4)
    assert snapshot.layers[0].dtype == "float16"
    assert snapshot.layers[0].device == "cuda:0"


@pytest.mark.parametrize(
    ("cache", "message"),
    [
        (FakeCache(layers=1, sequence_length=2), "has 1 layers; expected 2"),
        (FakeCache(layers=2, sequence_length=1), "cache sequence length 1; expected 2"),
        (
            FakeCache(layers=2, sequence_length=2, device="cpu"),
            "must remain on CUDA",
        ),
        (
            FakeCache(layers=2, sequence_length=2, dtype="float32"),
            "must use dtype float16",
        ),
    ],
)
def test_inspect_kv_cache_rejects_contract_mismatches(cache, message):
    with pytest.raises(ValueError, match=message):
        inspect_kv_cache(
            cache,
            phase="prefill",
            expected_layer_count=2,
            expected_batch_size=1,
            expected_sequence_length=2,
            expected_device="cuda:0",
            expected_dtype=FakeTorchForInspection.float16,
            torch=FakeTorchForInspection,
        )


def _successful_selection():
    return OneStepSelection(
        valid_token_count=2,
        raw_argmax_token_id=0,
        raw_argmax_text="x",
        raw_argmax_logit=10.0,
        raw_argmax_was_valid=False,
        selected_token_id=1,
        selected_token_text="1",
        selected_token_bytes_hex="31",
        selected_token_logit=3.0,
        selected_token_was_valid=True,
        invalid_raw_argmax_excluded=True,
        grammar_matched_after_selection=True,
        timings=SelectionTimings(
            valid_id_lookup_s=0.0,
            selection_call_s=0.0,
            result_sync_s=0.0,
            grammar_advance_s=0.0,
        ),
    )


def _install_fake_runtime(monkeypatch, *, fail_decode=False, cleanup_fails=False):
    references = {}
    cleanup_calls = []
    decode_failure = RuntimeError("synthetic cached decode failure")
    cleanup_failure = RuntimeError("synthetic empty-cache failure")

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
            return 4096

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

    class FakeTorch(FakeTorchForInspection):
        cuda = FakeCuda()
        int64 = "int64"

        @staticmethod
        def device(_kind, index):
            return FakeDevice(f"cuda:{index}")

        @staticmethod
        def no_grad():
            return nullcontext()

        @staticmethod
        def tensor(values, *, dtype, device):
            return FakeTensor(
                (len(values), len(values[0])), dtype=dtype, device=str(device), element_size=8
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

    class FakeEncoded(dict):
        def __init__(self):
            super().__init__(
                input_ids=FakeTensor((1, 2), dtype="int64", element_size=8),
                attention_mask=FakeTensor((1, 2), dtype="int64", element_size=8),
            )

        def to(self, _device):
            references["encoded"] = weakref.ref(self)
            return self

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return FakeEncoded()

    class FakeOutputs:
        def __init__(self, cache, label):
            self.logits = FakeTensor((1, 1, 4))
            self.past_key_values = cache
            references[f"{label}_outputs"] = weakref.ref(self)
            references[f"{label}_logits"] = weakref.ref(self.logits)

    class FakeModel:
        def __init__(self):
            self.cache = None
            self.calls = 0

        def eval(self):
            return self

        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs["use_cache"] is True
                self.cache = FakeCache(layers=2, sequence_length=2)
                references["cache"] = weakref.ref(self.cache)
                return FakeOutputs(self.cache, "prefill")

            assert kwargs["past_key_values"] is self.cache
            if fail_decode:
                raise decode_failure
            self.cache.grow(3)
            return FakeOutputs(self.cache, "decode")

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            model = FakeModel()
            references["model"] = weakref.ref(model)
            return model

    class FakeConfig:
        vocab_size = 4
        num_hidden_layers = 2

    def load_metadata(*_args, **_kwargs):
        config = FakeConfig()
        tokenizer = FakeTokenizer()
        references["config"] = weakref.ref(config)
        references["tokenizer"] = weakref.ref(tokenizer)
        return config, tokenizer, "resolved-revision"

    def inspect_tokenizer(*_args, **_kwargs):
        return SimpleNamespace(compatible=True, errors=[]), [b"x", b"1", b"2", b"3"]

    original_collect = kv_cache_probe.gc.collect

    def recording_collect():
        cleanup_calls.append("gc.collect")
        return original_collect()

    monkeypatch.setattr(
        kv_cache_probe,
        "_require_runtime",
        lambda: (FakeTorch, FakeModelLoader, lambda **_kwargs: object(), object()),
    )
    monkeypatch.setattr(kv_cache_probe, "_load_tokenizer_metadata", load_metadata)
    monkeypatch.setattr(kv_cache_probe, "inspect_loaded_tokenizer", inspect_tokenizer)
    monkeypatch.setattr(
        kv_cache_probe,
        "select_and_advance_one_token",
        lambda *_args, **_kwargs: _successful_selection(),
    )
    monkeypatch.setattr(kv_cache_probe.gc, "collect", recording_collect)

    return SimpleNamespace(
        references=references,
        cleanup_calls=cleanup_calls,
        decode_failure=decode_failure,
        cleanup_failure=cleanup_failure,
    )


def test_kv_cache_probe_validates_prefill_and_one_token_growth(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)

    report = run_kv_cache_probe()

    assert report.passed is True
    assert report.input_token_count == 2
    assert report.prefill_cache.sequence_length == 2
    assert report.decode_cache.sequence_length == 3
    assert report.cache_sequence_growth == 1
    assert report.cache_object_reused is True
    assert report.prefill_logits_shape == (1, 1, 4)
    assert report.decode_logits_shape == (1, 1, 4)
    assert report.memory_snapshots[-1].phase == "after_cleanup"
    assert all(reference() is None for reference in runtime.references.values())
    assert runtime.cleanup_calls[-3:] == ["gc.collect", "empty_cache", "synchronize"]


def test_kv_cache_probe_preserves_decode_failure_and_releases_resources(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch, fail_decode=True, cleanup_fails=True)

    with pytest.raises(RuntimeError, match="synthetic cached decode failure") as raised:
        run_kv_cache_probe()

    assert raised.value is runtime.decode_failure
    assert all(reference() is None for reference in runtime.references.values())
    assert getattr(raised.value, "_onyx_cleanup_failures") == (
        "torch.cuda.empty_cache: RuntimeError: synthetic empty-cache failure",
    )
    assert runtime.cleanup_calls[-3:] == ["gc.collect", "empty_cache", "synchronize"]


def test_kv_cache_probe_rejects_custom_model_before_runtime_loading(monkeypatch):
    monkeypatch.setattr(
        kv_cache_probe,
        "_require_runtime",
        lambda: pytest.fail("runtime loading must not start for an unsupported model"),
    )

    with pytest.raises(ValueError, match="only supports the validated model"):
        run_kv_cache_probe("other/model")


def test_kv_cache_cli_returns_dependency_error(monkeypatch):
    monkeypatch.setattr(
        probe_cuda_kv_cache,
        "run_kv_cache_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic load failure")),
    )

    assert probe_cuda_kv_cache.main([]) == 2


def test_kv_cache_cli_uses_fixed_qwen_model_and_revision(monkeypatch):
    calls = []

    def record(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(probe_cuda_kv_cache, "run_kv_cache_probe", record)
    monkeypatch.setattr(probe_cuda_kv_cache, "format_kv_cache_report", lambda _report: "ok")

    assert probe_cuda_kv_cache.main([]) == 0
    assert calls == [
        (
            kv_cache_probe.DEFAULT_MODEL_ID,
            {
                "revision": kv_cache_probe.DEFAULT_MODEL_REVISION,
                "prompt": kv_cache_probe.DEFAULT_PROMPT,
                "regex": kv_cache_probe.DEFAULT_REGEX,
                "local_files_only": False,
                "device_index": 0,
            },
        )
    ]
    assert not hasattr(probe_cuda_kv_cache.build_parser().parse_args([]), "model")


@pytest.mark.skipif(
    os.environ.get("ONYX_RUN_REAL_MODEL_TEST") != "1",
    reason="set ONYX_RUN_REAL_MODEL_TEST=1 to load the pinned quantized model",
)
def test_live_quantized_kv_cache_probe():
    report = run_kv_cache_probe(local_files_only=True)

    assert report.passed is True
    assert report.expected_logits_width == 151_936
    assert report.prefill_cache.layer_count == 24
    assert report.decode_cache.layer_count == 24
    assert report.cache_sequence_growth == 1
    assert report.memory_snapshots[-1].phase == "after_cleanup"
