from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import onyx_cuda.target_model as target_model_module
from onyx_cuda import DEFAULT_TARGET_PROFILE
from onyx_cuda.target_model import (
    QuantizedTargetCleanupError,
    QuantizedTargetImportError,
    QuantizedTargetLoadError,
    QuantizedTargetUnavailableError,
    _probe_quantized_target_modules,
    probe_quantized_target,
)


class FakeCuda:
    def __init__(self, *, available=True, device_count=1, cleanup_error=None):
        self.available = available
        self.detected_device_count = device_count
        self.cleanup_error = cleanup_error
        self.empty_cache_calls = 0
        self.synchronize_calls = 0
        self.mem_info = [(900, 1000), (500, 1000), (895, 1000)]
        self.allocated = [0, 400, 0]
        self.reserved = [0, 500, 0]

    def is_available(self):
        return self.available

    def device_count(self):
        return self.detected_device_count

    def get_device_properties(self, device):
        return SimpleNamespace(name="Fake RTX")

    def empty_cache(self):
        self.empty_cache_calls += 1
        if self.empty_cache_calls > 1 and self.cleanup_error is not None:
            raise self.cleanup_error

    def synchronize(self, device):
        self.synchronize_calls += 1

    def reset_peak_memory_stats(self, device):
        return None

    def mem_get_info(self, device):
        return self.mem_info.pop(0)

    def memory_allocated(self, device):
        return self.allocated.pop(0)

    def memory_reserved(self, device):
        return self.reserved.pop(0)

    def max_memory_allocated(self, device):
        return 450


class FakeTorch:
    __version__ = "2.6.0+cu124"
    float16 = "float16"

    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCuda()

    def device(self, name):
        return name


class FakeModel:
    is_loaded_in_4bit = True
    config = SimpleNamespace(vocab_size=8)

    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1

    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(shape=(8, 4)))

    def get_memory_footprint(self):
        return 350


class RecordingAutoModel:
    def __init__(self, model=None, error=None):
        self.model = model or FakeModel()
        self.error = error
        self.calls = []

    def from_pretrained(self, model_id, **kwargs):
        self.calls.append((model_id, kwargs))
        if self.error is not None:
            raise self.error
        return self.model


class FakeTransformers:
    __version__ = "4.57.6"

    def __init__(self, auto_model=None):
        self.AutoModelForCausalLM = auto_model or RecordingAutoModel()
        self.quantization_calls = []

    def BitsAndBytesConfig(self, **kwargs):
        self.quantization_calls.append(kwargs)
        return "quantization-config"


def fake_tokenizer_load(*args, **kwargs):
    return SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=6),
        load_seconds=0.25,
    )


def run_probe(monkeypatch, *, torch_module=None, transformers_module=None):
    monkeypatch.setattr(target_model_module, "load_qwen_tokenizer", fake_tokenizer_load)
    torch_module = torch_module or FakeTorch()
    transformers_module = transformers_module or FakeTransformers()
    readiness = _probe_quantized_target_modules(
        torch_module,
        transformers_module,
        SimpleNamespace(__version__="0.49.2"),
        profile=DEFAULT_TARGET_PROFILE,
        device_index=0,
        local_files_only=True,
    )
    return readiness, torch_module, transformers_module


def test_reports_measured_quantized_load_and_clean_unload(monkeypatch):
    readiness, torch_module, transformers_module = run_probe(monkeypatch)

    assert readiness.model_id == DEFAULT_TARGET_PROFILE.model_id
    assert readiness.revision == DEFAULT_TARGET_PROFILE.revision
    assert readiness.quantization == "bitsandbytes-nf4-double-quant"
    assert readiness.device_name == "Fake RTX"
    assert readiness.total_memory_bytes == 1000
    assert readiness.tokenizer_load_seconds == 0.25
    assert readiness.model_load_seconds >= 0
    assert readiness.unload_seconds >= 0
    assert readiness.tokenizer_vocab_size == 6
    assert readiness.model_vocab_size == 8
    assert readiness.model_memory_footprint_bytes == 350
    assert readiness.peak_allocated_bytes == 450
    assert readiness.reserved_before_bytes == 0
    assert readiness.allocated_after_unload_bytes == 0
    assert readiness.reserved_after_unload_bytes == 0
    assert torch_module.cuda.empty_cache_calls == 2
    assert torch_module.cuda.synchronize_calls == 3
    assert transformers_module.quantization_calls == [
        {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
        }
    ]


def test_uses_pinned_safe_model_loader_options(monkeypatch):
    readiness, _, transformers_module = run_probe(monkeypatch)

    assert readiness.model_vocab_size == 8
    assert transformers_module.AutoModelForCausalLM.calls == [
        (
            DEFAULT_TARGET_PROFILE.model_id,
            {
                "revision": DEFAULT_TARGET_PROFILE.revision,
                "local_files_only": True,
                "trust_remote_code": False,
                "quantization_config": "quantization-config",
                "device_map": {"": 0},
                "dtype": "float16",
                "low_cpu_mem_usage": True,
            },
        )
    ]


@pytest.mark.parametrize("missing_name", ["torch", "transformers", "bitsandbytes"])
def test_public_probe_reports_each_missing_dependency(monkeypatch, missing_name):
    def import_module(name):
        if name == missing_name:
            raise ModuleNotFoundError(f"{name} missing")
        return SimpleNamespace()

    monkeypatch.setattr(target_model_module.importlib, "import_module", import_module)

    with pytest.raises(QuantizedTargetImportError, match=f"{missing_name} missing"):
        probe_quantized_target()


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("device_index", -1, ValueError),
        ("device_index", True, TypeError),
        ("device_index", 0.5, TypeError),
        ("local_files_only", "yes", TypeError),
        ("profile", "model", TypeError),
    ],
)
def test_public_probe_rejects_invalid_input_before_import(monkeypatch, field, value, error_type):
    def unexpected_import(name):
        raise AssertionError("invalid input must fail before importing dependencies")

    monkeypatch.setattr(target_model_module.importlib, "import_module", unexpected_import)
    kwargs = {field: value}

    with pytest.raises(error_type):
        probe_quantized_target(**kwargs)


def test_reports_cuda_unavailable_before_loading_tokenizer(monkeypatch):
    monkeypatch.setattr(
        target_model_module,
        "load_qwen_tokenizer",
        lambda *args, **kwargs: pytest.fail("tokenizer must not load"),
    )

    with pytest.raises(QuantizedTargetUnavailableError, match="CUDA unavailable"):
        _probe_quantized_target_modules(
            FakeTorch(cuda=FakeCuda(available=False)),
            FakeTransformers(),
            SimpleNamespace(__version__="0.49.2"),
            profile=DEFAULT_TARGET_PROFILE,
            device_index=0,
            local_files_only=True,
        )


def test_reports_unavailable_device_index(monkeypatch):
    monkeypatch.setattr(target_model_module, "load_qwen_tokenizer", fake_tokenizer_load)

    with pytest.raises(QuantizedTargetUnavailableError, match="detected 1 device"):
        _probe_quantized_target_modules(
            FakeTorch(cuda=FakeCuda(device_count=1)),
            FakeTransformers(),
            SimpleNamespace(__version__="0.49.2"),
            profile=DEFAULT_TARGET_PROFILE,
            device_index=1,
            local_files_only=True,
        )


def test_load_failure_still_runs_cleanup(monkeypatch):
    auto_model = RecordingAutoModel(error=RuntimeError("quantization failed"))
    torch_module = FakeTorch()
    monkeypatch.setattr(target_model_module, "load_qwen_tokenizer", fake_tokenizer_load)

    with pytest.raises(QuantizedTargetLoadError, match="quantization failed"):
        _probe_quantized_target_modules(
            torch_module,
            FakeTransformers(auto_model),
            SimpleNamespace(__version__="0.49.2"),
            profile=DEFAULT_TARGET_PROFILE,
            device_index=0,
            local_files_only=True,
        )

    assert torch_module.cuda.empty_cache_calls == 2


def test_tokenizer_failure_still_runs_cleanup_without_baseline_metrics(monkeypatch):
    torch_module = FakeTorch()

    def fail_tokenizer(*args, **kwargs):
        raise RuntimeError("tokenizer failed")

    monkeypatch.setattr(target_model_module, "load_qwen_tokenizer", fail_tokenizer)

    with pytest.raises(QuantizedTargetLoadError, match="tokenizer failed"):
        _probe_quantized_target_modules(
            torch_module,
            FakeTransformers(),
            SimpleNamespace(__version__="0.49.2"),
            profile=DEFAULT_TARGET_PROFILE,
            device_index=0,
            local_files_only=True,
        )

    assert torch_module.cuda.empty_cache_calls == 1


def test_non_quantized_model_is_rejected_and_cleaned_up(monkeypatch):
    model = FakeModel()
    model.is_loaded_in_4bit = False
    torch_module = FakeTorch()
    monkeypatch.setattr(target_model_module, "load_qwen_tokenizer", fake_tokenizer_load)

    with pytest.raises(QuantizedTargetLoadError, match="did not report 4-bit"):
        _probe_quantized_target_modules(
            torch_module,
            FakeTransformers(RecordingAutoModel(model)),
            SimpleNamespace(__version__="0.49.2"),
            profile=DEFAULT_TARGET_PROFILE,
            device_index=0,
            local_files_only=True,
        )

    assert torch_module.cuda.empty_cache_calls == 2


def test_load_and_cleanup_failures_are_both_reported(monkeypatch):
    cuda = FakeCuda(cleanup_error=RuntimeError("cleanup failed"))
    auto_model = RecordingAutoModel(error=RuntimeError("load failed"))
    monkeypatch.setattr(target_model_module, "load_qwen_tokenizer", fake_tokenizer_load)

    with pytest.raises(
        QuantizedTargetCleanupError,
        match="load failed.*cleanup also failed.*cleanup failed",
    ):
        _probe_quantized_target_modules(
            FakeTorch(cuda=cuda),
            FakeTransformers(auto_model),
            SimpleNamespace(__version__="0.49.2"),
            profile=DEFAULT_TARGET_PROFILE,
            device_index=0,
            local_files_only=True,
        )


def test_excess_cleanup_residue_is_rejected(monkeypatch):
    cuda = FakeCuda()
    cuda.allocated = [0, 400, 2 * 1024 * 1024]
    monkeypatch.setattr(target_model_module, "load_qwen_tokenizer", fake_tokenizer_load)

    with pytest.raises(QuantizedTargetCleanupError, match="allowed runtime residue"):
        _probe_quantized_target_modules(
            FakeTorch(cuda=cuda),
            FakeTransformers(),
            SimpleNamespace(__version__="0.49.2"),
            profile=DEFAULT_TARGET_PROFILE,
            device_index=0,
            local_files_only=True,
        )


def test_readiness_is_immutable(monkeypatch):
    readiness, _, _ = run_probe(monkeypatch)

    with pytest.raises(FrozenInstanceError):
        readiness.model_id = "changed"
