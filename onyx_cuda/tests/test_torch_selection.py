import importlib
import sys

import pytest

import onyx_cuda.torch_selection as torch_selection_module
from onyx_cuda import MAX_SAMPLING_SEED, TemperatureTopPSelection
from onyx_cuda.torch_backend import (
    TorchBackendExecutionError,
    TorchBackendImportError,
    TorchBackendInvariantError,
    TorchBackendLoadError,
)
from onyx_cuda.torch_selection import create_cuda_sampler


def test_public_factory_rejects_invalid_inputs_before_import(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(torch_selection_module.importlib, "import_module", unexpected_import)

    with pytest.raises(TypeError, match="TemperatureTopPSelection"):
        create_cuda_sampler(None)
    with pytest.raises(TypeError, match="device_index"):
        create_cuda_sampler(
            TemperatureTopPSelection(1.0, 1.0, 0),
            device_index=True,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        create_cuda_sampler(
            TemperatureTopPSelection(1.0, 1.0, 0),
            device_index=-1,
        )


def test_public_factory_reports_missing_pytorch(monkeypatch):
    def missing_torch(name):
        raise ImportError("missing")

    monkeypatch.setattr(torch_selection_module.importlib, "import_module", missing_torch)

    with pytest.raises(TorchBackendImportError, match="PyTorch could not be imported"):
        create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0))


def test_public_factory_reports_cuda_unavailable(monkeypatch):
    class UnavailableCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = UnavailableCuda()

    monkeypatch.setattr(
        torch_selection_module.importlib,
        "import_module",
        lambda name: FakeTorch(),
    )

    with pytest.raises(TorchBackendLoadError, match="CUDA unavailable"):
        create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0))


try:
    torch = importlib.import_module("torch")
except (ImportError, OSError):
    torch = None
CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_sampler_replays_exact_seeded_sequence_on_private_generator():
    policy = TemperatureTopPSelection(temperature=1.0, top_p=1.0, seed=7)
    logits = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float16, device="cuda:0")

    first = create_cuda_sampler(policy)
    second = create_cuda_sampler(policy)

    assert tuple(first(logits) for _ in range(6)) == (0, 2, 2, 0, 2, 1)
    assert tuple(second(logits) for _ in range(6)) == (0, 2, 2, 0, 2, 1)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_sampler_accepts_the_portable_maximum_seed():
    sampler = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, MAX_SAMPLING_SEED))

    assert sampler(torch.tensor((0.0,), device="cuda:0")) == 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_sampler_does_not_advance_global_cuda_rng():
    original = torch.cuda.get_rng_state(0).clone()
    try:
        torch.cuda.manual_seed(1234)
        before = torch.cuda.get_rng_state(0).clone()
        sampler = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 7))
        logits = torch.tensor((0.0, 1.0, 2.0), device="cuda:0")

        sampler(logits)

        assert torch.equal(torch.cuda.get_rng_state(0), before)
    finally:
        torch.cuda.set_rng_state(original, 0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_top_p_uses_stable_token_id_ties_and_keeps_one_token():
    sampler = create_cuda_sampler(TemperatureTopPSelection(1.0, 0.5, 99))
    logits = torch.tensor((0.0, 0.0), device="cuda:0")

    assert tuple(sampler(logits) for _ in range(10)) == (0,) * 10


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_temperature_scaling_is_stable_for_extreme_finite_values():
    sampler = create_cuda_sampler(TemperatureTopPSelection(1e-20, 1.0, 0))
    logits = torch.tensor((1000.0, 999.0), dtype=torch.float32, device="cuda:0")

    assert sampler(logits) == 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_sampler_supports_infinity_semantics_and_masking():
    masked = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0))
    assert masked(
        torch.tensor((float("-inf"), 0.0, float("-inf")), device="cuda:0")
    ) == 1

    positive = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0))
    logits = torch.tensor((float("inf"), 0.0, float("inf")), device="cuda:0")
    assert tuple(positive(logits) for _ in range(6)) == (2, 0, 2, 0, 0, 2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (lambda: torch.empty(0, device="cuda:0"), "nonempty"),
        (lambda: torch.zeros((1, 2), device="cuda:0"), "one nonempty"),
        (lambda: torch.zeros(2), "non-CUDA"),
        (lambda: torch.tensor((0, 1), device="cuda:0"), "floating-point"),
        (lambda: torch.tensor((0.0, float("nan")), device="cuda:0"), "cannot contain NaN"),
        (
            lambda: torch.tensor((float("-inf"), float("-inf")), device="cuda:0"),
            "cannot all be -inf",
        ),
    ],
)
def test_cuda_sampler_rejects_invalid_or_degenerate_logits(logits, message):
    sampler = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0))

    with pytest.raises(TorchBackendInvariantError, match=message):
        sampler(logits())


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_sampler_stays_on_the_configured_device():
    sampler = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0), device_index=0)

    class WrongDeviceLogits:
        shape = (2,)
        is_cuda = True
        device = "cuda:1"

    with pytest.raises(TorchBackendInvariantError, match="expected cuda:0"):
        sampler(WrongDeviceLogits())


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native selector tests")
def test_cuda_sampler_wraps_tensor_execution_failures(monkeypatch):
    sampler = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0))
    logits = torch.tensor((0.0, 1.0), device="cuda:0")

    def fail_softmax(*args, **kwargs):
        raise RuntimeError("softmax failed")

    monkeypatch.setattr(torch, "softmax", fail_softmax)

    with pytest.raises(TorchBackendExecutionError, match="softmax failed"):
        sampler(logits)


def test_normal_module_import_does_not_import_torch(monkeypatch):
    requested = []
    original = importlib.import_module

    def record(name, *args, **kwargs):
        requested.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", record)
    sys.modules.pop("onyx_cuda.torch_selection", None)
    importlib.import_module("onyx_cuda.torch_selection")

    assert "torch" not in requested
