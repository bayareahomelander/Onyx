import importlib
import sys

import pytest

import onyx_cuda.torch_metrics as torch_metrics_module
from onyx_cuda import (
    TRANSFORMERS_DYNAMIC_CACHE_MODE,
    TorchMetricsExecutionError,
    TorchMetricsImportError,
    TorchMetricsInvariantError,
    TorchMetricsUnavailableError,
    create_torch_metrics_session,
)


class ScriptedClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class FakeDevice:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeCuda:
    def __init__(
        self,
        *,
        available=True,
        device_count=1,
        allocated=123,
        reserved=456,
        synchronize_error=None,
        reset_error=None,
        snapshot_error=None,
    ):
        self.available = available
        self.reported_device_count = device_count
        self.allocated = allocated
        self.reserved = reserved
        self.synchronize_error = synchronize_error
        self.reset_error = reset_error
        self.snapshot_error = snapshot_error
        self.calls = []

    def is_available(self):
        return self.available

    def device_count(self):
        return self.reported_device_count

    def synchronize(self, device):
        self.calls.append(("synchronize", str(device)))
        if self.synchronize_error is not None:
            raise self.synchronize_error

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", str(device)))
        if self.reset_error is not None:
            raise self.reset_error

    def max_memory_allocated(self, device):
        self.calls.append(("max_allocated", str(device)))
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return self.allocated

    def max_memory_reserved(self, device):
        self.calls.append(("max_reserved", str(device)))
        return self.reserved


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda
        self.devices = []

    def device(self, name):
        self.devices.append(name)
        return FakeDevice(name)


def test_public_factory_rejects_invalid_inputs_before_import(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(torch_metrics_module.importlib, "import_module", unexpected_import)

    with pytest.raises(TypeError, match="device_index"):
        create_torch_metrics_session(device_index=True)
    with pytest.raises(ValueError, match="cannot be negative"):
        create_torch_metrics_session(device_index=-1)
    with pytest.raises(TypeError, match="clock"):
        create_torch_metrics_session(clock=None)


def test_public_factory_reports_missing_pytorch(monkeypatch):
    monkeypatch.setattr(
        torch_metrics_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing")),
    )

    with pytest.raises(TorchMetricsImportError, match="could not be imported"):
        create_torch_metrics_session()


def test_public_factory_reports_unavailable_or_invalid_cuda_device(monkeypatch):
    fake_torch = FakeTorch(FakeCuda(available=False))
    monkeypatch.setattr(
        torch_metrics_module.importlib,
        "import_module",
        lambda name: fake_torch,
    )
    with pytest.raises(TorchMetricsUnavailableError, match="CUDA unavailable"):
        create_torch_metrics_session()

    fake_torch = FakeTorch(FakeCuda(device_count=1))
    monkeypatch.setattr(
        torch_metrics_module.importlib,
        "import_module",
        lambda name: fake_torch,
    )
    with pytest.raises(TorchMetricsUnavailableError, match="detected 1 device"):
        create_torch_metrics_session(device_index=1)


def test_public_factory_wraps_cuda_availability_check_failure(monkeypatch):
    cuda = FakeCuda()

    def fail_availability():
        raise RuntimeError("availability failed")

    cuda.is_available = fail_availability
    monkeypatch.setattr(
        torch_metrics_module.importlib,
        "import_module",
        lambda name: FakeTorch(cuda),
    )

    with pytest.raises(TorchMetricsExecutionError, match="availability failed"):
        create_torch_metrics_session()


def test_cuda_metrics_reset_and_snapshot_the_configured_device(monkeypatch):
    cuda = FakeCuda(allocated=321, reserved=654)
    fake_torch = FakeTorch(cuda)
    monkeypatch.setattr(
        torch_metrics_module.importlib,
        "import_module",
        lambda name: fake_torch,
    )
    session = create_torch_metrics_session(
        device_index=0,
        clock=ScriptedClock((0.0, 1.0, 2.0)),
    )

    session.begin()
    with session.active():
        session.mark_first_token()
    metrics = session.finish(1)

    assert metrics.cache_mode == TRANSFORMERS_DYNAMIC_CACHE_MODE
    assert metrics.ttft == 1.0
    assert metrics.generation_time == 2.0
    assert metrics.tokens_per_second == 0.5
    assert metrics.peak_allocated_vram_bytes == 321
    assert metrics.peak_reserved_vram_bytes == 654
    assert fake_torch.devices == ["cuda:0"]
    assert cuda.calls == [
        ("synchronize", "cuda:0"),
        ("reset", "cuda:0"),
        ("synchronize", "cuda:0"),
        ("max_allocated", "cuda:0"),
        ("max_reserved", "cuda:0"),
    ]


@pytest.mark.parametrize(
    ("cuda", "operation", "message"),
    [
        (FakeCuda(synchronize_error=RuntimeError("sync failed")), "begin", "sync failed"),
        (FakeCuda(reset_error=RuntimeError("reset failed")), "begin", "reset failed"),
        (
            FakeCuda(snapshot_error=RuntimeError("snapshot failed")),
            "finish",
            "snapshot failed",
        ),
    ],
)
def test_cuda_metrics_wrap_execution_failures(monkeypatch, cuda, operation, message):
    monkeypatch.setattr(
        torch_metrics_module.importlib,
        "import_module",
        lambda name: FakeTorch(cuda),
    )
    session = create_torch_metrics_session(
        clock=ScriptedClock((0.0, 1.0, 2.0)),
    )

    if operation == "begin":
        with pytest.raises(TorchMetricsExecutionError, match=message):
            session.begin()
        return

    session.begin()
    with session.active():
        session.mark_first_token()
    with pytest.raises(TorchMetricsExecutionError, match=message):
        session.finish(1)
    session.abort()


@pytest.mark.parametrize(
    ("allocated", "reserved", "message"),
    [
        (True, 1, "allocated.*integer"),
        (-1, 1, "allocated.*negative"),
        (1, 1.5, "reserved.*integer"),
        (1, -1, "reserved.*negative"),
    ],
)
def test_cuda_metrics_reject_malformed_allocator_results(
    monkeypatch,
    allocated,
    reserved,
    message,
):
    cuda = FakeCuda(allocated=allocated, reserved=reserved)
    monkeypatch.setattr(
        torch_metrics_module.importlib,
        "import_module",
        lambda name: FakeTorch(cuda),
    )
    session = create_torch_metrics_session(
        clock=ScriptedClock((0.0, 1.0, 2.0)),
    )
    session.begin()
    with session.active():
        session.mark_first_token()

    with pytest.raises(TorchMetricsInvariantError, match=message):
        session.finish(1)


def test_normal_module_import_does_not_import_torch(monkeypatch):
    requested = []
    original = importlib.import_module

    def record(name, *args, **kwargs):
        requested.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", record)
    sys.modules.pop("onyx_cuda.torch_metrics", None)
    importlib.import_module("onyx_cuda.torch_metrics")

    assert "torch" not in requested
