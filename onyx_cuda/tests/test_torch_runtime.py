from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from onyx_cuda import (
    TorchCudaImportError,
    TorchCudaProbeError,
    TorchCudaUnavailableError,
    probe_torch_cuda,
)
from onyx_cuda.torch_runtime import _probe_torch_module
from onyx_cuda._torch_install import PYTORCH_CUDA_INSTALL_COMMAND


class FakeTensor:
    def __init__(self, value=None):
        self.value = value

    def item(self):
        return self.value


class FakeCuda:
    def __init__(self, *, available=True, device_count=1, cleanup_error=None):
        self.available = available
        self.detected_device_count = device_count
        self.cleanup_error = cleanup_error
        self.empty_cache_calls = 0
        self.synchronize_calls = 0
        self.mem_info_calls = 0

    def is_available(self):
        return self.available

    def device_count(self):
        return self.detected_device_count

    def get_device_properties(self, device):
        return SimpleNamespace(name="Fake RTX", total_memory=6 * 1024**3)

    def mem_get_info(self, device):
        self.mem_info_calls += 1
        free = (5 * 1024**3) - self.mem_info_calls
        return free, 6 * 1024**3

    def reset_peak_memory_stats(self, device):
        return None

    def synchronize(self, device):
        self.synchronize_calls += 1

    def max_memory_allocated(self, device):
        return 2048

    def empty_cache(self):
        self.empty_cache_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error

    def get_allocator_backend(self):
        return "native"

    def get_device_capability(self, device):
        return 8, 9

    def memory_allocated(self, device):
        return 0

    def memory_reserved(self, device):
        return 0


class FakeTorch:
    __version__ = "2.6.0+cu124"
    version = SimpleNamespace(cuda="12.4")

    def __init__(self, cuda=None, *, tensor_error=None, selected_token=2):
        self.cuda = cuda or FakeCuda()
        self.tensor_error = tensor_error
        self.selected_token = selected_token

    def device(self, name):
        return name

    def inference_mode(self):
        return nullcontext()

    def tensor(self, values, *, device):
        if self.tensor_error is not None:
            raise self.tensor_error
        return FakeTensor()

    def where(self, condition, input_tensor, other):
        return FakeTensor()

    def argmax(self, input_tensor, *, dim):
        return FakeTensor(self.selected_token)


def test_model_free_probe_reports_runtime_device_and_cleanup():
    torch_module = FakeTorch()

    readiness = _probe_torch_module(torch_module, 0)

    assert readiness.torch_version == "2.6.0+cu124"
    assert readiness.compiled_cuda_version == "12.4"
    assert readiness.device_name == "Fake RTX"
    assert readiness.compute_capability == (8, 9)
    assert readiness.total_memory_mib == 6144
    assert readiness.peak_allocated_bytes == 2048
    assert readiness.allocated_after_bytes == 0
    assert readiness.reserved_after_bytes == 0
    assert readiness.allocator_backend == "native"
    assert readiness.selected_token_id == 2
    assert torch_module.cuda.empty_cache_calls == 1
    assert torch_module.cuda.synchronize_calls == 2


def test_public_probe_reports_missing_pytorch(monkeypatch):
    def missing_module(name):
        raise ModuleNotFoundError("torch is missing")

    monkeypatch.setattr("onyx_cuda.torch_runtime.importlib.import_module", missing_module)

    with pytest.raises(TorchCudaImportError, match="torch is missing") as raised:
        probe_torch_cuda()

    assert PYTORCH_CUDA_INSTALL_COMMAND in str(raised.value)


def test_probe_reports_cuda_runtime_unavailable():
    torch_module = FakeTorch(cuda=FakeCuda(available=False))

    with pytest.raises(TorchCudaUnavailableError, match="CUDA unavailable") as raised:
        _probe_torch_module(torch_module, 0)

    assert "compatible driver" in str(raised.value)


def test_probe_identifies_cpu_only_pytorch_and_reports_install_command():
    torch_module = FakeTorch(cuda=FakeCuda(available=False))
    torch_module.__version__ = "2.13.0+cpu"
    torch_module.version = SimpleNamespace(cuda=None)

    with pytest.raises(TorchCudaUnavailableError, match="CPU-only") as raised:
        _probe_torch_module(torch_module, 0)

    assert PYTORCH_CUDA_INSTALL_COMMAND in str(raised.value)


def test_probe_rejects_unavailable_device_index():
    torch_module = FakeTorch(cuda=FakeCuda(device_count=1))

    with pytest.raises(TorchCudaUnavailableError, match="detected 1 device"):
        _probe_torch_module(torch_module, 1)


@pytest.mark.parametrize("device_index", [-1, True, 0.5, "0"])
def test_public_probe_rejects_invalid_device_index_before_import(monkeypatch, device_index):
    def unexpected_import(name):
        raise AssertionError("invalid input must fail before importing PyTorch")

    monkeypatch.setattr("onyx_cuda.torch_runtime.importlib.import_module", unexpected_import)

    with pytest.raises((TypeError, ValueError)):
        probe_torch_cuda(device_index)


def test_tensor_execution_failure_still_cleans_allocator():
    torch_module = FakeTorch(tensor_error=RuntimeError("kernel failed"))

    with pytest.raises(TorchCudaProbeError, match="kernel failed"):
        _probe_torch_module(torch_module, 0)

    assert torch_module.cuda.empty_cache_calls == 1
    assert torch_module.cuda.synchronize_calls == 1


def test_unexpected_argmax_result_is_a_probe_error_and_cleans_up():
    torch_module = FakeTorch(selected_token=1)

    with pytest.raises(TorchCudaProbeError, match="expected token 2"):
        _probe_torch_module(torch_module, 0)

    assert torch_module.cuda.empty_cache_calls == 1


def test_cleanup_failure_is_reported_explicitly():
    torch_module = FakeTorch(cuda=FakeCuda(cleanup_error=RuntimeError("cleanup failed")))

    with pytest.raises(TorchCudaProbeError, match="cleanup failed"):
        _probe_torch_module(torch_module, 0)


def test_tensor_and_cleanup_failures_are_both_reported():
    cuda = FakeCuda(cleanup_error=RuntimeError("cleanup failed"))
    torch_module = FakeTorch(cuda=cuda, tensor_error=RuntimeError("kernel failed"))

    with pytest.raises(TorchCudaProbeError, match="kernel failed.*cleanup also failed.*cleanup failed"):
        _probe_torch_module(torch_module, 0)
