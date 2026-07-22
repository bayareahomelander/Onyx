"""Lazy, model-free readiness checks for the selected PyTorch CUDA runtime."""

from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass
from typing import Any

from ._torch_install import torch_cuda_unavailable_message, torch_import_error_message


class TorchCudaError(RuntimeError):
    """Base error raised by PyTorch CUDA readiness checks."""


class TorchCudaImportError(TorchCudaError):
    """Raised when PyTorch cannot be imported."""


class TorchCudaUnavailableError(TorchCudaError):
    """Raised when the installed PyTorch build cannot use a requested CUDA device."""


class TorchCudaProbeError(TorchCudaError):
    """Raised when CUDA tensor execution or cleanup fails."""


@dataclass(frozen=True, slots=True)
class TorchCudaReadiness:
    """Measured PyTorch CUDA runtime and allocator state from a model-free probe."""

    torch_version: str
    compiled_cuda_version: str
    device_index: int
    device_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    free_memory_before_bytes: int
    free_memory_after_bytes: int
    peak_allocated_bytes: int
    allocated_after_bytes: int
    reserved_after_bytes: int
    allocator_backend: str
    selected_token_id: int

    @property
    def total_memory_mib(self) -> int:
        return self.total_memory_bytes // (1024 * 1024)


def probe_torch_cuda(device_index: int = 0) -> TorchCudaReadiness:
    """Initialize PyTorch CUDA, run native mask/argmax, and report allocator cleanup."""

    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index cannot be negative")

    try:
        torch_module = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchCudaImportError(
            torch_import_error_message(exc, context="PyTorch could not be imported")
        ) from exc

    return _probe_torch_module(torch_module, device_index)


def _probe_torch_module(torch_module: Any, device_index: int) -> TorchCudaReadiness:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise TorchCudaUnavailableError(torch_cuda_unavailable_message(torch_module))

    device_count = cuda.device_count()
    if device_index >= device_count:
        raise TorchCudaUnavailableError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )

    device = torch_module.device(f"cuda:{device_index}")
    properties = cuda.get_device_properties(device)
    free_before, total_memory = cuda.mem_get_info(device)
    cuda.reset_peak_memory_stats(device)

    probe_failure = None
    try:
        selected_token_id, peak_allocated = _run_tensor_probe(torch_module, device)
        if selected_token_id != 2:
            raise TorchCudaProbeError(
                f"CUDA mask/argmax selected token {selected_token_id}; expected token 2"
            )
    except TorchCudaProbeError as exc:
        probe_failure = exc
    except Exception as exc:
        probe_failure = TorchCudaProbeError(f"PyTorch CUDA tensor probe failed: {exc}")

    gc.collect()
    cleanup_failure = None
    try:
        cuda.empty_cache()
        cuda.synchronize(device)
    except Exception as exc:
        cleanup_failure = TorchCudaProbeError(f"PyTorch CUDA cleanup failed: {exc}")

    if probe_failure is not None:
        if cleanup_failure is not None:
            raise TorchCudaProbeError(
                f"{probe_failure}; cleanup also failed: {cleanup_failure}"
            ) from probe_failure
        raise probe_failure
    if cleanup_failure is not None:
        raise cleanup_failure

    free_after, _ = cuda.mem_get_info(device)
    allocator_backend = str(cuda.get_allocator_backend())

    return TorchCudaReadiness(
        torch_version=str(torch_module.__version__),
        compiled_cuda_version=str(torch_module.version.cuda or "unknown"),
        device_index=device_index,
        device_name=str(properties.name),
        compute_capability=tuple(cuda.get_device_capability(device)),
        total_memory_bytes=int(total_memory),
        free_memory_before_bytes=int(free_before),
        free_memory_after_bytes=int(free_after),
        peak_allocated_bytes=int(peak_allocated),
        allocated_after_bytes=int(cuda.memory_allocated(device)),
        reserved_after_bytes=int(cuda.memory_reserved(device)),
        allocator_backend=allocator_backend,
        selected_token_id=selected_token_id,
    )


def _run_tensor_probe(torch_module: Any, device: Any) -> tuple[int, int]:
    """Run CUDA tensors in a short-lived scope so references are gone before cleanup."""

    with torch_module.inference_mode():
        logits = torch_module.tensor([[-2.0, 1.0, 7.0, 3.0]], device=device)
        valid_mask = torch_module.tensor([[False, True, True, False]], device=device)
        masked_logits = torch_module.where(valid_mask, logits, float("-inf"))
        selected = torch_module.argmax(masked_logits, dim=-1)
        selected_token_id = int(selected.item())
        torch_module.cuda.synchronize(device)
        peak_allocated = torch_module.cuda.max_memory_allocated(device)
    return selected_token_id, peak_allocated
