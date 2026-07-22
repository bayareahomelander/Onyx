"""Lazy PyTorch CUDA diagnostics for target-generation metrics."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from typing import Any

from ._torch_install import torch_cuda_unavailable_message, torch_import_error_message
from .metrics import MetricsError, TargetMetricsSession, create_target_metrics_session


TRANSFORMERS_DYNAMIC_CACHE_MODE = "transformers_dynamic"


class TorchMetricsError(MetricsError):
    """Base error raised by PyTorch CUDA generation diagnostics."""


class TorchMetricsImportError(TorchMetricsError):
    """Raised when PyTorch cannot be imported for explicit CUDA metrics."""


class TorchMetricsUnavailableError(TorchMetricsError):
    """Raised when the requested CUDA diagnostics device is unavailable."""


class TorchMetricsExecutionError(TorchMetricsError):
    """Raised when CUDA peak-memory diagnostics cannot execute."""


class TorchMetricsInvariantError(TorchMetricsError):
    """Raised when PyTorch returns malformed CUDA metric data."""


class TorchCUDADiagnosticsSession:
    """Reset and snapshot PyTorch peak memory for one CUDA generation."""

    def __init__(self, *, torch_module: Any, device: Any) -> None:
        self._torch = torch_module
        self._device = device
        self._state = "new"

    @property
    def cache_mode(self) -> str:
        return TRANSFORMERS_DYNAMIC_CACHE_MODE

    def begin(self) -> None:
        if self._state != "new":
            raise TorchMetricsInvariantError("CUDA diagnostics can only begin once")
        try:
            self._torch.cuda.synchronize(self._device)
            self._torch.cuda.reset_peak_memory_stats(self._device)
        except Exception as exc:
            raise TorchMetricsExecutionError(
                f"CUDA peak-memory reset failed: {exc}"
            ) from exc
        self._state = "running"

    def finish(self) -> tuple[int, int]:
        if self._state != "running":
            raise TorchMetricsInvariantError(
                "CUDA diagnostics must be running before snapshot"
            )
        try:
            self._torch.cuda.synchronize(self._device)
            allocated = self._torch.cuda.max_memory_allocated(self._device)
            reserved = self._torch.cuda.max_memory_reserved(self._device)
        except Exception as exc:
            raise TorchMetricsExecutionError(
                f"CUDA peak-memory snapshot failed: {exc}"
            ) from exc

        allocated_bytes = _nonnegative_int(
            allocated,
            label="peak allocated CUDA memory",
        )
        reserved_bytes = _nonnegative_int(
            reserved,
            label="peak reserved CUDA memory",
        )
        self._state = "finished"
        return allocated_bytes, reserved_bytes

    def abort(self) -> None:
        if self._state in {"finished", "aborted"}:
            return
        self._state = "aborted"


def create_torch_metrics_session(
    *,
    device_index: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> TargetMetricsSession:
    """Create one target metrics session bound to a CUDA device."""

    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index cannot be negative")
    if not callable(clock):
        raise TypeError("metrics clock must be callable")

    try:
        torch_module = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchMetricsImportError(
            torch_import_error_message(exc, context="PyTorch could not be imported")
        ) from exc

    try:
        cuda = torch_module.cuda
        cuda_available = cuda.is_available()
        device_count = cuda.device_count() if cuda_available else 0
    except Exception as exc:
        raise TorchMetricsExecutionError(
            f"CUDA metrics availability check failed: {exc}"
        ) from exc
    if not cuda_available:
        raise TorchMetricsUnavailableError(torch_cuda_unavailable_message(torch_module))
    if isinstance(device_count, bool) or not isinstance(device_count, int):
        raise TorchMetricsInvariantError("CUDA device count must be an integer")
    if device_index >= device_count:
        raise TorchMetricsUnavailableError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )
    try:
        device = torch_module.device(f"cuda:{device_index}")
    except Exception as exc:
        raise TorchMetricsExecutionError(
            f"CUDA metrics device creation failed: {exc}"
        ) from exc

    diagnostics = TorchCUDADiagnosticsSession(
        torch_module=torch_module,
        device=device,
    )
    return create_target_metrics_session(
        cache_mode=TRANSFORMERS_DYNAMIC_CACHE_MODE,
        clock=clock,
        diagnostics=diagnostics,
    )


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TorchMetricsInvariantError(f"{label} must be an integer")
    if value < 0:
        raise TorchMetricsInvariantError(f"{label} cannot be negative")
    return value
