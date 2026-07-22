"""Lazy stateless CUDA grammar-logit masking with sparse valid-token indices."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from typing import Any

from ._torch_install import torch_cuda_unavailable_message, torch_import_error_message
from .metrics import GrammarTimingSession


_SPARSE_VALID_INDEX_TRANSPORT = "sparse_valid_indices"


class TorchGrammarMaskError(RuntimeError):
    """Base error raised by the PyTorch CUDA grammar-logit mask boundary."""


class TorchGrammarMaskImportError(TorchGrammarMaskError):
    """Raised when PyTorch cannot be imported for explicit mask creation."""


class TorchGrammarMaskUnavailableError(TorchGrammarMaskError):
    """Raised when the requested CUDA mask device is unavailable."""


class TorchGrammarMaskInvariantError(TorchGrammarMaskError):
    """Raised when logits, valid IDs, or mask results violate the boundary contract."""


class TorchGrammarMaskExecutionError(TorchGrammarMaskError):
    """Raised when a PyTorch or CUDA mask operation cannot execute."""


class TorchCUDAGrammarLogitMask:
    """Apply one measured stateless valid-index transport to a CUDA logits row."""

    def __init__(
        self,
        *,
        torch_module: Any,
        vocab_size: int,
        device_index: int,
        device: Any,
        clock: Callable[[], float],
    ) -> None:
        self._torch = torch_module
        self._vocab_size = vocab_size
        self._device_index = device_index
        self._device = device
        self._transport_name = _SPARSE_VALID_INDEX_TRANSPORT
        self._clock = clock

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def transport_name(self) -> str:
        return self._transport_name

    def apply(
        self,
        logits: Any,
        valid_token_ids: tuple[int, ...],
        /,
    ) -> Any:
        """Return a new row preserving only logits named by a valid-ID tuple."""

        _validate_valid_token_ids(valid_token_ids, self._vocab_size)
        input_shape, input_device_text, input_dtype = self._validate_logits(logits)
        device_indices = self._materialize_device_indices(valid_token_ids)
        return self._apply_device_indices(
            logits,
            device_indices,
            input_shape=input_shape,
            input_device_text=input_device_text,
            input_dtype=input_dtype,
        )

    def apply_with_timing(
        self,
        logits: Any,
        valid_token_ids: tuple[int, ...],
        timing_session: GrammarTimingSession,
        /,
    ) -> Any:
        """Apply the mask and record synchronized transfer/application wall times."""

        if not isinstance(timing_session, GrammarTimingSession):
            raise TypeError("timing_session must be a GrammarTimingSession")
        _validate_valid_token_ids(valid_token_ids, self._vocab_size)
        input_shape, input_device_text, input_dtype = self._validate_logits(logits)

        self._synchronize("before valid-index transfer")
        transfer_start = self._read_clock()
        device_indices = self._materialize_device_indices(valid_token_ids)
        self._synchronize("after valid-index transfer")
        transfer_end = self._read_clock()
        transfer_time = self._elapsed(
            transfer_start,
            transfer_end,
            label="valid-index transfer",
        )

        application_start = self._read_clock()
        result = self._apply_device_indices(
            logits,
            device_indices,
            input_shape=input_shape,
            input_device_text=input_device_text,
            input_dtype=input_dtype,
        )
        self._synchronize("after mask application")
        application_end = self._read_clock()

        application_time = self._elapsed(
            application_start,
            application_end,
            label="mask application",
        )
        timing_session.record_mask_timing(transfer_time, application_time)
        return result

    def _materialize_device_indices(self, valid_token_ids: tuple[int, ...]) -> Any:
        try:
            host_indices = self._torch.tensor(
                valid_token_ids,
                dtype=self._torch.int64,
                device="cpu",
            )
            return host_indices.to(device=self._device)
        except Exception as exc:
            raise TorchGrammarMaskExecutionError(
                f"CUDA grammar-logit mask preparation failed: {exc}"
            ) from exc

    def _apply_device_indices(
        self,
        logits: Any,
        device_indices: Any,
        *,
        input_shape: tuple[int, ...],
        input_device_text: str,
        input_dtype: Any,
    ) -> Any:
        try:
            allowed_logits = logits.index_select(0, device_indices)
            result = self._torch.full_like(logits, float("-inf"))
        except Exception as exc:
            raise TorchGrammarMaskExecutionError(
                f"CUDA grammar-logit mask preparation failed: {exc}"
            ) from exc

        self._validate_result(
            result,
            logits=logits,
            input_shape=input_shape,
            input_device_text=input_device_text,
            input_dtype=input_dtype,
        )

        try:
            result.index_copy_(0, device_indices, allowed_logits)
            all_negative_infinity = self._torch.isneginf(result).all().item()
        except Exception as exc:
            raise TorchGrammarMaskExecutionError(
                f"CUDA grammar-logit mask application failed: {exc}"
            ) from exc

        if not isinstance(all_negative_infinity, bool):
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar-logit support check must return a boolean"
            )
        if all_negative_infinity:
            raise TorchGrammarMaskInvariantError(
                "valid token IDs have no logit support above -inf"
            )
        return result

    def _synchronize(self, boundary: str) -> None:
        try:
            self._torch.cuda.synchronize(self._device)
        except Exception as exc:
            raise TorchGrammarMaskExecutionError(
                f"CUDA grammar-logit mask synchronization failed {boundary}: {exc}"
            ) from exc

    def _read_clock(self) -> float:
        try:
            value = self._clock()
        except Exception as exc:
            raise TorchGrammarMaskExecutionError(
                f"CUDA grammar-logit timing clock failed: {exc}"
            ) from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar-logit timing clock must return a real number"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar-logit timing clock must return a finite value"
            )
        return numeric

    @staticmethod
    def _elapsed(start: float, end: float, *, label: str) -> float:
        duration = end - start
        if not math.isfinite(duration):
            raise TorchGrammarMaskInvariantError(f"{label} duration must be finite")
        if duration < 0.0:
            raise TorchGrammarMaskInvariantError(
                f"CUDA grammar-logit timing clock moved backwards during {label}"
            )
        return duration

    def _validate_logits(self, logits: Any) -> tuple[tuple[int, ...], Any, Any]:
        try:
            shape = tuple(logits.shape)
            shape_matches = shape == (self._vocab_size,)
        except Exception as exc:
            raise TorchGrammarMaskInvariantError(
                f"CUDA grammar logits shape could not be read: {exc}"
            ) from exc
        if not shape_matches:
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar mask requires one logits row with configured vocabulary shape "
                f"({self._vocab_size},); received {shape}"
            )

        try:
            is_cuda = bool(logits.is_cuda)
            device = logits.device
            device_text = str(device)
            expected_device_text = str(self._device)
        except Exception as exc:
            raise TorchGrammarMaskInvariantError(
                f"CUDA grammar logits device could not be read: {exc}"
            ) from exc
        if not is_cuda:
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar mask received a non-CUDA logits tensor"
            )
        if device_text != expected_device_text:
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar mask received logits on "
                f"{device_text}; expected {expected_device_text}"
            )

        try:
            is_floating_point = bool(logits.is_floating_point())
            dtype = logits.dtype
        except Exception as exc:
            raise TorchGrammarMaskInvariantError(
                f"CUDA grammar logits dtype could not be validated: {exc}"
            ) from exc
        if not is_floating_point:
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar logits must use a real floating-point dtype"
            )
        return shape, device_text, dtype

    def _validate_result(
        self,
        result: Any,
        *,
        logits: Any,
        input_shape: tuple[int, ...],
        input_device_text: str,
        input_dtype: Any,
    ) -> None:
        try:
            result_shape = tuple(result.shape)
            result_is_cuda = bool(result.is_cuda)
            result_device = result.device
            result_device_text = str(result_device)
            result_dtype = result.dtype
            shape_matches = result_shape == input_shape
            dtype_matches = bool(result_dtype == input_dtype)
        except Exception as exc:
            raise TorchGrammarMaskInvariantError(
                f"CUDA grammar mask result metadata could not be read: {exc}"
            ) from exc
        if not shape_matches:
            raise TorchGrammarMaskInvariantError(
                f"CUDA grammar mask changed logits shape from {input_shape} to {result_shape}"
            )
        if not result_is_cuda or result_device_text != input_device_text:
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar mask changed the logits device from "
                f"{input_device_text} to {result_device_text}"
            )
        if not dtype_matches:
            raise TorchGrammarMaskInvariantError(
                f"CUDA grammar mask changed logits dtype from {input_dtype} to {result_dtype}"
            )
        if result is logits:
            raise TorchGrammarMaskInvariantError(
                "CUDA grammar mask result must not alias the input tensor"
            )


def create_cuda_grammar_logit_mask(
    vocab_size: int,
    *,
    device_index: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> TorchCUDAGrammarLogitMask:
    """Create a stateless mask on one CUDA device, importing PyTorch only now."""

    _validate_factory_inputs(vocab_size, device_index, clock)
    try:
        torch_module = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchGrammarMaskImportError(
            torch_import_error_message(
                exc,
                context="PyTorch could not be imported for CUDA grammar masking",
            )
        ) from exc
    return _create_cuda_grammar_logit_mask(
        torch_module,
        vocab_size,
        device_index=device_index,
        clock=clock,
    )


def _create_cuda_grammar_logit_mask(
    torch_module: Any,
    vocab_size: int,
    *,
    device_index: int,
    clock: Callable[[], float] = time.perf_counter,
) -> TorchCUDAGrammarLogitMask:
    _validate_factory_inputs(vocab_size, device_index, clock)
    try:
        cuda = torch_module.cuda
        cuda_available = cuda.is_available()
        device_count = cuda.device_count() if cuda_available else 0
    except Exception as exc:
        raise TorchGrammarMaskExecutionError(
            f"CUDA grammar-mask availability check failed: {exc}"
        ) from exc
    if not isinstance(cuda_available, bool):
        raise TorchGrammarMaskInvariantError(
            "CUDA grammar-mask availability result must be a boolean"
        )
    if not cuda_available:
        raise TorchGrammarMaskUnavailableError(torch_cuda_unavailable_message(torch_module))
    if isinstance(device_count, bool) or not isinstance(device_count, int):
        raise TorchGrammarMaskInvariantError("CUDA device count must be an integer")
    if device_count < 0:
        raise TorchGrammarMaskInvariantError("CUDA device count cannot be negative")
    if device_index >= device_count:
        raise TorchGrammarMaskUnavailableError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )
    try:
        device = torch_module.device(f"cuda:{device_index}")
    except Exception as exc:
        raise TorchGrammarMaskExecutionError(
            f"CUDA grammar-mask device creation failed: {exc}"
        ) from exc
    return TorchCUDAGrammarLogitMask(
        torch_module=torch_module,
        vocab_size=vocab_size,
        device_index=device_index,
        device=device,
        clock=clock,
    )


def _validate_factory_inputs(
    vocab_size: int,
    device_index: int,
    clock: Callable[[], float],
) -> None:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")
    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index cannot be negative")
    if not callable(clock):
        raise TypeError("clock must be callable")


def _validate_valid_token_ids(
    valid_token_ids: tuple[int, ...],
    vocab_size: int,
) -> None:
    if not isinstance(valid_token_ids, tuple):
        raise TorchGrammarMaskInvariantError("valid_token_ids must be a tuple")
    if not valid_token_ids:
        raise TorchGrammarMaskInvariantError("valid_token_ids cannot be empty")

    previous = -1
    for token_id in valid_token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TorchGrammarMaskInvariantError(
                "valid_token_ids must contain only Python integers"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise TorchGrammarMaskInvariantError(
                f"valid token ID {token_id} is outside vocabulary range [0, {vocab_size})"
            )
        if token_id <= previous:
            raise TorchGrammarMaskInvariantError(
                "valid_token_ids must be strictly increasing and unique"
            )
        previous = token_id
