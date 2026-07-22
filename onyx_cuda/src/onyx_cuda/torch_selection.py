"""Lazy CUDA-native temperature and top-p token selection."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ._torch_install import torch_cuda_unavailable_message, torch_import_error_message
from .selection import TemperatureTopPSelection
from .torch_backend import (
    TorchBackendExecutionError,
    TorchBackendImportError,
    TorchBackendInvariantError,
    TorchBackendLoadError,
)


class _TorchCUDASamplingSession:
    """One private seeded CUDA RNG session for a single target generation."""

    def __init__(
        self,
        *,
        torch_module: Any,
        policy: TemperatureTopPSelection,
        device: Any,
        generator: Any,
    ) -> None:
        self._torch = torch_module
        self._policy = policy
        self._device = device
        self._generator = generator

    def __call__(self, logits: Any) -> int:
        shape = self._validate_logits(logits)
        try:
            contains_nan = bool(self._torch.isnan(logits).any().item())
        except Exception as exc:
            raise TorchBackendExecutionError(
                f"CUDA sampling logits validation failed: {exc}"
            ) from exc
        if contains_nan:
            raise TorchBackendInvariantError("CUDA sampling logits cannot contain NaN")

        try:
            positive_infinity = self._torch.isposinf(logits)
            has_positive_infinity = bool(positive_infinity.any().item())
            if has_positive_infinity:
                probabilities = positive_infinity.to(dtype=self._torch.float32)
                probabilities = probabilities / probabilities.sum()
            else:
                all_negative_infinity = bool(self._torch.isneginf(logits).all().item())
                if all_negative_infinity:
                    raise TorchBackendInvariantError(
                        "CUDA sampling logits cannot all be -inf"
                    )
                working_logits = logits.to(dtype=self._torch.float32)
                scaled = (working_logits - working_logits.max()) / float(
                    self._policy.temperature
                )
                probabilities = self._torch.softmax(scaled, dim=-1)

            probabilities_are_finite = bool(
                self._torch.isfinite(probabilities).all().item()
            )
            if not probabilities_are_finite:
                raise TorchBackendInvariantError(
                    "CUDA sampling produced non-finite probabilities"
                )

            sorted_probabilities, sorted_token_ids = self._torch.sort(
                probabilities,
                descending=True,
                stable=True,
            )
            previous_mass = (
                self._torch.cumsum(sorted_probabilities, dim=-1)
                - sorted_probabilities
            )
            # Comparing mass before each token retains the token that crosses the top-p boundary.
            retained = previous_mass < float(self._policy.top_p)
            filtered_probabilities = self._torch.where(
                retained,
                sorted_probabilities,
                self._torch.zeros_like(sorted_probabilities),
            )
            sampled_position = self._torch.multinomial(
                filtered_probabilities,
                num_samples=1,
                replacement=False,
                generator=self._generator,
            )
            selected = sorted_token_ids[sampled_position].item()
        except TorchBackendInvariantError:
            raise
        except Exception as exc:
            raise TorchBackendExecutionError(f"CUDA sampling failed: {exc}") from exc

        if isinstance(selected, bool) or not isinstance(selected, int):
            raise TorchBackendInvariantError(
                "CUDA sampling must return an integer token ID"
            )
        if selected < 0 or selected >= shape[0]:
            raise TorchBackendInvariantError(
                f"CUDA sampling returned token ID {selected} outside logits range "
                f"[0, {shape[0]})"
            )
        return selected

    def _validate_logits(self, logits: Any) -> tuple[int, ...]:
        try:
            shape = tuple(logits.shape)
        except Exception as exc:
            raise TorchBackendInvariantError(
                f"CUDA sampling logits shape could not be read: {exc}"
            ) from exc
        if len(shape) != 1 or not shape or shape[0] <= 0:
            raise TorchBackendInvariantError(
                "CUDA sampling requires one nonempty logits row; "
                f"received shape {shape}"
            )
        if not bool(getattr(logits, "is_cuda", False)):
            raise TorchBackendInvariantError(
                "CUDA sampling received a non-CUDA tensor"
            )
        if str(getattr(logits, "device", None)) != str(self._device):
            raise TorchBackendInvariantError(
                f"CUDA sampling received logits on {getattr(logits, 'device', None)}; "
                f"expected {self._device}"
            )
        try:
            is_floating_point = bool(logits.is_floating_point())
        except Exception as exc:
            raise TorchBackendInvariantError(
                f"CUDA sampling logits dtype could not be validated: {exc}"
            ) from exc
        if not is_floating_point:
            raise TorchBackendInvariantError(
                "CUDA sampling logits must use a floating-point dtype"
            )
        return shape


def create_cuda_sampler(
    policy: TemperatureTopPSelection,
    *,
    device_index: int = 0,
) -> Callable[[Any], int]:
    """Create a private seeded sampler without transferring vocabulary data to CPU."""

    if not isinstance(policy, TemperatureTopPSelection):
        raise TypeError("policy must be a TemperatureTopPSelection")
    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index cannot be negative")
    try:
        torch_module = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchBackendImportError(
            torch_import_error_message(exc, context="PyTorch could not be imported")
        ) from exc
    return _create_cuda_sampler(torch_module, policy, device_index=device_index)


def _create_cuda_sampler(
    torch_module: Any,
    policy: TemperatureTopPSelection,
    *,
    device_index: int,
) -> _TorchCUDASamplingSession:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise TorchBackendLoadError(torch_cuda_unavailable_message(torch_module))
    device_count = cuda.device_count()
    if device_index >= device_count:
        raise TorchBackendLoadError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )
    device = torch_module.device(f"cuda:{device_index}")
    try:
        generator = torch_module.Generator(device=device)
        generator.manual_seed(policy.seed)
    except Exception as exc:
        raise TorchBackendExecutionError(
            f"CUDA sampling generator creation failed: {exc}"
        ) from exc
    return _TorchCUDASamplingSession(
        torch_module=torch_module,
        policy=policy,
        device=device,
        generator=generator,
    )
