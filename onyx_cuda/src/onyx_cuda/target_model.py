"""Measured load-and-unload probe for the first quantized Qwen CUDA target."""

from __future__ import annotations

import gc
import importlib
import time
from dataclasses import dataclass
from typing import Any

from ._torch_install import (
    PYTORCH_CUDA_INSTALL_GUIDANCE,
    torch_cuda_unavailable_message,
)
from .model_profile import DEFAULT_TARGET_PROFILE, QwenModelProfile
from .production_tokenizer import load_qwen_tokenizer


class QuantizedTargetError(RuntimeError):
    """Base error raised by the quantized target lifecycle probe."""


class QuantizedTargetImportError(QuantizedTargetError):
    """Raised when an optional CUDA/model dependency cannot be imported."""


class QuantizedTargetUnavailableError(QuantizedTargetError):
    """Raised when the requested CUDA device cannot run the target probe."""


class QuantizedTargetLoadError(QuantizedTargetError):
    """Raised when the target model cannot be loaded or validated."""


class QuantizedTargetCleanupError(QuantizedTargetError):
    """Raised when model resources cannot be released and measured."""


@dataclass(frozen=True, slots=True)
class QuantizedTargetReadiness:
    """Measured tokenizer/model load state and allocator state after unload."""

    model_id: str
    revision: str
    quantization: str
    torch_version: str
    transformers_version: str
    bitsandbytes_version: str
    device_index: int
    device_name: str
    total_memory_bytes: int
    tokenizer_load_seconds: float
    model_load_seconds: float
    unload_seconds: float
    tokenizer_vocab_size: int
    model_vocab_size: int
    model_memory_footprint_bytes: int
    free_memory_before_bytes: int
    free_memory_after_load_bytes: int
    free_memory_after_unload_bytes: int
    allocated_before_bytes: int
    reserved_before_bytes: int
    allocated_after_load_bytes: int
    peak_allocated_bytes: int
    allocated_after_unload_bytes: int
    reserved_after_load_bytes: int
    reserved_after_unload_bytes: int

    @property
    def total_memory_mib(self) -> int:
        return self.total_memory_bytes // (1024 * 1024)

    @property
    def peak_allocated_mib(self) -> float:
        return self.peak_allocated_bytes / (1024 * 1024)

    @property
    def model_memory_footprint_mib(self) -> float:
        return self.model_memory_footprint_bytes / (1024 * 1024)


def probe_quantized_target(
    profile: QwenModelProfile = DEFAULT_TARGET_PROFILE,
    *,
    device_index: int = 0,
    local_files_only: bool = False,
) -> QuantizedTargetReadiness:
    """Load the pinned target in 4-bit NF4, measure it, and release all owned resources."""

    _validate_probe_inputs(profile, device_index, local_files_only)
    modules = {}
    for module_name in ("torch", "transformers", "bitsandbytes"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            message = f"{module_name} could not be imported: {exc}"
            if module_name == "torch":
                message = f"{message}. {PYTORCH_CUDA_INSTALL_GUIDANCE}"
            raise QuantizedTargetImportError(
                message
            ) from exc

    return _probe_quantized_target_modules(
        modules["torch"],
        modules["transformers"],
        modules["bitsandbytes"],
        profile=profile,
        device_index=device_index,
        local_files_only=local_files_only,
    )


def _probe_quantized_target_modules(
    torch_module: Any,
    transformers_module: Any,
    bitsandbytes_module: Any,
    *,
    profile: QwenModelProfile,
    device_index: int,
    local_files_only: bool,
) -> QuantizedTargetReadiness:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise QuantizedTargetUnavailableError(torch_cuda_unavailable_message(torch_module))
    device_count = cuda.device_count()
    if device_index >= device_count:
        raise QuantizedTargetUnavailableError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )

    device = torch_module.device(f"cuda:{device_index}")
    properties = cuda.get_device_properties(device)

    tokenizer_load = None
    model = None
    load_failure = None
    result_values = None
    allocated_before = None
    reserved_before = None

    try:
        tokenizer_load = load_qwen_tokenizer(
            profile,
            local_files_only=local_files_only,
        )

        gc.collect()
        cuda.empty_cache()
        cuda.synchronize(device)
        cuda.reset_peak_memory_stats(device)
        free_before, total_memory = cuda.mem_get_info(device)
        allocated_before = cuda.memory_allocated(device)
        reserved_before = cuda.memory_reserved(device)

        load_start = time.perf_counter()
        model = _load_nf4_model(
            torch_module,
            transformers_module,
            profile=profile,
            device_index=device_index,
            local_files_only=local_files_only,
        )
        model.eval()
        cuda.synchronize(device)
        model_load_seconds = time.perf_counter() - load_start

        if not bool(getattr(model, "is_loaded_in_4bit", False)):
            raise QuantizedTargetLoadError("the target model did not report 4-bit loading")

        model_vocab_size = _positive_int(
            getattr(model.config, "vocab_size", None), label="model vocabulary size"
        )
        embedding_rows = _positive_int(
            model.get_input_embeddings().weight.shape[0],
            label="model embedding vocabulary size",
        )
        if embedding_rows != model_vocab_size:
            raise QuantizedTargetLoadError(
                f"model embedding vocabulary size {embedding_rows} does not match "
                f"configuration vocabulary size {model_vocab_size}"
            )

        model_footprint = _positive_int(
            model.get_memory_footprint(), label="model memory footprint"
        )
        free_after_load, _ = cuda.mem_get_info(device)
        result_values = {
            "total_memory_bytes": int(total_memory),
            "tokenizer_load_seconds": float(tokenizer_load.load_seconds),
            "model_load_seconds": model_load_seconds,
            "tokenizer_vocab_size": tokenizer_load.tokenizer.vocab_size,
            "model_vocab_size": model_vocab_size,
            "model_memory_footprint_bytes": model_footprint,
            "free_memory_before_bytes": int(free_before),
            "free_memory_after_load_bytes": int(free_after_load),
            "allocated_before_bytes": int(allocated_before),
            "reserved_before_bytes": int(reserved_before),
            "allocated_after_load_bytes": int(cuda.memory_allocated(device)),
            "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
            "reserved_after_load_bytes": int(cuda.memory_reserved(device)),
        }
    except QuantizedTargetLoadError as exc:
        load_failure = exc
    except Exception as exc:
        load_failure = QuantizedTargetLoadError(
            f"failed to load quantized target {profile.pinned_id}: {exc}"
        )

    model = None
    tokenizer_load = None
    cleanup_start = time.perf_counter()
    cleanup_failure = None
    cleanup_values = None
    try:
        gc.collect()
        cuda.empty_cache()
        cuda.synchronize(device)
        free_after_unload, _ = cuda.mem_get_info(device)
        cleanup_values = {
            "unload_seconds": time.perf_counter() - cleanup_start,
            "free_memory_after_unload_bytes": int(free_after_unload),
            "allocated_after_unload_bytes": int(cuda.memory_allocated(device)),
            "reserved_after_unload_bytes": int(cuda.memory_reserved(device)),
        }
        if allocated_before is not None and reserved_before is not None:
            allocated_growth = (
                cleanup_values["allocated_after_unload_bytes"] - int(allocated_before)
            )
            reserved_growth = (
                cleanup_values["reserved_after_unload_bytes"] - int(reserved_before)
            )
            if allocated_growth > 1024 * 1024 or reserved_growth > 2 * 1024 * 1024:
                raise QuantizedTargetCleanupError(
                    "quantized target cleanup retained more than the allowed runtime residue: "
                    f"allocated growth={allocated_growth} bytes, "
                    f"reserved growth={reserved_growth} bytes"
                )
    except Exception as exc:
        cleanup_failure = QuantizedTargetCleanupError(
            f"quantized target cleanup failed: {exc}"
        )

    if load_failure is not None:
        if cleanup_failure is not None:
            raise QuantizedTargetCleanupError(
                f"{load_failure}; cleanup also failed: {cleanup_failure}"
            ) from load_failure
        raise load_failure
    if cleanup_failure is not None:
        raise cleanup_failure
    if result_values is None or cleanup_values is None:
        raise QuantizedTargetLoadError("quantized target probe produced no measurements")

    return QuantizedTargetReadiness(
        model_id=profile.model_id,
        revision=profile.revision,
        quantization="bitsandbytes-nf4-double-quant",
        torch_version=str(torch_module.__version__),
        transformers_version=str(transformers_module.__version__),
        bitsandbytes_version=str(bitsandbytes_module.__version__),
        device_index=device_index,
        device_name=str(properties.name),
        **result_values,
        **cleanup_values,
    )


def _validate_probe_inputs(
    profile: QwenModelProfile,
    device_index: int,
    local_files_only: bool,
) -> None:
    if not isinstance(profile, QwenModelProfile):
        raise TypeError("profile must be a QwenModelProfile")
    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index cannot be negative")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuantizedTargetLoadError(f"{label} must be an integer")
    if value <= 0:
        raise QuantizedTargetLoadError(f"{label} must be greater than zero")
    return value


def _load_nf4_model(
    torch_module: Any,
    transformers_module: Any,
    *,
    profile: QwenModelProfile,
    device_index: int,
    local_files_only: bool,
) -> Any:
    """Load one pinned model with the qualified NF4 configuration."""

    quantization_config = transformers_module.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_module.float16,
        bnb_4bit_use_double_quant=True,
    )
    return transformers_module.AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=profile.revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map={"": device_index},
        dtype=torch_module.float16,
        low_cpu_mem_usage=True,
    )
