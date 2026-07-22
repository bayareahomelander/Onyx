"""PyTorch CUDA target backend for pinned Qwen prefill, decode, and verification."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any

from ._torch_install import (
    PYTORCH_CUDA_INSTALL_GUIDANCE,
    torch_cuda_unavailable_message,
)
from .backend import AutoregressiveBackend, BackendError, BackendStateError, ModelStep
from .cache import CacheCheckpointStateError
from .model_profile import DEFAULT_TARGET_PROFILE, QwenModelProfile
from .production_tokenizer import QwenTokenizerAdapter, load_qwen_tokenizer
from .target_model import _load_nf4_model
from .torch_dynamic_cache import (
    DynamicCacheCropError,
    DynamicCacheLayoutSignature,
    DynamicCacheStructureError,
    inspect_pinned_dynamic_cache,
    rollback_pinned_dynamic_cache,
)
from .torch_metrics import TRANSFORMERS_DYNAMIC_CACHE_MODE
from .verification import BatchedTargetVerificationResult


class TorchBackendError(BackendError):
    """Base error raised by the production PyTorch target backend."""


class TorchBackendImportError(TorchBackendError):
    """Raised when an optional production backend dependency cannot be imported."""


class TorchBackendLoadError(TorchBackendError):
    """Raised when the pinned model backend cannot be loaded or validated."""


class TorchBackendExecutionError(TorchBackendError):
    """Raised when a CUDA prefill or decode forward pass fails."""


class TorchBackendInvariantError(TorchBackendError):
    """Raised when model tensors or cache state violate the backend contract."""


@dataclass(frozen=True, slots=True)
class _TorchCacheCheckpoint:
    """Opaque production-cache capability for one backend sequence epoch."""

    owner_id: int
    epoch: int
    allocation_id: int
    cache_length: int

    def __post_init__(self) -> None:
        _validate_checkpoint_integer(self.owner_id, label="owner_id", minimum=1)
        _validate_checkpoint_integer(self.epoch, label="epoch", minimum=1)
        _validate_checkpoint_integer(self.allocation_id, label="allocation_id", minimum=1)
        _validate_checkpoint_integer(self.cache_length, label="cache_length", minimum=0)


@dataclass(frozen=True, slots=True)
class _TorchCacheSnapshot:
    checkpoint: _TorchCacheCheckpoint
    token_ids: tuple[int, ...]
    layout: DynamicCacheLayoutSignature


_TORCH_CACHE_OWNER_IDS = count(1)


class TorchCUDATargetBackend(AutoregressiveBackend[Any]):
    """Persistent NF4 Qwen target with native CUDA logits and a dynamic KV cache."""

    def __init__(
        self,
        *,
        torch_module: Any,
        transformers_module: Any,
        model: Any,
        tokenizer: QwenTokenizerAdapter,
        profile: QwenModelProfile,
        device_index: int,
    ) -> None:
        _validate_load_inputs(profile, device_index, False)
        self._torch = torch_module
        self._transformers = transformers_module
        self._model = model
        self._tokenizer = tokenizer
        self._profile = profile
        self._device_index = device_index
        self._device = torch_module.device(f"cuda:{device_index}")
        self._cache = None
        self._closed = False
        self._owner_id = next(_TORCH_CACHE_OWNER_IDS)
        self._epoch = 1
        self._next_checkpoint_id = 1
        self._active_token_ids: list[int] = []
        self._active_cache_layout: DynamicCacheLayoutSignature | None = None
        self._cache_checkpoints: dict[int, _TorchCacheSnapshot] = {}
        self._checkpoint_profile_supported = (
            profile == DEFAULT_TARGET_PROFILE and device_index == 0
        )

        self._vocab_size = _positive_int(tokenizer.vocab_size, label="tokenizer vocabulary size")
        self._model_vocab_size = _positive_int(
            getattr(model.config, "vocab_size", None), label="model vocabulary size"
        )
        embedding_rows = _positive_int(
            model.get_input_embeddings().weight.shape[0],
            label="model embedding vocabulary size",
        )
        if embedding_rows != self._model_vocab_size:
            raise TorchBackendLoadError(
                f"model embedding vocabulary size {embedding_rows} does not match "
                f"configuration vocabulary size {self._model_vocab_size}"
            )
        if self._model_vocab_size < self._vocab_size:
            raise TorchBackendLoadError(
                f"model vocabulary size {self._model_vocab_size} is smaller than tokenizer "
                f"vocabulary size {self._vocab_size}"
            )
        if not bool(getattr(model, "is_loaded_in_4bit", False)):
            raise TorchBackendLoadError("the target model did not report 4-bit loading")
        self._model_memory_footprint_bytes = _positive_int(
            model.get_memory_footprint(), label="model memory footprint"
        )

    @property
    def model_id(self) -> str:
        return self._profile.pinned_id

    @property
    def vocab_size(self) -> int:
        """Return usable tokenizer IDs, excluding padded model-only output rows."""

        return self._vocab_size

    @property
    def model_vocab_size(self) -> int:
        return self._model_vocab_size

    @property
    def padded_vocab_rows(self) -> int:
        return self._model_vocab_size - self._vocab_size

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def cache_mode(self) -> str:
        return TRANSFORMERS_DYNAMIC_CACHE_MODE

    @property
    def model_memory_footprint_bytes(self) -> int:
        return self._model_memory_footprint_bytes

    @property
    def tokenizer(self) -> QwenTokenizerAdapter:
        """Return the paired production tokenizer while the backend is open."""

        self._require_open()
        return self._tokenizer

    @property
    def cache_length(self) -> int:
        if self._cache is None:
            return 0
        return self._read_cache_length(self._cache)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def prefill(self, prompt_token_ids: Sequence[int], /) -> ModelStep[Any]:
        self._require_open()
        prompt = tuple(prompt_token_ids)
        _validate_token_ids(prompt, self._vocab_size, label="prompt", allow_empty=False)

        self._transition_to_empty(advance_epoch=True)
        try:
            self._cache = self._transformers.DynamicCache(config=self._model.config)
        except Exception as exc:
            self._transition_to_empty(advance_epoch=False)
            raise TorchBackendExecutionError(
                f"PyTorch target cache creation failed: {exc}"
            ) from exc
        return self._forward(prompt, expected_cache_length=len(prompt), operation="prefill")

    def decode(self, token_id: int, /) -> ModelStep[Any]:
        self._require_open()
        _validate_token_id(token_id, self._vocab_size, label="decode token")
        previous_length = self.cache_length
        if previous_length == 0:
            raise BackendStateError("prefill must be called before decode")
        try:
            self._validate_forward_bookkeeping(previous_length)
        except TorchBackendInvariantError:
            self._transition_to_empty(advance_epoch=True)
            raise

        return self._forward(
            (token_id,),
            expected_cache_length=previous_length + 1,
            operation="decode",
        )

    def verify_proposal(
        self,
        current_token_id: int,
        proposal_token_ids: Sequence[int],
        /,
    ) -> BatchedTargetVerificationResult[Any]:
        """Evaluate one current token and a nonempty proposal in one pinned forward."""

        self._require_open()
        _validate_verification_token_id(
            current_token_id,
            self._vocab_size,
            label="current token",
        )
        try:
            proposal = tuple(proposal_token_ids)
        except TypeError as exc:
            raise TypeError("proposal_token_ids must be a sequence") from exc
        if not proposal:
            raise ValueError("proposal_token_ids cannot be empty")
        for position, token_id in enumerate(proposal):
            _validate_verification_token_id(
                token_id,
                self._vocab_size,
                label=f"proposal token at position {position}",
            )
        if not self._checkpoint_profile_supported:
            raise BackendStateError(
                "PyTorch batched target verification is qualified only for the pinned "
                "0.5B target on cuda:0"
            )

        try:
            previous_length, active_layout = self._validate_verification_active_state()
        except TorchBackendInvariantError:
            self._transition_to_empty(advance_epoch=True)
            raise

        input_suffix = (current_token_id, *proposal)
        return self._verification_forward(
            input_suffix,
            previous_length=previous_length,
            active_layout=active_layout,
        )

    def create_cache_checkpoint(self) -> _TorchCacheCheckpoint:
        """Record the exact active token prefix without retaining CUDA state."""

        self._require_open()
        cache_length, layout = self._validate_checkpoint_active_state()
        self._validate_checkpoint_counter()
        allocation_id = self._next_checkpoint_id
        if (
            isinstance(allocation_id, bool)
            or not isinstance(allocation_id, int)
            or allocation_id < 1
            or allocation_id in self._cache_checkpoints
        ):
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint allocation state is invalid"
            )
        if self._cache_checkpoints and allocation_id <= max(self._cache_checkpoints):
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint allocation counter is not monotonic"
            )

        checkpoint = _TorchCacheCheckpoint(
            owner_id=self._owner_id,
            epoch=self._epoch,
            allocation_id=allocation_id,
            cache_length=cache_length,
        )
        snapshot = _TorchCacheSnapshot(
            checkpoint=checkpoint,
            token_ids=tuple(self._active_token_ids),
            layout=layout,
        )
        self._cache_checkpoints[allocation_id] = snapshot
        self._next_checkpoint_id = allocation_id + 1
        return checkpoint

    def rollback_cache(self, checkpoint: _TorchCacheCheckpoint, /) -> None:
        """Restore the exact cache prefix represented by one active checkpoint."""

        self._require_open()
        self._validate_checkpoint_type_and_owner(checkpoint)
        if checkpoint.epoch != self._epoch:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint belongs to a stale sequence"
            )

        snapshot = self._cache_checkpoints.get(checkpoint.allocation_id)
        if snapshot is None:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint is unknown, released, or represents a discarded suffix"
            )
        if type(snapshot) is not _TorchCacheSnapshot:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint registry contains an invalid snapshot"
            )
        if checkpoint != snapshot.checkpoint:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint metadata does not match its canonical allocation"
            )

        current_length, active_layout = self._validate_checkpoint_active_state()
        self._validate_checkpoint_snapshot(snapshot, active_layout)
        target_length = snapshot.checkpoint.cache_length
        if target_length > current_length:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint is ahead of the current cache position"
            )
        if tuple(self._active_token_ids[:target_length]) != snapshot.token_ids:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint prefix no longer matches the active sequence"
            )

        self._validate_checkpoint_counter()
        retained_registry: dict[int, _TorchCacheSnapshot] = {}
        active_tokens = tuple(self._active_token_ids)
        for allocation_id, active_snapshot in self._cache_checkpoints.items():
            self._validate_checkpoint_snapshot(active_snapshot, active_layout)
            if allocation_id != active_snapshot.checkpoint.allocation_id:
                raise CacheCheckpointStateError(
                    "PyTorch cache checkpoint registry allocation is inconsistent"
                )
            active_length = active_snapshot.checkpoint.cache_length
            if active_length > current_length:
                raise CacheCheckpointStateError(
                    "PyTorch cache checkpoint registry contains a forward-position handle"
                )
            if active_length <= target_length:
                if snapshot.token_ids[:active_length] != active_snapshot.token_ids:
                    raise CacheCheckpointStateError(
                        "PyTorch cache checkpoint registry contains inconsistent retained prefixes"
                    )
                retained_registry[allocation_id] = active_snapshot
            elif active_tokens[:active_length] != active_snapshot.token_ids:
                raise CacheCheckpointStateError(
                    "PyTorch cache checkpoint registry contains an inconsistent suffix prefix"
                )

        restored_tokens = list(snapshot.token_ids)
        if target_length == current_length:
            return
        try:
            rollback_pinned_dynamic_cache(
                self._torch,
                self._transformers,
                self._cache,
                current_length=current_length,
                target_length=target_length,
                device_index=self._device_index,
                expected_layout=active_layout,
            )
        except DynamicCacheStructureError as exc:
            raise TorchBackendInvariantError(
                f"PyTorch cache rollback invariant failed: {exc}"
            ) from exc
        except DynamicCacheCropError as exc:
            raise TorchBackendExecutionError(
                f"PyTorch cache rollback execution failed: {exc}"
            ) from exc

        self._active_token_ids = restored_tokens
        self._cache_checkpoints = retained_registry

    def release_cache_checkpoint(self, checkpoint: _TorchCacheCheckpoint, /) -> None:
        """Release one checkpoint handle without changing the active cache."""

        self._require_open()
        self._validate_checkpoint_type_and_owner(checkpoint)
        if checkpoint.epoch != self._epoch:
            return
        snapshot = self._cache_checkpoints.get(checkpoint.allocation_id)
        if snapshot is None:
            return
        if type(snapshot) is not _TorchCacheSnapshot:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint registry contains an invalid snapshot"
            )
        if checkpoint != snapshot.checkpoint:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint metadata does not match its canonical allocation"
            )
        del self._cache_checkpoints[checkpoint.allocation_id]

    def reset(self) -> None:
        self._require_open()
        self._transition_to_empty(advance_epoch=True)

    def close(self) -> None:
        """Release model, tokenizer, and cache references; repeated calls are safe."""

        if self._closed:
            return

        torch_module = self._torch
        device = self._device
        self._transition_to_empty(advance_epoch=True)
        self._model = None
        self._tokenizer = None
        self._transformers = None
        self._closed = True

        try:
            gc.collect()
            torch_module.cuda.empty_cache()
            torch_module.cuda.synchronize(device)
        except Exception as exc:
            raise TorchBackendExecutionError(f"PyTorch target cleanup failed: {exc}") from exc

    def __enter__(self) -> TorchCUDATargetBackend:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _forward(
        self,
        token_ids: tuple[int, ...],
        *,
        expected_cache_length: int,
        operation: str,
    ) -> ModelStep[Any]:
        try:
            input_cache = self._cache
            input_ids = self._torch.tensor(
                [token_ids],
                dtype=self._torch.long,
                device=self._device,
            )
            with self._torch.inference_mode():
                output = self._model(
                    input_ids=input_ids,
                    past_key_values=self._cache,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
            self._torch.cuda.synchronize(self._device)
            output_cache = output.past_key_values
            if output_cache is not input_cache:
                raise TorchBackendInvariantError(
                    f"{operation} replaced the supported DynamicCache object"
                )
            self._cache = output_cache
            self._validate_output_shape(output.logits)
            cache_length = self._read_cache_length(self._cache)
            if cache_length != expected_cache_length:
                raise TorchBackendInvariantError(
                    f"{operation} produced cache length {cache_length}; "
                    f"expected {expected_cache_length}"
                )

            cache_layout = None
            if self._checkpoint_profile_supported:
                cache_layout = self._inspect_active_cache(
                    expected_cache_length,
                    operation=operation,
                )
                if operation == "decode" and cache_layout != self._active_cache_layout:
                    raise TorchBackendInvariantError(
                        "decode changed the qualified DynamicCache layout"
                    )

            usable_logits = output.logits[0, -1, : self._vocab_size]
            if tuple(usable_logits.shape) != (self._vocab_size,):
                raise TorchBackendInvariantError(
                    f"{operation} usable logits shape {tuple(usable_logits.shape)}; "
                    f"expected ({self._vocab_size},)"
                )
            if str(usable_logits.device) != str(self._device):
                raise TorchBackendInvariantError(
                    f"{operation} logits are on {usable_logits.device}; expected {self._device}"
                )
            step = ModelStep(logits=usable_logits, cache_length=cache_length)
            if operation == "prefill":
                self._active_token_ids = list(token_ids)
                self._active_cache_layout = cache_layout
            else:
                self._active_token_ids.append(token_ids[0])
            return step
        except TorchBackendInvariantError:
            self._transition_to_empty(advance_epoch=operation != "prefill")
            raise
        except Exception as exc:
            self._transition_to_empty(advance_epoch=operation != "prefill")
            raise TorchBackendExecutionError(f"PyTorch target {operation} failed: {exc}") from exc

    def _verification_forward(
        self,
        input_suffix: tuple[int, ...],
        *,
        previous_length: int,
        active_layout: DynamicCacheLayoutSignature,
    ) -> BatchedTargetVerificationResult[Any]:
        row_count = len(input_suffix)
        expected_cache_length = previous_length + row_count
        try:
            input_cache = self._cache
            input_ids = self._torch.tensor(
                [input_suffix],
                dtype=self._torch.long,
                device=self._device,
            )
            with self._torch.inference_mode():
                output = self._model(
                    input_ids=input_ids,
                    past_key_values=input_cache,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=row_count,
                )
            self._torch.cuda.synchronize(self._device)

            if output.past_key_values is not input_cache:
                raise TorchBackendInvariantError(
                    "verify_proposal replaced the supported DynamicCache object"
                )
            raw_logits = output.logits
            self._validate_verification_logits(
                raw_logits,
                row_count=row_count,
            )
            cache_length = self._read_cache_length(input_cache)
            if cache_length != expected_cache_length:
                raise TorchBackendInvariantError(
                    f"verify_proposal produced cache length {cache_length}; "
                    f"expected {expected_cache_length}"
                )
            cache_layout = self._inspect_active_cache(
                expected_cache_length,
                operation="verify_proposal",
            )
            if cache_layout != active_layout:
                raise TorchBackendInvariantError(
                    "verify_proposal changed the qualified DynamicCache layout"
                )

            rows = tuple(
                raw_logits[0, row_index, : self._vocab_size]
                for row_index in range(row_count)
            )
            self._validate_verification_rows(rows, row_count=row_count)
            try:
                result = self._build_verification_result(
                    logit_rows=rows,
                    cache_length=cache_length,
                )
            except (TypeError, ValueError) as exc:
                raise TorchBackendInvariantError(
                    f"verify_proposal result construction failed: {exc}"
                ) from exc
            if (
                type(result) is not BatchedTargetVerificationResult
                or type(result.logit_rows) is not tuple
                or len(result.logit_rows) != row_count
                or not all(
                    returned_row is expected_row
                    for returned_row, expected_row in zip(
                        result.logit_rows,
                        rows,
                        strict=True,
                    )
                )
                or result.cache_length != cache_length
            ):
                raise TorchBackendInvariantError(
                    "verify_proposal constructed an invalid verification result"
                )

            self._active_token_ids.extend(input_suffix)
            return result
        except TorchBackendInvariantError:
            self._transition_to_empty(advance_epoch=True)
            raise
        except Exception as exc:
            self._transition_to_empty(advance_epoch=True)
            raise TorchBackendExecutionError(
                f"PyTorch target verify_proposal failed: {exc}"
            ) from exc

    def _build_verification_result(
        self,
        *,
        logit_rows: tuple[Any, ...],
        cache_length: int,
    ) -> BatchedTargetVerificationResult[Any]:
        return BatchedTargetVerificationResult(
            logit_rows=logit_rows,
            cache_length=cache_length,
        )

    def _validate_verification_active_state(
        self,
    ) -> tuple[int, DynamicCacheLayoutSignature]:
        if self._cache is None:
            if self._active_token_ids or self._active_cache_layout is not None:
                raise TorchBackendInvariantError(
                    "active PyTorch cache bookkeeping exists without a DynamicCache"
                )
            raise BackendStateError("prefill must be called before verify_proposal")
        if not self._active_token_ids:
            raise TorchBackendInvariantError(
                "the active DynamicCache has no exact token prefix"
            )
        cache_length = self._read_cache_length(self._cache)
        if cache_length < 1:
            raise TorchBackendInvariantError(
                "the active DynamicCache must have a positive length"
            )
        if len(self._active_token_ids) != cache_length:
            raise TorchBackendInvariantError(
                "active token-prefix length does not match the DynamicCache length"
            )
        if self._active_cache_layout is None:
            raise TorchBackendInvariantError(
                "the active DynamicCache has no qualified layout signature"
            )
        current_layout = self._inspect_active_cache(
            cache_length,
            operation="verify_proposal preflight",
        )
        if current_layout != self._active_cache_layout:
            raise TorchBackendInvariantError(
                "the active DynamicCache layout changed within the sequence epoch"
            )
        return cache_length, current_layout

    def _validate_verification_logits(self, logits: Any, *, row_count: int) -> None:
        try:
            actual_shape = tuple(logits.shape)
        except Exception as exc:
            raise TorchBackendInvariantError(
                f"verify_proposal logits shape could not be read: {exc}"
            ) from exc
        expected_shape = (1, row_count, self._model_vocab_size)
        if actual_shape != expected_shape:
            raise TorchBackendInvariantError(
                f"verify_proposal model logits shape {actual_shape}; expected {expected_shape}"
            )
        if str(getattr(logits, "dtype", None)) != str(self._torch.float16):
            raise TorchBackendInvariantError(
                f"verify_proposal logits dtype {getattr(logits, 'dtype', None)}; "
                f"expected {self._torch.float16}"
            )
        if str(getattr(logits, "device", None)) != str(self._device):
            raise TorchBackendInvariantError(
                f"verify_proposal logits are on {getattr(logits, 'device', None)}; "
                f"expected {self._device}"
            )
        if getattr(logits, "is_cuda", None) is not True:
            raise TorchBackendInvariantError(
                "verify_proposal logits must be a CUDA tensor"
            )

    def _validate_verification_rows(
        self,
        rows: tuple[Any, ...],
        *,
        row_count: int,
    ) -> None:
        if type(rows) is not tuple or len(rows) != row_count:
            raise TorchBackendInvariantError(
                f"verify_proposal returned {len(rows)} rows; expected {row_count}"
            )
        for row_index, row in enumerate(rows):
            try:
                row_shape = tuple(row.shape)
            except Exception as exc:
                raise TorchBackendInvariantError(
                    f"verify_proposal row {row_index} shape could not be read: {exc}"
                ) from exc
            if row_shape != (self._vocab_size,):
                raise TorchBackendInvariantError(
                    f"verify_proposal row {row_index} shape {row_shape}; "
                    f"expected ({self._vocab_size},)"
                )
            if str(getattr(row, "dtype", None)) != str(self._torch.float16):
                raise TorchBackendInvariantError(
                    f"verify_proposal row {row_index} dtype {getattr(row, 'dtype', None)}; "
                    f"expected {self._torch.float16}"
                )
            if str(getattr(row, "device", None)) != str(self._device):
                raise TorchBackendInvariantError(
                    f"verify_proposal row {row_index} is on {getattr(row, 'device', None)}; "
                    f"expected {self._device}"
                )
            if getattr(row, "is_cuda", None) is not True:
                raise TorchBackendInvariantError(
                    f"verify_proposal row {row_index} must be a CUDA tensor"
                )

    def _validate_forward_bookkeeping(self, expected_cache_length: int) -> None:
        if len(self._active_token_ids) != expected_cache_length:
            raise TorchBackendInvariantError(
                "active token-prefix length does not match the DynamicCache length"
            )
        if self._checkpoint_profile_supported and self._active_cache_layout is None:
            raise TorchBackendInvariantError(
                "the active DynamicCache has no qualified layout signature"
            )

    def _validate_checkpoint_active_state(
        self,
    ) -> tuple[int, DynamicCacheLayoutSignature]:
        if not self._checkpoint_profile_supported:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoints are qualified only for the pinned 0.5B target on cuda:0"
            )
        if self._cache is None or not self._active_token_ids:
            raise CacheCheckpointStateError(
                "prefill must be called before creating or rolling back a PyTorch cache checkpoint"
            )
        cache_length = self._read_cache_length(self._cache)
        if cache_length < 1:
            raise CacheCheckpointStateError(
                "prefill must be called before creating or rolling back a PyTorch cache checkpoint"
            )
        if len(self._active_token_ids) != cache_length:
            raise CacheCheckpointStateError(
                "PyTorch cache length does not match its exact active token prefix"
            )
        if self._active_cache_layout is None:
            raise CacheCheckpointStateError(
                "the active PyTorch cache has no qualified layout signature"
            )
        current_layout = self._inspect_active_cache(
            cache_length,
            operation="checkpoint",
        )
        if current_layout != self._active_cache_layout:
            raise TorchBackendInvariantError(
                "the active DynamicCache layout changed within the sequence epoch"
            )
        return cache_length, current_layout

    def _inspect_active_cache(
        self,
        expected_cache_length: int,
        *,
        operation: str,
    ) -> DynamicCacheLayoutSignature:
        try:
            return inspect_pinned_dynamic_cache(
                self._torch,
                self._transformers,
                self._cache,
                expected_length=expected_cache_length,
                device_index=self._device_index,
            )
        except DynamicCacheStructureError as exc:
            raise TorchBackendInvariantError(
                f"{operation} DynamicCache invariant failed: {exc}"
            ) from exc

    def _validate_checkpoint_type_and_owner(
        self,
        checkpoint: _TorchCacheCheckpoint,
    ) -> None:
        if type(checkpoint) is not _TorchCacheCheckpoint:
            raise TypeError("checkpoint must be a _TorchCacheCheckpoint")
        for value, minimum in (
            (checkpoint.owner_id, 1),
            (checkpoint.epoch, 1),
            (checkpoint.allocation_id, 1),
            (checkpoint.cache_length, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise CacheCheckpointStateError(
                    "PyTorch cache checkpoint metadata is invalid"
                )
        if checkpoint.owner_id != self._owner_id:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint belongs to another backend"
            )

    def _validate_checkpoint_snapshot(
        self,
        snapshot: _TorchCacheSnapshot,
        active_layout: DynamicCacheLayoutSignature,
    ) -> None:
        if type(snapshot) is not _TorchCacheSnapshot:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint registry contains an invalid snapshot"
            )
        checkpoint = snapshot.checkpoint
        if type(checkpoint) is not _TorchCacheCheckpoint:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint registry is inconsistent"
            )
        for value, minimum in (
            (checkpoint.owner_id, 1),
            (checkpoint.epoch, 1),
            (checkpoint.allocation_id, 1),
            (checkpoint.cache_length, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise CacheCheckpointStateError(
                    "PyTorch cache checkpoint registry is inconsistent"
                )
        if (
            checkpoint.owner_id != self._owner_id
            or checkpoint.epoch != self._epoch
            or type(snapshot.token_ids) is not tuple
            or len(snapshot.token_ids) != checkpoint.cache_length
            or type(snapshot.layout) is not DynamicCacheLayoutSignature
            or snapshot.layout != active_layout
        ):
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint registry is inconsistent"
            )
        for token_id in snapshot.token_ids:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= self._vocab_size
            ):
                raise CacheCheckpointStateError(
                    "PyTorch cache checkpoint registry contains an invalid token prefix"
                )

    def _validate_checkpoint_counter(self) -> None:
        next_id = self._next_checkpoint_id
        if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 1:
            raise CacheCheckpointStateError(
                "PyTorch cache checkpoint allocation state is invalid"
            )
        for allocation_id in self._cache_checkpoints:
            if (
                isinstance(allocation_id, bool)
                or not isinstance(allocation_id, int)
                or allocation_id < 1
                or allocation_id >= next_id
            ):
                raise CacheCheckpointStateError(
                    "PyTorch cache checkpoint registry allocation is inconsistent"
                )

    def _transition_to_empty(self, *, advance_epoch: bool) -> None:
        self._cache = None
        self._active_token_ids = []
        self._active_cache_layout = None
        self._cache_checkpoints = {}
        self._next_checkpoint_id = 1
        if advance_epoch:
            self._epoch += 1

    def _validate_output_shape(self, logits: Any) -> None:
        actual_shape = tuple(logits.shape)
        expected_shape = (1, 1, self._model_vocab_size)
        if actual_shape != expected_shape:
            raise TorchBackendInvariantError(
                f"model logits shape {actual_shape}; expected {expected_shape}"
            )

    def _read_cache_length(self, cache: Any) -> int:
        try:
            value = cache.get_seq_length()
        except Exception as exc:
            raise TorchBackendInvariantError(f"cache length could not be read: {exc}") from exc
        if isinstance(value, bool) or not isinstance(value, int):
            raise TorchBackendInvariantError("cache length must be an integer")
        if value < 0:
            raise TorchBackendInvariantError("cache length cannot be negative")
        return value

    def _require_open(self) -> None:
        if self._closed:
            raise BackendStateError("the PyTorch target backend is closed")


def load_torch_cuda_target(
    profile: QwenModelProfile = DEFAULT_TARGET_PROFILE,
    *,
    device_index: int = 0,
    local_files_only: bool = False,
) -> TorchCUDATargetBackend:
    """Load the pinned NF4 target and production tokenizer as a persistent backend."""

    _validate_load_inputs(profile, device_index, local_files_only)
    modules = {}
    for module_name in ("torch", "transformers", "bitsandbytes"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            message = f"{module_name} could not be imported: {exc}"
            if module_name == "torch":
                message = f"{message}. {PYTORCH_CUDA_INSTALL_GUIDANCE}"
            raise TorchBackendImportError(message) from exc

    torch_module = modules["torch"]
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise TorchBackendLoadError(torch_cuda_unavailable_message(torch_module))
    device_count = cuda.device_count()
    if device_index >= device_count:
        raise TorchBackendLoadError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )

    device = torch_module.device(f"cuda:{device_index}")
    model = None
    tokenizer_load = None
    backend = None
    failure = None
    try:
        tokenizer_load = load_qwen_tokenizer(profile, local_files_only=local_files_only)
        model = _load_nf4_model(
            torch_module,
            modules["transformers"],
            profile=profile,
            device_index=device_index,
            local_files_only=local_files_only,
        )
        model.eval()
        cuda.synchronize(device)
        backend = TorchCUDATargetBackend(
            torch_module=torch_module,
            transformers_module=modules["transformers"],
            model=model,
            tokenizer=tokenizer_load.tokenizer,
            profile=profile,
            device_index=device_index,
        )
    except Exception as exc:
        failure = (
            type(exc)(str(exc))
            if isinstance(exc, TorchBackendError)
            else TorchBackendLoadError(
                f"failed to load PyTorch target {profile.pinned_id}: {exc}"
            )
        )

    if failure is None and backend is None:
        failure = TorchBackendLoadError("PyTorch target load produced no backend")
    if failure is None:
        return backend

    model = None
    tokenizer_load = None
    try:
        gc.collect()
        cuda.empty_cache()
        cuda.synchronize(device)
    except Exception as cleanup_exc:
        raise TorchBackendLoadError(
            f"{failure}; failed-load cleanup also failed: {cleanup_exc}"
        ) from failure
    raise failure


def select_cuda_argmax(logits: Any) -> int:
    """Select the first maximum on CUDA without transferring vocabulary logits to CPU."""

    try:
        shape = tuple(logits.shape)
    except Exception as exc:
        raise TorchBackendInvariantError(f"CUDA logits shape could not be read: {exc}") from exc
    if len(shape) != 1 or not shape or shape[0] <= 0:
        raise TorchBackendInvariantError(
            f"CUDA greedy selector requires one nonempty logits row; received shape {shape}"
        )
    if not bool(getattr(logits, "is_cuda", False)):
        raise TorchBackendInvariantError("CUDA greedy selector received a non-CUDA tensor")
    device = getattr(logits, "device", None)
    device_text = str(device)
    if device is None or not (device_text == "cuda" or device_text.startswith("cuda:")):
        raise TorchBackendInvariantError(
            f"CUDA greedy selector received logits on {device}; expected a CUDA device"
        )

    try:
        contains_nan = bool(logits.isnan().any().item())
    except Exception as exc:
        raise TorchBackendExecutionError(f"CUDA logits validation failed: {exc}") from exc
    if contains_nan:
        raise TorchBackendInvariantError("CUDA logits cannot contain NaN")

    try:
        selected = logits.argmax(dim=-1).item()
    except Exception as exc:
        raise TorchBackendExecutionError(f"CUDA argmax failed: {exc}") from exc
    if isinstance(selected, bool) or not isinstance(selected, int):
        raise TorchBackendInvariantError("CUDA argmax must return an integer token ID")
    if selected < 0 or selected >= shape[0]:
        raise TorchBackendInvariantError(
            f"CUDA argmax returned token ID {selected} outside logits range [0, {shape[0]})"
        )
    return selected


def _validate_load_inputs(
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
        raise TorchBackendLoadError(f"{label} must be an integer")
    if value <= 0:
        raise TorchBackendLoadError(f"{label} must be greater than zero")
    return value


def _validate_checkpoint_integer(value: int, *, label: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{label} cannot be negative")
        raise ValueError(f"{label} must be greater than zero")


def _validate_token_ids(
    token_ids: Sequence[int],
    vocab_size: int,
    *,
    label: str,
    allow_empty: bool,
) -> None:
    if not token_ids and not allow_empty:
        raise ValueError(f"{label} token IDs cannot be empty")
    for token_id in token_ids:
        _validate_token_id(token_id, vocab_size, label=f"{label} token")


def _validate_token_id(token_id: int, vocab_size: int, *, label: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} ID must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{label} ID {token_id} is outside vocabulary range [0, {vocab_size})")


def _validate_verification_token_id(
    token_id: int,
    vocab_size: int,
    *,
    label: str,
) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{label} must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"{label} {token_id} is outside vocabulary range [0, {vocab_size})"
        )
