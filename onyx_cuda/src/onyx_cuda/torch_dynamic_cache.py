"""Pinned Transformers ``DynamicCache`` inspection and transactional rollback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NATIVE_CROP_ROLLBACK_MECHANISM = "transformers_dynamic_cache_native_crop"

_PINNED_LAYER_COUNT = 24
_PINNED_BATCH_SIZE = 1
_PINNED_KEY_VALUE_HEADS = 2
_PINNED_HEAD_DIMENSION = 64
_PINNED_SEQUENCE_DIMENSION = 2
_PINNED_DEVICE_INDEX = 0
_CACHE_ATTRIBUTE_NAMES = ("layer_class_to_replicate", "layers", "offloading")
_LAYER_ATTRIBUTE_NAMES = ("device", "dtype", "is_initialized", "keys", "values")


class DynamicCacheStructureError(RuntimeError):
    """Raised when a cache does not match the characterized pinned layout."""


class DynamicCacheCropError(RuntimeError):
    """Raised when native cache cropping fails after successful staging."""


class DynamicCacheRecoveryError(DynamicCacheCropError):
    """Raised when a failed crop cannot be restored and revalidated."""


@dataclass(frozen=True, slots=True)
class DynamicCacheLayoutSignature:
    """Tensor-free identity of the one cache layout qualified by D29."""

    cache_type: str
    layer_type: str
    tensor_type: str
    layer_count: int
    batch_size: int
    key_value_heads: int
    head_dimension: int
    sequence_dimension: int
    device: str
    dtype: str
    stride_tail: tuple[int, int]
    storage_offset: int
    cache_attributes: tuple[str, ...]
    layer_attributes: tuple[str, ...]
    offloading: bool
    sliding: bool


@dataclass(slots=True)
class _RollbackStage:
    cache_state: dict[str, Any]
    layer_list: list[Any]
    layers: tuple[Any, ...]
    layer_states: tuple[dict[str, Any], ...]
    layout: DynamicCacheLayoutSignature
    cache_length: int


def inspect_pinned_dynamic_cache(
    torch_module: Any,
    transformers_module: Any,
    cache: Any,
    *,
    expected_length: int,
    device_index: int,
) -> DynamicCacheLayoutSignature:
    """Validate one initialized cache and return a tensor-free layout signature.

    This intentionally recognizes only the homogeneous, non-sliding, non-offloaded layout
    characterized with the pinned Qwen2.5 0.5B target and Transformers 4.57.6.
    """

    _require_plain_integer(expected_length, label="expected cache length", minimum=1)
    _require_plain_integer(device_index, label="CUDA device index", minimum=0)
    if device_index != _PINNED_DEVICE_INDEX:
        raise DynamicCacheStructureError(
            f"cache rollback is qualified only for cuda:{_PINNED_DEVICE_INDEX}"
        )

    try:
        cache_type = getattr(transformers_module, "DynamicCache")
        layer_type = getattr(transformers_module, "DynamicLayer")
        tensor_type = getattr(torch_module, "Tensor")
        expected_dtype = str(getattr(torch_module, "float16"))
    except Exception as exc:
        raise DynamicCacheStructureError(
            f"cache runtime types could not be resolved: {exc}"
        ) from exc
    if not isinstance(cache_type, type) or not isinstance(layer_type, type):
        raise DynamicCacheStructureError("Transformers cache runtime types must be concrete classes")
    if not isinstance(tensor_type, type):
        raise DynamicCacheStructureError("PyTorch Tensor must be a concrete class")
    if type(cache) is not cache_type:
        raise DynamicCacheStructureError(
            f"cache type {_qualified_type_name(type(cache))} is unsupported; "
            f"expected {_qualified_type_name(cache_type)}"
        )

    cache_state = _read_object_state(cache, label="cache")
    _require_attribute_names(cache_state, _CACHE_ATTRIBUTE_NAMES, label="cache")
    layers = cache_state["layers"]
    if type(layers) is not list:
        raise DynamicCacheStructureError("cache layers must be stored in a plain list")
    if len(layers) != _PINNED_LAYER_COUNT:
        raise DynamicCacheStructureError(
            f"cache has {len(layers)} layers; expected {_PINNED_LAYER_COUNT}"
        )
    if cache_state["layer_class_to_replicate"] is not None:
        raise DynamicCacheStructureError("cache cannot use a replicated lazy layer class")
    if cache_state["offloading"] is not False:
        raise DynamicCacheStructureError("offloaded caches are unsupported")

    expected_device = f"cuda:{device_index}"
    expected_shape = (
        _PINNED_BATCH_SIZE,
        _PINNED_KEY_VALUE_HEADS,
        expected_length,
        _PINNED_HEAD_DIMENSION,
    )
    expected_stride_tail = (_PINNED_HEAD_DIMENSION, 1)

    for layer_index, layer in enumerate(layers):
        if type(layer) is not layer_type:
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} has unsupported type "
                f"{_qualified_type_name(type(layer))}; expected {_qualified_type_name(layer_type)}"
            )
        layer_state = _read_object_state(layer, label=f"cache layer {layer_index}")
        _require_attribute_names(layer_state, _LAYER_ATTRIBUTE_NAMES, label=f"cache layer {layer_index}")
        if layer_state["is_initialized"] is not True:
            raise DynamicCacheStructureError(f"cache layer {layer_index} is not initialized")
        if getattr(layer, "is_sliding", None) is not False:
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} uses unsupported sliding behavior"
            )

        keys = layer_state["keys"]
        values = layer_state["values"]
        for tensor_label, tensor in (("keys", keys), ("values", values)):
            label = f"cache layer {layer_index} {tensor_label}"
            if type(tensor) is not tensor_type:
                raise DynamicCacheStructureError(
                    f"{label} have unsupported tensor type {_qualified_type_name(type(tensor))}"
                )
            shape = _read_tensor_shape(tensor, label=label)
            if shape != expected_shape:
                raise DynamicCacheStructureError(
                    f"{label} shape {shape} does not match expected {expected_shape}"
                )
            ndim = _read_plain_integer_attribute(tensor, "ndim", label=label, minimum=0)
            if ndim != 4:
                raise DynamicCacheStructureError(f"{label} rank {ndim} does not equal 4")
            if str(getattr(tensor, "device", None)) != expected_device:
                raise DynamicCacheStructureError(
                    f"{label} device {getattr(tensor, 'device', None)} does not equal {expected_device}"
                )
            if getattr(tensor, "is_cuda", None) is not True:
                raise DynamicCacheStructureError(f"{label} must be a CUDA tensor")
            if str(getattr(tensor, "dtype", None)) != expected_dtype:
                raise DynamicCacheStructureError(
                    f"{label} dtype {getattr(tensor, 'dtype', None)} does not equal {expected_dtype}"
                )
            _validate_prefix_view_strides(
                tensor,
                label=label,
                expected_length=expected_length,
                expected_stride_tail=expected_stride_tail,
            )

        if tuple(keys.shape) != tuple(values.shape):
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} key/value shapes disagree"
            )
        if str(keys.device) != str(values.device) or str(keys.dtype) != str(values.dtype):
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} key/value device or dtype disagrees"
            )
        if str(layer_state["device"]) != expected_device:
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} recorded device {layer_state['device']} "
                f"does not equal {expected_device}"
            )
        if str(layer_state["dtype"]) != expected_dtype:
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} recorded dtype {layer_state['dtype']} "
                f"does not equal {expected_dtype}"
            )
        layer_length = _read_sequence_length(layer, label=f"cache layer {layer_index}")
        if layer_length != expected_length:
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} length {layer_length} does not equal {expected_length}"
            )
        indexed_length = _read_sequence_length(
            cache,
            layer_index,
            label=f"cache layer {layer_index} through the cache",
        )
        if indexed_length != expected_length:
            raise DynamicCacheStructureError(
                f"cache layer {layer_index} reports length {indexed_length} through the cache; "
                f"expected {expected_length}"
            )

    cache_length = _read_sequence_length(cache, label="cache")
    if cache_length != expected_length:
        raise DynamicCacheStructureError(
            f"cache length {cache_length} does not equal expected {expected_length}"
        )

    return DynamicCacheLayoutSignature(
        cache_type=_qualified_type_name(cache_type),
        layer_type=_qualified_type_name(layer_type),
        tensor_type=_qualified_type_name(tensor_type),
        layer_count=_PINNED_LAYER_COUNT,
        batch_size=_PINNED_BATCH_SIZE,
        key_value_heads=_PINNED_KEY_VALUE_HEADS,
        head_dimension=_PINNED_HEAD_DIMENSION,
        sequence_dimension=_PINNED_SEQUENCE_DIMENSION,
        device=expected_device,
        dtype=expected_dtype,
        stride_tail=expected_stride_tail,
        storage_offset=0,
        cache_attributes=_CACHE_ATTRIBUTE_NAMES,
        layer_attributes=_LAYER_ATTRIBUTE_NAMES,
        offloading=False,
        sliding=False,
    )


def rollback_pinned_dynamic_cache(
    torch_module: Any,
    transformers_module: Any,
    cache: Any,
    *,
    current_length: int,
    target_length: int,
    device_index: int,
    expected_layout: DynamicCacheLayoutSignature,
) -> None:
    """Crop one qualified cache transactionally, restoring exact references on failure."""

    _require_plain_integer(current_length, label="current cache length", minimum=1)
    _require_plain_integer(target_length, label="target cache length", minimum=1)
    if target_length > current_length:
        raise DynamicCacheStructureError("cache rollback cannot move forward")
    if not isinstance(expected_layout, DynamicCacheLayoutSignature):
        raise DynamicCacheStructureError("expected cache layout signature is invalid")

    current_layout = inspect_pinned_dynamic_cache(
        torch_module,
        transformers_module,
        cache,
        expected_length=current_length,
        device_index=device_index,
    )
    if current_layout != expected_layout:
        raise DynamicCacheStructureError("active cache layout changed within the sequence epoch")
    if target_length == current_length:
        return

    stage = _stage_rollback(cache, current_layout, current_length)
    try:
        result = cache.crop(target_length)
        if result is not None:
            raise DynamicCacheCropError("native DynamicCache.crop() returned an unexpected value")
    except Exception as exc:
        _restore_after_failure(
            torch_module,
            transformers_module,
            cache,
            stage,
            failure=exc,
            device_index=device_index,
        )
        if isinstance(exc, DynamicCacheCropError):
            raise DynamicCacheCropError(
                "native DynamicCache.crop() returned an unexpected value; original cache restored"
            ) from exc
        raise DynamicCacheCropError(
            f"native DynamicCache.crop() failed; original cache restored: {exc}"
        ) from exc

    try:
        cropped_layout = inspect_pinned_dynamic_cache(
            torch_module,
            transformers_module,
            cache,
            expected_length=target_length,
            device_index=device_index,
        )
        if cropped_layout != expected_layout:
            raise DynamicCacheStructureError(
                "native cache crop changed the qualified layout signature"
            )
    except DynamicCacheStructureError as exc:
        _restore_after_failure(
            torch_module,
            transformers_module,
            cache,
            stage,
            failure=exc,
            device_index=device_index,
        )
        raise DynamicCacheStructureError(
            f"native cache crop produced an invalid cache state; original cache restored: {exc}"
        ) from exc


def _stage_rollback(
    cache: Any,
    layout: DynamicCacheLayoutSignature,
    cache_length: int,
) -> _RollbackStage:
    try:
        cache_state = dict(vars(cache))
        layer_list = cache_state["layers"]
        layers = tuple(layer_list)
        layer_states = tuple(dict(vars(layer)) for layer in layers)
    except Exception as exc:
        raise DynamicCacheStructureError(f"cache rollback could not be staged: {exc}") from exc
    return _RollbackStage(
        cache_state=cache_state,
        layer_list=layer_list,
        layers=layers,
        layer_states=layer_states,
        layout=layout,
        cache_length=cache_length,
    )


def _restore_after_failure(
    torch_module: Any,
    transformers_module: Any,
    cache: Any,
    stage: _RollbackStage,
    *,
    failure: Exception,
    device_index: int,
) -> None:
    try:
        for layer, layer_state in zip(stage.layers, stage.layer_states, strict=True):
            current_state = vars(layer)
            current_state.clear()
            current_state.update(layer_state)
        stage.layer_list[:] = stage.layers
        current_cache_state = vars(cache)
        current_cache_state.clear()
        current_cache_state.update(stage.cache_state)

        if cache.layers is not stage.layer_list:
            raise RuntimeError("cache layer-list reference was not restored")
        if tuple(cache.layers) != stage.layers:
            raise RuntimeError("cache layer references were not restored")
        for layer, layer_state in zip(stage.layers, stage.layer_states, strict=True):
            if layer.keys is not layer_state["keys"] or layer.values is not layer_state["values"]:
                raise RuntimeError("cache tensor references were not restored")

        restored_layout = inspect_pinned_dynamic_cache(
            torch_module,
            transformers_module,
            cache,
            expected_length=stage.cache_length,
            device_index=device_index,
        )
        if restored_layout != stage.layout:
            raise RuntimeError("restored cache layout does not match the staged layout")
    except Exception as recovery_exc:
        raise DynamicCacheRecoveryError(
            f"native cache crop failed and exact cache recovery could not be proved: {recovery_exc}"
        ) from failure


def _read_object_state(value: Any, *, label: str) -> dict[str, Any]:
    try:
        return vars(value)
    except Exception as exc:
        raise DynamicCacheStructureError(f"{label} attributes could not be inspected: {exc}") from exc


def _require_attribute_names(
    state: dict[str, Any],
    expected_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    actual_names = tuple(sorted(state))
    if actual_names != expected_names:
        raise DynamicCacheStructureError(
            f"{label} attributes {actual_names} do not match expected {expected_names}"
        )


def _read_tensor_shape(tensor: Any, *, label: str) -> tuple[int, ...]:
    try:
        shape = tuple(tensor.shape)
    except Exception as exc:
        raise DynamicCacheStructureError(f"{label} shape could not be read: {exc}") from exc
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise DynamicCacheStructureError(f"{label} shape contains an invalid dimension")
    return shape


def _validate_prefix_view_strides(
    tensor: Any,
    *,
    label: str,
    expected_length: int,
    expected_stride_tail: tuple[int, int],
) -> None:
    try:
        strides = tuple(tensor.stride())
        storage_offset = tensor.storage_offset()
    except Exception as exc:
        raise DynamicCacheStructureError(
            f"{label} stride or storage offset could not be read: {exc}"
        ) from exc
    if len(strides) != 4 or any(
        isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0
        for stride in strides
    ):
        raise DynamicCacheStructureError(f"{label} strides {strides} are invalid")
    if strides[-2:] != expected_stride_tail:
        raise DynamicCacheStructureError(
            f"{label} stride tail {strides[-2:]} does not match {expected_stride_tail}"
        )
    if strides[0] != _PINNED_KEY_VALUE_HEADS * strides[1]:
        raise DynamicCacheStructureError(f"{label} batch/head strides are inconsistent")
    minimum_head_stride = expected_length * _PINNED_HEAD_DIMENSION
    if strides[1] < minimum_head_stride or strides[1] % _PINNED_HEAD_DIMENSION != 0:
        raise DynamicCacheStructureError(f"{label} head stride is not a supported prefix view")
    if isinstance(storage_offset, bool) or not isinstance(storage_offset, int):
        raise DynamicCacheStructureError(f"{label} storage offset must be an integer")
    if storage_offset != 0:
        raise DynamicCacheStructureError(f"{label} storage offset must be zero")


def _read_plain_integer_attribute(
    value: Any,
    attribute: str,
    *,
    label: str,
    minimum: int,
) -> int:
    try:
        result = getattr(value, attribute)
    except Exception as exc:
        raise DynamicCacheStructureError(
            f"{label} {attribute} could not be read: {exc}"
        ) from exc
    _require_plain_integer(result, label=f"{label} {attribute}", minimum=minimum)
    return result


def _read_sequence_length(value: Any, *args: Any, label: str) -> int:
    try:
        length = value.get_seq_length(*args)
    except Exception as exc:
        raise DynamicCacheStructureError(f"{label} length could not be read: {exc}") from exc
    _require_plain_integer(length, label=f"{label} length", minimum=0)
    return length


def _require_plain_integer(value: Any, *, label: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DynamicCacheStructureError(f"{label} must be an integer")
    if value < minimum:
        raise DynamicCacheStructureError(f"{label} must be at least {minimum}")


def _qualified_type_name(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"
