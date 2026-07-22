import gc
import weakref
from dataclasses import replace

import pytest

import onyx_cuda.torch_dynamic_cache as cache_module
from onyx_cuda.torch_dynamic_cache import (
    DynamicCacheCropError,
    DynamicCacheLayoutSignature,
    DynamicCacheStructureError,
    NATIVE_CROP_ROLLBACK_MECHANISM,
    inspect_pinned_dynamic_cache,
    rollback_pinned_dynamic_cache,
)


class FakeDevice:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeTensor:
    def __init__(
        self,
        contents,
        *,
        device="cuda:0",
        dtype="torch.float16",
        shape=None,
        ndim=None,
        storage_length=None,
        strides=None,
        storage_offset=0,
    ):
        self.contents = tuple(contents)
        self.device = FakeDevice(device)
        self.dtype = dtype
        self.is_cuda = device.startswith("cuda:")
        self.shape = shape or (1, 2, len(self.contents), 64)
        self.ndim = len(self.shape) if ndim is None else ndim
        self.storage_length = len(self.contents) if storage_length is None else storage_length
        self.strides = strides
        self.offset = storage_offset

    def stride(self):
        if self.strides is not None:
            return self.strides
        return (2 * self.storage_length * 64, self.storage_length * 64, 64, 1)

    def storage_offset(self):
        return self.offset

    def cropped(self, length):
        return FakeTensor(
            self.contents[:length],
            device=str(self.device),
            dtype=self.dtype,
            storage_length=self.storage_length,
        )


class FakeDynamicLayer:
    is_sliding = False

    def __init__(self, length, *, initialized=True):
        self.keys = FakeTensor(range(length)) if initialized else None
        self.values = FakeTensor(range(100, 100 + length)) if initialized else None
        self.is_initialized = initialized
        self.dtype = "torch.float16" if initialized else None
        self.device = FakeDevice("cuda:0") if initialized else None

    def get_seq_length(self):
        if not self.is_initialized:
            return 0
        return self.keys.shape[-2]

    def crop(self, length):
        self.keys = self.keys.cropped(length)
        self.values = self.values.cropped(length)


class FakeDynamicCache:
    fail_after_layer = None
    corrupt_after_crop_layer = None
    crop_return = None

    def __init__(self, length=4, *, layer_count=24):
        self.layers = [FakeDynamicLayer(length) for _ in range(layer_count)]
        self.layer_class_to_replicate = None
        self.offloading = False

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()

    def crop(self, length):
        for layer_index, layer in enumerate(self.layers):
            layer.crop(length)
            if layer_index == type(self).fail_after_layer:
                raise RuntimeError(f"injected crop failure after layer {layer_index}")
        corrupt_index = type(self).corrupt_after_crop_layer
        if corrupt_index is not None:
            layer = self.layers[corrupt_index]
            layer.keys = FakeTensor(range(length), shape=(1, 2, length, 63))
        return type(self).crop_return


class FakeTorch:
    Tensor = FakeTensor
    float16 = "torch.float16"


class FakeTransformers:
    DynamicCache = FakeDynamicCache
    DynamicLayer = FakeDynamicLayer


@pytest.fixture(autouse=True)
def reset_fake_crop_behavior():
    FakeDynamicCache.fail_after_layer = None
    FakeDynamicCache.corrupt_after_crop_layer = None
    FakeDynamicCache.crop_return = None
    FakeDynamicLayer.is_sliding = False
    yield
    FakeDynamicCache.fail_after_layer = None
    FakeDynamicCache.corrupt_after_crop_layer = None
    FakeDynamicCache.crop_return = None
    FakeDynamicLayer.is_sliding = False


def inspect(cache, *, length=4):
    return inspect_pinned_dynamic_cache(
        FakeTorch,
        FakeTransformers,
        cache,
        expected_length=length,
        device_index=0,
    )


def rollback(cache, layout, *, current_length=4, target_length=2):
    return rollback_pinned_dynamic_cache(
        FakeTorch,
        FakeTransformers,
        cache,
        current_length=current_length,
        target_length=target_length,
        device_index=0,
        expected_layout=layout,
    )


def cache_references(cache):
    def contents(tensor):
        return getattr(tensor, "contents", None)

    return (
        cache.layers,
        tuple(cache.layers),
        tuple((layer.keys, layer.values) for layer in cache.layers),
        tuple((contents(layer.keys), contents(layer.values)) for layer in cache.layers),
    )


def assert_references_restored(cache, before):
    layer_list, layers, tensors, contents = before
    assert cache.layers is layer_list
    assert tuple(cache.layers) == layers
    for index, layer in enumerate(cache.layers):
        assert layer.keys is tensors[index][0]
        assert layer.values is tensors[index][1]
        assert getattr(layer.keys, "contents", None) == contents[index][0]
        assert getattr(layer.values, "contents", None) == contents[index][1]


def test_inspection_freezes_the_characterized_tensor_free_layout():
    cache = FakeDynamicCache()

    layout = inspect(cache)

    assert isinstance(layout, DynamicCacheLayoutSignature)
    assert layout.layer_count == 24
    assert layout.batch_size == 1
    assert layout.key_value_heads == 2
    assert layout.head_dimension == 64
    assert layout.sequence_dimension == 2
    assert layout.device == "cuda:0"
    assert layout.dtype == "torch.float16"
    assert layout.stride_tail == (64, 1)
    assert layout.storage_offset == 0
    assert layout.offloading is False
    assert layout.sliding is False
    assert NATIVE_CROP_ROLLBACK_MECHANISM == "transformers_dynamic_cache_native_crop"


def test_successful_native_crop_preserves_exact_prefix_and_uses_prefix_views():
    cache = FakeDynamicCache()
    layout = inspect(cache)
    original_storage_lengths = tuple(layer.keys.storage_length for layer in cache.layers)

    rollback(cache, layout)

    assert cache.get_seq_length() == 2
    assert inspect(cache, length=2) == layout
    for index, layer in enumerate(cache.layers):
        assert layer.keys.contents == (0, 1)
        assert layer.values.contents == (100, 101)
        assert layer.keys.storage_length == original_storage_lengths[index] == 4


def test_same_position_rollback_is_a_valid_non_mutating_success():
    cache = FakeDynamicCache()
    layout = inspect(cache)
    before = cache_references(cache)

    rollback(cache, layout, target_length=4)

    assert_references_restored(cache, before)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cache: setattr(cache, "offloading", True), "offloaded"),
        (lambda cache: setattr(cache, "layer_class_to_replicate", FakeDynamicLayer), "replicated"),
        (lambda cache: cache.layers.pop(), "23 layers"),
        (lambda cache: setattr(cache.layers[0], "is_initialized", False), "not initialized"),
        (lambda cache: setattr(cache.layers[0], "keys", object()), "tensor type"),
        (lambda cache: setattr(cache.layers[0].keys, "shape", (2, 2, 4, 64)), "shape"),
        (lambda cache: setattr(cache.layers[0].keys, "shape", (1, 3, 4, 64)), "shape"),
        (lambda cache: setattr(cache.layers[0].values, "shape", (1, 2, 4, 63)), "shape"),
        (lambda cache: setattr(cache.layers[0].keys, "ndim", 3), "rank"),
        (lambda cache: setattr(cache.layers[0].keys, "device", FakeDevice("cpu")), "device"),
        (lambda cache: setattr(cache.layers[0].values, "dtype", "torch.float32"), "dtype"),
        (lambda cache: setattr(cache.layers[0].keys, "strides", (512, 256, 1, 4)), "stride"),
        (lambda cache: setattr(cache.layers[0].keys, "offset", 1), "storage offset"),
        (lambda cache: setattr(cache, "unexpected_counter", 1), "attributes"),
        (lambda cache: setattr(cache.layers[0], "unexpected_counter", 1), "attributes"),
    ],
)
def test_each_structural_invariant_fails_closed_without_mutation(mutation, message):
    cache = FakeDynamicCache()
    mutation(cache)
    before = cache_references(cache)

    with pytest.raises(DynamicCacheStructureError, match=message):
        inspect(cache)

    assert_references_restored(cache, before)


def test_wrong_cache_and_layer_concrete_types_are_rejected():
    class OtherCache(FakeDynamicCache):
        pass

    cache = OtherCache()
    with pytest.raises(DynamicCacheStructureError, match="cache type"):
        inspect(cache)

    cache = FakeDynamicCache()

    class OtherLayer(FakeDynamicLayer):
        pass

    cache.layers[0] = OtherLayer(4)
    with pytest.raises(DynamicCacheStructureError, match="layer 0.*type"):
        inspect(cache)


def test_sliding_layer_and_wrong_device_index_are_rejected():
    cache = FakeDynamicCache()
    FakeDynamicLayer.is_sliding = True

    with pytest.raises(DynamicCacheStructureError, match="sliding"):
        inspect(cache)

    FakeDynamicLayer.is_sliding = False
    with pytest.raises(DynamicCacheStructureError, match="only for cuda:0"):
        inspect_pinned_dynamic_cache(
            FakeTorch,
            FakeTransformers,
            cache,
            expected_length=4,
            device_index=1,
        )


@pytest.mark.parametrize("reported_length", [True, -1, "4"])
def test_malformed_layer_sequence_length_is_rejected(monkeypatch, reported_length):
    cache = FakeDynamicCache()
    bad_layer = cache.layers[7]
    original = FakeDynamicLayer.get_seq_length

    def malformed_length(layer):
        if layer is bad_layer:
            return reported_length
        return original(layer)

    monkeypatch.setattr(FakeDynamicLayer, "get_seq_length", malformed_length)

    with pytest.raises(DynamicCacheStructureError, match="length"):
        inspect(cache)


def test_inconsistent_per_layer_length_is_rejected(monkeypatch):
    cache = FakeDynamicCache()
    bad_layer = cache.layers[12]
    original = FakeDynamicLayer.get_seq_length

    def inconsistent_length(layer):
        if layer is bad_layer:
            return 3
        return original(layer)

    monkeypatch.setattr(FakeDynamicLayer, "get_seq_length", inconsistent_length)

    with pytest.raises(DynamicCacheStructureError, match="layer 12 length 3"):
        inspect(cache)


def test_layout_mismatch_and_forward_crop_are_precommit_failures():
    cache = FakeDynamicCache()
    layout = inspect(cache)
    before = cache_references(cache)
    different_layout = replace(layout, head_dimension=32)

    with pytest.raises(DynamicCacheStructureError, match="layout changed"):
        rollback(cache, different_layout)
    assert_references_restored(cache, before)
    with pytest.raises(DynamicCacheStructureError, match="cannot move forward"):
        rollback(cache, layout, target_length=5)
    assert_references_restored(cache, before)

    with pytest.raises(DynamicCacheStructureError, match="layout signature is invalid"):
        rollback_pinned_dynamic_cache(
            FakeTorch,
            FakeTransformers,
            cache,
            current_length=4,
            target_length=2,
            device_index=0,
            expected_layout=object(),
        )
    assert_references_restored(cache, before)


def test_staging_failure_is_precommit_and_cache_neutral(monkeypatch):
    cache = FakeDynamicCache()
    layout = inspect(cache)
    before = cache_references(cache)

    def fail_staging(*args, **kwargs):
        raise DynamicCacheStructureError("injected staging failure")

    monkeypatch.setattr(cache_module, "_stage_rollback", fail_staging)

    with pytest.raises(DynamicCacheStructureError, match="staging failure"):
        rollback(cache, layout)
    assert_references_restored(cache, before)


def test_partial_commit_failure_restores_exact_references_and_contents():
    cache = FakeDynamicCache()
    layout = inspect(cache)
    before = cache_references(cache)
    FakeDynamicCache.fail_after_layer = 5

    with pytest.raises(DynamicCacheCropError, match="original cache restored") as raised:
        rollback(cache, layout)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert_references_restored(cache, before)
    assert inspect(cache) == layout


def test_post_validation_failure_restores_exact_references_and_contents():
    cache = FakeDynamicCache()
    layout = inspect(cache)
    before = cache_references(cache)
    FakeDynamicCache.corrupt_after_crop_layer = 9

    with pytest.raises(DynamicCacheStructureError, match="original cache restored") as raised:
        rollback(cache, layout)

    assert isinstance(raised.value.__cause__, DynamicCacheStructureError)
    assert_references_restored(cache, before)
    assert inspect(cache) == layout


def test_unexpected_crop_return_is_restored_and_reported_as_execution_failure():
    cache = FakeDynamicCache()
    layout = inspect(cache)
    before = cache_references(cache)
    FakeDynamicCache.crop_return = object()

    with pytest.raises(DynamicCacheCropError, match="unexpected value"):
        rollback(cache, layout)

    assert_references_restored(cache, before)


def test_layout_and_completed_transaction_retain_no_cache_or_tensor_reference():
    cache = FakeDynamicCache()
    cache_ref = weakref.ref(cache)
    tensor_ref = weakref.ref(cache.layers[0].keys)
    layout = inspect(cache)

    rollback(cache, layout)
    del cache
    gc.collect()

    assert cache_ref() is None
    assert tensor_ref() is None
    assert layout.layer_count == 24
