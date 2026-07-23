from contextlib import nullcontext
from dataclasses import FrozenInstanceError, fields, replace
import inspect
from types import SimpleNamespace
import gc
import weakref

import pytest

import onyx_cuda.torch_backend as torch_backend_module
from onyx_cuda import (
    TRANSFORMERS_DYNAMIC_CACHE_MODE,
    AutoregressiveBackend,
    BackendStateError,
    BatchedTargetVerificationBackend,
    BatchedTargetVerificationResult,
    CacheCheckpoint,
    CacheCheckpointStateError,
    CheckpointableAutoregressiveBackend,
    DEFAULT_TARGET_PROFILE,
    DraftProposalCleanupError,
    QWEN_3B_CANDIDATE_PROFILE,
    generate_draft_proposal,
)
from onyx_cuda.torch_backend import (
    TorchBackendExecutionError,
    TorchBackendImportError,
    TorchBackendInvariantError,
    TorchBackendLoadError,
    TorchCUDATargetBackend,
    _TorchCacheCheckpoint,
    load_torch_cuda_target,
)


class FakeDevice:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeInputTensor:
    def __init__(self, values, device, dtype):
        self.values = values
        self.device = device
        self.dtype = dtype
        self.shape = (len(values), len(values[0]))


class FakeLogitVector:
    def __init__(
        self,
        size,
        device,
        marker,
        *,
        dtype="torch.float16",
        is_cuda=None,
        parent=None,
    ):
        self.shape = (size,)
        self.device = device
        self.dtype = dtype
        self.is_cuda = str(device).startswith("cuda:") if is_cuda is None else is_cuda
        self.marker = marker
        self._parent = parent

    def cpu(self):
        raise AssertionError("verification logits must not be transferred to the host")

    def tolist(self):
        raise AssertionError("verification logits must not be converted to a Python container")


class FakeLogits:
    def __init__(
        self,
        shape,
        device,
        markers=(),
        *,
        dtype="torch.float16",
        is_cuda=None,
        row_size=None,
        row_device=None,
        row_dtype=None,
        row_is_cuda=None,
        row_error_at=None,
    ):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.is_cuda = str(device).startswith("cuda:") if is_cuda is None else is_cuda
        self.markers = tuple(markers)
        self.row_size = row_size
        self.row_device = device if row_device is None else FakeDevice(row_device)
        self.row_dtype = dtype if row_dtype is None else row_dtype
        self.row_is_cuda = row_is_cuda
        self.row_error_at = row_error_at
        self.slice_calls = []

    def __getitem__(self, key):
        self.slice_calls.append(key)
        row_index = key[1]
        if row_index == self.row_error_at:
            raise RuntimeError(f"injected row extraction failure at {row_index}")
        token_slice = key[2]
        size = token_slice.stop if self.row_size is None else self.row_size
        return FakeLogitVector(
            size,
            self.row_device,
            self.markers[row_index],
            dtype=self.row_dtype,
            is_cuda=self.row_is_cuda,
            parent=self,
        )

    def cpu(self):
        raise AssertionError("verification logits must not be transferred to the host")

    def tolist(self):
        raise AssertionError("verification logits must not be converted to a Python container")


class FakeCacheTensor:
    def __init__(
        self,
        token_ids=(),
        *,
        device="cuda:0",
        dtype="torch.float16",
        storage_length=None,
    ):
        self.token_ids = tuple(token_ids)
        self.device = FakeDevice(device)
        self.dtype = dtype
        self.is_cuda = device.startswith("cuda:")
        self.shape = (1, 2, len(self.token_ids), 64)
        self.ndim = 4
        self._storage_length = len(self.token_ids) if storage_length is None else storage_length

    def stride(self):
        return (2 * self._storage_length * 64, self._storage_length * 64, 64, 1)

    def storage_offset(self):
        return 0

    def crop(self, length):
        return FakeCacheTensor(
            self.token_ids[:length],
            device=str(self.device),
            dtype=self.dtype,
            storage_length=self._storage_length,
        )

    def append(self, token_ids):
        combined = self.token_ids + tuple(token_ids)
        return FakeCacheTensor(combined, device=str(self.device), dtype=self.dtype)


class FakeDynamicLayer:
    is_sliding = False

    def __init__(self):
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.dtype = None
        self.device = None

    def update(self, token_ids):
        if not self.is_initialized:
            self.keys = FakeCacheTensor(token_ids)
            self.values = FakeCacheTensor(token_ids)
            self.is_initialized = True
            self.dtype = self.keys.dtype
            self.device = self.keys.device
            return
        self.keys = self.keys.append(token_ids)
        self.values = self.values.append(token_ids)

    def crop(self, length):
        self.keys = self.keys.crop(length)
        self.values = self.values.crop(length)

    def get_seq_length(self):
        if not self.is_initialized:
            return 0
        return self.keys.shape[-2]


class FakeCache:
    def __init__(self, *, config):
        self.layers = [FakeDynamicLayer() for _ in range(config.num_hidden_layers)]
        self.layer_class_to_replicate = None
        self.offloading = False

    def append(self, token_ids):
        for layer in self.layers:
            layer.update(token_ids)

    def crop(self, length):
        for layer in self.layers:
            layer.crop(length)

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()


class FailingDynamicCache:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def __call__(self, *, config):
        self.calls.append(config)
        if self.error is not None:
            raise self.error
        return FakeCache(config=config)


class FakeModel:
    is_loaded_in_4bit = True

    def __init__(
        self,
        *,
        model_vocab_size=8,
        embedding_rows=None,
        error=None,
        output_shape=None,
        output_device="cuda:0",
        output_dtype="torch.float16",
        output_is_cuda=None,
        row_size=None,
        row_device=None,
        row_dtype=None,
        row_is_cuda=None,
        row_error_at=None,
        cache_increment=None,
        error_after_cache_update=None,
        post_cache_mutation=None,
        memory_footprint=4096,
    ):
        self.config = SimpleNamespace(
            vocab_size=model_vocab_size,
            num_hidden_layers=24,
            num_key_value_heads=2,
            hidden_size=896,
            num_attention_heads=14,
            use_sliding_window=False,
            sliding_window=None,
            layer_types=["full_attention"] * 24,
        )
        self.embedding_rows = embedding_rows or model_vocab_size
        self.error = error
        self.output_shape = output_shape
        self.output_device = FakeDevice(output_device)
        self.output_dtype = output_dtype
        self.output_is_cuda = output_is_cuda
        self.row_size = row_size
        self.row_device = row_device
        self.row_dtype = row_dtype
        self.row_is_cuda = row_is_cuda
        self.row_error_at = row_error_at
        self.cache_increment = cache_increment
        self.error_after_cache_update = error_after_cache_update
        self.post_cache_mutation = post_cache_mutation
        self.memory_footprint = memory_footprint
        self.calls = []
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1

    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(shape=(self.embedding_rows, 4)))

    def get_memory_footprint(self):
        return self.memory_footprint

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        input_length = kwargs["input_ids"].shape[1]
        increment = input_length if self.cache_increment is None else self.cache_increment
        input_token_ids = tuple(kwargs["input_ids"].values[0])
        appended_token_ids = input_token_ids[:increment]
        existing_token_ids = ()
        if kwargs["past_key_values"].layers[0].is_initialized:
            existing_token_ids = kwargs["past_key_values"].layers[0].keys.token_ids
        kwargs["past_key_values"].append(appended_token_ids)
        if self.post_cache_mutation is not None:
            self.post_cache_mutation(kwargs["past_key_values"])
        if self.error_after_cache_update is not None:
            raise self.error_after_cache_update
        row_count = kwargs["logits_to_keep"]
        markers = tuple(
            existing_token_ids + input_token_ids[:position]
            for position in range(input_length - row_count + 1, input_length + 1)
        )
        shape = self.output_shape or (1, row_count, self.config.vocab_size)
        return SimpleNamespace(
            logits=FakeLogits(
                shape,
                self.output_device,
                markers=markers,
                dtype=self.output_dtype,
                is_cuda=self.output_is_cuda,
                row_size=self.row_size,
                row_device=self.row_device,
                row_dtype=self.row_dtype,
                row_is_cuda=self.row_is_cuda,
                row_error_at=self.row_error_at,
            ),
            past_key_values=kwargs["past_key_values"],
        )


class FakeCuda:
    def __init__(
        self,
        *,
        available=True,
        device_count=1,
        cleanup_error=None,
        synchronize_error=None,
    ):
        self.available = available
        self.detected_device_count = device_count
        self.cleanup_error = cleanup_error
        self.synchronize_error = synchronize_error
        self.empty_cache_calls = 0
        self.synchronize_calls = 0

    def is_available(self):
        return self.available

    def device_count(self):
        return self.detected_device_count

    def empty_cache(self):
        self.empty_cache_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error

    def synchronize(self, device):
        self.synchronize_calls += 1
        if self.synchronize_error is not None:
            raise self.synchronize_error


class FakeTorch:
    long = "long"
    float16 = "torch.float16"
    Tensor = FakeCacheTensor

    def __init__(self, cuda=None, *, tensor_error=None):
        self.cuda = cuda or FakeCuda()
        self.tensor_error = tensor_error
        self.tensor_calls = []

    def device(self, name):
        return FakeDevice(name)

    def tensor(self, values, *, dtype, device):
        self.tensor_calls.append((values, dtype, device))
        if self.tensor_error is not None:
            raise self.tensor_error
        return FakeInputTensor(values, device, dtype)

    def inference_mode(self):
        return nullcontext()


class FakeTransformers:
    def __init__(self, dynamic_cache=None):
        self.DynamicCache = FakeCache if dynamic_cache is None else dynamic_cache
        self.DynamicLayer = FakeDynamicLayer


class OnePassProposal:
    def __init__(self, token_ids):
        self._token_ids = token_ids
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("proposal was materialized more than once")
        return iter(self._token_ids)


def make_backend(
    *,
    model=None,
    torch_module=None,
    transformers_module=None,
    tokenizer_size=6,
    profile=DEFAULT_TARGET_PROFILE,
    device_index=0,
):
    return TorchCUDATargetBackend(
        torch_module=torch_module or FakeTorch(),
        transformers_module=transformers_module or FakeTransformers(),
        model=model or FakeModel(),
        tokenizer=SimpleNamespace(
            vocab_size=tokenizer_size,
            tokenizer_id=profile.pinned_id,
        ),
        profile=profile,
        device_index=device_index,
    )


def backend_checkpoint_state(backend):
    cache_state = None
    if backend._cache is not None:
        cache_state = (
            id(backend._cache),
            id(backend._cache.layers),
            tuple(
                (
                    id(layer),
                    id(layer.keys),
                    id(layer.values),
                    layer.keys.token_ids,
                    layer.values.token_ids,
                    tuple(layer.keys.shape),
                    tuple(layer.values.shape),
                )
                for layer in backend._cache.layers
            ),
        )
    return (
        backend._epoch,
        tuple(backend._active_token_ids),
        backend._active_cache_layout,
        backend._next_checkpoint_id,
        tuple(backend._cache_checkpoints.items()),
        cache_state,
    )


def assert_cache_prefix(backend, token_ids):
    assert tuple(backend._active_token_ids) == tuple(token_ids)
    assert backend.cache_length == len(token_ids)
    for layer in backend._cache.layers:
        assert layer.keys.token_ids == tuple(token_ids)
        assert layer.values.token_ids == tuple(token_ids)


def assert_terminal_verification_state(backend, *, previous_epoch, checkpoint):
    assert backend._epoch == previous_epoch + 1
    assert backend._cache is None
    assert backend.cache_length == 0
    assert backend._active_token_ids == []
    assert backend._active_cache_layout is None
    assert backend._cache_checkpoints == {}
    assert backend._next_checkpoint_id == 1
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(checkpoint)


def test_backend_satisfies_contract_and_exposes_padded_vocab_boundary():
    backend = make_backend()
    verification_parameters = tuple(
        inspect.signature(TorchCUDATargetBackend.verify_proposal).parameters.values()
    )

    assert isinstance(backend, AutoregressiveBackend)
    assert isinstance(backend, CheckpointableAutoregressiveBackend)
    assert isinstance(backend, BatchedTargetVerificationBackend)
    assert tuple(parameter.name for parameter in verification_parameters) == (
        "self",
        "current_token_id",
        "proposal_token_ids",
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in verification_parameters
    )
    assert backend.model_id == DEFAULT_TARGET_PROFILE.pinned_id
    assert backend.vocab_size == 6
    assert backend.model_vocab_size == 8
    assert backend.padded_vocab_rows == 2
    assert backend.device_index == 0
    assert backend.cache_mode == TRANSFORMERS_DYNAMIC_CACHE_MODE
    assert backend.model_memory_footprint_bytes == 4096
    assert backend.cache_length == 0
    assert not backend.is_closed


def test_prefill_and_decode_return_native_cropped_logits_and_cache_lengths():
    model = FakeModel()
    torch_module = FakeTorch()
    backend = make_backend(model=model, torch_module=torch_module)

    prefill = backend.prefill([1, 2])
    decode = backend.decode(3)

    assert prefill.cache_length == 2
    assert prefill.logits.shape == (6,)
    assert str(prefill.logits.device) == "cuda:0"
    assert decode.cache_length == 3
    assert decode.logits.shape == (6,)
    assert backend.cache_length == 3
    assert torch_module.tensor_calls[0][0] == [(1, 2)]
    assert torch_module.tensor_calls[1][0] == [(3,)]
    assert all(call["use_cache"] is True for call in model.calls)
    assert all(call["return_dict"] is True for call in model.calls)
    assert all(call["logits_to_keep"] == 1 for call in model.calls)
    assert model.calls[0]["past_key_values"] is model.calls[1]["past_key_values"]
    assert model.calls[0]["input_ids"].shape == (1, 2)
    assert model.calls[1]["input_ids"].shape == (1, 1)


def test_one_token_proposal_uses_one_batched_forward_and_returns_two_native_rows():
    model = FakeModel()
    torch_module = FakeTorch()
    backend = make_backend(model=model, torch_module=torch_module)
    backend.prefill([1, 2])
    cache = backend._cache
    calls_before = len(model.calls)

    result = backend.verify_proposal(3, (4,))

    assert isinstance(result, BatchedTargetVerificationResult)
    assert result.cache_length == backend.cache_length == 4
    assert len(result.logit_rows) == 2
    assert tuple(row.marker for row in result.logit_rows) == (
        (1, 2, 3),
        (1, 2, 3, 4),
    )
    assert all(row.shape == (6,) for row in result.logit_rows)
    assert all(row.dtype == "torch.float16" for row in result.logit_rows)
    assert all(row.is_cuda is True for row in result.logit_rows)
    assert all(str(row.device) == "cuda:0" for row in result.logit_rows)
    assert len(model.calls) == calls_before + 1
    call = model.calls[-1]
    assert call["input_ids"].shape == (1, 2)
    assert call["input_ids"].values == [(3, 4)]
    assert call["input_ids"].dtype == torch_module.long
    assert str(call["input_ids"].device) == "cuda:0"
    assert call["past_key_values"] is cache is backend._cache
    assert call["use_cache"] is True
    assert call["return_dict"] is True
    assert call["logits_to_keep"] == 2
    assert set(call) == {
        "input_ids",
        "past_key_values",
        "use_cache",
        "return_dict",
        "logits_to_keep",
    }
    assert len(torch_module.tensor_calls) == 2
    assert model.calls[0]["logits_to_keep"] == 1
    assert_cache_prefix(backend, (1, 2, 3, 4))


def test_multi_token_proposal_exposes_all_rows_in_native_order_without_padded_vocab():
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([1])

    result = backend.verify_proposal(2, (3, 4, 5))

    assert tuple(row.marker for row in result.logit_rows) == (
        (1, 2),
        (1, 2, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
    )
    assert len(result.logit_rows) == 4
    assert all(row.shape == (backend.vocab_size,) for row in result.logit_rows)
    assert model.calls[-1]["logits_to_keep"] == 4
    assert model.calls[-1]["input_ids"].shape == (1, 4)
    assert_cache_prefix(backend, (1, 2, 3, 4, 5))


def test_proposal_is_materialized_once_and_consecutive_batches_have_no_hidden_limit():
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([0])
    proposal = OnePassProposal((2, 3))

    first = backend.verify_proposal(1, proposal)
    second = backend.verify_proposal(4, (5, 0, 1, 2, 3))

    assert proposal.iterations == 1
    assert len(first.logit_rows) == 3
    assert len(second.logit_rows) == 6
    assert model.calls[-2]["logits_to_keep"] == 3
    assert model.calls[-1]["logits_to_keep"] == 6
    assert_cache_prefix(backend, (0, 1, 2, 3, 4, 5, 0, 1, 2, 3))


@pytest.mark.parametrize(
    ("current_token", "proposal", "error", "message"),
    [
        (True, (1,), TypeError, "current token must be an integer"),
        ("1", (1,), TypeError, "current token must be an integer"),
        (-1, (1,), ValueError, "current token -1"),
        (6, (1,), ValueError, "current token 6"),
        (1, None, TypeError, "proposal_token_ids must be a sequence"),
        (1, (), ValueError, "proposal_token_ids cannot be empty"),
        (1, (0, True), TypeError, "proposal token at position 1 must be an integer"),
        (1, (0, "2"), TypeError, "proposal token at position 1 must be an integer"),
        (1, (0, -1), ValueError, "proposal token at position 1 -1"),
        (1, (0, 6), ValueError, "proposal token at position 1 6"),
    ],
)
def test_invalid_verification_inputs_are_pre_execution_and_state_preserving(
    current_token,
    proposal,
    error,
    message,
):
    model = FakeModel()
    torch_module = FakeTorch()
    backend = make_backend(model=model, torch_module=torch_module)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    before = backend_checkpoint_state(backend)
    tensor_calls = len(torch_module.tensor_calls)
    model_calls = len(model.calls)

    with pytest.raises(error, match=message):
        backend.verify_proposal(current_token, proposal)

    assert backend_checkpoint_state(backend) == before
    assert len(torch_module.tensor_calls) == tensor_calls
    assert len(model.calls) == model_calls
    backend.rollback_cache(checkpoint)
    assert backend_checkpoint_state(backend) == before


def test_verify_before_prefill_is_pre_execution_and_state_preserving():
    model = FakeModel()
    torch_module = FakeTorch()
    backend = make_backend(model=model, torch_module=torch_module)
    before = backend_checkpoint_state(backend)

    with pytest.raises(BackendStateError, match="prefill"):
        backend.verify_proposal(1, (2,))

    assert backend_checkpoint_state(backend) == before
    assert torch_module.tensor_calls == []
    assert model.calls == []


@pytest.mark.parametrize(
    ("profile", "device_index"),
    [
        (DEFAULT_TARGET_PROFILE, 1),
        (QWEN_3B_CANDIDATE_PROFILE, 0),
    ],
)
def test_unqualified_profile_or_device_rejects_verification_before_execution(
    profile,
    device_index,
):
    model = FakeModel(output_device=f"cuda:{device_index}")
    torch_module = FakeTorch()
    backend = make_backend(
        model=model,
        torch_module=torch_module,
        profile=profile,
        device_index=device_index,
    )
    backend.prefill([1])
    decoded = backend.decode(2)
    before = backend_checkpoint_state(backend)
    tensor_calls = len(torch_module.tensor_calls)
    model_calls = len(model.calls)

    with pytest.raises(BackendStateError, match="pinned 0.5B target on cuda:0"):
        backend.verify_proposal(3, (4,))

    assert decoded.cache_length == 2
    assert backend_checkpoint_state(backend) == before
    assert len(torch_module.tensor_calls) == tensor_calls
    assert len(model.calls) == model_calls


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda backend: backend._active_token_ids.append(3),
            "token-prefix length",
        ),
        (
            lambda backend: setattr(backend, "_active_cache_layout", None),
            "no qualified layout signature",
        ),
        (
            lambda backend: setattr(
                backend,
                "_active_cache_layout",
                replace(backend._active_cache_layout, head_dimension=63),
            ),
            "layout changed",
        ),
        (
            lambda backend: setattr(
                backend._cache.layers[3].keys,
                "shape",
                (1, 2, 2, 63),
            ),
            "DynamicCache invariant",
        ),
    ],
)
def test_corrupt_verification_preflight_is_terminal_without_model_execution(
    mutation,
    message,
):
    model = FakeModel()
    torch_module = FakeTorch()
    backend = make_backend(model=model, torch_module=torch_module)
    backend.prefill([1, 2])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    tensor_calls = len(torch_module.tensor_calls)
    model_calls = len(model.calls)
    mutation(backend)

    with pytest.raises(TorchBackendInvariantError, match=message):
        backend.verify_proposal(3, (4,))

    assert len(torch_module.tensor_calls) == tensor_calls
    assert len(model.calls) == model_calls
    assert_terminal_verification_state(
        backend,
        previous_epoch=previous_epoch,
        checkpoint=checkpoint,
    )


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (
            lambda model: setattr(model, "error", RuntimeError("kernel failed")),
            TorchBackendExecutionError,
            "kernel failed",
        ),
        (
            lambda model: setattr(
                model,
                "error_after_cache_update",
                RuntimeError("partial cache failure"),
            ),
            TorchBackendExecutionError,
            "partial cache failure",
        ),
        (
            lambda model: setattr(model, "output_shape", (3, 8)),
            TorchBackendInvariantError,
            "logits shape",
        ),
        (
            lambda model: setattr(model, "output_shape", (2, 3, 8)),
            TorchBackendInvariantError,
            "logits shape",
        ),
        (
            lambda model: setattr(model, "output_shape", (1, 2, 8)),
            TorchBackendInvariantError,
            "logits shape",
        ),
        (
            lambda model: setattr(model, "output_shape", (1, 3, 7)),
            TorchBackendInvariantError,
            "logits shape",
        ),
        (
            lambda model: setattr(model, "output_dtype", "torch.float32"),
            TorchBackendInvariantError,
            "logits dtype",
        ),
        (
            lambda model: setattr(model, "output_device", FakeDevice("cpu")),
            TorchBackendInvariantError,
            "logits are on",
        ),
        (
            lambda model: setattr(model, "output_is_cuda", False),
            TorchBackendInvariantError,
            "must be a CUDA tensor",
        ),
        (
            lambda model: setattr(model, "cache_increment", 2),
            TorchBackendInvariantError,
            "cache length",
        ),
        (
            lambda model: setattr(
                model,
                "post_cache_mutation",
                lambda cache: cache.append((0,)),
            ),
            TorchBackendInvariantError,
            "cache length",
        ),
        (
            lambda model: setattr(
                model,
                "post_cache_mutation",
                lambda cache: setattr(
                    cache.layers[8].keys,
                    "shape",
                    (1, 2, 5, 63),
                ),
            ),
            TorchBackendInvariantError,
            "DynamicCache invariant",
        ),
        (
            lambda model: setattr(model, "row_size", 5),
            TorchBackendInvariantError,
            "row 0 shape",
        ),
        (
            lambda model: setattr(model, "row_dtype", "torch.float32"),
            TorchBackendInvariantError,
            "row 0 dtype",
        ),
        (
            lambda model: setattr(model, "row_device", "cpu"),
            TorchBackendInvariantError,
            "row 0 is on",
        ),
        (
            lambda model: setattr(model, "row_is_cuda", False),
            TorchBackendInvariantError,
            "row 0 must be a CUDA tensor",
        ),
        (
            lambda model: setattr(model, "row_error_at", 1),
            TorchBackendExecutionError,
            "row extraction failure",
        ),
    ],
)
def test_verification_execution_and_output_failures_are_terminal(
    mutation,
    error_type,
    message,
):
    model = FakeModel()
    torch_module = FakeTorch()
    backend = make_backend(model=model, torch_module=torch_module)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    calls_before = len(model.calls)
    mutation(model)

    with pytest.raises(error_type, match=message):
        backend.verify_proposal(2, (3, 4))

    assert len(model.calls) == calls_before + 1
    assert_terminal_verification_state(
        backend,
        previous_epoch=previous_epoch,
        checkpoint=checkpoint,
    )


def test_verification_output_cache_replacement_is_terminal():
    class CacheReplacingModel(FakeModel):
        replace_cache = False

        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            if self.replace_cache:
                output.past_key_values = FakeCache(config=self.config)
            return output

    model = CacheReplacingModel()
    backend = make_backend(model=model)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    model.replace_cache = True

    with pytest.raises(TorchBackendInvariantError, match="replaced"):
        backend.verify_proposal(2, (3,))

    assert_terminal_verification_state(
        backend,
        previous_epoch=previous_epoch,
        checkpoint=checkpoint,
    )


@pytest.mark.parametrize("failure_stage", ("tensor", "synchronize"))
def test_tensor_creation_or_synchronization_failure_is_terminal(failure_stage):
    cuda = FakeCuda()
    torch_module = FakeTorch(cuda=cuda)
    model = FakeModel()
    backend = make_backend(model=model, torch_module=torch_module)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    if failure_stage == "tensor":
        torch_module.tensor_error = RuntimeError("tensor creation failed")
    else:
        cuda.synchronize_error = RuntimeError("synchronization failed")

    message = "tensor creation" if failure_stage == "tensor" else "synchronization"
    with pytest.raises(TorchBackendExecutionError, match=message):
        backend.verify_proposal(2, (3,))

    assert_terminal_verification_state(
        backend,
        previous_epoch=previous_epoch,
        checkpoint=checkpoint,
    )


@pytest.mark.parametrize("result_failure", ("raise", "malformed"))
def test_result_construction_failure_is_terminal(monkeypatch, result_failure):
    backend = make_backend()
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    if result_failure == "raise":
        def build_result(**kwargs):
            raise RuntimeError("injected result construction failure")

        error_type = TorchBackendExecutionError
        message = "result construction failure"
    else:
        def build_result(**kwargs):
            return BatchedTargetVerificationResult(
                logit_rows=(object(),),
                cache_length=kwargs["cache_length"],
            )

        error_type = TorchBackendInvariantError
        message = "invalid verification result"
    monkeypatch.setattr(backend, "_build_verification_result", build_result)

    with pytest.raises(error_type, match=message):
        backend.verify_proposal(2, (3,))

    assert_terminal_verification_state(
        backend,
        previous_epoch=previous_epoch,
        checkpoint=checkpoint,
    )


def test_checkpoint_is_immutable_slotted_cpu_only_and_satisfies_public_contract():
    backend = make_backend()
    backend.prefill([1, 2])

    checkpoint = backend.create_cache_checkpoint()
    copied_checkpoint = replace(checkpoint)
    snapshot = backend._cache_checkpoints[checkpoint.allocation_id]

    assert isinstance(checkpoint, CacheCheckpoint)
    assert checkpoint.cache_length == backend.cache_length == 2
    assert copied_checkpoint == checkpoint
    assert copied_checkpoint is not checkpoint
    assert not hasattr(checkpoint, "__dict__")
    assert not hasattr(snapshot, "__dict__")
    assert all(
        isinstance(value, int)
        for value in (
            checkpoint.owner_id,
            checkpoint.epoch,
            checkpoint.allocation_id,
            checkpoint.cache_length,
        )
    )
    assert snapshot.token_ids == (1, 2)
    assert all(
        isinstance(value, (str, int, bool, tuple))
        for field_name in snapshot.layout.__slots__
        for value in (getattr(snapshot.layout, field_name),)
    )
    with pytest.raises(FrozenInstanceError):
        checkpoint.cache_length = 3

    backend.rollback_cache(copied_checkpoint)


def test_verification_composes_with_exact_rollback_replay_and_alternative_suffix():
    backend = make_backend()
    backend.prefill([0, 1])
    cache = backend._cache
    layout = backend._active_cache_layout
    root = backend.create_cache_checkpoint()
    same_position = backend.create_cache_checkpoint()

    first = backend.verify_proposal(2, (3, 4))
    first_markers = tuple(row.marker for row in first.logit_rows)
    post_batch = backend.create_cache_checkpoint()
    assert first.cache_length == 5
    assert cache is backend._cache
    assert backend._active_cache_layout == layout
    assert_cache_prefix(backend, (0, 1, 2, 3, 4))

    backend.rollback_cache(root)

    assert cache is backend._cache
    assert backend._active_cache_layout == layout
    assert_cache_prefix(backend, (0, 1))
    assert tuple(backend._cache_checkpoints) == (
        root.allocation_id,
        same_position.allocation_id,
    )
    backend.rollback_cache(same_position)
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(post_batch)

    replay = backend.verify_proposal(2, (3, 4))
    assert tuple(row.marker for row in replay.logit_rows) == first_markers
    assert replay.cache_length == first.cache_length
    assert_cache_prefix(backend, (0, 1, 2, 3, 4))

    backend.rollback_cache(root)
    alternative = backend.verify_proposal(5, (4, 3))
    assert tuple(row.marker for row in alternative.logit_rows) == (
        (0, 1, 5),
        (0, 1, 5, 4),
        (0, 1, 5, 4, 3),
    )
    assert_cache_prefix(backend, (0, 1, 5, 4, 3))


def test_verification_after_prefix_view_crop_compacts_and_preserves_layout():
    backend = make_backend()
    backend.prefill([0, 1])
    root = backend.create_cache_checkpoint()
    layout = backend._active_cache_layout
    backend.verify_proposal(2, (3, 4))

    backend.rollback_cache(root)

    cropped_keys = backend._cache.layers[0].keys
    assert cropped_keys._storage_length == 5
    assert cropped_keys.shape == (1, 2, 2, 64)
    replay = backend.verify_proposal(2, (3, 4))
    compacted_keys = backend._cache.layers[0].keys
    assert replay.cache_length == 5
    assert compacted_keys is not cropped_keys
    assert compacted_keys._storage_length == 5
    assert compacted_keys.stride() == (640, 320, 64, 1)
    assert backend._active_cache_layout == layout
    assert_cache_prefix(backend, (0, 1, 2, 3, 4))


def test_nested_and_same_position_checkpoint_lifetimes_hold_around_verification():
    backend = make_backend()
    backend.prefill([0])
    earliest = backend.create_cache_checkpoint()
    backend.decode(1)
    before_batch = backend.create_cache_checkpoint()
    same_position = backend.create_cache_checkpoint()
    backend.verify_proposal(2, (3, 4))
    after_batch = backend.create_cache_checkpoint()

    backend.rollback_cache(before_batch)

    assert_cache_prefix(backend, (0, 1))
    assert tuple(backend._cache_checkpoints) == (
        earliest.allocation_id,
        before_batch.allocation_id,
        same_position.allocation_id,
    )
    backend.release_cache_checkpoint(before_batch)
    backend.release_cache_checkpoint(before_batch)
    backend.rollback_cache(same_position)
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(after_batch)

    backend.rollback_cache(earliest)
    assert_cache_prefix(backend, (0,))
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(same_position)


def test_bounded_verification_rollback_release_has_no_registry_or_row_order_drift():
    backend = make_backend()
    backend.prefill([0])
    layout = backend._active_cache_layout

    for iteration in range(1_000):
        checkpoint = backend.create_cache_checkpoint()
        current_token = iteration % backend.vocab_size
        proposal_token = (iteration + 1) % backend.vocab_size
        result = backend.verify_proposal(current_token, (proposal_token,))
        assert tuple(row.marker for row in result.logit_rows) == (
            (0, current_token),
            (0, current_token, proposal_token),
        )
        backend.rollback_cache(checkpoint)
        backend.release_cache_checkpoint(checkpoint)
        assert backend._cache_checkpoints == {}
        assert backend._active_cache_layout == layout
        assert_cache_prefix(backend, (0,))

    assert backend._next_checkpoint_id == 1_001


@pytest.mark.parametrize("replacement", (None, (4,)))
def test_reset_or_replacement_prefill_invalidates_verification_era_handles(replacement):
    backend = make_backend()
    backend.prefill([0])
    before_batch = backend.create_cache_checkpoint()
    backend.verify_proposal(1, (2,))
    after_batch = backend.create_cache_checkpoint()

    if replacement is None:
        backend.reset()
    else:
        backend.prefill(replacement)

    assert backend._cache_checkpoints == {}
    for checkpoint in (before_batch, after_batch):
        with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
            backend.rollback_cache(checkpoint)
        backend.release_cache_checkpoint(checkpoint)


def test_verification_rows_and_parent_logits_are_caller_owned_only():
    class NonRetainingFakeModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.input_refs = []

        def __call__(self, **kwargs):
            self.input_refs.append(weakref.ref(kwargs["input_ids"]))
            output = super().__call__(**kwargs)
            self.calls.clear()
            return output

    model = NonRetainingFakeModel()
    backend = make_backend(model=model)
    backend.prefill([0])
    checkpoint = backend.create_cache_checkpoint()
    state_before = backend_checkpoint_state(backend)

    result = backend.verify_proposal(1, (2,))
    result_state = backend_checkpoint_state(backend)
    row_refs = tuple(weakref.ref(row) for row in result.logit_rows)
    parent = result.logit_rows[0]._parent
    assert all(row._parent is parent for row in result.logit_rows)
    parent_ref = weakref.ref(parent)
    assert model.input_refs[-1]() is None
    assert all(
        value is not result and value is not parent
        for value in vars(backend).values()
    )
    snapshot = backend._cache_checkpoints[checkpoint.allocation_id]
    assert all(
        not isinstance(
            value,
            (FakeInputTensor, FakeLogits, FakeLogitVector, FakeCacheTensor, FakeModel),
        )
        for value in (
            checkpoint.owner_id,
            checkpoint.epoch,
            checkpoint.allocation_id,
            checkpoint.cache_length,
            snapshot.token_ids,
            snapshot.layout,
        )
    )

    del parent
    del result
    gc.collect()

    assert parent_ref() is None
    assert all(row_ref() is None for row_ref in row_refs)
    assert backend_checkpoint_state(backend) == result_state
    assert backend_checkpoint_state(backend) != state_before
    backend.rollback_cache(checkpoint)
    assert_cache_prefix(backend, (0,))


# D34 production draft-proposal composition


def _release_d34_result(backend, result):
    for checkpoint in result.rollback_checkpoints:
        backend.release_cache_checkpoint(checkpoint)


def _d34_recording_selector(token_ids, seen_rows):
    selected = iter(token_ids)

    def select(row):
        seen_rows.append(row)
        return next(selected)

    return select


def _assert_d34_checkpoint_metadata_is_tensor_free(backend, result, selector):
    forbidden_types = (
        FakeInputTensor,
        FakeLogits,
        FakeLogitVector,
        FakeCacheTensor,
        FakeModel,
        type(selector),
    )
    assert not hasattr(result, "__dict__")
    assert all(
        not isinstance(value, forbidden_types)
        for value in (
            result.proposal_token_ids,
            result.rollback_checkpoints,
            result.initial_cache_length,
            result.final_cache_length,
        )
    )
    for checkpoint in result.rollback_checkpoints:
        snapshot = backend._cache_checkpoints[checkpoint.allocation_id]
        assert all(
            isinstance(getattr(checkpoint, field.name), int)
            for field in fields(checkpoint)
        )
        assert all(isinstance(token_id, int) for token_id in snapshot.token_ids)
        assert all(
            not isinstance(getattr(snapshot.layout, field.name), forbidden_types)
            for field in fields(snapshot.layout)
        )


def test_d34_one_token_proposal_uses_two_one_token_decodes_and_no_verification(
    monkeypatch,
):
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([0, 1])
    cache = backend._cache
    layout = backend._active_cache_layout
    model_calls_before = len(model.calls)
    seen_rows = []

    def fail_verification(*args, **kwargs):
        raise AssertionError("D34 must not invoke batched target verification")

    monkeypatch.setattr(backend, "verify_proposal", fail_verification)
    result = generate_draft_proposal(
        backend,
        2,
        proposal_length=1,
        select_token=_d34_recording_selector((3,), seen_rows),
    )

    decode_calls = model.calls[model_calls_before:]
    assert [tuple(call["input_ids"].values[0]) for call in decode_calls] == [
        (2,),
        (3,),
    ]
    assert [call["logits_to_keep"] for call in decode_calls] == [1, 1]
    assert [row.marker for row in seen_rows] == [(0, 1, 2)]
    assert result.proposal_token_ids == (3,)
    assert result.initial_cache_length == 2
    assert result.final_cache_length == 4
    assert tuple(cp.cache_length for cp in result.rollback_checkpoints) == (3,)
    assert tuple(backend._cache_checkpoints) == (
        result.rollback_checkpoints[0].allocation_id,
    )
    assert backend._cache is cache
    assert backend._active_cache_layout == layout
    assert_cache_prefix(backend, (0, 1, 2, 3))

    final_prefix = tuple(backend._active_token_ids)
    _release_d34_result(backend, result)
    _release_d34_result(backend, result)
    assert backend._cache_checkpoints == {}
    assert tuple(backend._active_token_ids) == final_prefix
    assert backend._cache is cache
    assert backend._active_cache_layout == layout
    assert_cache_prefix(backend, final_prefix)


def test_d34_multi_token_proposal_aligns_rows_checkpoints_and_production_cache():
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([0, 1])
    cache = backend._cache
    layout = backend._active_cache_layout
    model_calls_before = len(model.calls)
    seen_rows = []
    selector = _d34_recording_selector((3, 4, 5), seen_rows)

    result = generate_draft_proposal(
        backend,
        2,
        proposal_length=3,
        select_token=selector,
    )

    decode_calls = model.calls[model_calls_before:]
    assert [tuple(call["input_ids"].values[0]) for call in decode_calls] == [
        (2,),
        (3,),
        (4,),
        (5,),
    ]
    assert all(call["logits_to_keep"] == 1 for call in decode_calls)
    assert [row.marker for row in seen_rows] == [
        (0, 1, 2),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 4),
    ]
    assert all(row.marker != (0, 1, 2, 3, 4, 5) for row in seen_rows)
    assert result.proposal_token_ids == (3, 4, 5)
    assert result.initial_cache_length == 2
    assert result.final_cache_length == 6
    assert tuple(cp.cache_length for cp in result.rollback_checkpoints) == (3, 4, 5)
    assert tuple(backend._cache_checkpoints) == tuple(
        cp.allocation_id for cp in result.rollback_checkpoints
    )
    assert backend._next_checkpoint_id == 5
    assert backend._cache is cache
    assert backend._active_cache_layout == layout
    assert len(backend._cache.layers) == 24
    assert_cache_prefix(backend, (0, 1, 2, 3, 4, 5))
    _assert_d34_checkpoint_metadata_is_tensor_free(backend, result, selector)

    _release_d34_result(backend, result)


@pytest.mark.parametrize("rejection_index", [0, 1, 2])
def test_d34_each_rejection_checkpoint_restores_the_exact_production_prefix(
    rejection_index,
):
    backend = make_backend()
    prompt = (0, 1)
    backend.prefill(prompt)
    root = backend.create_cache_checkpoint()
    result = generate_draft_proposal(
        backend,
        2,
        proposal_length=3,
        select_token=_d34_recording_selector((3, 4, 5), []),
    )
    target = result.rollback_checkpoints[rejection_index]

    backend.rollback_cache(target)

    expected_prefix = (*prompt, 2, *result.proposal_token_ids[:rejection_index])
    assert target.cache_length == len(prompt) + 1 + rejection_index
    assert_cache_prefix(backend, expected_prefix)
    assert tuple(backend._cache_checkpoints) == (
        root.allocation_id,
        *(cp.allocation_id for cp in result.rollback_checkpoints[: rejection_index + 1]),
    )
    same_position_state = backend_checkpoint_state(backend)
    backend.rollback_cache(target)
    assert backend_checkpoint_state(backend) == same_position_state
    for checkpoint in result.rollback_checkpoints[rejection_index + 1 :]:
        with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
            backend.rollback_cache(checkpoint)

    cache_state = (
        id(backend._cache),
        backend._active_cache_layout,
        tuple(backend._active_token_ids),
    )
    _release_d34_result(backend, result)
    _release_d34_result(backend, result)
    assert (
        id(backend._cache),
        backend._active_cache_layout,
        tuple(backend._active_token_ids),
    ) == cache_state
    assert tuple(backend._cache_checkpoints) == (root.allocation_id,)
    backend.rollback_cache(root)
    assert_cache_prefix(backend, prompt)
    backend.release_cache_checkpoint(root)


def test_d34_root_rollback_replays_with_fresh_selector_and_new_allocations():
    backend = make_backend()
    prompt = (0, 1)
    backend.prefill(prompt)
    root = backend.create_cache_checkpoint()
    cache = backend._cache
    layout = backend._active_cache_layout

    first = generate_draft_proposal(
        backend,
        2,
        proposal_length=3,
        select_token=_d34_recording_selector((3, 4, 5), []),
    )
    first_allocations = tuple(cp.allocation_id for cp in first.rollback_checkpoints)
    backend.rollback_cache(root)
    _release_d34_result(backend, first)

    replay = generate_draft_proposal(
        backend,
        2,
        proposal_length=3,
        select_token=_d34_recording_selector((3, 4, 5), []),
    )

    replay_allocations = tuple(cp.allocation_id for cp in replay.rollback_checkpoints)
    assert replay.proposal_token_ids == first.proposal_token_ids == (3, 4, 5)
    assert tuple(cp.cache_length for cp in replay.rollback_checkpoints) == (3, 4, 5)
    assert min(replay_allocations) > max(first_allocations)
    assert backend._cache is cache
    assert backend._active_cache_layout == layout
    assert_cache_prefix(backend, (*prompt, 2, *replay.proposal_token_ids))

    backend.rollback_cache(root)
    _release_d34_result(backend, replay)
    backend.release_cache_checkpoint(root)


def test_d34_bounded_propose_rollback_release_keeps_one_root_and_monotonic_ids():
    backend = make_backend()
    prompt = (0,)
    backend.prefill(prompt)
    root = backend.create_cache_checkpoint()
    cache = backend._cache
    layout = backend._active_cache_layout
    epoch = backend._epoch
    previous_allocation = root.allocation_id

    for _ in range(250):
        result = generate_draft_proposal(
            backend,
            1,
            proposal_length=1,
            select_token=lambda row: 2,
        )
        allocation = result.rollback_checkpoints[0].allocation_id
        assert allocation > previous_allocation
        previous_allocation = allocation
        backend.rollback_cache(root)
        _release_d34_result(backend, result)
        _release_d34_result(backend, result)
        assert tuple(backend._cache_checkpoints) == (root.allocation_id,)
        assert backend._epoch == epoch
        assert backend._cache is cache
        assert backend._active_cache_layout == layout
        assert_cache_prefix(backend, prompt)

    assert backend._next_checkpoint_id == 502
    backend.release_cache_checkpoint(root)
    assert backend._cache_checkpoints == {}


def test_d34_unsupported_production_profile_fails_before_decode_or_selection():
    model = FakeModel()
    backend = make_backend(model=model, profile=QWEN_3B_CANDIDATE_PROFILE)
    backend.prefill([0, 1])
    before = backend_checkpoint_state(backend)
    model_calls_before = len(model.calls)
    selector_calls = []

    with pytest.raises(BackendStateError, match="pinned 0.5B target on cuda:0"):
        generate_draft_proposal(
            backend,
            2,
            proposal_length=1,
            select_token=lambda row: selector_calls.append(row),
        )

    assert len(model.calls) == model_calls_before
    assert selector_calls == []
    assert backend_checkpoint_state(backend) == before
    assert_cache_prefix(backend, (0, 1))


def test_d34_selector_failure_restores_root_and_backend_is_immediately_reusable():
    model = FakeModel()
    backend = make_backend(model=model)
    prompt = (0, 1)
    backend.prefill(prompt)
    root = backend.create_cache_checkpoint()
    cache = backend._cache
    layout = backend._active_cache_layout
    epoch = backend._epoch
    failure = LookupError("injected D34 selector failure")
    seen_rows = []

    def failing_selector(row):
        seen_rows.append(row)
        if len(seen_rows) == 2:
            raise failure
        return 3

    with pytest.raises(LookupError, match="injected D34 selector failure") as raised:
        generate_draft_proposal(
            backend,
            2,
            proposal_length=3,
            select_token=failing_selector,
        )

    assert raised.value is failure
    assert [row.marker for row in seen_rows] == [(0, 1, 2), (0, 1, 2, 3)]
    assert backend._cache is cache
    assert backend._active_cache_layout == layout
    assert backend._epoch == epoch
    assert tuple(backend._cache_checkpoints) == (root.allocation_id,)
    assert backend._next_checkpoint_id == 5
    assert_cache_prefix(backend, prompt)

    result = generate_draft_proposal(
        backend,
        2,
        proposal_length=3,
        select_token=_d34_recording_selector((3, 4, 5), []),
    )
    assert min(cp.allocation_id for cp in result.rollback_checkpoints) > 4
    assert_cache_prefix(backend, (*prompt, 2, *result.proposal_token_ids))
    backend.rollback_cache(root)
    _release_d34_result(backend, result)
    backend.release_cache_checkpoint(root)


def test_d34_terminal_decode_failure_reports_cleanup_error_and_safe_empty_epoch():
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([0, 1])
    root = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    injected = RuntimeError("injected D34 post-mutation failure")

    def arm_terminal_failure(row):
        model.error_after_cache_update = injected
        return 3

    with pytest.raises(DraftProposalCleanupError) as raised:
        generate_draft_proposal(
            backend,
            2,
            proposal_length=2,
            select_token=arm_terminal_failure,
        )

    error = raised.value
    assert isinstance(error.original_failure, TorchBackendExecutionError)
    assert error.original_failure.__cause__ is injected
    assert [operation for operation, _ in error.cleanup_failures] == [
        "start checkpoint rollback"
    ]
    assert isinstance(error.cleanup_failures[0][1], CacheCheckpointStateError)
    assert backend._epoch == previous_epoch + 1
    assert backend._cache is None
    assert backend.cache_length == 0
    assert backend._active_token_ids == []
    assert backend._active_cache_layout is None
    assert backend._cache_checkpoints == {}
    assert backend._next_checkpoint_id == 1
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(root)
    backend.release_cache_checkpoint(root)

    model.error_after_cache_update = None
    backend.prefill([0, 1])
    assert_cache_prefix(backend, (0, 1))


def test_d34_result_and_checkpoint_registry_do_not_retain_transient_objects():
    class NonRetainingFakeModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.input_refs = []

        def __call__(self, **kwargs):
            self.input_refs.append(weakref.ref(kwargs["input_ids"]))
            output = super().__call__(**kwargs)
            self.calls.clear()
            return output

    class WeakRecordingSelector:
        def __init__(self):
            self.row_refs = []
            self.token_ids = iter((3, 4, 5))

        def __call__(self, row):
            self.row_refs.append(weakref.ref(row))
            return next(self.token_ids)

    model = NonRetainingFakeModel()
    backend = make_backend(model=model)
    backend.prefill([0, 1])
    selector = WeakRecordingSelector()
    selector_ref = weakref.ref(selector)

    result = generate_draft_proposal(
        backend,
        2,
        proposal_length=3,
        select_token=selector,
    )
    row_refs = tuple(selector.row_refs)
    _assert_d34_checkpoint_metadata_is_tensor_free(backend, result, selector)
    assert all(reference() is None for reference in model.input_refs)

    del selector
    gc.collect()

    assert selector_ref() is None
    assert all(reference() is None for reference in row_refs)
    assert_cache_prefix(backend, (0, 1, 2, 3, 4, 5))
    _release_d34_result(backend, result)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"owner_id": True}, TypeError, "owner_id must be an integer"),
        ({"owner_id": 0}, ValueError, "owner_id must be greater than zero"),
        ({"epoch": 0}, ValueError, "epoch must be greater than zero"),
        ({"allocation_id": 0}, ValueError, "allocation_id must be greater than zero"),
        ({"cache_length": True}, TypeError, "cache_length must be an integer"),
        ({"cache_length": -1}, ValueError, "cache_length cannot be negative"),
    ],
)
def test_private_checkpoint_constructor_rejects_invalid_metadata(changes, error, message):
    values = {"owner_id": 1, "epoch": 1, "allocation_id": 1, "cache_length": 0}
    values.update(changes)

    with pytest.raises(error, match=message):
        _TorchCacheCheckpoint(**values)


def test_checkpoint_requires_a_successful_nonempty_prefill_atomically():
    backend = make_backend()
    before = backend_checkpoint_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="prefill"):
        backend.create_cache_checkpoint()

    assert backend_checkpoint_state(backend) == before


def test_exact_advance_rollback_replay_and_alternative_suffix():
    backend = make_backend()
    prefill = backend.prefill([1, 2])
    checkpoint = backend.create_cache_checkpoint()

    first = backend.decode(3)
    second = backend.decode(4)
    assert prefill.logits.marker == (1, 2)
    assert first.logits.marker == (1, 2, 3)
    assert second.logits.marker == (1, 2, 3, 4)
    assert_cache_prefix(backend, (1, 2, 3, 4))

    backend.rollback_cache(checkpoint)
    assert_cache_prefix(backend, (1, 2))
    replay = backend.decode(3)
    assert replay.logits.marker == first.logits.marker
    assert_cache_prefix(backend, (1, 2, 3))

    backend.rollback_cache(checkpoint)
    alternative = backend.decode(5)
    assert alternative.logits.marker == (1, 2, 5)
    assert alternative.logits.marker != first.logits.marker
    assert_cache_prefix(backend, (1, 2, 5))


def test_nested_rollback_retains_prefix_handles_and_discards_deeper_handles():
    backend = make_backend()
    backend.prefill([1])
    earliest = backend.create_cache_checkpoint()
    backend.decode(2)
    middle = backend.create_cache_checkpoint()
    backend.decode(3)
    deepest = backend.create_cache_checkpoint()
    backend.decode(4)

    backend.rollback_cache(middle)

    assert_cache_prefix(backend, (1, 2))
    assert tuple(backend._cache_checkpoints) == (
        earliest.allocation_id,
        middle.allocation_id,
    )
    before = backend_checkpoint_state(backend)
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(deepest)
    assert backend_checkpoint_state(backend) == before
    backend.rollback_cache(middle)

    backend.rollback_cache(earliest)
    assert_cache_prefix(backend, (1,))
    with pytest.raises(CacheCheckpointStateError, match="discarded suffix"):
        backend.rollback_cache(middle)


def test_same_length_handles_are_independent_and_release_is_idempotent_cache_neutral():
    backend = make_backend()
    backend.prefill([1, 2])
    first = backend.create_cache_checkpoint()
    second = backend.create_cache_checkpoint()

    assert first != second
    assert first.cache_length == second.cache_length == 2
    assert first.allocation_id != second.allocation_id
    cache_before_release = backend_checkpoint_state(backend)[-1]

    backend.release_cache_checkpoint(first)
    after_release = backend_checkpoint_state(backend)
    backend.release_cache_checkpoint(first)

    assert backend_checkpoint_state(backend) == after_release
    assert backend_checkpoint_state(backend)[-1] == cache_before_release
    with pytest.raises(CacheCheckpointStateError, match="released"):
        backend.rollback_cache(first)

    before_same_position = backend_checkpoint_state(backend)
    backend.rollback_cache(second)
    assert backend_checkpoint_state(backend) == before_same_position

    backend.release_cache_checkpoint(second)
    backend.release_cache_checkpoint(second)
    assert backend._cache_checkpoints == {}
    assert_cache_prefix(backend, (1, 2))


def test_wrong_type_and_foreign_handles_are_rejected_atomically():
    owner = make_backend()
    owner.prefill([1])
    foreign = owner.create_cache_checkpoint()

    backend = make_backend()
    backend.prefill([2])
    local = backend.create_cache_checkpoint()
    backend.decode(3)

    for operation in (backend.rollback_cache, backend.release_cache_checkpoint):
        before = backend_checkpoint_state(backend)
        with pytest.raises(TypeError, match="_TorchCacheCheckpoint"):
            operation(object())
        assert backend_checkpoint_state(backend) == before

        with pytest.raises(CacheCheckpointStateError, match="another backend"):
            operation(foreign)
        assert backend_checkpoint_state(backend) == before

    backend.rollback_cache(local)
    owner.rollback_cache(foreign)


def test_unknown_fabricated_mismatched_and_released_handles_are_atomic():
    backend = make_backend()
    backend.prefill([1, 2])
    valid = backend.create_cache_checkpoint()
    unknown = replace(valid, allocation_id=valid.allocation_id + 100)
    mismatched = replace(valid, cache_length=valid.cache_length + 1)

    for invalid in (unknown, mismatched):
        before = backend_checkpoint_state(backend)
        with pytest.raises(CacheCheckpointStateError):
            backend.rollback_cache(invalid)
        assert backend_checkpoint_state(backend) == before

    before_unknown_release = backend_checkpoint_state(backend)
    backend.release_cache_checkpoint(unknown)
    assert backend_checkpoint_state(backend) == before_unknown_release

    before_mismatched_release = backend_checkpoint_state(backend)
    with pytest.raises(CacheCheckpointStateError, match="metadata"):
        backend.release_cache_checkpoint(mismatched)
    assert backend_checkpoint_state(backend) == before_mismatched_release

    backend.release_cache_checkpoint(valid)
    before_released = backend_checkpoint_state(backend)
    with pytest.raises(CacheCheckpointStateError, match="released"):
        backend.rollback_cache(valid)
    assert backend_checkpoint_state(backend) == before_released


def test_corrupted_checkpoint_registry_is_rejected_with_typed_atomic_errors():
    backend = make_backend()
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    backend._cache_checkpoints[checkpoint.allocation_id] = object()

    for operation in (backend.rollback_cache, backend.release_cache_checkpoint):
        before = backend_checkpoint_state(backend)
        with pytest.raises(CacheCheckpointStateError, match="invalid snapshot"):
            operation(checkpoint)
        assert backend_checkpoint_state(backend) == before

    backend = make_backend()
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    backend._cache_checkpoints[0] = backend._cache_checkpoints[checkpoint.allocation_id]
    before = backend_checkpoint_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="registry allocation"):
        backend.rollback_cache(checkpoint)
    assert backend_checkpoint_state(backend) == before
    with pytest.raises(CacheCheckpointStateError, match="registry allocation"):
        backend.create_cache_checkpoint()
    assert backend_checkpoint_state(backend) == before


def test_forward_position_and_prefix_mismatch_are_rejected_atomically():
    backend = make_backend()
    backend.prefill([1, 2])
    checkpoint = backend.create_cache_checkpoint()
    backend._cache.crop(1)
    backend._active_token_ids = [1]
    before_forward = backend_checkpoint_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="ahead"):
        backend.rollback_cache(checkpoint)
    assert backend_checkpoint_state(backend) == before_forward

    backend = make_backend()
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    backend.decode(2)
    backend._active_token_ids[0] = 0
    before_prefix = backend_checkpoint_state(backend)

    with pytest.raises(CacheCheckpointStateError, match="prefix"):
        backend.rollback_cache(checkpoint)
    assert backend_checkpoint_state(backend) == before_prefix


@pytest.mark.parametrize("token_id", [-1, 6, True, "1"])
def test_invalid_decode_preserves_active_checkpoint_and_complete_state(token_id):
    backend = make_backend()
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    before = backend_checkpoint_state(backend)

    with pytest.raises((TypeError, ValueError)):
        backend.decode(token_id)

    assert backend_checkpoint_state(backend) == before
    backend.rollback_cache(checkpoint)


@pytest.mark.parametrize("prompt", [[], [-1], [6], [True], ["1"]])
def test_invalid_prefill_preserves_active_checkpoint_and_complete_state(prompt):
    backend = make_backend()
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    backend.decode(2)
    before = backend_checkpoint_state(backend)

    with pytest.raises((TypeError, ValueError)):
        backend.prefill(prompt)

    assert backend_checkpoint_state(backend) == before
    backend.rollback_cache(checkpoint)
    assert_cache_prefix(backend, (1,))


def test_identical_replacement_invalidates_old_epoch_and_recycles_allocation_safely():
    backend = make_backend()
    backend.prefill([1, 2])
    old = backend.create_cache_checkpoint()

    backend.prefill([1, 2])
    current = backend.create_cache_checkpoint()

    assert old.allocation_id == current.allocation_id == 1
    assert old.cache_length == current.cache_length == 2
    assert old.epoch != current.epoch
    before = backend_checkpoint_state(backend)
    backend.release_cache_checkpoint(old)
    assert backend_checkpoint_state(backend) == before
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(old)
    assert backend_checkpoint_state(backend) == before
    backend.rollback_cache(current)


def test_reset_invalidates_handles_and_clears_all_checkpoint_bookkeeping():
    backend = make_backend()
    backend.prefill([1])
    first = backend.create_cache_checkpoint()
    backend.decode(2)
    second = backend.create_cache_checkpoint()

    backend.reset()

    assert backend.cache_length == 0
    assert backend._active_token_ids == []
    assert backend._active_cache_layout is None
    assert backend._cache_checkpoints == {}
    assert backend._next_checkpoint_id == 1
    for checkpoint in (first, second):
        before = backend_checkpoint_state(backend)
        with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
            backend.rollback_cache(checkpoint)
        assert backend_checkpoint_state(backend) == before
        backend.release_cache_checkpoint(checkpoint)
        assert backend_checkpoint_state(backend) == before


def test_bounded_checkpoint_reuse_keeps_registry_empty_and_allocation_monotonic():
    backend = make_backend()
    backend.prefill([1])

    for iteration in range(1_000):
        checkpoint = backend.create_cache_checkpoint()
        step = backend.decode(iteration % backend.vocab_size)
        assert step.cache_length == 2
        backend.rollback_cache(checkpoint)
        backend.release_cache_checkpoint(checkpoint)
        assert backend._cache_checkpoints == {}
        assert_cache_prefix(backend, (1,))

    assert backend._next_checkpoint_id == 1_001


def test_repeated_epochs_never_alias_stale_handles():
    backend = make_backend()
    previous = None

    for epoch_number in range(100):
        backend.prefill([epoch_number % backend.vocab_size])
        current = backend.create_cache_checkpoint()
        if previous is not None:
            assert current.allocation_id == previous.allocation_id == 1
            assert current.epoch != previous.epoch
            before = backend_checkpoint_state(backend)
            backend.release_cache_checkpoint(previous)
            assert backend_checkpoint_state(backend) == before
            with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
                backend.rollback_cache(previous)
            assert backend_checkpoint_state(backend) == before
        backend.release_cache_checkpoint(current)
        previous = current

    assert backend._cache_checkpoints == {}


def test_checkpoint_metadata_does_not_retain_cache_tensors_or_model():
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([1, 2])
    checkpoint = backend.create_cache_checkpoint()
    cache_ref = weakref.ref(backend._cache)
    tensor_ref = weakref.ref(backend._cache.layers[0].keys)
    model_ref = weakref.ref(model)

    backend.close()
    del model
    del backend
    gc.collect()

    assert checkpoint.cache_length == 2
    assert cache_ref() is None
    assert tensor_ref() is None
    assert model_ref() is None


def test_nondefault_candidate_target_path_remains_usable_but_is_not_d29_qualified():
    backend = make_backend(profile=QWEN_3B_CANDIDATE_PROFILE)

    step = backend.prefill([1, 2])
    decoded = backend.decode(3)

    assert step.cache_length == 2
    assert decoded.cache_length == 3
    with pytest.raises(CacheCheckpointStateError, match="pinned 0.5B"):
        backend.create_cache_checkpoint()


def test_prefill_restarts_with_a_fresh_dynamic_cache():
    class RecordingFakeCache(FakeCache):
        calls = []

        def __init__(self, *, config):
            self.calls.append(config)
            super().__init__(config=config)

    backend = make_backend(transformers_module=FakeTransformers(RecordingFakeCache))

    backend.prefill([1, 2])
    first_cache = backend._cache
    restarted = backend.prefill([3])

    assert restarted.cache_length == 1
    assert backend._cache is not first_cache
    assert len(RecordingFakeCache.calls) == 2


def test_reset_clears_cache_and_decode_requires_prefill():
    backend = make_backend()
    backend.prefill([1])

    backend.reset()

    assert backend.cache_length == 0
    with pytest.raises(BackendStateError, match="prefill"):
        backend.decode(1)


@pytest.mark.parametrize("prompt", [[], [-1], [6], [True], ["1"]])
def test_invalid_prefill_does_not_replace_existing_cache(prompt):
    backend = make_backend()
    backend.prefill([1, 2])
    existing_cache = backend._cache

    with pytest.raises((TypeError, ValueError)):
        backend.prefill(prompt)

    assert backend._cache is existing_cache
    assert backend.cache_length == 2


@pytest.mark.parametrize("token_id", [-1, 6, True, "1"])
def test_invalid_decode_does_not_change_cache(token_id):
    backend = make_backend()
    backend.prefill([1, 2])

    with pytest.raises((TypeError, ValueError)):
        backend.decode(token_id)

    assert backend.cache_length == 2


def test_forward_failure_resets_cache_to_safe_empty_state():
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([1])
    model.error = RuntimeError("kernel failed")

    with pytest.raises(TorchBackendExecutionError, match="kernel failed"):
        backend.decode(2)

    assert backend.cache_length == 0


def test_cache_creation_failure_invalidates_old_handles_and_keeps_replacement_epoch_empty():
    backend = make_backend()
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    backend._transformers.DynamicCache = FailingDynamicCache(RuntimeError("cache failed"))

    with pytest.raises(TorchBackendExecutionError, match="cache failed"):
        backend.prefill([2])

    assert backend._epoch == previous_epoch + 1
    assert backend.cache_length == 0
    assert backend._active_token_ids == []
    assert backend._active_cache_layout is None
    assert backend._cache_checkpoints == {}
    assert backend._next_checkpoint_id == 1
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(checkpoint)
    before = backend_checkpoint_state(backend)
    backend.release_cache_checkpoint(checkpoint)
    assert backend_checkpoint_state(backend) == before


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (lambda model: setattr(model, "error", RuntimeError("kernel failed")), TorchBackendExecutionError, "kernel failed"),
        (lambda model: setattr(model, "output_shape", (1, 2, 8)), TorchBackendInvariantError, "logits shape"),
        (lambda model: setattr(model, "output_device", FakeDevice("cpu")), TorchBackendInvariantError, "logits are on"),
        (lambda model: setattr(model, "cache_increment", 0), TorchBackendInvariantError, "cache length"),
    ],
)
def test_decode_terminal_failures_invalidate_checkpoint_epoch_and_clear_all_state(
    mutation,
    error_type,
    message,
):
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    mutation(model)

    with pytest.raises(error_type, match=message):
        backend.decode(2)

    assert backend._epoch == previous_epoch + 1
    assert backend.cache_length == 0
    assert backend._active_token_ids == []
    assert backend._active_cache_layout is None
    assert backend._cache_checkpoints == {}
    assert backend._next_checkpoint_id == 1
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(checkpoint)


def test_failed_valid_replacement_prefill_invalidates_old_handles_without_double_epoch_advance():
    model = FakeModel()
    backend = make_backend(model=model)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    model.output_shape = (1, 2, 8)

    with pytest.raises(TorchBackendInvariantError, match="logits shape"):
        backend.prefill([2])

    assert backend._epoch == previous_epoch + 1
    assert backend.cache_length == 0
    assert backend._cache_checkpoints == {}
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(checkpoint)


def test_output_cache_replacement_is_an_invariant_failure_with_terminal_cleanup():
    class CacheReplacingModel(FakeModel):
        replace_cache = False

        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            if self.replace_cache:
                output.past_key_values = FakeCache(config=self.config)
            return output

    model = CacheReplacingModel()
    backend = make_backend(model=model)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    previous_epoch = backend._epoch
    model.replace_cache = True

    with pytest.raises(TorchBackendInvariantError, match="replaced"):
        backend.decode(2)

    assert backend._epoch == previous_epoch + 1
    assert backend.cache_length == 0
    assert backend._active_token_ids == []
    assert backend._cache_checkpoints == {}
    with pytest.raises(CacheCheckpointStateError, match="stale sequence"):
        backend.rollback_cache(checkpoint)


def test_checkpoint_creation_cache_invariant_failure_is_atomic():
    backend = make_backend()
    backend.prefill([1, 2])
    backend._cache.layers[3].keys.shape = (1, 2, 2, 63)
    before = backend_checkpoint_state(backend)

    with pytest.raises(TorchBackendInvariantError, match="DynamicCache invariant"):
        backend.create_cache_checkpoint()

    assert backend_checkpoint_state(backend) == before


def test_partial_native_crop_failure_maps_to_execution_error_and_restores_everything():
    class PartiallyFailingCache(FakeCache):
        def crop(self, length):
            for layer_index, layer in enumerate(self.layers):
                layer.crop(length)
                if layer_index == 5:
                    raise RuntimeError("injected native crop failure")

    backend = make_backend(transformers_module=FakeTransformers(PartiallyFailingCache))
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    backend.decode(2)
    before = backend_checkpoint_state(backend)

    with pytest.raises(TorchBackendExecutionError, match="native crop failure") as raised:
        backend.rollback_cache(checkpoint)

    assert isinstance(raised.value.__cause__, Exception)
    assert backend_checkpoint_state(backend) == before
    assert_cache_prefix(backend, (1, 2))


def test_native_crop_post_validation_failure_maps_to_invariant_and_restores_everything():
    class InvalidPostCropCache(FakeCache):
        def crop(self, length):
            super().crop(length)
            self.layers[8].keys.shape = (1, 2, length, 63)

    backend = make_backend(transformers_module=FakeTransformers(InvalidPostCropCache))
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()
    backend.decode(2)
    before = backend_checkpoint_state(backend)

    with pytest.raises(TorchBackendInvariantError, match="original cache restored"):
        backend.rollback_cache(checkpoint)

    assert backend_checkpoint_state(backend) == before
    assert_cache_prefix(backend, (1, 2))


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (FakeModel(output_shape=(1, 2, 8)), "logits shape"),
        (FakeModel(output_device="cpu"), "logits are on"),
        (FakeModel(cache_increment=0), "cache length"),
    ],
)
def test_output_or_cache_invariant_failure_resets_state(model, message):
    backend = make_backend(model=model)

    with pytest.raises(TorchBackendInvariantError, match=message):
        backend.prefill([1, 2])

    assert backend.cache_length == 0


def test_cache_creation_failure_is_explicit_and_leaves_empty_state():
    caches = FailingDynamicCache(error=RuntimeError("cache failed"))
    backend = make_backend(transformers_module=FakeTransformers(caches))

    with pytest.raises(TorchBackendExecutionError, match="cache failed"):
        backend.prefill([1])

    assert backend.cache_length == 0


@pytest.mark.parametrize(
    ("model", "tokenizer_size", "message"),
    [
        (FakeModel(model_vocab_size=5), 6, "smaller than tokenizer"),
        (FakeModel(model_vocab_size=8, embedding_rows=7), 6, "embedding vocabulary"),
    ],
)
def test_constructor_rejects_incompatible_vocabularies(model, tokenizer_size, message):
    with pytest.raises(TorchBackendLoadError, match=message):
        make_backend(model=model, tokenizer_size=tokenizer_size)


def test_constructor_rejects_model_not_loaded_in_4bit():
    model = FakeModel()
    model.is_loaded_in_4bit = False

    with pytest.raises(TorchBackendLoadError, match="4-bit"):
        make_backend(model=model)


@pytest.mark.parametrize("memory_footprint", [0, -1, True, "4096"])
def test_constructor_rejects_invalid_model_memory_footprint(memory_footprint):
    with pytest.raises(TorchBackendLoadError, match="model memory footprint"):
        make_backend(model=FakeModel(memory_footprint=memory_footprint))


@pytest.mark.parametrize(
    ("profile", "device_index", "error_type"),
    [
        ("model", 0, TypeError),
        (DEFAULT_TARGET_PROFILE, -1, ValueError),
        (DEFAULT_TARGET_PROFILE, True, TypeError),
    ],
)
def test_constructor_rejects_invalid_profile_or_device(profile, device_index, error_type):
    with pytest.raises(error_type):
        TorchCUDATargetBackend(
            torch_module=FakeTorch(),
            transformers_module=FakeTransformers(),
            model=FakeModel(),
            tokenizer=SimpleNamespace(vocab_size=6, tokenizer_id="test"),
            profile=profile,
            device_index=device_index,
        )


def test_close_is_idempotent_and_blocks_further_operations():
    torch_module = FakeTorch()
    backend = make_backend(torch_module=torch_module)
    backend.prefill([1])
    checkpoint = backend.create_cache_checkpoint()

    backend.close()
    backend.close()

    assert backend.is_closed
    assert backend.cache_length == 0
    assert backend._active_token_ids == []
    assert backend._active_cache_layout is None
    assert backend._cache_checkpoints == {}
    assert torch_module.cuda.empty_cache_calls == 1
    assert torch_module.cuda.synchronize_calls == 2
    with pytest.raises(BackendStateError, match="closed"):
        backend.prefill([1])
    with pytest.raises(BackendStateError, match="closed"):
        backend.decode(1)
    with pytest.raises(BackendStateError, match="closed"):
        backend.verify_proposal(1, (2,))
    with pytest.raises(BackendStateError, match="closed"):
        backend.reset()
    with pytest.raises(BackendStateError, match="closed"):
        backend.create_cache_checkpoint()
    with pytest.raises(BackendStateError, match="closed"):
        backend.rollback_cache(checkpoint)
    with pytest.raises(BackendStateError, match="closed"):
        backend.release_cache_checkpoint(checkpoint)
    with pytest.raises(BackendStateError, match="closed"):
        _ = backend.tokenizer


def test_context_manager_closes_backend():
    backend = make_backend()

    with backend as entered:
        assert entered is backend
        entered.prefill([1])

    assert backend.is_closed


def test_close_reports_cleanup_failure_after_marking_closed():
    torch_module = FakeTorch(cuda=FakeCuda(cleanup_error=RuntimeError("cleanup failed")))
    backend = make_backend(torch_module=torch_module)

    with pytest.raises(TorchBackendExecutionError, match="cleanup failed"):
        backend.close()

    assert backend.is_closed


def test_factory_loads_pinned_components_lazily(monkeypatch):
    torch_module = FakeTorch()
    transformers_module = FakeTransformers()
    bitsandbytes_module = SimpleNamespace(__version__="0.49.2")
    modules = {
        "torch": torch_module,
        "transformers": transformers_module,
        "bitsandbytes": bitsandbytes_module,
    }
    tokenizer_calls = []
    model_calls = []
    model = FakeModel()

    monkeypatch.setattr(
        torch_backend_module.importlib,
        "import_module",
        lambda name: modules[name],
    )

    def load_tokenizer(profile, *, local_files_only):
        tokenizer_calls.append((profile, local_files_only))
        return SimpleNamespace(
            tokenizer=SimpleNamespace(
                vocab_size=6,
                tokenizer_id=profile.pinned_id,
            )
        )

    def load_model(torch_arg, transformers_arg, **kwargs):
        model_calls.append((torch_arg, transformers_arg, kwargs))
        return model

    monkeypatch.setattr(torch_backend_module, "load_qwen_tokenizer", load_tokenizer)
    monkeypatch.setattr(torch_backend_module, "_load_nf4_model", load_model)

    backend = load_torch_cuda_target(local_files_only=True)

    assert backend.model_vocab_size == 8
    assert tokenizer_calls == [(DEFAULT_TARGET_PROFILE, True)]
    assert model_calls == [
        (
            torch_module,
            transformers_module,
            {
                "profile": DEFAULT_TARGET_PROFILE,
                "device_index": 0,
                "local_files_only": True,
            },
        )
    ]
    assert model.eval_calls == 1
    backend.close()


@pytest.mark.parametrize("missing_name", ["torch", "transformers", "bitsandbytes"])
def test_factory_reports_each_missing_dependency(monkeypatch, missing_name):
    def import_module(name):
        if name == missing_name:
            raise ModuleNotFoundError(f"{name} missing")
        return SimpleNamespace()

    monkeypatch.setattr(torch_backend_module.importlib, "import_module", import_module)

    with pytest.raises(TorchBackendImportError, match=f"{missing_name} missing"):
        load_torch_cuda_target()


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("device_index", -1, ValueError),
        ("device_index", True, TypeError),
        ("device_index", 0.5, TypeError),
        ("local_files_only", "yes", TypeError),
        ("profile", "model", TypeError),
    ],
)
def test_factory_rejects_invalid_input_before_import(monkeypatch, field, value, error_type):
    def unexpected_import(name):
        raise AssertionError("invalid input must fail before importing dependencies")

    monkeypatch.setattr(torch_backend_module.importlib, "import_module", unexpected_import)

    with pytest.raises(error_type):
        load_torch_cuda_target(**{field: value})


def test_factory_rejects_unavailable_cuda_before_loading(monkeypatch):
    torch_module = FakeTorch(cuda=FakeCuda(available=False))
    modules = {
        "torch": torch_module,
        "transformers": FakeTransformers(),
        "bitsandbytes": SimpleNamespace(),
    }
    monkeypatch.setattr(
        torch_backend_module.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        torch_backend_module,
        "load_qwen_tokenizer",
        lambda *args, **kwargs: pytest.fail("tokenizer must not load"),
    )

    with pytest.raises(TorchBackendLoadError, match="CUDA unavailable"):
        load_torch_cuda_target()


def test_factory_validation_failure_releases_loaded_components(monkeypatch):
    torch_module = FakeTorch()
    transformers_module = FakeTransformers()
    modules = {
        "torch": torch_module,
        "transformers": transformers_module,
        "bitsandbytes": SimpleNamespace(),
    }
    monkeypatch.setattr(
        torch_backend_module.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        torch_backend_module,
        "load_qwen_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(
            tokenizer=SimpleNamespace(vocab_size=6, tokenizer_id="test")
        ),
    )
    monkeypatch.setattr(
        torch_backend_module,
        "_load_nf4_model",
        lambda *args, **kwargs: FakeModel(model_vocab_size=5),
    )

    with pytest.raises(TorchBackendLoadError, match="smaller than tokenizer"):
        load_torch_cuda_target()

    assert torch_module.cuda.empty_cache_calls == 1
    assert torch_module.cuda.synchronize_calls == 2


def test_factory_load_and_cleanup_failures_are_both_reported(monkeypatch):
    torch_module = FakeTorch(cuda=FakeCuda(cleanup_error=RuntimeError("cleanup failed")))
    modules = {
        "torch": torch_module,
        "transformers": FakeTransformers(),
        "bitsandbytes": SimpleNamespace(),
    }
    monkeypatch.setattr(
        torch_backend_module.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        torch_backend_module,
        "load_qwen_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(
            tokenizer=SimpleNamespace(vocab_size=6, tokenizer_id="test")
        ),
    )
    def fail_model(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(torch_backend_module, "_load_nf4_model", fail_model)

    with pytest.raises(
        TorchBackendLoadError,
        match="load failed.*failed-load cleanup also failed.*cleanup failed",
    ):
        load_torch_cuda_target()
