import gc
import importlib
import math
import sys
import weakref

import pytest

import onyx_cuda.torch_grammar_mask as grammar_mask_module
from onyx_cuda import (
    GrammarTimingMetrics,
    TemperatureTopPSelection,
    TorchGrammarMaskExecutionError,
    TorchGrammarMaskImportError,
    TorchGrammarMaskInvariantError,
    TorchGrammarMaskUnavailableError,
    create_cuda_grammar_logit_mask,
    create_cuda_sampler,
    create_grammar_timing_session,
    select_cuda_argmax,
)
from onyx_cuda.torch_backend import TorchBackendInvariantError
from onyx_cuda.torch_grammar_mask import _create_cuda_grammar_logit_mask


class FakeDevice:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeCuda:
    def __init__(self, *, available=True, device_count=1):
        self.available = available
        self.reported_device_count = device_count
        self.availability_error = None
        self.device_count_error = None
        self.synchronize_calls = []
        self.synchronize_errors = {}

    def is_available(self):
        if self.availability_error is not None:
            raise self.availability_error
        return self.available

    def device_count(self):
        if self.device_count_error is not None:
            raise self.device_count_error
        return self.reported_device_count

    def synchronize(self, device):
        call_number = len(self.synchronize_calls) + 1
        self.synchronize_calls.append(str(device))
        error = self.synchronize_errors.get(call_number)
        if error is not None:
            raise error


class ScriptedClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakePredicate:
    def __init__(self, values, *, scalar_override=None):
        self.values = values
        self.scalar_override = scalar_override

    def all(self):
        value = all(self.values) if self.scalar_override is None else self.scalar_override
        return FakeScalar(value)


class FakeTensor:
    _next_pointer = 1

    def __init__(
        self,
        values,
        *,
        device,
        dtype,
        floating,
        shape=None,
        to_error=None,
        index_copy_error=None,
    ):
        self.values = list(values)
        self.device = device
        self.dtype = dtype
        self._floating = floating
        self.shape = (len(self.values),) if shape is None else shape
        self.is_cuda = str(device).startswith("cuda")
        self.to_error = to_error
        self.index_copy_error = index_copy_error
        self.pointer = FakeTensor._next_pointer
        FakeTensor._next_pointer += 1

    def is_floating_point(self):
        return self._floating

    def to(self, *, device):
        if self.to_error is not None:
            raise self.to_error
        return FakeTensor(
            self.values,
            device=device,
            dtype=self.dtype,
            floating=self._floating,
        )

    def index_select(self, dimension, indices):
        assert dimension == 0
        return FakeTensor(
            (self.values[index] for index in indices.values),
            device=self.device,
            dtype=self.dtype,
            floating=self._floating,
        )

    def index_copy_(self, dimension, indices, source):
        if self.index_copy_error is not None:
            raise self.index_copy_error
        assert dimension == 0
        for index, value in zip(indices.values, source.values, strict=True):
            self.values[index] = value
        return self


class FakeTorch:
    int64 = "int64"

    def __init__(self, cuda=None):
        self.cuda = FakeCuda() if cuda is None else cuda
        self.created_devices = []
        self.tensor_calls = []
        self.device_error = None
        self.tensor_error = None
        self.host_to_error = None
        self.full_like_error = None
        self.full_like_calls = 0
        self.result_shape = None
        self.result_device = None
        self.result_dtype = None
        self.result_index_copy_error = None
        self.return_input_from_full_like = False
        self.support_scalar_override = None

    def device(self, name):
        if self.device_error is not None:
            raise self.device_error
        self.created_devices.append(name)
        return FakeDevice(name)

    def tensor(self, values, *, dtype, device):
        if self.tensor_error is not None:
            raise self.tensor_error
        self.tensor_calls.append((tuple(values), dtype, device))
        return FakeTensor(
            values,
            device=FakeDevice(device),
            dtype=dtype,
            floating=False,
            to_error=self.host_to_error,
        )

    def full_like(self, logits, fill_value):
        self.full_like_calls += 1
        if self.full_like_error is not None:
            raise self.full_like_error
        if self.return_input_from_full_like:
            return logits
        return FakeTensor(
            (fill_value,) * len(logits.values),
            device=logits.device if self.result_device is None else self.result_device,
            dtype=logits.dtype if self.result_dtype is None else self.result_dtype,
            floating=True,
            shape=self.result_shape,
            index_copy_error=self.result_index_copy_error,
        )

    def isneginf(self, tensor):
        values = (math.isinf(value) and value < 0 for value in tensor.values)
        return FakePredicate(values, scalar_override=self.support_scalar_override)


def fake_logits(values=(3.0, 8.0, -2.0), *, device="cuda:0"):
    return FakeTensor(
        values,
        device=FakeDevice(device),
        dtype="float32",
        floating=True,
    )


def test_public_factory_rejects_invalid_inputs_before_import(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(grammar_mask_module.importlib, "import_module", unexpected_import)

    for value in (True, 1.5, "3"):
        with pytest.raises(TypeError, match="vocab_size"):
            create_cuda_grammar_logit_mask(value)
    for value in (0, -1):
        with pytest.raises(ValueError, match="greater than zero"):
            create_cuda_grammar_logit_mask(value)
    for value in (True, 1.5, "0"):
        with pytest.raises(TypeError, match="device_index"):
            create_cuda_grammar_logit_mask(3, device_index=value)
    with pytest.raises(ValueError, match="cannot be negative"):
        create_cuda_grammar_logit_mask(3, device_index=-1)
    with pytest.raises(TypeError, match="clock must be callable"):
        create_cuda_grammar_logit_mask(3, clock=None)


@pytest.mark.parametrize("error", [ImportError("missing"), OSError("broken loader")])
def test_public_factory_maps_pytorch_import_failures(monkeypatch, error):
    monkeypatch.setattr(
        grammar_mask_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(error),
    )

    with pytest.raises(TorchGrammarMaskImportError, match="could not be imported") as raised:
        create_cuda_grammar_logit_mask(3)

    assert raised.value.__cause__ is error


def test_injected_factory_binds_exact_device_and_properties():
    fake_torch = FakeTorch()

    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)

    assert mask.vocab_size == 3
    assert mask.device_index == 0
    assert mask.transport_name == "sparse_valid_indices"
    assert fake_torch.created_devices == ["cuda:0"]
    assert set(mask.__dict__) == {
        "_torch",
        "_vocab_size",
        "_device_index",
        "_device",
        "_transport_name",
        "_clock",
    }


def test_injected_factory_reports_unavailable_cuda_and_device():
    with pytest.raises(TorchGrammarMaskUnavailableError, match="CUDA unavailable"):
        _create_cuda_grammar_logit_mask(FakeTorch(FakeCuda(available=False)), 3, device_index=0)

    with pytest.raises(TorchGrammarMaskUnavailableError, match="detected 1 device"):
        _create_cuda_grammar_logit_mask(FakeTorch(FakeCuda(device_count=1)), 3, device_index=1)


def test_injected_factory_wraps_availability_and_device_creation_failures():
    cuda = FakeCuda()
    availability_error = RuntimeError("availability failed")
    cuda.availability_error = availability_error
    with pytest.raises(TorchGrammarMaskExecutionError, match="availability failed") as raised:
        _create_cuda_grammar_logit_mask(FakeTorch(cuda), 3, device_index=0)
    assert raised.value.__cause__ is availability_error

    fake_torch = FakeTorch()
    device_error = RuntimeError("device failed")
    fake_torch.device_error = device_error
    with pytest.raises(TorchGrammarMaskExecutionError, match="device failed") as raised:
        _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)
    assert raised.value.__cause__ is device_error


@pytest.mark.parametrize(
    ("available", "device_count", "message"),
    [(1, 1, "availability.*boolean"), (True, True, "device count.*integer"), (True, -1, "negative")],
)
def test_injected_factory_rejects_malformed_cuda_metadata(available, device_count, message):
    with pytest.raises(TorchGrammarMaskInvariantError, match=message):
        _create_cuda_grammar_logit_mask(
            FakeTorch(FakeCuda(available=available, device_count=device_count)),
            3,
            device_index=0,
        )


def test_injected_mask_applies_sparse_indices_without_mutating_or_retaining_calls():
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)
    logits = fake_logits()

    result = mask.apply(logits, (1, 2))

    assert result.values == [float("-inf"), 8.0, -2.0]
    assert logits.values == [3.0, 8.0, -2.0]
    assert result is not logits
    assert fake_torch.tensor_calls == [((1, 2), "int64", "cpu")]
    assert all(value is not result for value in mask.__dict__.values())


def test_timed_mask_matches_untimed_output_and_records_completed_stage_totals():
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(
        fake_torch,
        3,
        device_index=0,
        clock=ScriptedClock((10.0, 12.0, 20.0, 25.0)),
    )
    logits = fake_logits()
    untimed = mask.apply(logits, (1, 2))
    assert fake_torch.cuda.synchronize_calls == []

    timing_session = create_grammar_timing_session(clock=ScriptedClock((0.0, 1.0)))
    with timing_session.state_scan():
        pass
    timed = mask.apply_with_timing(logits, (1, 2), timing_session)
    timing = timing_session.finish(1)

    assert timed.values == untimed.values
    assert timed.shape == untimed.shape
    assert str(timed.device) == str(untimed.device)
    assert timed.dtype == untimed.dtype
    assert logits.values == [3.0, 8.0, -2.0]
    assert fake_torch.cuda.synchronize_calls == ["cuda:0", "cuda:0", "cuda:0"]
    assert timing == GrammarTimingMetrics(
        compilation_time=0.0,
        state_scan_time=1.0,
        valid_index_transfer_time=2.0,
        mask_application_time=5.0,
    )
    assert all(value is not timing_session for value in mask.__dict__.values())
    assert all(value is not timed for value in mask.__dict__.values())


def test_repeated_timed_calls_record_one_atomic_pair_each_without_retention():
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(
        fake_torch,
        3,
        device_index=0,
        clock=ScriptedClock((10.0, 11.0, 20.0, 22.0, 30.0, 33.0, 40.0, 44.0)),
    )
    timing_session = create_grammar_timing_session(
        clock=ScriptedClock((0.0, 1.0, 2.0, 4.0))
    )

    for valid_ids in ((0,), (2,)):
        with timing_session.state_scan():
            pass
        mask.apply_with_timing(fake_logits(), valid_ids, timing_session)

    assert timing_session.finish(2) == GrammarTimingMetrics(0.0, 3.0, 4.0, 6.0)
    assert fake_torch.cuda.synchronize_calls == ["cuda:0"] * 6
    assert set(mask.__dict__) == {
        "_torch",
        "_vocab_size",
        "_device_index",
        "_device",
        "_transport_name",
        "_clock",
    }


@pytest.mark.parametrize("synchronization_call", [1, 2, 3])
def test_timed_mask_maps_each_synchronization_failure_without_recording(
    synchronization_call,
):
    fake_torch = FakeTorch()
    error = RuntimeError(f"sync {synchronization_call} failed")
    fake_torch.cuda.synchronize_errors[synchronization_call] = error
    mask = _create_cuda_grammar_logit_mask(
        fake_torch,
        3,
        device_index=0,
        clock=ScriptedClock((0.0, 1.0, 2.0, 3.0)),
    )
    timing_session = create_grammar_timing_session()

    with pytest.raises(
        TorchGrammarMaskExecutionError,
        match=f"sync {synchronization_call} failed",
    ) as raised:
        mask.apply_with_timing(fake_logits(), (1,), timing_session)

    assert raised.value.__cause__ is error
    assert len(fake_torch.cuda.synchronize_calls) == synchronization_call
    timing_session.abort()


@pytest.mark.parametrize(
    ("clock", "error_type", "message"),
    [
        (
            lambda: (_ for _ in ()).throw(RuntimeError("clock failed")),
            TorchGrammarMaskExecutionError,
            "clock failed",
        ),
        (ScriptedClock((True,)), TorchGrammarMaskInvariantError, "real number"),
        (
            ScriptedClock((float("nan"),)),
            TorchGrammarMaskInvariantError,
            "finite",
        ),
        (
            ScriptedClock((2.0, 1.0, 3.0, 4.0)),
            TorchGrammarMaskInvariantError,
            "backwards during valid-index transfer",
        ),
        (
            ScriptedClock((0.0, 1.0, 4.0, 3.0)),
            TorchGrammarMaskInvariantError,
            "backwards during mask application",
        ),
    ],
)
def test_timed_mask_rejects_failing_invalid_and_backwards_clocks(
    clock,
    error_type,
    message,
):
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(
        fake_torch,
        3,
        device_index=0,
        clock=clock,
    )
    timing_session = create_grammar_timing_session()

    with pytest.raises(error_type, match=message):
        mask.apply_with_timing(fake_logits(), (1,), timing_session)
    timing_session.abort()


def test_backwards_transfer_clock_fails_before_mask_application_starts():
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(
        fake_torch,
        3,
        device_index=0,
        clock=ScriptedClock((2.0, 1.0)),
    )
    timing_session = create_grammar_timing_session()

    with pytest.raises(TorchGrammarMaskInvariantError, match="backwards.*transfer"):
        mask.apply_with_timing(fake_logits(), (1,), timing_session)

    assert fake_torch.cuda.synchronize_calls == ["cuda:0", "cuda:0"]
    assert fake_torch.full_like_calls == 0
    timing_session.abort()


def test_timed_mask_rejects_invalid_session_before_synchronization_or_allocation():
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)

    with pytest.raises(TypeError, match="GrammarTimingSession"):
        mask.apply_with_timing(fake_logits(), (1,), object())

    assert fake_torch.cuda.synchronize_calls == []
    assert fake_torch.tensor_calls == []


def test_timed_mask_operation_failure_does_not_record_a_partial_pair():
    fake_torch = FakeTorch()
    fake_torch.full_like_error = RuntimeError("full_like failed")
    mask = _create_cuda_grammar_logit_mask(
        fake_torch,
        3,
        device_index=0,
        clock=ScriptedClock((0.0, 1.0, 2.0)),
    )
    timing_session = create_grammar_timing_session()

    with pytest.raises(TorchGrammarMaskExecutionError, match="full_like failed"):
        mask.apply_with_timing(fake_logits(), (1,), timing_session)

    assert fake_torch.cuda.synchronize_calls == ["cuda:0", "cuda:0"]
    timing_session.abort()


@pytest.mark.parametrize(
    "valid_token_ids",
    [
        [1],
        {1},
        iter((1,)),
        (),
        (True,),
        (1.0,),
        (-1,),
        (3,),
        (1, 1),
        (2, 1),
    ],
)
def test_injected_mask_rejects_invalid_ids_before_tensor_allocation(valid_token_ids):
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)
    logits = fake_logits()

    with pytest.raises(TorchGrammarMaskInvariantError):
        mask.apply(logits, valid_token_ids)

    assert fake_torch.tensor_calls == []
    assert logits.values == [3.0, 8.0, -2.0]


def test_injected_mask_rejects_bad_shape_device_and_dtype_before_allocation():
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)
    bad_inputs = (
        FakeTensor((1.0, 2.0), device=FakeDevice("cuda:0"), dtype="float32", floating=True),
        fake_logits(device="cpu"),
        fake_logits(device="cuda:1"),
        FakeTensor((1, 2, 3), device=FakeDevice("cuda:0"), dtype="int64", floating=False),
    )

    for logits in bad_inputs:
        with pytest.raises(TorchGrammarMaskInvariantError):
            mask.apply(logits, (1,))

    assert fake_torch.tensor_calls == []


def test_injected_mask_rejects_all_negative_infinity_support():
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)

    with pytest.raises(TorchGrammarMaskInvariantError, match="no logit support"):
        mask.apply(fake_logits((float("-inf"),) * 3), (0, 2))


@pytest.mark.parametrize("operation", ["tensor", "transfer", "full_like", "index_copy"])
def test_injected_mask_wraps_operation_failures_with_cause(operation):
    fake_torch = FakeTorch()
    error = RuntimeError(f"{operation} failed")
    if operation == "tensor":
        fake_torch.tensor_error = error
    elif operation == "transfer":
        fake_torch.host_to_error = error
    elif operation == "full_like":
        fake_torch.full_like_error = error
    else:
        fake_torch.result_index_copy_error = error
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)

    with pytest.raises(TorchGrammarMaskExecutionError, match=f"{operation} failed") as raised:
        mask.apply(fake_logits(), (1,))

    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("result_shape", (1, 3), "changed logits shape"),
        ("result_device", FakeDevice("cuda:1"), "changed the logits device"),
        ("result_dtype", "float16", "changed logits dtype"),
    ],
)
def test_injected_mask_rejects_malformed_result_metadata(attribute, value, message):
    fake_torch = FakeTorch()
    setattr(fake_torch, attribute, value)
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)

    with pytest.raises(TorchGrammarMaskInvariantError, match=message):
        mask.apply(fake_logits(), (1,))


def test_injected_mask_rejects_malformed_support_scalar():
    fake_torch = FakeTorch()
    fake_torch.support_scalar_override = 0
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)

    with pytest.raises(TorchGrammarMaskInvariantError, match="must return a boolean"):
        mask.apply(fake_logits(), (1,))


def test_injected_mask_rejects_an_alias_before_writing_to_the_input():
    fake_torch = FakeTorch()
    fake_torch.return_input_from_full_like = True
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)
    logits = fake_logits()

    with pytest.raises(TorchGrammarMaskInvariantError, match="must not alias"):
        mask.apply(logits, (1,))

    assert logits.values == [3.0, 8.0, -2.0]


def test_injected_mask_maps_unreadable_tensor_metadata_to_invariant_errors():
    class BadShape:
        @property
        def shape(self):
            raise RuntimeError("shape failed")

    class BadDtype:
        shape = (3,)
        is_cuda = True
        device = FakeDevice("cuda:0")

        def is_floating_point(self):
            raise RuntimeError("dtype failed")

    class BadDeviceText:
        def __str__(self):
            raise RuntimeError("device text failed")

    bad_device = fake_logits()
    bad_device.device = BadDeviceText()
    fake_torch = FakeTorch()
    mask = _create_cuda_grammar_logit_mask(fake_torch, 3, device_index=0)

    for logits, message in (
        (BadShape(), "shape failed"),
        (BadDtype(), "dtype failed"),
        (bad_device, "device text failed"),
    ):
        with pytest.raises(TorchGrammarMaskInvariantError, match=message) as raised:
            mask.apply(logits, (1,))
        assert isinstance(raised.value.__cause__, RuntimeError)


try:
    torch = importlib.import_module("torch")
except (ImportError, OSError):
    torch = None
CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()


def _bitwise_equal(first, second):
    return torch.equal(first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_cuda_mask_bitwise_preserves_allowed_edge_values_and_masks_everything_else(dtype):
    mask = create_cuda_grammar_logit_mask(8)
    logits = torch.tensor(
        (-0.0, 0.0, 5.0, float("inf"), float("nan"), -7.0, float("-inf"), 99.0),
        dtype=getattr(torch, dtype),
        device="cuda:0",
    )
    original = logits.clone()
    valid = (0, 1, 3, 4, 6)

    result = mask.apply(logits, valid)

    indices = torch.tensor(valid, device="cuda:0")
    disallowed = torch.tensor((2, 5, 7), device="cuda:0")
    assert _bitwise_equal(result[indices], original[indices])
    assert torch.isneginf(result[disallowed]).all().item()
    assert _bitwise_equal(logits, original)
    assert result.shape == logits.shape
    assert result.device == logits.device
    assert result.dtype == logits.dtype
    assert result.untyped_storage().data_ptr() != logits.untyped_storage().data_ptr()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_timed_mask_matches_untimed_values_and_reports_completed_positive_stages():
    mask = create_cuda_grammar_logit_mask(4, device_index=0)
    logits = torch.tensor((1.0, 5.0, 3.0, -2.0), device="cuda:0")
    timing_session = create_grammar_timing_session()
    with timing_session.state_scan():
        pass

    untimed = mask.apply(logits, (1, 3))
    timed = mask.apply_with_timing(logits, (1, 3), timing_session)
    timing = timing_session.finish(1)

    assert _bitwise_equal(timed, untimed)
    assert timing.valid_index_transfer_time > 0.0
    assert timing.mask_application_time > 0.0
    assert _bitwise_equal(logits, torch.tensor((1.0, 5.0, 3.0, -2.0), device="cuda:0"))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_cuda_mask_preserves_finite_extremes_with_all_and_all_but_one_support(dtype):
    torch_dtype = getattr(torch, dtype)
    limits = torch.finfo(torch_dtype)
    logits = torch.tensor(
        (limits.min, limits.max, -1.0, 1.0),
        dtype=torch_dtype,
        device="cuda:0",
    )
    original = logits.clone()
    mask = create_cuda_grammar_logit_mask(4)

    complete = mask.apply(logits, (0, 1, 2, 3))
    almost_complete = mask.apply(logits, (0, 1, 3))

    allowed_indices = torch.tensor((0, 1, 3), device="cuda:0")
    assert _bitwise_equal(complete, original)
    assert _bitwise_equal(almost_complete[allowed_indices], original[allowed_indices])
    assert torch.isneginf(almost_complete[2]).item()
    assert _bitwise_equal(logits, original)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_handles_production_vocabulary_ceiling_and_rejects_model_padding():
    mask = create_cuda_grammar_logit_mask(151_665)
    logits = torch.zeros(151_665, dtype=torch.float16, device="cuda:0")
    logits[0] = 100.0

    result = mask.apply(logits, (151_664,))

    assert select_cuda_argmax(result) == 151_664
    with pytest.raises(TorchGrammarMaskInvariantError, match="151665"):
        mask.apply(torch.zeros(151_936, dtype=torch.float16, device="cuda:0"), (151_664,))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (lambda: torch.empty(0, device="cuda:0"), "configured vocabulary shape"),
        (lambda: torch.zeros((1, 4), device="cuda:0"), "configured vocabulary shape"),
        (lambda: torch.zeros(4), "non-CUDA"),
        (lambda: torch.zeros(4, dtype=torch.int64, device="cuda:0"), "floating-point"),
        (lambda: torch.zeros(4, dtype=torch.bool, device="cuda:0"), "floating-point"),
        (lambda: torch.zeros(4, dtype=torch.complex64, device="cuda:0"), "floating-point"),
    ],
)
def test_cuda_mask_rejects_invalid_logits(logits, message):
    mask = create_cuda_grammar_logit_mask(4)

    with pytest.raises(TorchGrammarMaskInvariantError, match=message):
        mask.apply(logits(), (1,))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_rejects_all_negative_infinity_for_partial_and_complete_support():
    mask = create_cuda_grammar_logit_mask(3)
    logits = torch.full((3,), float("-inf"), device="cuda:0")

    for valid in ((1,), (0, 1, 2)):
        with pytest.raises(TorchGrammarMaskInvariantError, match="no logit support"):
            mask.apply(logits, valid)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_confines_greedy_and_seeded_sampling_and_preserves_replay():
    mask = create_cuda_grammar_logit_mask(6)
    logits = torch.tensor((99.0, 7.0, 8.0, 8.0, 8.0, 100.0), device="cuda:0")
    valid = (1, 3, 4)
    masked = mask.apply(logits, valid)

    assert select_cuda_argmax(masked) == 3
    policy = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=23)
    first = create_cuda_sampler(policy)
    second = create_cuda_sampler(policy)
    first_sequence = tuple(first(masked) for _ in range(20))
    second_sequence = tuple(second(masked) for _ in range(20))
    assert first_sequence == second_sequence
    assert set(first_sequence) <= set(valid)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_one_id_support_and_infinity_semantics_confine_selectors():
    mask = create_cuda_grammar_logit_mask(4)
    logits = torch.tensor((float("inf"), 4.0, float("inf"), -3.0), device="cuda:0")
    masked = mask.apply(logits, (2,))

    assert select_cuda_argmax(masked) == 2
    for seed in (0, 1, 99):
        for top_p in (0.1, 0.5, 1.0):
            sampler = create_cuda_sampler(TemperatureTopPSelection(1.0, top_p, seed))
            assert tuple(sampler(masked) for _ in range(5)) == (2,) * 5

    equal_positive_infinity = mask.apply(logits, (0, 2))
    assert select_cuda_argmax(equal_positive_infinity) == 0
    first = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 7))
    second = create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 7))
    first_sequence = tuple(first(equal_positive_infinity) for _ in range(12))
    second_sequence = tuple(second(equal_positive_infinity) for _ in range(12))
    assert first_sequence == second_sequence
    assert set(first_sequence) <= {0, 2}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_removes_disallowed_nan_but_preserves_allowed_nan_for_selector_rejection():
    mask = create_cuda_grammar_logit_mask(3)
    logits = torch.tensor((float("nan"), 2.0, 1.0), device="cuda:0")

    disallowed_nan = mask.apply(logits, (1, 2))
    assert select_cuda_argmax(disallowed_nan) == 1
    allowed_nan = mask.apply(logits, (0, 1))
    assert torch.isnan(allowed_nan[0]).item()
    with pytest.raises(TorchBackendInvariantError, match="NaN"):
        select_cuda_argmax(allowed_nan)
    with pytest.raises(TorchBackendInvariantError, match="NaN"):
        create_cuda_sampler(TemperatureTopPSelection(1.0, 1.0, 0))(allowed_nan)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_does_not_advance_global_rng_or_leak_state_between_calls():
    mask = create_cuda_grammar_logit_mask(5)
    logits = torch.arange(5, dtype=torch.float32, device="cuda:0")
    original_rng = torch.cuda.get_rng_state(0).clone()
    try:
        torch.cuda.manual_seed(723)
        before = torch.cuda.get_rng_state(0).clone()
        first = mask.apply(logits, (0, 2))
        second = mask.apply(logits, (1, 4))
        assert torch.equal(torch.cuda.get_rng_state(0), before)
    finally:
        torch.cuda.set_rng_state(original_rng, 0)

    assert torch.isneginf(first[1]).item()
    assert torch.isneginf(second[2]).item()
    assert select_cuda_argmax(first) == 2
    assert select_cuda_argmax(second) == 4


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_retains_no_input_or_output_tensor_references():
    mask = create_cuda_grammar_logit_mask(3)
    logits = torch.tensor((1.0, 2.0, 3.0), device="cuda:0")
    result = mask.apply(logits, (1,))
    logits_reference = weakref.ref(logits)
    result_reference = weakref.ref(result)

    del logits, result
    gc.collect()

    assert logits_reference() is None
    assert result_reference() is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_stays_on_the_configured_device():
    mask = create_cuda_grammar_logit_mask(3, device_index=0)

    with pytest.raises(TorchGrammarMaskInvariantError, match="expected cuda:0"):
        mask.apply(fake_logits(device="cuda:1"), (1,))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for native grammar-mask tests")
def test_cuda_mask_wraps_tensor_execution_failures_without_mutating_input(monkeypatch):
    mask = create_cuda_grammar_logit_mask(3)
    logits = torch.tensor((1.0, 2.0, 3.0), device="cuda:0")
    original = logits.clone()

    def fail_full_like(*args, **kwargs):
        raise RuntimeError("full_like failed")

    monkeypatch.setattr(torch, "full_like", fail_full_like)
    with pytest.raises(TorchGrammarMaskExecutionError, match="full_like failed") as raised:
        mask.apply(logits, (1,))

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert torch.equal(logits, original)


def test_normal_module_import_does_not_import_torch(monkeypatch):
    requested = []
    original = importlib.import_module

    def record(name, *args, **kwargs):
        requested.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", record)
    sys.modules.pop("onyx_cuda.torch_grammar_mask", None)
    importlib.import_module("onyx_cuda.torch_grammar_mask")

    assert "torch" not in requested
