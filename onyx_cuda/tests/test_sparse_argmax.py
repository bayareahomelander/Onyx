"""Differential GPU checks for the optional fused greedy selector."""

import sys

import pytest
import torch

from onyx_cuda.masking import apply_grammar_mask, grammar_argmax

pytestmark = pytest.mark.gpu


@pytest.fixture
def kernel_available():
    pytest.importorskip("cupy", reason="install onyx-cuda[kernels] to test the CUDA kernel")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("count", [1, 31, 256, 257, 4096, 152064])
def test_sparse_matches_dense_random_and_ties(kernel_available, dtype, count):
    generator = torch.Generator(device="cuda").manual_seed(17)
    logits = torch.randn((3, 152064), dtype=dtype, device="cuda", generator=generator)
    ids = torch.randperm(logits.shape[-1], device="cuda", generator=generator)[:count].tolist()
    # Duplicate, unsorted IDs and exact ties must select the smallest token ID.
    ids += ids[:2]
    logits[1, ids] = 42
    original = logits.clone()
    expected = apply_grammar_mask(logits, ids).argmax(dim=-1)
    actual = grammar_argmax(logits, ids, backend="cuda")
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.long and actual.device == logits.device
    assert torch.equal(logits, original)


@pytest.mark.parametrize("values,ids", [
    ([0., float("nan"), 1., float("nan")], [3, 1]),
    ([float("nan"), 2., 1., 2.], [3, 1]),
    ([0., float("inf"), 1., float("inf")], [3, 1]),
    ([0., -float("inf"), 0., -float("inf")], [3, 1]),
    ([0., -0., 0., 0.], [3, 1]),
])
def test_sparse_preserves_dense_nonfinite_semantics(kernel_available, values, ids):
    logits = torch.tensor(values, device="cuda")
    assert torch.equal(
        grammar_argmax(logits, ids, backend="cuda"),
        apply_grammar_mask(logits, ids).argmax(dim=-1),
    )


def test_sparse_strided_views_and_nondefault_stream(kernel_available):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for logits in (
            torch.randn((17, 6), device="cuda").T[1::2, 1::2],
            torch.randn(34, device="cuda")[1::2],
            torch.randn((1, 17), device="cuda").expand(3, -1),
            torch.empty((0, 17), device="cuda"),
        ):
            ids = [5, 2, 1]
            expected = apply_grammar_mask(logits, ids).argmax(dim=-1)
            actual = grammar_argmax(logits, ids, backend="cuda")
            assert torch.equal(actual, expected)
    stream.synchronize()


@pytest.mark.parametrize("dtype,shape", [(torch.float64, (2, 8)), (torch.bfloat16, (8,)), (torch.float32, (2, 3, 8))])
def test_unhandled_types_and_ranks_preserve_dense_precision(dtype, shape):
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    assert torch.equal(
        grammar_argmax(logits, [7, 2], backend="cuda"),
        apply_grammar_mask(logits, [7, 2]).argmax(dim=-1),
    )


@pytest.mark.parametrize("ids", [[], [-1], [8], [True], [1.5]])
def test_sparse_rejects_invalid_ids_before_launch(ids):
    with pytest.raises(ValueError):
        grammar_argmax(torch.zeros(8, device="cuda"), ids, backend="cuda")


def test_backend_configuration(monkeypatch):
    logits = torch.tensor([1., 2.], device="cuda")
    monkeypatch.setenv("ONYX_GREEDY_BACKEND", "invalid")
    with pytest.raises(ValueError, match="ONYX_GREEDY_BACKEND"):
        grammar_argmax(logits, [0])
    assert grammar_argmax(logits, [0], backend="torch").item() == 0


def test_explicit_cuda_backend_reports_missing_dependency(monkeypatch):
    from onyx_cuda._sparse_argmax import _kernel

    _kernel.cache_clear()
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match=r"onyx-cuda\[kernels\]"):
        grammar_argmax(torch.zeros(8, device="cuda"), [1], backend="cuda")
