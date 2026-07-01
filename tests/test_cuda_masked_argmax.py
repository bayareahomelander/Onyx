import pytest


def test_onyx_cuda_imports_without_loading_extension():
    import onyx_cuda

    available, error = onyx_cuda.extension_status()
    assert available in (True, False)
    assert error is None or isinstance(error, str)


def _cuda_module_or_skip():
    import importlib

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    module = importlib.import_module("onyx_cuda.masked_argmax")
    module._load_extension()

    return torch, module


def test_masked_argmax_matches_reference_float32():
    torch, module = _cuda_module_or_skip()
    torch.manual_seed(7)

    vocab_size = 4096
    logits = torch.randn(vocab_size, device="cuda", dtype=torch.float32)
    valid_ids = torch.randperm(vocab_size, device="cuda", dtype=torch.long)[:257]

    actual = module.masked_argmax_tensor(logits, valid_ids)
    expected = module.torch_reference_masked_argmax(logits, valid_ids)

    assert int(actual.item()) == int(expected.item())


def test_masked_argmax_matches_reference_float16():
    torch, module = _cuda_module_or_skip()
    torch.manual_seed(11)

    vocab_size = 2048
    logits = torch.randn(vocab_size, device="cuda", dtype=torch.float16)
    valid_ids = torch.randperm(vocab_size, device="cuda", dtype=torch.long)[:113]

    actual = module.masked_argmax_tensor(logits, valid_ids)
    expected = module.torch_reference_masked_argmax(logits, valid_ids)

    assert int(actual.item()) == int(expected.item())


def test_masked_argmax_diagnostics_preserve_selection_result():
    torch, module = _cuda_module_or_skip()

    logits = torch.tensor([0.0, 4.0, 2.0, 9.0], device="cuda", dtype=torch.float32)
    valid_ids = torch.tensor([1, 2], device="cuda", dtype=torch.long)

    selected, diagnostics = module.masked_argmax_tensor_with_diagnostics(logits, valid_ids)

    assert int(selected.item()) == 1
    assert diagnostics.input_preparation_s >= 0
    assert diagnostics.input_validation_s >= 0
    assert diagnostics.extension_load_s >= 0
    assert diagnostics.selector_launch_s >= 0
    assert diagnostics.total_s == pytest.approx(
        diagnostics.input_preparation_s
        + diagnostics.input_validation_s
        + diagnostics.extension_load_s
        + diagnostics.selector_launch_s
    )


def test_masked_argmax_tie_chooses_smallest_token_id():
    torch, module = _cuda_module_or_skip()

    logits = torch.zeros(16, device="cuda", dtype=torch.float32)
    logits[3] = 9.0
    logits[7] = 9.0
    logits[9] = 9.0
    valid_ids = torch.tensor([9, 3, 7], device="cuda", dtype=torch.long)

    assert module.masked_argmax(logits, valid_ids) == 3


def test_masked_argmax_supports_batched_logits():
    torch, module = _cuda_module_or_skip()

    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0, 4.0, 3.0],
            [0.0, 6.0, 1.0, 4.0, 6.0],
            [5.0, 2.0, 8.0, 4.0, 3.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    valid_ids = torch.tensor([1, 3, 4], device="cuda", dtype=torch.long)

    actual = module.masked_argmax_tensor(logits, valid_ids)
    expected = module.torch_reference_masked_argmax(logits, valid_ids)

    assert actual.cpu().tolist() == expected.cpu().tolist()


def test_masked_argmax_rejects_invalid_valid_ids():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available")

    from onyx_cuda.masked_argmax import masked_argmax_tensor

    logits = torch.zeros(8, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="valid_token_ids cannot be empty"):
        masked_argmax_tensor(logits, [], check_inputs=True)

    with pytest.raises(ValueError, match="valid_token_ids must be in"):
        masked_argmax_tensor(logits, [0, 8], check_inputs=True)

    from onyx_cuda.masked_argmax import masked_argmax_tensor_with_diagnostics

    with pytest.raises(ValueError, match="valid_token_ids must be in"):
        masked_argmax_tensor_with_diagnostics(logits, [0, 8], check_inputs=True)
