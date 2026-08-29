import pytest
import torch

from onyx_cuda.masking import apply_grammar_mask


@pytest.mark.parametrize(
    ("valid_token_ids", "expected"),
    [
        ([2], [-torch.inf, -torch.inf, 3.0, -torch.inf]),
        ([0, 3], [1.0, -torch.inf, -torch.inf, 4.0]),
    ],
)
def test_apply_grammar_mask_preserves_only_valid_cuda_logits(
    valid_token_ids, expected
):
    logits = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16, device="cuda:0"
    )
    original = logits.clone()

    masked = apply_grammar_mask(logits, valid_token_ids)

    assert torch.equal(
        masked,
        torch.tensor(expected, dtype=logits.dtype, device="cuda:0")[None],
    )
    assert masked.dtype == logits.dtype
    assert masked.device == logits.device
    assert torch.equal(logits, original)


def test_apply_grammar_mask_rejects_empty_valid_ids_without_mutation():
    logits = torch.arange(4, dtype=torch.float32, device="cuda:0")
    original = logits.clone()

    with pytest.raises(ValueError, match="cannot be empty"):
        apply_grammar_mask(logits, [])

    assert torch.equal(logits, original)


@pytest.mark.parametrize("valid_token_ids", [[-1], [4]])
def test_apply_grammar_mask_rejects_out_of_range_ids_without_mutation(
    valid_token_ids,
):
    logits = torch.arange(4, dtype=torch.float32, device="cuda:0")
    original = logits.clone()

    with pytest.raises(ValueError, match=r"integers in \[0, 4\)"):
        apply_grammar_mask(logits, valid_token_ids)

    assert torch.equal(logits, original)
