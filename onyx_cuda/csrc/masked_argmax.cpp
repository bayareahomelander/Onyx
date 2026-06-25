#include <torch/extension.h>

torch::Tensor masked_argmax_cuda(torch::Tensor logits, torch::Tensor valid_ids);

torch::Tensor masked_argmax(torch::Tensor logits, torch::Tensor valid_ids) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(valid_ids.is_cuda(), "valid_ids must be a CUDA tensor");
    TORCH_CHECK(
        logits.device() == valid_ids.device(),
        "logits and valid_ids must be on the same CUDA device"
    );
    TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be 1D or 2D");
    TORCH_CHECK(valid_ids.dim() == 1, "valid_ids must be 1D");
    TORCH_CHECK(valid_ids.numel() > 0, "valid_ids cannot be empty");
    TORCH_CHECK(valid_ids.scalar_type() == at::kLong, "valid_ids must be int64");
    TORCH_CHECK(
        logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kHalf,
        "logits must be float32 or float16"
    );

    return masked_argmax_cuda(logits.contiguous(), valid_ids.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "masked_argmax",
        &masked_argmax,
        "sparse masked argmax over grammar-valid token IDs (CUDA)"
    );
}
