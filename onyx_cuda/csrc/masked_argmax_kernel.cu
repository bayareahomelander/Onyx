#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

template <typename scalar_t>
__global__ void masked_argmax_kernel(const scalar_t *__restrict__ logits,
                                     const int64_t *__restrict__ valid_ids,
                                     int64_t *__restrict__ output, int64_t rows,
                                     int64_t vocab_size, int64_t valid_count) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const scalar_t *row_logits = logits + row * vocab_size;

  float best_value = -FLT_MAX;
  int64_t best_id = LLONG_MAX;

  for (int64_t i = threadIdx.x; i < valid_count; i += blockDim.x) {
    const int64_t token_id = valid_ids[i];
    const float value = static_cast<float>(row_logits[token_id]);

    if (best_id == LLONG_MAX || value > best_value ||
        (value == best_value && token_id < best_id)) {
      best_value = value;
      best_id = token_id;
    }
  }

  __shared__ float shared_values[kBlockSize];
  __shared__ int64_t shared_ids[kBlockSize];

  shared_values[threadIdx.x] = best_value;
  shared_ids[threadIdx.x] = best_id;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = shared_values[threadIdx.x + stride];
      const int64_t other_id = shared_ids[threadIdx.x + stride];
      const int64_t current_id = shared_ids[threadIdx.x];

      if (other_id != LLONG_MAX &&
          (current_id == LLONG_MAX ||
           other_value > shared_values[threadIdx.x] ||
           (other_value == shared_values[threadIdx.x] &&
            other_id < current_id))) {
        shared_values[threadIdx.x] = other_value;
        shared_ids[threadIdx.x] = other_id;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[row] = shared_ids[0];
  }
}

} // namespace

at::Tensor masked_argmax_cuda(at::Tensor logits, at::Tensor valid_ids) {
  const c10::cuda::CUDAGuard device_guard(logits.device());

  const int64_t rows = logits.dim() == 1 ? 1 : logits.size(0);
  const int64_t vocab_size =
      logits.dim() == 1 ? logits.size(0) : logits.size(1);
  const int64_t valid_count = valid_ids.numel();

  auto output = at::empty({rows}, valid_ids.options());

  const dim3 blocks(rows);
  const dim3 threads(kBlockSize);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logits.scalar_type(), "masked_argmax_cuda", [&] {
        masked_argmax_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            logits.data_ptr<scalar_t>(), valid_ids.data_ptr<int64_t>(),
            output.data_ptr<int64_t>(), rows, vocab_size, valid_count);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
