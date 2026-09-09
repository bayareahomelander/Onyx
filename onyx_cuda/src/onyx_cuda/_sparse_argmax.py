"""Optional NVRTC-compiled CUDA reduction; all tensor storage belongs to PyTorch."""

from functools import lru_cache

import numpy as np
import torch


_SOURCE = r"""
#include <cuda_fp16.h>
#include <math_constants.h>

// Match dense torch.argmax: NaN wins, then highest value, then lowest token ID.
__device__ bool better(float value, long long id, float best, long long best_id) {
    if (isnan(value)) return !isnan(best) || id < best_id;
    if (isnan(best)) return false;
    return value > best || (value == best && id < best_id);
}

extern "C" __global__ void sparse_argmax(
    const LOGIT_TYPE* logits, const long long* ids, long long* output,
    long long count, long long row_stride, long long column_stride) {
    __shared__ float values[256];
    __shared__ long long indices[256];
    int tid = threadIdx.x;
    const LOGIT_TYPE* row = logits + blockIdx.x * row_stride;
    // Invalid positions in the dense reference are -inf, including token 0.
    float best = -CUDART_INF_F;
    long long best_id = 0;
    for (long long offset = tid; offset < count; offset += blockDim.x) {
        long long id = ids[offset];
        float value = (float)row[id * column_stride];
        if (better(value, id, best, best_id)) {
            best = value;
            best_id = id;
        }
    }
    values[tid] = best;
    indices[tid] = best_id;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && better(values[tid + stride], indices[tid + stride],
                                  values[tid], indices[tid])) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = indices[0];
}
"""


@lru_cache(maxsize=2)
def _kernel(dtype: torch.dtype):
    try:
        import cupy
    except ImportError as exc:
        raise RuntimeError(
            "CUDA greedy selection requires the optional onyx-cuda[kernels] extra"
        ) from exc
    scalar = "half" if dtype == torch.float16 else "float"
    return cupy.RawKernel(_SOURCE.replace("LOGIT_TYPE", scalar), "sparse_argmax")


def sparse_argmax(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Internal launch for validated FP16/FP32 vectors or matrices and int64 IDs."""
    kernel = _kernel(logits.dtype)
    import cupy

    result = torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
    if result.numel() == 0:
        return result
    # Use PyTorch's current stream, including non-default streams, so its
    # allocator can safely recycle IDs/output after queued work has completed.
    with cupy.cuda.Device(logits.device.index), cupy.cuda.ExternalStream(
        torch.cuda.current_stream(logits.device).cuda_stream,
        device_id=logits.device.index,
    ):
        kernel(
            (result.numel(),), (256,),
            (
                # RawKernel marshals scalar arguments by value. An explicit
                # uint64 carries each device address in the CUDA pointer ABI;
                # tensors stay alive here and all work uses their current stream.
                # This avoids creating three DLPack wrappers for every token.
                np.uint64(logits.data_ptr()),
                np.uint64(token_ids.data_ptr()),
                np.uint64(result.data_ptr()),
                token_ids.numel(),
                logits.stride(0) if logits.ndim == 2 else 0,
                logits.stride(-1),
            ),
        )
    return result
