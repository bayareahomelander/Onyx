# Onyx CUDA Experiments

This package contains optional CUDA components for constrained-decoding
experiments.

## Current Component: Sparse Masked Argmax

`onyx_cuda.masked_argmax.masked_argmax_tensor(logits, valid_token_ids)` takes
CUDA logits and a sparse set of grammar-valid token IDs, then returns the
highest-logit valid token. It is designed to model the inner selection step
used after the Rust grammar engine computes valid token IDs.

`onyx_cuda.grammar_handoff.masked_argmax_from_grammar_state(...)` is a small
bridge helper that asks a grammar constraint for valid token IDs, transfers that
set to CUDA, and runs the masked-argmax kernel.

`onyx_cuda.grammar_handoff.CudaValidIdCache` caches per-state valid-token tensors
on CUDA devices so repeated visits to the same grammar state skip repeated
host-to-device valid-ID uploads. a cache is valid for one compiled grammar state
space; clear or recreate it after recompiling or otherwise changing the grammar
constraint.

Scope:

- greedy selection only
- CUDA tensors only
- `float32` and `float16` logits
- logits shaped `[vocab]` or `[batch, vocab]`
- one shared valid-token set across the batch
- deterministic ties: smallest token ID wins

Install PyTorch with CUDA support, then build lazily on first use:

```bash
python -m pip install -e ".[cuda]"
python benchmark_cuda_masked_argmax.py
python benchmark_cuda_grammar_handoff.py
```

For local tests, install the development dependencies too:

```bash
python -m pip install -e ".[cuda,dev]"
python -m pytest tests/test_cuda_masked_argmax.py -q
python -m pytest tests/test_cuda_grammar_handoff.py -q
```

On Windows, building the extension also requires the NVIDIA CUDA Toolkit with
`nvcc` on `PATH` and compatible Microsoft C++ Build Tools. The `cuda` extra
declares PyTorch; use a CUDA-enabled PyTorch build, because the extra does not
install the CUDA compiler toolchain.

Packaging note: this experimental backend currently expects an editable source
tree because PyTorch builds the extension from `onyx_cuda/csrc` at runtime. Wheel
or source-distribution packaging for the CUDA sources is intentionally not
claimed yet.

Import note: the callable convenience wrapper is available from the submodule:

```python
from onyx_cuda.masked_argmax import masked_argmax, masked_argmax_tensor
```

The benchmark uses synthetic logits and valid-token sets, so it does not load
an LLM and is suitable for small GPUs.
