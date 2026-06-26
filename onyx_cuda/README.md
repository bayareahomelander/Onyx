# Onyx CUDA Experiments

This package contains optional CUDA components for constrained-decoding
experiments.

## Current Components

### Sparse Masked Argmax

`onyx_cuda.masked_argmax.masked_argmax_tensor(logits, valid_token_ids)` takes
CUDA logits and a sparse set of grammar-valid token IDs, then returns the
highest-logit valid token. It is designed to model the inner selection step
used after the Rust grammar engine computes valid token IDs.

`onyx_cuda.grammar_handoff.masked_argmax_from_grammar_state(...)` is a small
bridge helper that asks a grammar constraint for valid token IDs, transfers that
set to CUDA, and runs the masked-argmax kernel.

`onyx_cuda.grammar_handoff.CudaValidIdCache` caches per-state valid-token tensors
on CUDA devices so repeated visits to the same grammar state skip repeated
host-to-device valid-ID uploads. Each lookup re-reads the grammar-valid IDs and
compares their fingerprint with the cached entry. If an opaque handle is reused
after reset for a different grammar path, the CUDA tensor is refreshed instead
of reused. `discard(state)` removes one state's tensors and `clear()` removes all
entries.

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
declares PyTorch and Ninja; use a CUDA-enabled PyTorch build, because the extra
does not install the CUDA compiler toolchain. Run the commands from an activated
virtual environment inside an MSVC developer shell so `ninja`, `cl`, and `nvcc`
are all discoverable.

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

### Model-Free Multi-Token Decode Loop

`onyx_cuda.decode_loop.decode_greedy_from_logits(...)` connects the existing
pieces into a bounded single-sequence decode loop:

1. fetch grammar-valid token IDs for the current Rust grammar state;
2. transfer or retrieve those IDs through `CudaValidIdCache`;
3. select the highest-logit valid token with the CUDA kernel;
4. copy that one token ID to the host and advance the grammar state;
5. stop when the grammar matches, logits are exhausted, or `max_steps` is met.

The helper consumes precomputed CUDA logits and performs no model inference.
This keeps the integration deterministic and small while establishing the
stateful boundary needed by a future Windows inference backend.

```bash
python -m pytest tests/test_cuda_decode_loop.py -q
python benchmark_cuda_decode_loop.py
```

The returned result includes selected token IDs, the final live grammar state,
the termination reason, and host-observed timing totals for valid-ID lookup,
selection, result synchronization, grammar advancement, and pre-cleanup decode
work. The benchmark measures the complete public call separately for its
end-to-end throughput calculation.
The loop owns a short-lived valid-ID cache and discards each entry after the
selected token has synchronized, keeping its cache footprint bounded.

State ownership is transactional. A normal return consumes the initial and
intermediate states and returns one live final state. If any step raises, every
successor created by the loop is released and the caller's initial state remains
live.

The loop currently supports one sequence at a time. Each logits tensor must be
shaped `[vocab]` or `[1, vocab]`, selection is greedy, and one token ID is copied
to the host per step so the CPU-based Rust grammar can advance.
