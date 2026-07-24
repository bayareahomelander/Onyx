# Onyx for Windows

Onyx for Windows is a local language-model inference engine for NVIDIA GPUs. It constrains model
output to a regular expression or supported JSON Schema while keeping generation, sampling,
streaming, and diagnostics under application control.

This directory contains the independent Windows/CUDA port of Onyx. It uses PyTorch and
Transformers instead of the Apple MLX runtime used by the macOS package.

> **Project status:** pre-alpha. Target-only generation is working and qualified on the development
> machine. Fixed speculative decoding and the OpenAI-compatible server are not implemented yet.

## What works today

- A pinned, quantized Qwen2.5 0.5B target running on CUDA.
- Greedy and explicitly seeded temperature/top-p generation.
- Ordered single-token and multi-token stops.
- Non-streaming and lifecycle-owned streaming output.
- Regex-constrained and JSON Schema-constrained generation.
- Time-to-first-token, throughput, grammar-stage timing, and peak-VRAM metrics.
- Exact checkpoint and transactional rollback of the pinned production `DynamicCache`.
- A framework-neutral, model-free draft-proposal primitive with exact rejection rollback
  checkpoints, deterministic replay, and failure-atomic cleanup.
- Direct isolated qualification of the pinned 0.5B production backend as a D32 proposal producer
  on `cuda:0`, with greedy and fresh-seeded CUDA selection plus exact rejection rollback.
- Pinned production batched target verification in one Transformers forward, with exact checkpoint
  rollback and replay composition.
- A framework-neutral, model-free match/replace acceptance decision that accepts target-matching
  draft tokens, returns the first target-selected mismatch replacement, and deliberately leaves the
  post-proposal final row unused.
- A framework-neutral, model-free one-iteration coordinator that composes proposal, verification,
  and acceptance over two already-prefilled checkpointable roles. A mismatch restores both caches
  to the exact accepted prefix and leaves the target replacement uncached; full acceptance keeps
  both complete proposal suffixes cached without selecting a bonus token.
- A lazy independent Rust grammar runtime; normal package import does not initialize CUDA or load a
  model.

The package is currently a Python library. It does not yet provide a CLI or HTTP server.

## Requirements

- Windows with an NVIDIA CUDA-capable GPU.
- An NVIDIA driver compatible with the PyTorch CUDA 12.4 wheel.
- Python 3.10 or newer; current qualification uses Python 3.12.
- A Rust MSVC toolchain and Visual Studio Build Tools when building the native grammar extension
  from source.
- Internet access on first model use, unless the pinned model and tokenizer snapshots are already
  cached locally.

The acceptance configuration is an NVIDIA GeForce RTX 4050 Laptop GPU with 6,141 MiB dedicated
VRAM and 16 GiB system RAM. Other NVIDIA configurations may work, but they have not been qualified.

## Install from source

Use a short virtual-environment path. On Windows systems without long-path support, deeply nested
environment or temporary paths can make PyTorch installation fail with `WinError 206`.

```powershell
git clone https://github.com/bayareahomelander/Onyx.git
Set-Location Onyx\onyx_cuda

py -3.12 -m venv C:\venvs\onyx-cuda
& C:\venvs\onyx-cuda\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the official CUDA build of PyTorch first. The default package index can otherwise resolve a
CPU-only wheel or reject the exact CUDA-local version required by the validated stack.

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[cuda]"
```

The `cuda` extra pins the complete validated top-level stack: Accelerate 1.14.0, bitsandbytes
0.49.2, Hugging Face Hub 0.36.2, psutil 7.2.2, Tokenizers 0.22.2, PyTorch 2.6.0+cu124, and
Transformers 4.57.6.

### Verify CUDA

```powershell
python -c "from onyx_cuda import probe_torch_cuda; print(probe_torch_cuda())"
```

The command should report the selected CUDA device and a successful on-device masked-token
selection. It should not report a CPU-only PyTorch build.

You can also inspect NVIDIA devices without importing PyTorch or creating a CUDA context:

```python
from onyx_cuda import discover_nvidia_devices

for device in discover_nvidia_devices():
    print(device.name, device.memory_total_mib, device.compute_capability)
```

## Quickstart

The first run downloads the pinned Qwen2.5 0.5B model and tokenizer if they are not already cached.
Model loading and first CUDA initialization can take noticeably longer than later generations.

```python
from onyx_cuda import load_production_target_engine

with load_production_target_engine() as engine:
    result = engine.generate("Hello world", max_new_tokens=4)

print(result.text)
print(result.sampled_token_ids)
```

With the validated stack, the deterministic smoke result is:

```text
! I'm a
(0, 358, 2776, 264)
```

After the snapshot is cached, use `local_files_only=True` for reproducible offline loading:

```python
with load_production_target_engine(local_files_only=True) as engine:
    result = engine.generate("Hello world", max_new_tokens=4)
```

Prompts are currently encoded as plain text. The tokenizer's stored chat template is not applied
implicitly.

## Constrained JSON generation

The production engine can constrain every selected token to the supported JSON Schema subset:

```python
import json

from onyx_cuda import JsonSchemaGrammar, load_production_target_engine

grammar = JsonSchemaGrammar('{"type":"string","enum":["ready"]}')

with load_production_target_engine() as engine:
    result = engine.generate_constrained(
        "Return the status as JSON:",
        grammar=grammar,
        max_new_tokens=8,
    )

value = json.loads(result.text)
print(value)  # ready
print(result.generation.finish_reason)
```

The JSON grammar accepts structural whitespace around a completed root. A matching result can
therefore finish either as `grammar_complete` when completion EOS is selected or as `length` when
the token budget ends while valid whitespace continuations remain. `json.loads()` verifies the
user-visible result in either case. Regex constraints use the same interface through
`RegexGrammar(pattern)`.

## Streaming

Use the production stream as a context manager whenever the consumer might stop early. This ensures
partial CUDA cache and grammar state are released immediately.

```python
from onyx_cuda import TextGenerationDelta, load_production_target_engine

with load_production_target_engine() as engine:
    with engine.stream("Hello world", max_new_tokens=16) as stream:
        for event in stream:
            if isinstance(event, TextGenerationDelta):
                print(event.text, end="", flush=True)
```

Only one constrained or unconstrained production stream may be active on an engine at a time.

## Sampling and stops

Greedy generation is the default. Seeded temperature/top-p sampling must be requested explicitly:

```python
from onyx_cuda import TemperatureTopPSelection, load_production_target_engine

selection = TemperatureTopPSelection(temperature=0.8, top_p=0.9, seed=7)

with load_production_target_engine() as engine:
    result = engine.generate(
        "Hello world",
        max_new_tokens=16,
        selection=selection,
        stop_token_sequences=((151645,),),
    )
```

Stop sequences are token-ID sequences, not text strings. A matched stop remains in sampled-token
metadata but is removed from visible output. The removed `stop_token_ids` keyword is not accepted;
represent a single-token stop as a one-element sequence.

## Metrics

Every successful generation exposes immutable metrics at `result.generation.metrics`:

```python
metrics = result.generation.metrics
print(metrics.ttft)
print(metrics.tokens_per_second)
print(metrics.peak_allocated_vram_bytes)
print(metrics.peak_reserved_vram_bytes)
```

Constrained requests additionally report compilation, grammar-state scan, valid-index transfer,
and CUDA mask-application totals in `metrics.grammar_timing`.

## Current limitations

The Windows package does not yet provide:

- a selected two-model draft/target pair or a separate production draft engine;
- a cache-coordinated iterative speculative engine or user-visible speculative decoding;
- production draft/target evidence pairing, an iterative full-acceptance handoff, or a
  final-row/bonus-token policy;
- grammar-state speculation or speculative metrics;
- final context, output, concurrency, or fixed-`gamma` operating limits;
- CPU offload;
- implicit chat-template formatting;
- a CLI or FastAPI/OpenAI-compatible server;
- support for non-NVIDIA GPUs.

The 0.5B target is the only production default. A separately pinned 3B model has bounded
target-only qualification but is not the default, is not qualified for cache checkpoints or
simultaneous draft/target residency, and carries a research/evaluation license rather than Apache
2.0. The isolated D34 proposal-role qualification does not select the 0.5B model as a release draft.

## Development

Install the development tools after the CUDA installation:

```powershell
python -m pip install -e ".[dev]"
```

Run the Python and Rust checks:

```powershell
python -m pytest
python -m ruff check src tests
cargo test --manifest-path rust/Cargo.toml
```

Build a source distribution with the independent Rust manifest explicit:

```powershell
python -m maturin sdist --manifest-path rust/Cargo.toml
```

All Windows implementation work belongs under `onyx_cuda/`. It must not add CUDA dependencies to
or change the behavior of the macOS `onyx` package.

## Technical reference

See [docs/technical_reference.md](docs/technical_reference.md) for the backend and tokenizer
contracts, production cache layout and rollback transaction, grammar ABI and state semantics, CUDA
mask behavior, metric boundaries, qualification evidence, packaging isolation, and deferred
architecture.

## License

The source package declares the MIT license in `pyproject.toml`. Downloaded model assets are not
bundled and retain their own licenses. The pinned 0.5B default is Apache 2.0 licensed; the optional
3B candidate is governed by the Qwen Research License and is limited to the project's personal,
noncommercial evaluation scope unless separate permission is obtained.
