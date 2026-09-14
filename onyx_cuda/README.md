# Onyx CUDA

The Windows/NVIDIA implementation of [Onyx](../README.md): structured LLM output
with regex and JSON Schema constraints, an OpenAI-compatible API, and streaming.

By default, Onyx CUDA loads only the Qwen2.5 1.5B target in FP16. A 0.5B draft model and
custom CUDA token selection are optional. Windows v1 (package 0.1.0) targets
a 6 GB RTX 4050; see [validation](#validation).

**To use a larger model**, select it explicitly with `ONYX_TARGET_MODEL` before
starting the server. For a speculative pair, also set `ONYX_DRAFT_MODEL` and a
positive `ONYX_SPECULATIVE_GAMMA`. Onyx does not automatically switch models or
prompt you based on available VRAM. Use [preflight and GPU validation](#validation)
to check your chosen configuration, then apply the [startup settings](#model-selection).

Weights load directly onto `cuda:0` through Accelerate, avoiding a complete FP16
model in system RAM before GPU transfer. Loading still needs host memory for
checkpoint I/O and temporary tensors; the model and generation caches must fit
in VRAM. CPU/disk offloading and quantized loading are not supported.

## Requirements

- Windows x64 and Python 3.12 x64
- NVIDIA GPU and driver compatible with CUDA 12.4 PyTorch
- For source builds: Rust MSVC toolchain and Visual Studio Build Tools with the C++ workload

The optional custom selector also requires CUDA Toolkit 12.4 with NVRTC/headers
and CuPy. Install `.[kernels]` and set `ONYX_GREEDY_BACKEND=cuda` to opt in;
the default `torch` selector needs neither.

## Setup

Install from source; no prebuilt Windows release is currently published.

### Build from a clone

From the repository root, in PowerShell:

```powershell
cd onyx_cuda
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==26.1.2
.\.venv\Scripts\python.exe -m pip install -c requirements-validation.txt torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -c requirements-validation.txt -e ".[dev]"
```

### Start the server

From `onyx_cuda`, after installation:

```powershell
.\.venv\Scripts\python.exe -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

Model weights download on first startup. Use one worker and trusted local
clients; stop with Ctrl+C. The server defaults to target-only generation.
Optional [startup model settings](#model-selection) let you
choose a compatible target and draft; existing defaults stay unchanged.

## Model selection

Set these environment variables in the server's terminal **before startup**:

| Variable | Default / purpose |
| --- | --- |
| `ONYX_TARGET_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `ONYX_DRAFT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `ONYX_TARGET_REVISION`, `ONYX_DRAFT_REVISION` | Bundled models use pinned revisions; set commit hashes to reproduce a custom selection. |
| `ONYX_SPECULATIVE_GAMMA` | `0` loads only the target; a positive value loads the draft for greedy speculation. |

For example, after validating the selected pair on your GPU:

```powershell
$env:ONYX_TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct"
$env:ONYX_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
$env:ONYX_SPECULATIVE_GAMMA = "2"
```

Models must be compatible Hugging Face repository IDs. More VRAM alone does
not establish compatibility. Run [validation](#validation) before switching;
restart the server to apply changes. Benchmark to choose gamma:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --compare --output benchmarks/results/custom-comparison.json
```

For Qwen3, add `--disable-thinking --max-tokens 256 --require-complete` to compare
completed non-thinking answers. These flags affect the benchmark only.

## Example

In another PowerShell window:

```powershell
$body = @{
    model = "onyx-speculative"
    messages = @(@{ role = "user"; content = "Generate a product code." })
    regex = "[A-Z]{3}-[0-9]{4}"
    max_tokens = 16
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType "application/json" -Body $body
```

Use `json_schema` for JSON constraints and `stream: true` for SSE. The API allows
up to 1024 output tokens within a 2048-token prompt-plus-output budget. Partial
output has `finish_reason: "length"` and may not satisfy the constraint.

To see live SSE output in PowerShell, set `$body` to a request with `stream = $true`
before converting it to JSON, then send it using:

```powershell
$body | curl.exe --no-buffer http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@-"
```

Generated text is in each event's `choices[0].delta.content`. For constrained
JSON, wait for a successful `stop` finish event; `[DONE]` alone is insufficient.

## Validation

For a source/development install, run the complete local suite from `onyx_cuda`:

```powershell
.\.venv\Scripts\python.exe -m pytest --require-cuda
```

For a fresh Windows build/install check, run
`./scripts/validate_windows.ps1 -Mode Cuda -Profile Core -Python ./.venv/Scripts/python.exe`.
Add `-Benchmarks` to run the same-process comparison. `-Profile Full` additionally
checks the optional CUDA kernels; Core is the default.
Routine GitHub CI builds the package and runs Rust/CPU tests. The manual
[release workflow](../.github/workflows/windows-release.yml) verifies a candidate
wheel on CPU and GPU, with kernel checks available as an opt-in input.

To inspect another model without loading weights, then test it on a suitable GPU:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.preflight --target-model Qwen/Qwen2.5-7B-Instruct --vram-gib 32 --output validation/7b-preflight.json
# On the machine that will run the model; this downloads and loads its weights:
.\.venv\Scripts\python.exe -m onyx_cuda.validate_model --target-model Qwen/Qwen2.5-7B-Instruct --output validation/7b-runtime.json
```

Preflight establishes eligibility, not GPU fit. Add `--gamma 2` and
`--draft-model` to inspect/test a pair. Use the resolved revisions for repeatable
runs. A successful preflight checks metadata compatibility; runtime validation
checks generation, constraints, API streaming, and caches on that GPU and OS.
Neither guarantees a speedup. Reports must use a new output filename.

## Measured performance

Target-only generation remains the default: the tested Qwen2.5 0.5B draft +
1.5B target workload was slower with speculation on the 6 GB RTX 4050 laptop.
More VRAM allows a larger target alongside a small draft, but speedup still
depends on draft cost, accepted proposals, and response length.

On September 14, 2026 (UTC), commit `6cf435f` passed 497 Python tests and
43 Rust tests on a Linux server with an RTX 2080 Ti reporting 22 GiB VRAM.
Three Windows-only tests were skipped. CUDA and optional kernel checks were
enabled; all 44 GPU tests returned to their starting allocated-memory level
after cleanup. This Linux run does not replace Windows validation.

An exploratory retest paired **Qwen3-8B** with **Qwen2.5-0.5B-Instruct** and
**Qwen2.5-1.5B-Instruct**, all in FP16. The numeric cases reproduced the Mac
benchmark's raw prompts and regex constraints. The counting, JSON, and prose
cases used the target's chat template with thinking disabled.

| Workload | 0.5B draft speedup (gamma) | 1.5B draft speedup (gamma) |
| --- | --- | --- |
| Four-digit year | 1.27x (2) | 1.19x (2) |
| 32 constrained digits | 1.91x (8) | 2.07x (8) |
| Counting 1 through 10 | 2.41x (8) | 2.14x (8) |
| Short JSON response | 1.41x (2) | 1.32x (2) |
| One-sentence GPU explanation | 1.09x (2) | 1.02x (2) |

Each entry selects the best observed draft length from 1, 2, 4, and 8; it is
not the performance of one fixed configuration. Measurements used one warmup
and three repetitions, synchronized full-call wall time, and disabled internal
timing instrumentation. Every speculative output matched its same-run
target-only baseline token-for-token, completed within its budget, and passed
the applicable constraint checks. Counting reached 69.7 tokens/s with the 0.5B
draft versus 28.9 tokens/s target-only. Very short fixed replies still favored
target-only generation.

**These cross-family pairs are experimental and rejected by the production
loader's tokenizer compatibility checks.** The isolated benchmark loaded the
models directly without changing those checks. Four tool/thinking token byte
mappings differ between the families. Passing these cases does not establish
general tokenizer compatibility or API support. The weights also differ from
the Mac's MLX 4-bit artifacts; this comparison does not isolate operating-system
or quantization effects. The results demonstrate workload-specific CUDA
speedups, not a general 2x improvement or a reason to change the default.

## Structure

```text
src/onyx_cuda/  CUDA inference, token selection, and API
rust/          Native regex and JSON Schema engine
tests/         Unit, API, and real-GPU tests
scripts/       Windows package validation
```

## License

[MIT](../LICENSE).
