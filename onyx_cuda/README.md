# Onyx CUDA

The NVIDIA CUDA implementation of [Onyx](../README.md) generates structured LLM
output with regex and JSON Schema constraints. It provides an OpenAI-compatible
chat-completions API and streaming on Windows and Linux.

## Requirements

- Windows or Linux x64 and Python 3.12
- An NVIDIA GPU with approximately **22 GiB VRAM** for the default models, and a driver compatible with CUDA 12.4 PyTorch
- Rust for source builds; on Windows, the MSVC toolchain and Visual Studio C++ Build Tools

Both models and their caches must fit on the GPU. Quantization and CPU/disk
offloading are not supported. No prebuilt Windows release is published.

## Setup

### Windows

From the repository root, in PowerShell:

```powershell
cd onyx_cuda
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==26.1.2
.\.venv\Scripts\python.exe -m pip install -c requirements-validation.txt torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -c requirements-validation.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

### Linux

From the repository root, with Python 3.12 and Rust installed:

```sh
cd onyx_cuda
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install pip==26.1.2
python -m pip install -c requirements-validation.txt torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements-validation.txt -e ".[dev]"
python -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

Model weights download on first startup. Use one worker and trusted local
clients; stop with Ctrl+C. One inference worker reuses CUDA workspaces and
serializes GPU execution; extra Uvicorn workers would duplicate both models.

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
up to 4096 output tokens within an 8192-token prompt-plus-output budget by default.
Constrained output ends with `finish_reason: "stop"` once it fully matches and
either cannot be extended or the model chooses to end it, so open-ended patterns
such as `[0-9]+` are not cut off at their first match. Partial output has
`finish_reason: "length"` and may not satisfy the constraint. Custom `stop`
strings cannot be combined with regex or JSON Schema constraints; these
combinations return HTTP 422.

For live SSE output, set `stream = $true` before converting the body to JSON:

```powershell
$body | curl.exe --no-buffer http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@-"
```

Generated text is in `choices[0].delta.content`. For constrained JSON, wait for
a successful `stop` finish event; `[DONE]` alone is insufficient.

## Defaults

| Setting | Default |
| --- | --- |
| Target / draft | Pinned Qwen3-8B / Qwen2.5-0.5B-Instruct, FP16 |
| Decoding | Fixed gamma 2 speculation for greedy requests; target-only sampling for positive temperature |
| Thinking | Disabled |
| Context / maximum output | 8192 tokens including prompt / 4096 output tokens |
| Token selection / recovery | Torch / scalar |

No model overrides are needed for the default setup.

Opt-in graph recovery processes unconstrained emitted history in eight-token
blocks, with two/three-token blocks and scalar steps for remainders. Numerical
checks and scalar fallback remain in place; speculative gamma stays at 2.

## Measured performance

On a Linux RTX 2080 Ti reporting 22 GiB, the September 22, 2026 comparison covered
48 cases with one warmup and three measured runs per mode. Aggregates sum
per-case median generation times, excluding model loading and graph setup:

| Mode | Aggregate speedup over target-only |
| --- | ---: |
| Fixed gamma 2, scalar recovery (default) | 1.112x |
| Fixed gamma 2, graph recovery (opt-in) | **1.190x** |
| Adaptive speculation, graph recovery (opt-in) | 1.160x |

An independent ten-repetition comparison confirmed 1.92% lower fixed-mode
latency and 2.56% lower adaptive-mode latency versus the previous two/three-token
graph backend. No case crossed the predeclared regression threshold of both
5% and 5 ms. Gains vary by workload; some short and recovery-heavy requests
remain slower than target-only.

Graph preparation took 45.3 seconds versus 32.1 seconds for the previous backend
in separate startup measurements. Preallocating graph outputs reduced prepared
reserved memory from 18.15 to 17.22 GiB; allocated memory increased slightly,
from 16.47 to 16.52 GiB. These startup observations are not context-capacity bounds.

The full CUDA suite passed 657 tests, with three Windows-specific skips. All
3,264 benchmark generations matched reference tokens and finish reasons.
Separate checks matched full logits and KV caches bitwise through 8192 tokens,
and the API completed a 4096-prompt/4096-output capacity test. Graph recovery
remains restricted to its validated runtime and opt-in; memory exhaustion still
releases graphs and falls back to scalar recovery.

Graph recovery remains opt-in.

To reproduce the comparison on your GPU, use a new output filename for each run:

```sh
python -m onyx_cuda.benchmark_adaptive --split all --repetitions 3 --output validation/speedups.json
```

One interleaved run times target-only, fixed gamma 2, and adaptive speculation,
and adds graph-recovery modes when the GPU supports them (compute capability 7.5).
Every speculative output must match target-only generation token for token.
The report records which modes ran and why graph recovery was unavailable.

### Speedup by workload

The September 17, 2026 comparison reported the following category speedups with
**fixed gamma 2 and opt-in graph recovery**, using the Qwen3-8B FP16 target and
Qwen2.5-0.5B-Instruct draft on the same Linux RTX 2080 Ti configuration:

| Workload | Speedup over target-only |
| --- | ---: |
| Code generation | **1.65x** |
| Regex-constrained output | **1.48x** |
| Information extraction | **1.39x** |
| JSON Schema output | **1.13x** |
| Prose | 1.02x |
| Short replies | 0.96x (slower) |

Each baseline uses the same target model, prompt, precision, and output budget.
Regex and JSON baselines enforce the same constraints; these figures measure
the benefit of speculation over already-constrained target-only generation.
Model loading and graph setup are excluded.

This breakdown belongs to the September 17 **1.168x aggregate** result and has
not been remeasured for later graph-recovery revisions. Category results
describe the tested cases and do not guarantee a speedup for every request.

## Structure

```text
src/onyx_cuda/  CUDA inference, token selection, and API
rust/          Native regex and JSON Schema engine
tests/         Unit, API, and real-GPU tests
scripts/       Windows package validation
```

## License

[MIT](../LICENSE).
