# Onyx CUDA

The Windows/NVIDIA implementation of [Onyx](../README.md): structured LLM output
with regex and JSON Schema constraints, an OpenAI-compatible API, and streaming.

By default, Onyx CUDA loads only the Qwen2.5 1.5B target in FP16. A 0.5B draft model and
custom CUDA token selection are optional. Windows v1 (package 0.1.0) targets
a 6 GB RTX 4050; see the [validation guide](REPORT.md#repeatable-windows-delivery-check).

**To use a larger model**, select it explicitly with `ONYX_TARGET_MODEL` before
starting the server. For a speculative pair, also set `ONYX_DRAFT_MODEL` and a
positive `ONYX_SPECULATIVE_GAMMA`. Onyx does not automatically switch models or
prompt you based on available VRAM. Use [preflight and GPU validation](REPORT.md#model-validation-and-support-levels)
to check your chosen configuration, then apply the [startup settings](REPORT.md#selecting-models-at-startup).

## Requirements

- Windows x64 and Python 3.12 x64
- NVIDIA GPU and driver compatible with CUDA 12.4 PyTorch
- Rust MSVC toolchain and Visual Studio Build Tools with the C++ workload

The optional custom selector also requires CUDA Toolkit 12.4 with NVRTC/headers
and CuPy. See the [kernel setup](REPORT.md#optional-sparse-cuda-token-selection).

## Setup

From the repository root, in PowerShell:

```powershell
cd onyx_cuda
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==26.1.2
.\.venv\Scripts\python.exe -m pip install -c requirements-validation.txt torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -c requirements-validation.txt -e ".[dev]"
```

Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

Model weights download on first startup. Use one worker and trusted local
clients; stop with Ctrl+C. The server defaults to target-only generation.
Optional [startup model settings](REPORT.md#selecting-models-at-startup) let you
choose a compatible target and draft; existing defaults stay unchanged.

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
output has `finish_reason: "length"`; see the [API contract](REPORT.md#api-options-and-completion-status).

## Validation

From `onyx_cuda`, run the complete local suite:

```powershell
.\.venv\Scripts\python.exe -m pytest --require-cuda
```

Fresh-install gates and benchmark commands are in the
[validation guide](REPORT.md#repeatable-windows-delivery-check).

To inspect another model without loading weights, then test it on a suitable GPU:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.preflight --target-model Qwen/Qwen2.5-7B-Instruct --vram-gib 32 --output validation/7b-preflight.json
# On the machine that will run the model; this downloads and loads its weights:
.\.venv\Scripts\python.exe -m onyx_cuda.validate_model --target-model Qwen/Qwen2.5-7B-Instruct --output validation/7b-runtime.json
```

Preflight establishes eligibility, not GPU fit. Add `--gamma 2` and
`--draft-model` to inspect/test a pair. Use the resolved revisions for repeatable
runs; see [model validation and support levels](REPORT.md#model-validation-and-support-levels).

## Structure

```text
src/onyx_cuda/  CUDA inference, token selection, and API
rust/          Native regex and JSON Schema engine
tests/         Unit, API, and real-GPU tests
scripts/       Windows package validation
```

## Documentation

- [Technical report: API, limits, kernels, and benchmarks](REPORT.md)

## License

[MIT](../LICENSE).
