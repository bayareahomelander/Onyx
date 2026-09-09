# Onyx CUDA

The Windows/NVIDIA implementation of [Onyx](../README.md): structured LLM output
with regex and JSON Schema constraints, an OpenAI-compatible API, and streaming.

The default is Qwen2.5 1.5B target-only FP16 generation. A 0.5B draft model and
custom CUDA token selection are optional. Windows v1 (package 0.1.0) targets
a 6 GB RTX 4050; see the [validation guide](REPORT.md#repeatable-windows-delivery-check).

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
