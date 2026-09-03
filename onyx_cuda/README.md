# Onyx CUDA

Onyx CUDA is the Windows and NVIDIA GPU edition of [Onyx](../README.md), an
inference engine for fast, structured LLM output. It combines speculative
decoding with token-level regex and JSON Schema constraints and exposes an
OpenAI-compatible chat-completions API.

This package is under active development. The current implementation uses
Qwen2.5 0.5B as the draft model and Qwen2.5 1.5B as the target model.

## Features

- NVIDIA CUDA inference with no CPU fallback
- Grammar-constrained generation using regex or JSON Schema
- Fixed-gamma speculative decoding
- OpenAI-compatible non-streaming and streaming responses
- Generation timing and token-acceptance metrics

## Requirements

- 64-bit Windows
- Python 3.12 recommended
- NVIDIA GPU and current driver (validated on a 6 GB RTX 4050)
- Rust MSVC toolchain
- Visual Studio Build Tools with the C++ workload

Model weights are downloaded to the Hugging Face cache on first use.

## Quick start

From the `onyx_cuda` directory, create a virtual environment and install the
CUDA build of PyTorch before installing the package:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Start the API server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

The first startup may take a while while the draft and target models download.

## Make a request

In another PowerShell window:

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"onyx-speculative","messages":[{"role":"user","content":"Generate a product code."}],"regex":"[A-Z]{3}-[0-9]{4}","max_tokens":16}'
```

For JSON output, pass a JSON Schema instead of a regex:

```json
{
  "model": "onyx-speculative",
  "messages": [{"role": "user", "content": "Generate a user record."}],
  "json_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"}
    },
    "required": ["name", "age"]
  }
}
```

Set `"stream": true` to receive server-sent events. The server also provides:

- `GET /` for health information
- `GET /v1/models` for available models
- `POST /v1/chat/completions` for generation

## Development

Run the test suite from the `onyx_cuda` directory:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the benchmark commands as needed:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --constraints
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --target
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --speculative
```

Benchmark results are written under `benchmarks/results/` and are ignored by
Git.

## Current limitations

- The model pair is currently fixed to Qwen2.5 0.5B and 1.5B Instruct.
- Speculative decoding is greedy; positive-temperature requests use the target
  model directly.
- Beam search, batching, and repetition penalties are not implemented.
- CUDA is required. Model loading fails rather than falling back to CPU.

## Project layout

- `src/onyx_cuda/` — model loading, generation, speculative decoding, and API
- `rust/` — native regex and JSON Schema grammar engine
- `tests/` — unit and API tests

## License

Onyx is available under the [MIT License](../LICENSE).
