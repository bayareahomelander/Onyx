# Onyx

Onyx generates structured LLM output by enforcing regex and supported JSON
Schema constraints during token generation. It provides an OpenAI-compatible
chat-completions API, streaming, and grammar-aware speculative decoding.

There are two independent implementations:

| Platform | Runtime | Setup |
| --- | --- | --- |
| macOS / Apple Silicon | MLX | [Mac setup below](#mac-setup) |
| Windows or Linux / NVIDIA GPU | PyTorch + CUDA | [CUDA setup](onyx_cuda/README.md) |

## Setup

Clone the repository, then follow the instructions for your platform:

```sh
git clone https://github.com/bayareahomelander/Onyx.git
cd Onyx
```

### Mac setup

Requires Apple Silicon with Metal GPU access, Python 3.12, and Rust.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip "maturin>=1.4,<2.0"
python -m pip install -e ".[server]"
python -m maturin develop --release
python -m uvicorn onyx.server:app --host 127.0.0.1 --port 8000
```

For NVIDIA CUDA setup, model selection, and benchmarks, see the
[Onyx CUDA README](onyx_cuda/README.md).

## Example

With the Mac server running, request a constrained product code:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"onyx-speculative","messages":[{"role":"user","content":"Generate a product code."}],"regex":"[A-Z]{3}-[0-9]{4}","max_tokens":16}'
```

Both editions accept `json_schema` for structured JSON and `stream: true` for
streaming. See the [Windows PowerShell example](onyx_cuda/README.md#example).
A token budget may leave output incomplete. Check the finish reason before using
constrained output; see the [Windows example](onyx_cuda/README.md#example).

## Repository structure

```text
onyx/          Apple Silicon inference and API
rust/          Apple Silicon package's native grammar engine
examples/      Python usage examples
tests/         Apple Silicon tests
onyx_cuda/     NVIDIA CUDA package: Python source, native engine, tests, and build tools
```

## License

[MIT](LICENSE).
