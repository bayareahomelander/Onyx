# Onyx

Onyx generates structured LLM output by enforcing regex and supported JSON
Schema constraints during token generation. It provides an OpenAI-compatible
chat-completions API, streaming, and grammar-aware speculative decoding.

There are two independent implementations:

| Platform | Runtime | Setup |
| --- | --- | --- |
| macOS / Apple Silicon | MLX | [Mac setup below](#mac-setup) |
| Windows / NVIDIA GPU | PyTorch + CUDA | [Windows setup](onyx_cuda/README.md) |

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

### Windows setup

Use the separate package in `onyx_cuda`. It requires Python 3.12 x64, an NVIDIA
GPU, CUDA-enabled PyTorch, Rust MSVC, and Visual Studio C++ Build Tools.
See the [Windows guide](onyx_cuda/README.md) for installation and startup commands.

Windows defaults to Qwen2.5 1.5B target-only FP16 generation. The 0.5B draft
model and custom CUDA token selector are optional; the tested Windows workload
is fastest with target-only generation on the 6 GB RTX 4050.

Model weights download on first use. Run one server at a time on port 8000.

## Example

With the Mac server running, request a constrained product code:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"onyx-speculative","messages":[{"role":"user","content":"Generate a product code."}],"regex":"[A-Z]{3}-[0-9]{4}","max_tokens":16}'
```

Both editions accept `json_schema` for structured JSON and `stream: true` for
streaming. See the [Windows PowerShell example](onyx_cuda/README.md#example).
A token budget may leave output incomplete; consult the relevant API contract.

## Repository structure

```text
onyx/          Apple Silicon inference and API
rust/          Apple Silicon package's native grammar engine
examples/      Python usage examples
tests/         Apple Silicon tests
onyx_cuda/     Windows package: Python source, native engine, tests, and build tools
REPORT.md      Apple Silicon architecture, API reference, and benchmarks
```

## Documentation

- [Apple Silicon technical report](REPORT.md)
- [Windows technical report](onyx_cuda/REPORT.md)
- [Windows reproducibility guide](onyx_cuda/REPORT.md#repeatable-windows-delivery-check)

Performance results and reproduction commands live in the reports. Speedups
vary with hardware, model configuration, and workload.

## License

[MIT](LICENSE).
