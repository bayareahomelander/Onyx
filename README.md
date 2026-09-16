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

## Mac API contract

The default Mac model pair remains the 4-bit 0.5B draft / 1.5B target. Selecting
`onyx-speculative-8b` loads the 8B target lazily. Both API modes still use gamma 4.
These API changes do **not** alter the independent CUDA service contract.

For Mac development tests, install `python -m pip install -e ".[server,dev]"`
and run `python -m pytest tests`. The HTTP contract tests use fake engines;
the generation-loop unit tests use simulated MLX operations. They do not
replace inference and native grammar validation on Apple Silicon.

## Mac benchmarks

These benchmarks used Apple Silicon and 4-bit quantized Qwen models.

### Grammar-aware speculative decoding

| Target | Baseline | Grammar-aware draft | Speedup |
| --- | ---: | ---: | ---: |
| 1.5B | 73.5 tokens/s | 69.2 tokens/s | 0.94× |
| 8B | 15.6 tokens/s | 22.6 tokens/s | **1.45×** |

For the `[0-9]{4}` grammar, reported draft acceptance increased from **75%**
with an unconstrained draft to **100%** with a grammar-aware draft, for both
target sizes. Speculation was slower than baseline with the 1.5B target.

### Experimental adaptive gamma

This 8B benchmark used the `[0-9]{32}` constraint. Warmup runs were excluded
from the reported averages.

| Configuration | Throughput | Speedup over baseline | Acceptance |
| --- | ---: | ---: | ---: |
| Target-only | 21.9 tokens/s | 1.00× | — |
| Fixed gamma 2 | 25.3 tokens/s | 1.16× | 93.8% |
| Fixed gamma 4 | 29.1 tokens/s | 1.33× | 88.2% |
| Fixed gamma 8 | 27.6 tokens/s | 1.26× | 78.9% |
| Adaptive gamma | 29.2 tokens/s | **1.34×** | 88.2% |

The adaptive controller started at gamma 4 with bounds of 1–8, averaged 4.2,
and finished at 8. It essentially matched fixed gamma 4 on this forced-digits
workload; these figures do not establish a general adaptive advantage.

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
