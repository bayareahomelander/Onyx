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

The Mac service exposes `GET /` for status, `GET /v1/models` for model discovery,
and `POST /v1/chat/completions` for generation. Swagger UI is available at
`/docs`; set `ONYX_API_DOCS=0` before starting the server to disable it.
`/redoc` is removed. `/openapi.json` remains available in either configuration.

| Request field | Supported values / default |
| --- | --- |
| `messages` | Required nonempty list of string-content messages; roles: `system`, `user`, `assistant` |
| `model` | `onyx-speculative` (default) or `onyx-speculative-8b` |
| `max_tokens` | Positive integer; defaults to 256 |
| `temperature` | Finite number from 0 to 2; defaults to 0 |
| `top_p` | Finite number greater than 0 and at most 1; defaults to 1 |
| `stream` | Boolean; defaults to false |
| `regex` | Optional regex constraint |
| `json_schema` | Optional supported JSON Schema object; mutually exclusive with `regex` |
| `stop` | Optional list of nonempty strings; model EOS remains active |

Compatibility changes for Mac clients:

- Only one completion is supported. Omit `n`; `n=1` remains accepted as a
  deprecated compatibility field. Other values return HTTP 422.
- JSON is no longer reformatted by the server. Omit `compact_json` and format
  parsed JSON in the client if needed. `compact_json=false` remains accepted
  as a deprecated compatibility field; `true` returns HTTP 422.
- Unknown fields, unsupported roles, conflicting constraints, and invalid
  numeric values return HTTP 422 instead of being ignored or replaced with
  defaults. Explicit null is only accepted for the optional constraint/stop
  fields; omit other fields to use their defaults.
- Both response modes report `length` for an exhausted budget, `stop` for EOS
  or a requested delimiter, and `grammar_complete` for a completed constraint.
  Stop delimiters and EOS tokens are excluded from returned text. A `length`
  result can contain incomplete JSON or a partial regex match.

The default Mac model pair remains the 4-bit 0.5B draft / 1.5B target. Selecting
`onyx-speculative-8b` loads the 8B target lazily. Both API modes still use gamma 4.
These API changes do not alter the independent CUDA service contract.

For Mac development tests, install `python -m pip install -e ".[server,dev]"`
and run `python -m pytest tests`. The HTTP contract tests use fake engines;
the generation-loop unit tests use simulated MLX operations. They do not
replace inference and native grammar validation on Apple Silicon.

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
