# Onyx CUDA

The NVIDIA CUDA implementation of [Onyx](../README.md) constrains LLM output to a
regex or a JSON Schema as it generates, and speeds generation up with
grammar-aware speculative decoding. Greedy output matches the target model
running alone, token for token. It serves an OpenAI-style chat-completions API
with streaming on Windows and Linux.

## Demo

Each clip is the [`/demo` race page](#example) on an RTX 2080 Ti: one prompt, run
with Qwen3-8B alone and then with Onyx, played side by side from 0 s. Every token
is a mark at its arrival time. Onyx's tokens arrive a verification round at a
time, and both outputs match exactly.

**Python function**, free-form text:

https://github.com/user-attachments/assets/615559ff-cbeb-4468-b74b-c8e33a00f2bd

**JSON array**, constrained by a JSON Schema:

https://github.com/user-attachments/assets/4b654cb7-8bca-466b-9ed9-e7fc9de7a8c2

**32 digits**, constrained by the regex `[0-9]{32}`:

https://github.com/user-attachments/assets/e0e805ad-8085-4640-b515-620c59d7c744

## Measured performance

On a Linux RTX 2080 Ti, across a fixed 48-case benchmark, Onyx generated 1.49x
faster in aggregate than Qwen3-8B alone, and every output matched the target
alone token for token, including the finish reason:

| Workload | Cases | Speedup over target-only |
| --- | ---: | ---: |
| Code | 6 | 2.44x |
| JSON Schema output | 4 | 2.11x |
| Regex-constrained output | 4 | 2.07x |
| Information extraction | 6 | 1.85x |
| Original mixed prompts | 9 | 1.62x |
| Output that changes pattern | 3 | 1.39x |
| Prose | 8 | 1.22x |
| Short replies | 8 | 1.13x |
| **All cases** | **48** | **1.49x** |

Constrained baselines enforce the same regex or schema, so these figures measure
the gain from speculation alone. Nine cases remain slower than target-only: six
with very short outputs (at most 21 ms slower) and three longer requests that
needed numerical recovery. [Performance details](docs/performance.md) cover the
method, scalar-recovery results, history, and validation.

## How it works

- **Draft, then verify.** Each round, the Qwen2.5-0.5B draft proposes up to three
  tokens. The Qwen3-8B target scores the current token and every proposal in one
  forward pass, keeps the longest prefix it agrees with, and adds its own next
  token, so a round emits one to four tokens. Both KV caches are then cropped back
  to the accepted tokens.
- **The target decides every token.** Greedy output therefore matches the target
  running alone, token for token and including the finish reason, and real-model
  tests check it. Requests with a positive temperature sample from the target
  alone.
- **FP16 near-ties are repaired.** Scoring several positions in one batch can round
  differently from one-token decoding. When the two best eligible scores at a
  position are within FP16 rounding of each other, Onyx recomputes them with
  one-token steps from a clean checkpoint cache. On the validated GPU, CUDA graphs
  replay the emitted history of unconstrained requests in blocks of up to eight
  tokens, matching one-token steps bitwise.
- **The draft runs as CUDA graphs.** The draft's decoder step is reimplemented over
  its own weights with a static KV cache and captured as CUDA graphs, cutting draft
  time from about 7.5-9.8 ms to 4.1-4.4 ms per token. The draft prefills the prompt
  only after the target picks the first token, so speculation adds almost nothing
  to time to first token.
- **Grammar masks live on the GPU.** A Rust engine (regex DFAs and a stack-based JSON
  Schema parser, exposed through PyO3) works out which tokens are legal in each
  grammar state from the target tokenizer's raw bytes. Its scans are cached by
  grammar state and reused as GPU masks, and both models select only legal tokens.

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

In another terminal on Linux:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"onyx-speculative","messages":[{"role":"user","content":"Generate a product code."}],"regex":"[A-Z]{3}-[0-9]{4}","max_tokens":16}'
```

Or in another PowerShell window on Windows:

```powershell
$body = @{
    model = "onyx-speculative"
    messages = @(@{ role = "user"; content = "Generate a product code." })
    regex = "[A-Z]{3}-[0-9]{4}"
    max_tokens = 16
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType "application/json" -Body $body
```

The [API reference](docs/api.md) covers JSON Schema constraints, streaming,
request fields, errors, metrics, defaults, and settings.

Open `http://127.0.0.1:8000/demo` while the server runs to race both modes on a
preset prompt: the page runs the target alone, then speculation, and plays the two
runs side by side with each token placed at its arrival time. Opened as a local
file, the page plays recorded sample runs instead.

## Structure

```text
src/onyx_cuda/  CUDA inference, token selection, and API
rust/          Native regex and JSON Schema engine
tests/         Unit, API, and real-GPU tests
docs/          API reference and performance details
```

## License

[MIT](../LICENSE).
