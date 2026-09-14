# Onyx CUDA

The NVIDIA CUDA implementation of [Onyx](../README.md): structured LLM output
with regex and JSON Schema constraints, an OpenAI-compatible API, and streaming.
Windows and Linux use the same inference engine.

The default targets a **22 GiB GPU**: pinned **Qwen3 8B** plus a **Qwen2.5 0.5B
Instruct draft**, FP16, speculative decoding with **gamma 2**, and non-thinking
answers. Positive-temperature requests use target-model sampling; speculation
is greedy. No model settings are needed to select the default pair.

Weights load directly onto `cuda:0` through Accelerate. Both models and their
caches must fit in VRAM. Cached prefill processes long prompts in chunks and
keeps only the final position's logits where supported, bounding attention
workspace without discarding context. CPU/disk offloading and quantized loading
are not supported.

## Requirements

- Windows or Linux x64 and Python 3.12
- NVIDIA GPU with approximately 22 GiB VRAM for the default configuration and a driver compatible with CUDA 12.4 PyTorch
- Source builds: Rust; on Windows, the MSVC toolchain and Visual Studio C++ Build Tools

The optional custom selector requires CUDA Toolkit 12.4 with NVRTC/headers and
CuPy. Install `.[kernels]` and set `ONYX_GREEDY_BACKEND=cuda` to opt in; the
default `torch` selector needs neither. No prebuilt Windows release is published.

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

## Model selection

Set overrides in the server's terminal **before startup**:

| Variable | Default / purpose |
| --- | --- |
| `ONYX_TARGET_MODEL` | `Qwen/Qwen3-8B` |
| `ONYX_DRAFT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `ONYX_TARGET_REVISION`, `ONYX_DRAFT_REVISION` | Bundled models use pinned revisions; use commit hashes for custom selections |
| `ONYX_SPECULATIVE_GAMMA` | `2` draft tokens per iteration; `0` explicitly selects target-only generation |

The target tokenizer defines chat prompts, grammar bytes, decoding, and EOS for
both models. Compatible drafts may omit target-added tokens in unused embedding
slots, but existing token byte meanings and special tokens must agree and logits
widths must match. Every proposed token is verified by the target. This supports
the default Qwen2.5/Qwen3 pair without bypassing compatibility checks.

For another configuration, set model IDs and run [validation](#validation)
before restarting the server. Available VRAM does not trigger automatic model
selection. To compare proposal lengths 0, 1, 2, 4, and 8:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --compare --require-complete --output benchmarks/results/custom-comparison.json
```

The benchmark defaults to non-thinking answers and a 256-token budget. Use
`--enable-thinking` or `--max-tokens` to change those settings. `--disable-thinking`
remains accepted. It reports individual regressions and selects the mode with
the lowest total generation wall time over its corpus, not a universal winner.

## Capacity and thinking

Capacity is resolved once at application creation. Set overrides before startup:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ONYX_MAX_CONTEXT_TOKENS` | `8192` | Prompt plus requested output tokens, also bounded by model context limits |
| `ONYX_MAX_OUTPUT_TOKENS` | `4096` | Maximum requested output; omitted budgets use the smaller of 1024 and this limit |
| `ONYX_MAX_ACTIVE_REQUESTS` | `8` | Running or queued completions; excess requests receive HTTP 429 |
| `ONYX_STREAM_BUFFER_CHUNKS` | `64` | Buffered SSE chunks before the producer waits for the reader |

All values must be positive integers; output must leave room for a prompt within
context. The health endpoint reports active limits. Larger overrides require
validation on the selected GPU. Queue depth and buffering bound waiting work;
they do not allocate concurrent model copies. Choices within `n` run sequentially.

Requests default to `"enable_thinking": false`. Set it to `true` to request the
model's thinking template. Thinking consumes the same output/context budget.

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
Partial output has `finish_reason: "length"` and may not satisfy the constraint.

For live SSE output, set `stream = $true` before converting the body to JSON:

```powershell
$body | curl.exe --no-buffer http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@-"
```

Generated text is in `choices[0].delta.content`. For constrained JSON, wait for
a successful `stop` finish event; `[DONE]` alone is insufficient.

## Validation

From a development install in `onyx_cuda` (on Linux, use the activated `python`):

```powershell
.\.venv\Scripts\python.exe -m pytest --require-cuda
.\.venv\Scripts\python.exe -m onyx_cuda.preflight --vram-gib 22 --output validation/default-preflight.json
.\.venv\Scripts\python.exe -m onyx_cuda.validate_model --output validation/default-runtime.json
```

Preflight checks metadata without loading weights; it does not establish GPU fit.
Both tools default to the configured pair, gamma 2, and 8192 context tokens.
Use `--gamma 0` for target-only validation or `--context-tokens` to probe another
capacity. Runtime validation checks generation, constraints, API/SSE, and caches.
Neither tool guarantees speedup. Reports require a new output filename.

For a fresh Windows build/install check, run
`./scripts/validate_windows.ps1 -Mode Cuda -Profile Core -Python ./.venv/Scripts/python.exe`.
Add `-Benchmarks` for performance comparisons or `-Profile Full` for the optional
CUDA kernels. Routine CI builds the package and runs Rust/CPU tests. The manual
[Windows release workflow](../.github/workflows/windows-release.yml) verifies the
candidate wheel on Windows CPU and GPU. Linux validation does not replace that
Windows delivery check.

## Measured performance

The revised default passed **511 Python tests and 44 Rust tests** on the Linux
RTX 2080 Ti reporting 22 GiB VRAM, with three Windows-only tests skipped. The
Windows CPU suite passed 468 tests. An additional regression for an omitted
output budget under a smaller configured limit passed on both platforms.

The API completed a **4096-token prompt plus 4096 generated tokens** with gamma
2, peaking at **17.55 GiB allocated / 18.56 GiB reserved**. Streaming, sampled,
and thinking requests also succeeded after that long generation. Separate
runtime validation passed all 20 checks, including both 8192-token caches.
These checks cover the working-tree implementation; a new Windows GPU wheel
release still requires the Windows delivery gate above.

With the production loader and **fixed gamma 2**, the bundled five-case corpus
ran **1.29x faster in aggregate** than target-only (sum of median full-generation
wall times). Median reported output throughput was 31.48 versus 28.17 tokens/s;
the median per-case speedup was 1.09x, and some short cases regressed.

Four additional prompts outside that corpus used the same fixed setting:

| Workload | Speedup versus target-only |
| --- | --- |
| Two-sentence cache explanation | 1.17x |
| Python function | 1.48x |
| City-name extraction | 1.47x |
| Customer-support reply | 1.17x |

Their aggregate speedup was **1.27x**. All outputs completed and matched the
same-run target-only baseline token-for-token. Both comparisons used FP16,
non-thinking target chat formatting, one warmup, three measured repetitions,
a 256-token budget, Torch selection, and synchronized full-generation wall
time with timing instrumentation enabled. These small corpora establish gains
for the tested workloads, not a general speedup guarantee.

Historically, Qwen2.5 0.5B draft + 1.5B target speculation was slower on a 6 GB
RTX 4050 laptop, so that configuration used target-only generation. It is no
longer the default. A larger target makes speculation worthwhile when enough
draft tokens are accepted; short replies can still be slower.

On September 14, 2026 (UTC), commit `6cf435f` passed 497 Python tests and 43 Rust
tests on a Linux RTX 2080 Ti reporting 22 GiB VRAM. Three Windows-only tests were
skipped. All 44 GPU tests returned to their starting allocation after cleanup.

An exploratory FP16 retest paired Qwen3 8B with Qwen2.5 0.5B and 1.5B drafts:

| Workload | 0.5B draft speedup (gamma) | 1.5B draft speedup (gamma) |
| --- | --- | --- |
| Four-digit year | 1.27x (2) | 1.19x (2) |
| 32 constrained digits | 1.91x (8) | 2.07x (8) |
| Counting 1 through 10 | 2.41x (8) | 2.14x (8) |
| Short JSON response | 1.41x (2) | 1.32x (2) |
| One-sentence GPU explanation | 1.09x (2) | 1.02x (2) |

These entries select the best observed gamma per case, not one fixed setting.
The numeric cases reproduced the Mac's raw prompts and regex constraints; the
other cases used non-thinking target chat formatting. Each used one warmup,
three measured repetitions, synchronized full-call wall time, and disabled
internal timing instrumentation. Every output matched the same-run target-only
baseline token-for-token and completed successfully. Counting reached 69.7
output tokens/s with the 0.5B draft versus 28.9 target-only.

The historical retest used an isolated loader because production then rejected
the cross-family pair. The current loader validates shared token meanings and
uses the target tokenizer throughout. The weights still differ from the Mac's
MLX 4-bit artifacts; the comparison does not isolate operating-system or
quantization effects. These are workload-specific gains, not a general 2x claim.

## Structure

```text
src/onyx_cuda/  CUDA inference, token selection, and API
rust/          Native regex and JSON Schema engine
tests/         Unit, API, and real-GPU tests
scripts/       Windows package validation
```

## License

[MIT](../LICENSE).
