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
| Draft decoding | CUDA graphs for the pinned draft; ordinary forward otherwise |
| Thinking | Disabled |
| Context / maximum output | 8192 tokens including prompt / 4096 output tokens |
| Token selection / recovery | Torch / scalar |

No model overrides are needed for the default setup.

Draft graphs replay each draft token as one CUDA graph over the draft's own
weights. The target still verifies every proposed token, so they change speed,
not output. Set `ONYX_DRAFT_BACKEND=eager` to use the ordinary draft forward.
Opt-in graph recovery processes unconstrained emitted history in eight-token
blocks, with two/three-token blocks and scalar steps for remainders; set
`ONYX_REPLAY_BACKEND=graph` to enable it.

## Settings

Set environment variables in the server's terminal before startup:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ONYX_TARGET_MODEL`, `ONYX_DRAFT_MODEL` | `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model IDs |
| `ONYX_TARGET_REVISION`, `ONYX_DRAFT_REVISION` | Pinned revisions | Commit hashes for custom models |
| `ONYX_SPECULATIVE_GAMMA` | `2` | Draft tokens per step; `0` selects target-only generation |
| `ONYX_SPECULATIVE_MODE` | `fixed` | `adaptive` opts into the experimental adaptive controller |
| `ONYX_DRAFT_BACKEND` | `graph` | `eager` uses the ordinary draft forward |
| `ONYX_REPLAY_BACKEND` | `scalar` | `graph` opts into graph recovery, validated on CUDA capability 7.5 |
| `ONYX_GREEDY_BACKEND` | `torch` | `cuda` uses the optional custom selector (install `.[kernels]`; needs CUDA Toolkit 12.4) |
| `ONYX_MAX_CONTEXT_TOKENS` | `8192` | Prompt plus requested output tokens |
| `ONYX_MAX_OUTPUT_TOKENS` | `4096` | Maximum requested output; omitted budgets use the smaller of 1024 and this limit |
| `ONYX_MAX_ACTIVE_REQUESTS` | `8` | Running or queued completions; excess requests receive HTTP 429 |
| `ONYX_STREAM_BUFFER_CHUNKS` | `64` | Buffered SSE chunks before the producer waits for the reader |

Limits must be positive integers, and the output limit must leave room for a
prompt. The root endpoint (`GET /`) reports the active limits, models, and
backends. Validate custom models or larger limits on the target GPU first with
`python -m pytest --require-cuda` and
`python -m onyx_cuda.validate_model --output validation/runtime.json`.

## Measured performance

On a Linux RTX 2080 Ti reporting 22 GiB, the September 25, 2026 comparison covered
48 cases with one warmup and three measured runs per mode. Aggregates sum
per-case median generation times, excluding model loading and graph setup:

| Mode | Aggregate speedup over target-only |
| --- | ---: |
| Fixed gamma 2, draft graphs, scalar recovery (default) | 1.292x |
| Fixed gamma 2, draft graphs, graph recovery (opt-in) | **1.398x** |

In the same run, the ordinary draft forward measured 1.130x and 1.211x; draft
graphs reduced total generation time by 12.6% and 13.4%. Every output in every
mode matched target-only tokens and finish reasons, and no case was slower with
draft graphs. Nine cases remain slower than target-only: six very short
requests (at most 23 ms slower) and three recovery-heavy prose requests.

Draft graphs cut draft cost from about 7.5-9.8 ms to 4.1-4.4 ms per token and
add about 2 seconds of startup; graph recovery preparation adds about 45 seconds.
Peak memory in the comparison was 17.29 GiB allocated.

The full CUDA suite passed 872 tests with draft graphs enabled, with three
Windows-specific skips. September 22 checks matched full logits and KV caches
bitwise through 8192 tokens with graph recovery, and the API completed a
4096-prompt/4096-output capacity test. Memory exhaustion during graph recovery
still releases its graphs and falls back to scalar recovery. Draft graphs have
not yet been validated on a Windows GPU. Adaptive speculation remains
experimental and was not remeasured with draft graphs.

To reproduce the comparison on your GPU, use a new output filename for each run:

```sh
python -m onyx_cuda.benchmark_adaptive --split all --repetitions 3 --output validation/speedups.json
```

One interleaved run times target-only, fixed gamma 2, and adaptive speculation,
and adds graph-recovery modes when the GPU supports them (compute capability 7.5).
Every mode uses the configured draft backend, draft graphs by default. Every
speculative output must match target-only generation token for token. The report
records which modes ran and why a graph backend was unavailable. The September 25
figures come from an equivalent validation run that also timed the ordinary draft
forward side by side.

### Speedup by workload

Category results from the same September 25 comparison:

| Workload | Default | With graph recovery |
| --- | ---: | ---: |
| Code generation | **2.02x** | **2.02x** |
| Regex-constrained output | **1.79x** | **1.79x** |
| Information extraction | **1.61x** | **1.61x** |
| JSON Schema output | **1.58x** | **1.58x** |
| Prose | 1.08x | 1.20x |
| Short replies | 1.06x | 1.06x |

Each baseline uses the same target model, prompt, precision, and output budget.
Regex and JSON baselines enforce the same constraints; these figures measure
the benefit of speculation over already-constrained target-only generation.
Graph recovery changes only requests that need numerical recovery, which in this
corpus were prose and changing-pattern requests. Category results describe the
tested cases and do not guarantee a speedup for every request.

## Structure

```text
src/onyx_cuda/  CUDA inference, token selection, and API
rust/          Native regex and JSON Schema engine
tests/         Unit, API, and real-GPU tests
scripts/       Windows package validation
```

## License

[MIT](../LICENSE).
