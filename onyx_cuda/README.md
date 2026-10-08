# Onyx CUDA

The NVIDIA CUDA implementation of [Onyx](../README.md) generates structured LLM
output with regex and JSON Schema constraints. It provides an OpenAI-style
chat-completions API and streaming on Windows and Linux.

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

Use `json_schema` for JSON constraints and `stream: true` for SSE. The API allows
up to 4096 output tokens within an 8192-token prompt-plus-output budget by default.
Constrained output ends with `finish_reason: "stop"` once it fully matches and
either cannot be extended or the model chooses to end it, so open-ended patterns
such as `[0-9]+` are not cut off at their first match. Partial output has
`finish_reason: "length"` and may not satisfy the constraint. Custom `stop`
strings cannot be combined with regex or JSON Schema constraints; these
combinations return HTTP 422.

JSON Schema support is a checked subset: `type` (a name or a list of names),
`properties`, `required`, boolean `additionalProperties`, `items`, `enum`,
`pattern`, `minLength`/`maxLength`, and `minItems`/`maxItems`, plus the
annotations `title`, `description`, `default`, `examples`, and `$comment`. Other
keywords, such as `$ref`, `oneOf`, `format`, or `minimum`, are rejected rather
than ignored. Generated objects contain only declared properties, however
`additionalProperties` is set. A schema `pattern` matches anywhere in the
string, as JSON Schema specifies, while a `regex` constraint must match the
whole output.

The API follows OpenAI's request and response shapes but is not a drop-in
replacement. Messages take plain-string `content` with the `system`, `user`, and
`assistant` roles. Besides `model` and `messages`, requests accept `max_tokens`
(or `max_completion_tokens`), `temperature`, `top_p`, `seed`, `n` (up to four,
without streaming), `stop`, `stream`, and `response_format` (`text` or
`json_schema`), plus Onyx's `regex`, `json_schema`, `compact_json`,
`enable_thinking`, and `speculative`. Any other field, such as `tools`,
`stream_options`, or `presence_penalty`, returns HTTP 422 instead of being
ignored. A regex that does not compile, an unsupported JSON Schema, or an
unknown `model` returns HTTP 400. Request errors use FastAPI's `detail` body
rather than OpenAI's `error` object; an error during a stream arrives as an
`error` object followed by `[DONE]`.

For live SSE output on Linux, add `"stream":true` to the body and pass `-N` to
`curl`. In PowerShell, set `stream = $true` before converting the body to JSON:

```powershell
$body | curl.exe --no-buffer http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@-"
```

Generated text is in `choices[0].delta.content`. For constrained JSON, wait for
a successful `stop` finish event; `[DONE]` alone is insufficient. The finishing
chunk also carries `usage` and `onyx_metrics`, as non-streaming responses do:
time to first token, total generation time, decode rate, whether speculation ran,
and for speculative requests the acceptance rate and verification rounds.

To compare both modes on one server, send `"speculative": false` to run the
target alone for that request; omitting the field follows the server setting.
`"speculative": true` returns HTTP 422 when the server runs the target alone
(`ONYX_SPECULATIVE_GAMMA=0`) or the request samples with a positive temperature.

Open `http://127.0.0.1:8000/demo` while the server runs to race both modes on a
preset prompt: the page runs the target alone, then speculation, and plays the two
runs side by side with each token placed at its arrival time. Opened as a local
file, the page plays recorded sample runs instead.

## Defaults

| Setting | Default |
| --- | --- |
| Target / draft | Pinned Qwen3-8B / Qwen2.5-0.5B-Instruct, FP16 |
| Decoding | Fixed gamma 3 speculation for greedy requests; target-only sampling for positive temperature |
| Draft decoding | CUDA graphs for the pinned draft; ordinary forward otherwise |
| Thinking | Disabled |
| Context / maximum output | 8192 tokens including prompt / 4096 output tokens |
| Numerical recovery | Graph recovery on the validated Linux RTX 2080 Ti, scalar elsewhere |

No model overrides are needed for the default setup.

Draft graphs replay each draft step of one or two tokens as one CUDA graph over
the draft's own weights; after a fully accepted round, one two-token step
consumes the last proposal and the next token together. The target still
verifies every proposed token, so they change speed, not output. Set `ONYX_DRAFT_BACKEND=eager` to use the ordinary draft forward.
Graph recovery processes unconstrained emitted history in eight-token blocks,
with two/three-token blocks and scalar steps for remainders. The default
(`auto`) enables it only where it was qualified: Linux, an RTX 2080 Ti (compute
capability 7.5, 68 SMs), and the pinned target and library versions. Elsewhere,
or if preparing its graphs runs out of GPU memory, the server starts with scalar
recovery and `GET /` reports why. Set `ONYX_REPLAY_BACKEND=scalar` to disable
it, or `graph` to require it on any capability 7.5 GPU. Recovery beyond 6144
tokens of context uses scalar steps: each graph block briefly holds an extra
copy of the KV cache, which no longer fits beside the draft graphs.

## Settings

Set environment variables in the server's terminal before startup:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ONYX_TARGET_MODEL`, `ONYX_DRAFT_MODEL` | `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model IDs |
| `ONYX_TARGET_REVISION`, `ONYX_DRAFT_REVISION` | Pinned revisions | Commit hashes for custom models |
| `ONYX_SPECULATIVE_GAMMA` | `3` | Draft tokens per step; `0` selects target-only generation |
| `ONYX_DRAFT_BACKEND` | `graph` | `eager` uses the ordinary draft forward |
| `ONYX_REPLAY_BACKEND` | `auto` | `scalar` disables graph recovery; `graph` requires it on any CUDA capability 7.5 GPU and fails startup if it cannot be prepared |
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

On a Linux RTX 2080 Ti reporting 22 GiB, the latest September 28, 2026 comparison
covered 48 cases with one warmup and three measured runs per mode. Aggregates sum
per-case median generation times, excluding model loading and graph setup:

| Mode | Aggregate speedup over target-only |
| --- | ---: |
| Fixed gamma 3, draft graphs, scalar recovery (`ONYX_REPLAY_BACKEND=scalar`) | 1.370x |
| Fixed gamma 3, draft graphs, graph recovery (default on this GPU) | **1.492x** |

Every output in every mode matched target-only tokens and finish reasons. Nine cases
remain slower than target-only: six very short requests (at most 21 ms slower)
and three recovery-heavy prose requests. Earlier the same day, before draft
catch-up steps were folded into the next draft step (2.1% less time), gamma 3
measured 1.346x / 1.462x and the previous default, gamma 2, 1.292x / 1.397x.
Gamma 3 improved or held every workload category over gamma 2 and needed no
additional numerical recoveries. The
September 25 comparison, which also timed the ordinary draft forward, measured
1.130x and 1.211x at gamma 2 without draft graphs; draft graphs reduced total
generation time by 12.6% and 13.4%.

Speculative requests reach their first token as quickly as target-only requests:
the draft prefills the prompt only after the target has selected the first token,
and not at all when generation ends there. Across the 48 cases, speculation adds
0.3 ms in total to time to first token, down from 637 ms on September 27.

Draft graphs cut draft cost from about 7.5-9.8 ms to 4.1-4.4 ms per token and
add about 4 seconds of startup; graph recovery preparation adds about 45 seconds.
Peak memory in the comparison was 17.31 GiB allocated. On September 27, forced
recoveries with 5,000 to 8,000 prompt tokens and both graph sets loaded matched
target-only output and peaked at 19.54 GiB allocated.

On October 6 the full CUDA suite passed all 848 tests with gamma 3, draft
graphs, and default graph recovery. September 22 checks
matched full logits and KV caches bitwise through 8192 tokens with graph
recovery, and the API completed a 4096-prompt/4096-output capacity test. If a
graph block still exhausts GPU memory, that recovery continues with scalar steps
and later requests keep the graphs. Draft graphs have not yet been validated on
a Windows GPU.

To reproduce the comparison on your GPU, use a new output filename for each run:

```sh
python -m onyx_cuda.benchmark --output validation/speedups.json
```

One interleaved run times target-only and fixed gamma 3 generation, and adds a
graph-recovery mode when the GPU supports it (compute capability 7.5).
`--gamma 2 3` compares several gammas in the same run.
Every mode uses the configured draft backend, draft graphs by default. Every
speculative output must match target-only generation token for token. The report
records which modes ran and why a graph backend was unavailable. Each speculative
run also records its proposal, acceptance, and recovery counts, and a graph mode
fails if any of its recoveries fell back to scalar steps.

### Speedup by workload

Gamma 3 category results from the same latest September 28 comparison:

| Workload | Scalar recovery | Graph recovery (default) |
| --- | ---: | ---: |
| Code generation | **2.44x** | **2.44x** |
| JSON Schema output | **2.11x** | **2.11x** |
| Regex-constrained output | **2.07x** | **2.07x** |
| Information extraction | **1.85x** | **1.85x** |
| Prose | 1.10x | 1.22x |
| Short replies | 1.13x | 1.13x |

Each baseline uses the same target model, prompt, precision, and output budget.
Regex and JSON baselines enforce the same constraints; these figures measure
the benefit of speculation over already-constrained target-only generation.
Graph recovery changes only requests that need numerical recovery, which in this
corpus were prose and changing-pattern requests. Category results describe the
tested cases and do not guarantee a speedup for every request.

Constrained generation caches each grammar state's valid tokens and reuses them
as GPU masks, so long text inside a JSON string no longer rescans the vocabulary
at every token. In a separate September 27 check of four long-text requests (a
JSON summary, JSON records with descriptions, code in a JSON string, and a broad
regex), speculation measured 0.74x-2.01x against target-only, up from
0.49x-0.99x before the cache, with identical output. The JSON summary stays
slower because its one numerical recovery runs in scalar steps: graph recovery
does not apply to constrained requests.

## Structure

```text
src/onyx_cuda/  CUDA inference, token selection, and API
rust/          Native regex and JSON Schema engine
tests/         Unit, API, and real-GPU tests
```

## License

[MIT](../LICENSE).
