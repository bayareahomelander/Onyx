# Onyx CUDA technical report

API contracts, operating limits, implementation details, benchmarks, and
reproduction instructions for Windows/NVIDIA. Start with the
[setup guide](README.md). The repeatable delivery check below produces local
validation results; generated artifacts and implementation notes are excluded
from version control.

## Installing a prebuilt wheel

If using a wheel from `validation/results/<run-id>/artifacts/`, install that
wheel with its `[server]` extra after installing CUDA PyTorch. A prebuilt wheel
does not require Rust or Visual Studio Build Tools on the receiving machine:

```powershell
.\.venv\Scripts\python.exe -m pip install ".\path\to\onyx_cuda-0.1.0-cp312-cp312-win_amd64.whl[server]"
```


## Running the local service

Use one Uvicorn worker. Each worker loads its own model weights, so adding
workers can exhaust the 6 GB device. This release is intended for trusted local
clients on loopback; it does not include authentication for public hosting.
Stop the server with Ctrl+C. An in-progress model operation may finish before
shutdown releases its GPU resources. Avoid `--reload` for ordinary use.

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

## API options and completion status

Requests reject unknown fields, including nested message and response-format
fields. Message content must be text, with a `system`, `user`, or `assistant`
role. Send at least one message. Values must have the declared JSON types;
strings such as `"0.8"` are not accepted as numbers.

| Option | Behavior |
| --- | --- |
| `max_tokens` or `max_completion_tokens` | Output-token budget from 1 to 1024; default 256. Send only one name. |
| `temperature`, `top_p`, `seed` | Greedy decoding at temperature 0; target-model sampling above 0. `top_p` and `seed` apply to sampling. The seed is an integer in PyTorch's supported range, −2⁶³ through 2⁶⁴−1. |
| `response_format` | `{"type":"text"}`, or `json_schema` as shown below. |
| `json_schema`, `regex` | Onyx constraint extensions. Choose only one of these or `response_format`. |
| `stop` | One nonempty string or a list of up to four. Supported for text/regex output; JSON completion is determined by its schema. |
| `n` | 1 to 4 sequential choices; streaming supports only 1. A fixed seed is reused for each choice. |
| `stream` | SSE content deltas during generation, for both greedy and sampled decoding. |
| `compact_json` | Compact completed non-streaming JSON; default true. Explicit `true` with streaming is rejected. Partial JSON is never compacted. |

For the standard structured-output request shape:

```json
{
  "messages": [{"role": "user", "content": "Generate a user record."}],
  "max_completion_tokens": 64,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "user_record",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
      }
    }
  }
}
```

The schema name and optional description are metadata. `strict` defaults to
true; false is unsupported. `json_object` mode, tools, multimodal messages,
penalties, log probabilities, `stream_options`, and other unlisted options are
rejected explicitly. Use `json_schema` with declared properties for JSON objects.

The API exposes two finish reasons:

- `stop`: normal completion, including EOS or a stop string. For JSON requests,
  the returned document has passed schema validation.
- `length`: the token budget was exhausted. HTTP status is 200 and the response
  contains the partial output and usage. JSON may be incomplete and must not be
  treated as validated. Increase the budget and retry. A document completed on
  the last available token still finishes with `stop`.

Streams end with a terminal choice and `[DONE]`, including when the finish reason
is `length`. Deltas preserve split Unicode characters and withhold possible stop
strings. They are emitted as tokens become available; sampled generation does
not wait for the whole answer. Disconnecting closes the generation iterator.

Invalid fields, types, and option combinations return HTTP 422 before generation.
Unsupported schemas, invalid regexes, and unknown models return HTTP 400. Runtime
service failures return 503 and unexpected failures return 500. Once an SSE
response has begun, errors use an `error` event followed by `[DONE]` instead of
changing the HTTP status; there is no normal terminal choice on that path.

### Local service limits

The API reserves a conservative token envelope for the 6 GB device. The
chat-template prompt plus the requested output budget must fit within **2048
tokens**, or a smaller model context limit if one applies. The server rejects
excess context with HTTP 422 before generation or streaming headers; it does
not silently truncate the conversation. All choices use the same per-choice
context budget and run sequentially. Requests accept at most 128 messages and
32,768 total message-content characters, checked before tokenization.

At most **8 completion requests** may be active or waiting for the model in one
process. Further requests receive HTTP 429 with `Retry-After: 1`; retry after
that interval, with backoff if the service remains busy. Health and model-list
requests remain available while generation is busy. Model execution remains
serial, and non-streaming cancellation holds the model lock until the current
generation finishes.

All generation modes reuse one dedicated inference worker for the application's
lifetime. This also reuses CUDA/cuBLAS thread-local workspaces, preventing
per-request thread creation from accumulating live GPU workspace allocations.

Streams buffer at most **64 chunks** between generation and delivery. A slow
reader pauses the producer once that buffer is full. Disconnecting releases
the producer even when the buffer is full, and model ownership is retained
until its iterator has closed. These limits bound application work, rather
than guarantee a particular VRAM total under competing desktop applications.
The health endpoint reports the API limits. They apply to the HTTP API;
programmatic generation callers manage their own budgets.

For HTTP 503, check the server console and `nvidia-smi`, close other GPU-heavy
processes, and retry with a shorter prompt/output budget. Keep gamma 0 and the
default Torch selector for the smallest dependency footprint. Kernel startup
failures require checking the optional CUDA Toolkit/CuPy installation or
selecting `ONYX_GREEDY_BACKEND=torch` and restarting the server.

## Structured-output contract

A JSON-schema response with `finish_reason: "stop"` contains a complete document that passes
validation against the supported schema subset below. Unsupported keywords and
malformed schemas raise an error; they are never silently ignored. For example,
`{"type":"integer","minimum":18}` is rejected because `minimum` is not supported.

| Keyword | Supported behavior |
| --- | --- |
| `type` | Object, array, string, number, integer, boolean, null, or a nonempty array of these names |
| `properties`, `required` | Nested schemas; required names must be declared in properties |
| `additionalProperties` | Boolean only; generation normally chooses declared property names |
| `items` | One schema applied to every array element |
| `minItems`, `maxItems` | Nonnegative integer bounds |
| `minLength`, `maxLength` | Nonnegative bounds measured in Unicode code points, including escaped characters |
| `pattern` | Search semantics over decoded string contents, using the portable regex subset described below |
| `enum` | Nonempty, unique values intersected with all sibling constraints |

Object, array, and string keywords require the corresponding explicit `type`
(which may be part of a union). Empty schema objects are accepted. Boolean
schemas, references, composition keywords, numeric bounds, `format`, and all
other validation keywords are unsupported. `title`, `description`, `default`,
`examples`, and `$comment` are accepted as annotations and do not affect output.

Patterns support literals, character ranges, grouping, alternation, repetition,
anchors, and ASCII `\d` / `\w` classes and their complements. They are searched
within the string; use `^` and `$` to constrain the whole string. Lookaround,
backreferences, inline flags, Unicode property escapes, character-class set
operations, and other nonportable escapes are rejected. Grammar generation may
choose a narrower set of values than the schema permits; it does not promise to
generate every valid representation.

Constraints require a ByteLevel tokenizer, as used by the bundled Qwen models.
Raw token bytes preserve Unicode characters split across tokens, and special
tokens cannot satisfy a grammar. Completion is checked both against the schema
and against the final decoded API text. Numeric output retains its precision
during validation and compaction. HTTP schemas with decimal values that would
lose precision in the request parser are rejected. Token exhaustion returns
unvalidated partial content with `finish_reason: "length"`, as described above.
Non-streaming requests return HTTP 400 for schema or validation errors.
Streaming requests can emit partial content before an error event; clients must
wait for a terminal choice with `finish_reason: "stop"` before treating the assembled JSON as
validated. `[DONE]` alone does not indicate success.

## Development

Run the test suite from the `onyx_cuda` directory:

```powershell
.\.venv\Scripts\python.exe -m pytest --require-cuda
```

This runs unit, native, and real-model/API tests together in one process. GPU
tests are marked `gpu`; `pytest -m "not gpu"` runs the CPU/native subset. Without
`--require-cuda`, GPU tests skip when CUDA is unavailable. The explicit flag
prevents an absent GPU from turning a hardware validation run into a passing
CPU-only run. Run GPU tests serially on a dedicated device, without pytest-xdist.

The suite collects Python reference cycles after fixture teardown before
releasing cuBLAS workspaces and the CUDA allocator cache. Each successful GPU test must return to its
starting allocation, preventing model copies from accumulating across tests.

`create_app(engine=...)` transfers an injected engine to a single application
lifespan and releases its reference at shutdown. Use `create_app(load_engine=...)`
when an application must restart with a newly loaded engine; restarting an app
with an already-consumed injected engine raises an error.

### Repeatable Windows delivery check

With Python 3.12, Rustup, and the Visual Studio C++ Build Tools installed, run:

```powershell
# Build native tests, sdist, and wheel; install the wheel in a fresh environment;
# then run all CPU/native tests against the installed package.
./scripts/validate_windows.ps1 -Mode Cpu

# Default suite, required custom-kernel GPU suite, and both benchmark comparisons.
./scripts/validate_windows.ps1 -Mode Cuda -Benchmarks
```

The script uses new build and test environments with user-site packages disabled,
rebuilds the wheel from the source archive, checks installation paths and `pip
check`, and runs copied tests outside the source tree. It uses the committed
Rust lockfile, Rust 1.96.1, and `requirements-validation.txt` dependency constraints.
The CPU check installs the CPU PyTorch wheel; the hardware check installs cu124.
The hardware check also installs `[kernels]`, runs the suite with the default
backend, then repeats GPU tests with `ONYX_GREEDY_BACKEND=cuda` and
`--require-kernels`. That flag fails if kernel compilation or startup fails,
rather than accepting skipped kernel tests. The script fixes both backend
settings for each run and restores the caller's environment afterward.
Downloaded package and Hugging Face caches may be reused; virtual environments
and installed packages are always fresh.

Each run retains its artifacts and evidence under `validation/results/<run-id>/`:
wheel/source archive and SHA-256 hashes, command log, resolved dependencies,
JUnit results, and `validation.json` with outcomes, hardware, observed model
revisions, and per-test GPU allocation measurements. A CPU run records no
observed model revisions because it does not load real models.

The Windows GitHub Actions workflow runs the CPU build/install/test gate on
pushes and pull requests. A manual dispatch with `gpu=true` also runs the full
gate and benchmarks on a self-hosted Windows x64 NVIDIA runner labeled
`onyx-cuda`. That runner needs PowerShell 7 and the build prerequisites above.
Install CUDA Toolkit 12.4 with NVRTC and headers on the GPU runner for the
required custom-kernel pass.
Workflow artifacts retain the reports and distributions; the workflow does not
publish packages.
The required-kernel pass writes `pytest-kernels.xml` and
`validation-kernels.json`, including the backend and CuPy version.

### Model snapshots

The bundled loaders use these immutable revisions from `onyx_cuda.revisions`:

| Model | Validation revision |
| --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

Config, tokenizer, and model weights use the same resolved revision. Tests check
the actual loaded revision against these pins and record it in the validation
report. Updating a pin requires rerunning the full suite and regenerating any
benchmark baselines. Programmatic custom-model callers can pass an explicit
`revision` to `load_model`.

### Benchmarks

Compare target-only generation with gamma 1, 2, and 4 in one process:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --compare
```

This loads both models to compare them on the same device, with one warmup and
three measured runs per case. It checks exact token IDs and finish reasons
against the target, records model revisions and dependency versions, and reports
generation wall time, draft/verification/masking costs, VRAM, and cold/warm
vocabulary setup. Output is `benchmarks/results/performance_comparison.json`.
The recommendation stays at gamma 0 unless another mode is at least 5% faster
on every tested case. Individual case results remain available for workload-specific
tuning. Model loading, prompt formatting, and HTTP transport are outside the
generation timing; vocabulary setup is reported separately.

The file-based speculative gate requires the **target constraint baseline**, which
itself requires the target baseline. Run this complete sequence from `onyx_cuda`:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --target
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --target --constraints
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --speculative
```

These produce `phase4_target_baseline.json`, `phase4_target_constraint_gate.json`,
and `phase4_speculative_gate.json` under `benchmarks/results/`. `--speculative`
checks that the baseline model revision, device, settings, token IDs, and finish
reasons match. `--compare` is the alternative that needs no prerequisite files.
All benchmark result files are ignored by Git.
Baselines record the greedy backend, FP16 precision, optional CuPy version, and
timing-contract version. Comparisons reject mismatched or older baselines;
regenerate the full prerequisite chain after changing these settings. Both
target-only and speculative internal generation times now include final JSON
validation. HTTP formatting and transport remain outside those measurements.

## Performance defaults

`ONYX_SPECULATIVE_GAMMA=0` is the default: both streaming and non-streaming
requests use the target model directly, and the API does not load a draft model.
To opt into speculation, set a positive gamma before starting the server:

```powershell
$env:ONYX_SPECULATIVE_GAMMA = "2"
.\.venv\Scripts\python.exe -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

Programmatic callers can use `create_app(gamma=2)` or
`generate_speculative(..., gamma=0)` to select the mode explicitly. Gamma is a
server setting, not a request field. Positive-temperature requests always use
target sampling. `GET /` reports the configured `speculative_gamma`. The API model
ID remains `onyx-speculative` for compatibility.

The API reuses token-byte vocabularies for immutable loaded tokenizers, with a
cache bounded to two tokenizer/width pairs. JSON token enumeration rejects
impossible first bytes before checking whole tokens; the full parser and final
schema validation still determine which tokens are valid.

Measured on the 6 GB RTX 4050 laptop on 2026-09-06, with the bundled model pair,
one warmup, three repetitions, and a 32-token budget:

| Case | Previous default (gamma 4) | New default (target-only) | Speedup |
| --- | ---: | ---: | ---: |
| Short reply | 0.269 s | 0.157 s | 1.72× |
| GPU summary | 1.712 s | 0.858 s | 2.00× |
| Number sequence | 0.952 s | 0.880 s | 1.08× |
| Regex | 0.167 s | 0.084 s | 1.99× |
| JSON enum object | 1.853 s | 0.461 s | 4.02× |

These are median generation times from separate before/after runs, excluding
request setup. JSON enumeration itself fell from 0.567 s to 0.090 s. Repeated
vocabulary setup measured 0.149 s cold and 0.000013 s cached. In the final
same-process comparison, gamma 1/2/4 reached median speed ratios of
0.75×/0.72×/0.61× versus target-only; none won on these five cases. Re-run the
comparison on other hardware and workloads before enabling speculation.

### Optional sparse CUDA token selection

Constrained greedy generation can use a custom CUDA kernel that reads only
allowed logits and reduces them directly to a token ID. It avoids allocating a
full masked vocabulary tensor. Both target-only and speculative generation use
this selector, including draft proposals and target verification. Sampling keeps
the existing dense mask. Model weights remain FP16; this option does not quantize
models or change the grammar engine.

Install the optional kernel dependency and opt in before starting the API:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[kernels]"
$env:ONYX_GREEDY_BACKEND = "cuda"
```

This backend requires CUDA 12.x NVRTC and headers (validated with CUDA Toolkit
12.4 on Windows), plus `cupy-cuda12x==13.6.0`. CuPy compiles the bundled kernel on
first use and caches it outside the repository by default. Set
`CUPY_CACHE_IN_MEMORY=1` before running to avoid a persistent compilation cache.
The API resolves `greedy_backend` at application creation, validates and warms
the selected kernels during startup, and reports it alongside `speculative_gamma`
in `GET /`. `create_app(greedy_backend="cuda")` explicitly overrides the environment.
Later environment changes do not change an existing application's backend.
Cold import/compilation is excluded from warmed benchmark timings. Missing
dependencies or compilation failures are reported; they do not silently select
another backend. Set `ONYX_GREEDY_BACKEND=torch` to restore the default.

The kernel supports FP16/FP32 vectors and matrices, including strided views,
and launches on PyTorch's current CUDA stream. Other dtypes and tensor ranks
use the dense reference. It preserves dense argmax tie-breaking and nonfinite
logit behavior. Grammar enumeration and transferring allowed token IDs remain
separate costs. For greedy requests, the existing `mask_transfer_seconds` and
speculative `mask_seconds` measurements now include argmax selection as well.

Run the permanent comparison benchmark and optional-kernel tests:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark_masking --models
.\.venv\Scripts\python.exe -m pytest tests/test_sparse_argmax.py --require-cuda
```

The benchmark checks exact token agreement and records selection latency with
and without ID validation/upload, plus interleaved dense/sparse generation runs
for regex and JSON at gamma 0 and 2. Generation timing includes final validation
but excludes model loading, vocabulary setup, HTTP, and kernel warmup. Results
are written to the Git-ignored `benchmarks/results/masking_comparison.json`.
Optional-kernel tests skip when CuPy is absent; install the extra to run them.

On the RTX 4050 laptop on 2026-09-07, FP16 selection with 256 already-uploaded
allowed IDs measured 79.9 us for the dense path and 23.7 us for the custom path.
Including Python validation and ID upload reduced that advantage to
162.2 versus 135.5 us. Other candidate counts showed mixed results.

Five interleaved measured repetitions per backend produced these medians:

| Case | Gamma | Dense selection | Custom selection |
| --- | ---: | ---: | ---: |
| Regex | 0 | 106.4 ms | 105.3 ms |
| Regex | 2 | 164.6 ms | 162.2 ms |
| JSON enum object | 0 | 387.9 ms | 369.0 ms |
| JSON enum object | 2 | 565.3 ms | 605.3 ms |

Token IDs and completion reasons agreed across both backends and gamma values.
These short cases showed substantial timing variation and no consistent
generation speedup, so the custom backend remains opt-in. Faster selection
alone has not established faster speculative generation.

## Current limitations

- The model pair is currently fixed to Qwen2.5 0.5B and 1.5B Instruct.
- Speculative decoding is greedy; positive-temperature requests use the target
  model directly.
- Beam search, batching, and repetition penalties are not implemented.
- CUDA is required. Model loading fails rather than falling back to CPU.
