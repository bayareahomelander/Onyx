# Onyx CUDA technical report

API contracts, operating limits, implementation details, benchmarks, and
reproduction instructions for Windows/NVIDIA. Start with the
[setup guide](README.md). The repeatable delivery check below produces local
validation results; generated artifacts and implementation notes are excluded
from version control.

## Selecting models at startup

Model selection applies only to Windows. With no settings, the API still loads
the pinned Qwen2.5 1.5B target; gamma 0 avoids loading any draft. A positive
`ONYX_SPECULATIVE_GAMMA` loads the selected draft as well.

Available VRAM does not change these defaults, and the server does not prompt
you to choose a larger model. To use a larger compatible target or draft/target
pair, explicitly configure the settings below before startup. Start with
[preflight and GPU validation](#model-validation-and-support-levels) to check
your chosen configuration.

| Environment variable | Default |
| --- | --- |
| `ONYX_TARGET_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `ONYX_TARGET_REVISION` | The bundled target's pinned snapshot |
| `ONYX_DRAFT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `ONYX_DRAFT_REVISION` | The bundled draft's pinned snapshot |
| `ONYX_SPECULATIVE_GAMMA` | `0` |

For example, this uses the bundled 0.5B model as a smaller target without a draft:

```powershell
$env:ONYX_TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
$env:ONYX_SPECULATIVE_GAMMA = "0"
.\.venv\Scripts\python.exe -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

Set the model variables to other Hugging Face repository IDs to experiment with
compatible models. Local directories, URLs, prequantized models, and models
requiring custom remote code are outside this loader's supported scope. All
weights use FP16 on CUDA; the loader does not fall back to CPU or substitute a
different model if loading fails. Changing selection requires a server restart.

Python callers can use `create_app(target_model=..., target_revision=...,
draft_model=..., draft_revision=..., gamma=...)`. Each explicit non-`None`
argument overrides its corresponding environment variable. Settings are captured
when the application is created. Explicit model settings cannot be combined
with an injected `engine` or `load_engine`; injected engines supply their own
models and their actual identities are reported where available.

Revision defaults are looked up for the selected repository ID, not its role.
A custom repository without a revision resolves its current default-branch
commit once; that immutable commit is then used for tokenizer and weight loads.
Branches and tags supplied as revisions are resolved the same way. Pin the
reported commit for repeatable runs. Clear inherited revision variables when
switching repositories, since each setting has independent precedence. Missing
files download automatically; cached files are reused.

Before accepting requests, each loaded model must have a supported ByteLevel
tokenizer, usable chat template and EOS token, and pass a short forward/cache
extension/rollback/replay check. A speculative pair must also have matching
token bytes, vocabulary widths, special tokens, EOS IDs, and compatible chat
formatting. These checks catch unsupported execution contracts; they do not
establish correctness or a speedup for every model, prompt, or context length.

`GET /` reports `models.target` and `models.draft` (null when absent).
`GET /v1/models` exposes the same data under `configuration`. Startup logs also
record these identities, actual resolved revisions, precision, gamma, and
selector backend. `validated_snapshot` means an individual repository/revision
matches a bundled validation pin; it does not certify a custom pair or its
performance. The request model ID remains `onyx-speculative`; HTTP requests
cannot switch models or initiate model downloads.

The existing 2048-token API context envelope and other service limits still
apply, even on larger GPUs. Select models that fit alongside KV caches and
temporary tensors; parameter count alone is not a memory guarantee.

### Benchmarking a selected configuration

Both benchmark tools read the same model/revision environment settings as the
server. `benchmark --target` uses the selected target; its default single-model
mode uses the selected draft. `--compare` and `--speculative` load both models
and compare gamma 0/1/2/4 regardless of the server's gamma setting.
`benchmark_masking --models` loads the selected pair for gamma 0/2 comparisons.

```powershell
# After setting the desired model/revision environment variables:
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --compare --output benchmarks/results/custom-comparison.json
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark_masking --models --output benchmarks/results/custom-masking.json
```

Reports record actual selected repository IDs and resolved revisions. The
file-based target/constraint/speculation gates reject baselines with a different
target identity or revision; regenerate the chain after changing models or
settings. Use separate output paths for experiments. The full regression suite
and `validate_windows.ps1` deliberately isolate model-selection variables so
release validation continues to use bundled pins. The script restores the
caller's environment afterward. Custom configurations remain experimental until
their own target-baseline, structured-output, and resource checks pass.

## Model validation and support levels

Model size and family name alone do not establish compatibility. These commands
let a small-GPU machine check metadata and a larger-GPU machine supply runtime
evidence. They do not change server defaults or automatically choose models.

| Level | Evidence |
| --- | --- |
| Prechecked | Configuration/tokenizer checks, causal-LM construction on the meta device, and pair compatibility if requested; no weights or CUDA execution |
| Startup-verified | Actual selected weights passed the loader's CUDA forward, cache extension, crop, and replay checks |
| Tested | The selected-model validator completed every recorded case on the reported OS/GPU, revisions, and settings |

These are scoped results, not universal certifications. A failed report can show
`support_level: startup-verified` because startup passed before a later case
failed. Always inspect `status` and the individual `checks`. A precheck failure
may be an access/download problem or a limitation of the installed runtime;
it does not necessarily mean the model can never work.

### Preflight without model weights

From an installed `onyx_cuda` environment (CPU PyTorch is sufficient):

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.preflight --target-model Qwen/Qwen2.5-7B-Instruct --vram-gib 32 --output validation/7b-preflight.json
```

Only configuration and tokenizer assets download. The command reuses the loader's
repository-ID, revision, quantization, tokenizer, and chat-template checks. It
constructs a weightless model skeleton with the installed Transformers to check
architecture support and count parameters. Custom remote code is disabled.
It does not download pretrained weights, run a forward pass, or initialize CUDA.

`--gamma 0` (the default) checks only the target. Positive gamma also checks the
draft and applies the exact pair compatibility checks used at startup. Both
commands accept `--target-model`, `--target-revision`, `--draft-model`, and
`--draft-revision`; explicit values override their corresponding environment
variables, then bundled defaults apply. Gamma defaults to 0 and the runtime
validator's selector defaults to `torch`, independently of environment variables.
Select speculation with `--gamma`; select the optional kernel explicitly with
`--greedy-backend cuda` on the runtime validator.

The memory report separates FP16 parameter storage, model buffers, and a
conventional full-attention KV estimate for known cache layouts. Unknown KV
layouts report null. The KV estimate assumes batch 1 and the chosen context
size plus gamma; sliding/hybrid implementations can use different amounts.
`--context-tokens` defaults to 2048. Activations, attention workspaces, allocator
reserve, driver/desktop use, and other processes are not estimated. `--vram-gib`
is an optional hypothetical capacity in binary GiB: weights exceeding that
capacity are flagged, but remaining capacity never produces a guarantee of fit.
A successful precheck returns exit code 0 even when its memory advisory says
the weights exceed capacity; failed checks return 1.

### Runtime validation on the intended GPU

Install the package with its `[server]` extra and CUDA PyTorch, or use the
documented development environment. Stop other model processes and run serially.
The runtime command performs its own preflight and uses those resolved immutable
revisions for all subsequent weight loads. To reproduce a preflight from another
machine, transfer its JSON report and explicitly reuse its model IDs/revisions:

```powershell
$preflight = Get-Content validation/7b-preflight.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe -m onyx_cuda.validate_model --target-model $preflight.models.target.id --target-revision $preflight.models.target.revision --output validation/7b-runtime.json

# Bundled speculative pair; use explicit draft ID/revision flags for a custom pair:
.\.venv\Scripts\python.exe -m onyx_cuda.validate_model --gamma 2 --output validation/bundled-pair-runtime.json
```

Every report path must be new; existing evidence is never overwritten. An
interrupted run leaves an incomplete marker. On failure, the terminal gives the
exception details and the report records the failing stage and exception type.
An unsuccessful runtime validation returns exit code 1.

The runtime checks use production code and the existing benchmark prompt corpus:

- Model loading and startup cache probes, with actual identities and precision.
- A synthetic batch-1 cache extension/crop/replay probe at `--context-tokens`
  (default 2048; runtime range 64 through 2048), separately for target and draft.
- Ordinary greedy generation, regex, supported JSON Schema, and seeded top-p
  sampling; constrained cases must finish successfully, not merely return partial
  output with a `length` finish reason.
- Exact selected-generator token IDs and finish reasons against a same-process
  target-only oracle, plus incremental text decoding agreement.
- In-process HTTP completion and SSE agreement, including a required successful
  finish event before `[DONE]`; an SSE error never counts as success.
- For positive gamma, a deliberately rejected proposal followed by target and
  draft cache replay comparisons, even if the two models are identical.

All generation checks share the API's persistent inference worker. Reports
contain dependency versions, OS, GPU/VRAM, CUDA runtime, model revisions,
settings, per-case outcomes, and peak allocated/reserved PyTorch memory. They
include the source commit and dirty flag when run from a checkout; installed
wheels without Git metadata report null and always include hashes of the actual
Python/native package files. They omit local paths, private prompts, generated
text, and raw exception messages. Output hashes and token counts permit comparison.
Reports stay under the Git-ignored `validation/` directory unless another path
is explicitly supplied. Nothing is uploaded automatically.

This selected-model check is distinct from the full bundled regression suite and
fresh-install delivery gate. It does not exhaust the schema subset, stress every
context/workload, measure network streaming latency/disconnections, or establish
a performance gain. Use the existing benchmark commands for timings and the
full regression suite for cancellation, backpressure, and other service contracts.
A Linux CUDA result is Linux evidence; Windows support needs a Windows run.

### Current evidence and larger-model contributions

The bundled Qwen2.5 1.5B target and 0.5B draft have prior full Windows validation
on the 6 GB RTX 4050 at the pins in `src/onyx_cuda/revisions.py`. Target-only
remains the default. The selected-model workflow can extend the evidence to new
configurations without changing those release tests.

A metadata-only check of Qwen2.5 7B Instruct at revision
`a09a35458c702b33eeacc393d103063234e8bc28` passed locally. This is **prechecked**,
not GPU-tested. Its estimated FP16 weights alone occupy 15,231,233,024 bytes.
Pairing that target with the bundled 0.5B draft fails the current compatibility
contract: their logits widths are 152064 and 151936. More VRAM does not resolve
that mismatch, and the engine does not remap or pad their vocabularies.

To contribute a larger-model result, run the validator on the intended Windows
GPU and share its JSON report after reviewing it. Identify the exact revisions,
gamma, selector, context size, and source version. A result becomes an entry in
the tested list only after its evidence is reviewed; model-family similarity or
a preflight pass alone does not promote it.

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

### One candidate across CPU and CUDA

The delivery script also writes `artifacts/candidate.json`, binding the wheel
and source archive hashes to the source commit, source inputs, package version,
build environment, and bundled model pins. Development builds are permitted,
but a candidate built from a dirty checkout cannot pass release promotion.

After the CPU build, validate the exact same wheel on the Windows GPU machine:

```powershell
$manifest = "C:\candidate\artifacts\candidate.json"
$candidate = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
$wheel = Join-Path (Split-Path -Parent $manifest) $candidate.artifacts.wheel.name
./scripts/validate_windows.ps1 -Mode Cuda -WheelPath $wheel -WheelSha256 $candidate.artifacts.wheel.sha256 -CandidateManifest $manifest
```

Transfer the entire artifacts directory, including the source archive. Use the
same source checkout for the validation harness. Hash and source checks run
before installation; the existing-wheel mode does not rebuild or need Rust.
Each environment is fresh, and installed package hashes must match the candidate.

Both modes exercise the installed preflight command. CPU mode requires runtime
validation to refuse CPU-only execution. CUDA mode also runs the selected-model
validator with gamma 0 and 2, then creates a separate `[server]` consumer
environment without pytest or maturin. It launches the normal Uvicorn CLI as
an owned child, checks text/regex/JSON/SSE over loopback HTTP, and requires the
child to shut down. Runtime logs stay local; generated reports omit raw errors.
This smoke check complements the existing controlled incremental-stream test.

`-Profile Full` is the default and retains required custom-kernel validation.
`-Profile Core` is an explicit development option that omits CuPy and permits
only optional sparse-kernel test skips. It cannot satisfy a full release gate.
`-Benchmarks` requires CUDA and the Full profile. CUDA tests and consumer model
processes run sequentially, including on the 6 GB device.

Successful runs write `release-run.json`, binding the relevant report hashes to
the candidate digest. Failed or incomplete runs cannot produce a passing receipt.
To verify both receipts and assemble a release bundle without publishing:

```powershell
$commit = git rev-parse HEAD
python scripts/verify_release.py release --candidate C:\candidate\artifacts\candidate.json --cpu C:\evidence\cpu --cuda C:\evidence\cuda --expected-commit $commit --output validation/release-bundle
```

The verifier requires Windows/Python 3.12 evidence, expected model pins/settings,
all required checks, matching installed files and artifact hashes, and clean
source provenance. It rejects changed reports and missing/skipped hardware
requirements. The bundle copies the already-tested wheel and source archive,
adds `release-evidence.json`, `SHA256SUMS.txt`, and release notes, and does not rebuild.
Receipts are consistency checks within the trusted validation process, not
cryptographic attestations of arbitrary third-party reports.

### Prerelease workflow

`Windows release candidate` is a separate manual workflow, restricted to `main`.
It builds once on hosted Windows, sends the candidate to the existing trusted
self-hosted Windows CUDA runner, verifies both results, and uploads a verified
bundle. The GPU runner still needs the documented CUDA Toolkit/CuPy prerequisites
for the full kernel gate. It never runs automatically on pull requests.

By default this workflow only validates. Explicit `create_draft=true` creates a
draft prerelease tagged `onyx-cuda-v<package-version>-rc.<candidate-number>` after
every gate passes. Only that final job has repository write permission. A failed
or skipped prerequisite prevents it from running. Publishing the draft remains
a deliberate maintainer action; no public release is created by ordinary pushes.
The package stays at 0.1.0; "Windows v1" is the existing product milestone name.

All local environments, caches, plans (including `windows-v1-release.md`),
validation logs, candidate manifests, and release bundles belong under the
ignored `onyx_cuda/validation/` directory. Commit permanent scripts, tests,
workflow definitions, and user documentation only.

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

- Bundled validation covers Qwen2.5 0.5B and 1.5B Instruct; other startup-selected models are experimental.
- Speculative decoding is greedy; positive-temperature requests use the target
  model directly.
- Beam search, batching, and repetition penalties are not implemented.
- CUDA is required. Model loading fails rather than falling back to CPU.
