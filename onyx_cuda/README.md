# Onyx CUDA

Onyx CUDA is the Windows and NVIDIA GPU edition of [Onyx](../README.md), an
inference engine for fast, structured LLM output. It combines speculative
decoding with token-level regex and JSON Schema constraints and exposes an
OpenAI-compatible chat-completions API.

This package is under active development. The current implementation uses
Qwen2.5 0.5B as the draft model and Qwen2.5 1.5B as the target model.

## Features

- NVIDIA CUDA inference with no CPU fallback
- Grammar-constrained generation using regex or JSON Schema
- Fixed-gamma speculative decoding
- OpenAI-compatible non-streaming and streaming responses
- Generation timing and token-acceptance metrics

## Requirements

- 64-bit Windows
- Python 3.12 recommended
- NVIDIA GPU and current driver (validated on a 6 GB RTX 4050)
- Rust MSVC toolchain
- Visual Studio Build Tools with the C++ workload

Model weights are downloaded to the Hugging Face cache on first use.

## Quick start

From the `onyx_cuda` directory, create a virtual environment and install the
CUDA build of PyTorch before installing the package:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Start the API server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn onyx_cuda.server:create_app --factory --host 127.0.0.1 --port 8000
```

The first startup may take a while while the draft and target models download.

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
| `max_tokens` or `max_completion_tokens` | Positive output-token budget; default 256. Send only one name. |
| `temperature`, `top_p`, `seed` | Greedy decoding at temperature 0; target-model sampling above 0. `top_p` and `seed` apply to sampling. The seed is an integer in PyTorch's supported range, −2⁶³ through 2⁶⁴−1. |
| `response_format` | `{"type":"text"}`, or `json_schema` as shown below. |
| `json_schema`, `regex` | Onyx constraint extensions. Choose only one of these or `response_format`. |
| `stop` | One nonempty string or a list of up to four. Supported for text/regex output; JSON completion is determined by its schema. |
| `n` | Positive number of sequential choices; streaming supports only 1. A fixed seed is reused for each choice. |
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
.\.venv\Scripts\python.exe -m pytest
```

Run the benchmark commands as needed:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --constraints
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --target
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --speculative
```

Benchmark results are written under `benchmarks/results/` and are ignored by
Git.

## Current limitations

- The model pair is currently fixed to Qwen2.5 0.5B and 1.5B Instruct.
- Speculative decoding is greedy; positive-temperature requests use the target
  model directly.
- Beam search, batching, and repetition penalties are not implemented.
- CUDA is required. Model loading fails rather than falling back to CPU.

## Project layout

- `src/onyx_cuda/` — model loading, generation, speculative decoding, and API
- `rust/` — native regex and JSON Schema grammar engine
- `tests/` — unit and API tests

## License

Onyx is available under the [MIT License](../LICENSE).
