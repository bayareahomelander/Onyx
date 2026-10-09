# Onyx CUDA API reference

The server from the [README](../README.md#setup) serves
`POST /v1/chat/completions`, `GET /v1/models`, a status endpoint at `GET /`, and
the `/demo` race page. See the [README example](../README.md#example) for a first
request.

## Requests

The API follows OpenAI's request and response shapes but is not a drop-in
replacement. Messages take plain-string `content` with the `system`, `user`, and
`assistant` roles. Besides `model` and `messages`, requests accept `max_tokens`
(or `max_completion_tokens`), `temperature`, `top_p`, `seed`, `n` (up to four,
without streaming), `stop`, `stream`, and `response_format` (`text` or
`json_schema`), plus Onyx's `regex`, `json_schema`, `compact_json`,
`enable_thinking`, and `speculative`. Any other field, such as `tools`,
`stream_options`, or `presence_penalty`, returns HTTP 422 instead of being
ignored.

The API allows up to 4096 output tokens within an 8192-token prompt-plus-output
budget by default; see [Settings](#settings) to change the limits.

## Constraints

Use `regex` for a regular expression and `json_schema`, or `response_format` with
`type: "json_schema"`, for JSON. Constrained output ends with
`finish_reason: "stop"` once it fully matches and either cannot be extended or the
model chooses to end it, so open-ended patterns such as `[0-9]+` are not cut off
at their first match. Partial output has `finish_reason: "length"` and may not
satisfy the constraint. Custom `stop` strings cannot be combined with regex or
JSON Schema constraints; these combinations return HTTP 422.

JSON Schema support is a checked subset: `type` (a name or a list of names),
`properties`, `required`, boolean `additionalProperties`, `items`, `enum`,
`pattern`, `minLength`/`maxLength`, and `minItems`/`maxItems`, plus the
annotations `title`, `description`, `default`, `examples`, and `$comment`. Other
keywords, such as `$ref`, `oneOf`, `format`, or `minimum`, are rejected rather
than ignored. Generated objects contain only declared properties, however
`additionalProperties` is set. A schema `pattern` matches anywhere in the
string, as JSON Schema specifies, while a `regex` constraint must match the
whole output.

## Streaming and metrics

Set `stream: true` for SSE. For live output on Linux, pass `-N` to `curl`. In
PowerShell, set `stream = $true` before converting the body to JSON:

```powershell
$body | curl.exe --no-buffer http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@-"
```

Generated text is in `choices[0].delta.content`. For constrained JSON, wait for
a successful `stop` finish event; `[DONE]` alone is insufficient. The finishing
chunk also carries `usage` and `onyx_metrics`, as non-streaming responses do:
time to first token, total generation time, decode rate, whether speculation ran,
and for speculative requests the acceptance rate and verification rounds.

## Comparing modes

To compare both modes on one server, send `"speculative": false` to run the
target alone for that request; omitting the field follows the server setting.
`"speculative": true` returns HTTP 422 when the server runs the target alone
(`ONYX_SPECULATIVE_GAMMA=0`) or the request samples with a positive temperature.
The `/demo` page uses this to race both modes on a preset prompt.

## Errors

Unknown fields, unsupported option combinations, and token budgets beyond the
limits return HTTP 422. A regex that
does not compile, an unsupported JSON Schema, or an unknown `model` returns
HTTP 400. Requests beyond the active-request limit receive HTTP 429. Request
errors use FastAPI's `detail` body rather than OpenAI's `error` object; an error
during a stream arrives as an `error` object followed by `[DONE]`.

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
verifies every proposed token, so they change speed, not output. Set
`ONYX_DRAFT_BACKEND=eager` to use the ordinary draft forward.

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
