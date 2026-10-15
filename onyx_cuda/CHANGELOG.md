# Changelog

## 0.1.2 - 2026-10-14

A fix for schema strings that combine `pattern` with `minLength`, and the
license text in the source package. Models, defaults, and request fields are
unchanged.

### Fixes

- A schema `pattern` with a `minLength` now allows only characters after which
  the string can still match at an allowed length. For `^(A|BBB)$` with
  `minLength: 2`, generation could previously write `A`, which matches but is
  too short and cannot grow, and then end with an error. A `pattern` that
  cannot match within its `minLength` and `maxLength`, such as `^[A-Z]{3}$`
  with `minLength: 5`, now returns HTTP 400. This removes the known limitation
  listed for 0.1.1.

### Packaging

- The source distribution now includes the MIT license text, and the
  package README links to it. Before, the package declared the MIT license
  but shipped without the file.

### Validation

- The full suite passed all 902 tests on a Linux RTX 2080 Ti (22 GiB) on
  October 14, 2026, with 53 native Rust tests. A test review removed checks
  that could not fail and one duplicate test; no coverage changed.
- Greedy output that completed before is unchanged. The benchmark was not
  rerun.

### Known limitations

- The 0.1.0 limitations still apply. The accepted request fields they refer
  to are listed in `docs/api.md`.

## 0.1.1 - 2026-10-13

Fixes for requests that could break the server or never finish, and a shorter
README. Models, defaults, and request fields are unchanged.

### Fixes

- A generation result's `past_key_values` now always holds the prompt and every
  returned token but the last. A stop sequence, or during speculation an
  accepted EOS or grammar-completing draft token, previously left extra tokens
  in it. Output tokens were never affected, and the API does not expose the
  cache.
- A positive temperature below about 1e-37 no longer produces NaN sampling
  probabilities. On CUDA these tripped a device-side assert that left every
  later request in the process failing until a restart.
- A `regex` now allows only tokens after which a match is still reachable. A
  branch that can never match, such as `b$c` in `a|b$c`, previously failed
  partway through generation, and a regex that can never match, such as `a$b`,
  failed only after the prompt prefill; it now returns HTTP 400 up front.
- A schema `pattern` without `maxLength` now allows only characters after which
  the string can still match. A string such as `"X` for the pattern `^AB` could
  previously never close, so generation ran to `length`. A `pattern` that can
  never match, or cannot match within its `maxLength`, now returns HTTP 400
  instead of generating until `length`.

### Documentation

- The CUDA README is now a short overview: the demo, one results table, a new
  "How it works" section, setup, and a first request. The API reference moves
  to `docs/api.md`, and benchmark method, history, and validation move to
  `docs/performance.md`.
- The API reference lists the supported JSON Schema keywords and which request
  errors return HTTP 400 rather than 422.
- The performance page breaks the 48-case benchmark down by workload category
  for both recovery modes, with each category's prompts.

### Cleanup

- Removed the unused `fallback_reason` from graph recovery, left over from
  when running out of memory closed the recovery graphs.

### Validation

- The full suite passed 881 tests on a Linux RTX 2080 Ti (22 GiB) on
  October 12, 2026, with 51 native Rust tests.
- Greedy output that completed before is unchanged. The benchmark was not
  rerun.

### Known limitations

- The 0.1.0 limitations still apply.
- A `minLength` longer than any string a `pattern` can match is not rejected
  up front, and generation ends with an error when it reaches that point.

## 0.1.0 - 2026-10-08

First tagged release of the NVIDIA CUDA implementation.

### Features

- Regex and JSON Schema constraints enforced during generation by a native Rust
  engine. JSON Schema support is a checked subset; completed JSON is validated
  again before it is returned.
- Grammar-aware speculative decoding with a Qwen2.5-0.5B-Instruct draft and a
  Qwen3-8B target in FP16, three draft tokens per round. Greedy output matches
  the target running alone token for token, including the finish reason.
- CUDA graphs for draft steps, and graph-based numerical recovery on the
  validated GPU.
- An OpenAI-style `/v1/chat/completions` API with SSE streaming, `/v1/models`,
  and a status endpoint at `/`. Responses carry `onyx_metrics`, and
  `"speculative": false` runs the target alone for side-by-side comparisons.
- A `/demo` race page that plays target-only and speculative runs side by side.
- `python -m onyx_cuda.benchmark` for speed comparisons, and `preflight` and
  `validate_model` for qualifying other models.

### Validation

- The full suite passed 848 tests on a Linux RTX 2080 Ti (22 GiB) on
  October 6, 2026, with 48 native Rust tests.
- Windows CI builds the package and runs the Rust and non-GPU Python tests.
- Across 48 benchmark cases, speculation ran 1.492x faster than the target alone
  in aggregate (1.370x with scalar recovery), with identical output.

### Known limitations

- The default model pair needs about 22 GiB of VRAM. Quantization and CPU or
  disk offload are not supported.
- Graph recovery is qualified only on a Linux RTX 2080 Ti; elsewhere it falls
  back to scalar recovery. Draft graphs have not been validated on a Windows GPU.
- Positive temperature samples from the target alone, without speculation.
- The API is not a drop-in OpenAI replacement: unsupported request fields
  return HTTP 422. See the README for the accepted fields.
- One inference worker serves trusted local clients; there is no
  authentication.
- The `/demo` page loads its fonts from Google Fonts and falls back to system
  fonts offline.
- No prebuilt wheels are published. Source builds need Rust, plus the MSVC
  toolchain on Windows.
