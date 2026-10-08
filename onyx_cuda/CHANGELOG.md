# Changelog

## Unreleased

### Fixes

- A generation result's `past_key_values` now always holds the prompt and every
  returned token but the last. A stop sequence, or during speculation an
  accepted EOS or grammar-completing draft token, previously left extra tokens
  in it. Output tokens were never affected, and the API does not expose the
  cache.

### Documentation

- The README lists the supported JSON Schema keywords and which request errors
  return HTTP 400 rather than 422.

### Cleanup

- Removed the unused `fallback_reason` from graph recovery, left over from
  when running out of memory closed the recovery graphs.

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
