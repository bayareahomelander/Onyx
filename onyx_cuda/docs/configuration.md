# CUDA configuration

[Setup and first request](../README.md) · [Benchmarks](benchmarks.md) · [Validation](validation.md)

Run commands from `onyx_cuda` after following the setup guide. Set environment
variables in the server terminal before startup.

Weights load directly onto `cuda:0` through Accelerate. Both models and their
caches must fit in VRAM. Cached prefill processes long prompts in chunks and
keeps only the final position's logits where supported, bounding attention
workspace without discarding context. CPU/disk offloading and quantized loading
are not supported.

## Model selection

Set overrides in the server's terminal **before startup**:

| Variable | Default / purpose |
| --- | --- |
| `ONYX_TARGET_MODEL` | `Qwen/Qwen3-8B` |
| `ONYX_DRAFT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `ONYX_TARGET_REVISION`, `ONYX_DRAFT_REVISION` | Bundled models use pinned revisions; use commit hashes for custom selections |
| `ONYX_SPECULATIVE_GAMMA` | `2` draft tokens per iteration; `0` explicitly selects target-only generation |
| `ONYX_REPLAY_BACKEND` | `scalar` by default; `graph` opts into model-owned CUDA graphs for historical recovery on the supported target |

The target tokenizer defines chat prompts, grammar bytes, decoding, and EOS for
both models. Compatible drafts may omit target-added tokens in unused embedding
slots, but existing token byte meanings and special tokens must agree and logits
widths must match. Every proposed token is verified by the target. This supports
the default Qwen2.5/Qwen3 pair without bypassing compatibility checks.

For another configuration, set model IDs and run [validation](validation.md)
before restarting the server. Available VRAM does not trigger automatic model
selection.

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

## Token selection

The default `torch` selector needs neither CuPy nor a CUDA Toolkit installation.
The optional custom selector requires CUDA Toolkit 12.4 with NVRTC/headers
and CuPy. Install `.[kernels]` and set `ONYX_GREEDY_BACKEND=cuda` to opt in.

## Adaptive decoding (experimental)

Fixed gamma 2 remains the default. Opt into the candidate controller with
`ONYX_SPECULATIVE_MODE=adaptive`, or pass `adaptive=True` to
`generate_speculative` / `generate_speculative_events`. The controller starts
at gamma 2, measures a target-only calibration step, and compares execution
time per committed token over recent rounds. Measured draft cost and acceptance
estimate whether expanding a proposal is worthwhile; a failed expansion can
return to a measured profitable shorter proposal. It can use gamma 1/2/4/8 or pause
drafting entirely. Failed recovery probes back off from 8 to at most 64 target
tokens; successful probes resume speculation. Catch-up uses only accepted
history and its cost is charged to recovery. Decisions use no prompt features
and reset for each generation.

`ONYX_SPECULATIVE_MODE=fixed` preserves numeric gamma settings.
`ONYX_SPECULATIVE_GAMMA=0` always disables speculation, and positive-temperature
requests still use target-only sampling. Adaptive mode uses the same models,
FP16 precision, grammar rules, and streaming path. Its diagnostic counters are
available in `result.timings.adaptive_stats`, including when `measure=False`;
detailed stage profiling still requires `measure=True`.

## Numerical recovery

Both fixed and adaptive verification now guard ambiguous FP16 token scores.
When scores are close at the dtype's rounding scale, a separate target cache
replays the accepted prefix with single-token forwards and rechecks the batch.
The checkpoint preserves the original prompt cache, advances monotonically, and
reuses subsequent clean target-only work. For supported full-attention dynamic
caches, independent cache containers share unchanged tensor storage; other
cache types use independent copies. Repair stops at the first rejected proposal
instead of evaluating its discarded suffix. Its execution cost is included in
adaptive decisions. It can require an additional target KV cache, including on
requests that never need repair. A replay that discovers an already
emitted noncanonical token fails explicitly. The rounding-scale trigger is not
a proof of a universal numerical error bound; exact corpus comparison remains
a release requirement.

With `measure=True`, `result.timings.replay_stats` separates historical token
catch-up, proposal replay, skipped proposal positions, and time spent in replay
stages and checkpoint snapshots. The adaptive benchmark preserves this field
for both fixed and adaptive runs when invoked with `--measure`; use ordinary
unprofiled runs for performance comparisons.

## Optional graph recovery

Set `ONYX_REPLAY_BACKEND=graph` before starting the server, or pass
`create_app(replay_backend="graph")`. Startup prepares reusable graphs once for
the target model. The root endpoint reports the requested and active backend,
setup time, and the reason for scalar fallback on an unsupported target.

This backend supports the pinned Qwen3-8B FP16 target with full SDPA attention,
default rotary embeddings, PyTorch 2.6.0, Transformers 4.57.6, and CUDA capability
7.5. It was validated on an RTX 2080 Ti with 22 GiB. Other configurations retain
scalar recovery. Explicit graph setup errors fail startup after releasing partial
graph state. The default remains `scalar`; no graphs are built during a request.

Graph recovery consumes already-known **unconstrained historical tokens** in
blocks of two or three. Ordinary speculative verification, its ambiguity guard,
proposal recovery, and constrained requests keep their existing paths. Each
block checks all next-token IDs. A mismatch discards the tentative cache and
returns to scalar recovery for that generation; an invalid emitted prefix still
raises. If graph recovery exhausts CUDA memory, its tentative cache is discarded
and its model-owned graphs are released; subsequent recovery stays scalar until
explicitly prepared again. The root endpoint reports that change. `replay_stats`
adds `chunked_history_tokens` and `graph_replay_fallbacks`.

For programmatic use, prepare the loaded target explicitly and release its graphs
when finished. Keep weights and model configuration unchanged while prepared:

```python
from onyx_cuda.model import load_model_pair
from onyx_cuda.replay_backend import prepare_replay_backend, close_replay_backend

pair = load_model_pair()
configuration = prepare_replay_backend(pair.target.model, "graph")
try:
    # Existing generate_speculative / generate_speculative_events calls using
    # this target automatically use the prepared recovery backend when eligible.
    ...
finally:
    close_replay_backend(pair.target.model)
```

Graph buffers belong to the target, are serialized across CUDA streams, and are
released by server shutdown. Normal model forwards and the global attention
registry are not patched. Closing a generation stream releases its request
caches; model-owned graphs remain available for the next request.

Graphs retain additional VRAM alongside model weights and recovery caches.
See [benchmark results and capacity checks](benchmarks.md) for measured costs.
