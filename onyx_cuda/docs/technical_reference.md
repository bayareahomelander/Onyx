# Onyx for Windows technical reference

This document records the architecture, contracts, implementation boundaries, and measured
qualification state of the `onyx_cuda` package. For installation and first use, start with the
[project README](../README.md).

## Scope and isolation boundary

Onyx for Windows is the independently maintained NVIDIA CUDA variant of Onyx. All Windows-owned
Python, Rust, package metadata, tests, qualifiers, and documentation live under `onyx_cuda/`.
Windows development does not import from or modify the Apple MLX execution layer in `onyx/`.

The package uses a src layout and the `onyx_cuda` import namespace. Normal `import onyx_cuda` is
side-effect free with respect to optional runtimes: it does not import MLX, PyTorch, Transformers,
bitsandbytes, Accelerate, Hugging Face Hub, Tokenizers, psutil, ONNX Runtime, or the native grammar
extension; it does not initialize CUDA, load a model, access the network, or read tokenizer assets.
Optional runtimes load only through explicit factories.

The selected production stack is:

- PyTorch 2.6.0+cu124 for CUDA tensor execution and allocator measurements;
- Transformers 4.57.6 for causal-model integration and `DynamicCache`;
- bitsandbytes 0.49.2 for 4-bit NF4 double quantization;
- Accelerate 1.14.0 for model-loading integration;
- Hugging Face Hub 0.36.2 and Tokenizers 0.22.2 for framework-independent tokenizer loading;
- psutil 7.2.2 for bounded candidate-qualification process measurements;
- an independent Maturin/PyO3 Rust extension for regex and JSON grammar execution.

## Framework-neutral inference boundaries

### Autoregressive backend

`AutoregressiveBackend` defines the minimum target-only contract. A backend exposes `model_id`,
`vocab_size`, and `cache_length`; starts a sequence with `prefill(prompt_token_ids)`; consumes one
selected token with `decode(token_id)`; and clears active state with `reset()`.

Both `prefill()` and `decode()` return an immutable `ModelStep` containing backend-native next-token
logits and the logical cache length that produced them. Tensor types remain outside the protocol, so
the generation algorithm is not coupled to PyTorch, MLX, or another execution framework.

`generate_target()` and its cancellable iterator implement target-only generation over this
boundary. Token selection is injected so CUDA implementations can reduce logits on-device instead
of copying a vocabulary-sized row to the CPU. The final selected token is not appended to the cache
when no later logits row is required; the reported final cache length therefore normally equals the
prompt length plus all sampled tokens except the last.

`onyx_cuda.testing.FakeAutoregressiveBackend` supplies deterministic scripted logits and exact cache
transitions for model-free tests.

### Cache checkpoints

`CheckpointableAutoregressiveBackend` is an optional capability layered beside the minimum backend.
Its opaque `CacheCheckpoint` identifies one exact active-sequence prefix. The public operations are:

- `create_cache_checkpoint()`;
- `rollback_cache(checkpoint)`;
- `release_cache_checkpoint(checkpoint)`.

The contract fixes owner, sequence-epoch, allocation, direction, lifetime, and exact-prefix rules.
Checkpoints at the rollback position and earlier remain reusable; checkpoints in a discarded suffix
become invalid. Same-position rollback is successful and idempotent. Release is cache-neutral and
idempotent for a valid same-owner handle. Reset, sequence replacement, and terminal backend failure
invalidate the affected epoch. Invalid requests must be failure-atomic.

The fake backend proves exact token-prefix and scripted-logit restoration, alternative suffixes,
nested and same-position handles, stale/foreign/fabricated/released rejection, repeated epochs, and
bounded registry reuse without requiring a model or GPU.

### Draft proposal

`generate_draft_proposal(backend, current_token_id, *, proposal_length, select_token)` is a
framework-neutral operation over an already-prefilled `CheckpointableAutoregressiveBackend`. It
returns a frozen, slotted `DraftProposalResult` with exactly four fields:

- `proposal_token_ids`, the exact nonempty ordered token tuple;
- `rollback_checkpoints`, one opaque checkpoint per proposal token;
- `initial_cache_length`; and
- `final_cache_length`.

For starting cache length `P`, positive caller-supplied proposal length `n`, and proposal
`(d0, ..., d[n-1])`, the operation decodes the uncached current token, selects `d0` from that row,
then decodes and selects in strict autoregressive order. It calls the borrowed selector exactly
`n` times and `decode()` exactly `n + 1` times. The final decode consumes `d[n-1]` and validates its
post-proposal row without selecting from or retaining that row. A successful call leaves the cache
at `P + n + 1` with the exact suffix `(current_token, *proposal_token_ids)`.

Checkpoint `rollback_checkpoints[k]` records length `P + 1 + k`, immediately before `d[k]` is
selected. Rejection after accepting `k < n` proposal tokens rolls back to that handle and restores
the exact prefix through `proposal_token_ids[:k]`. Full acceptance needs no final checkpoint because
the cache already contains the complete suffix. The caller owns every returned handle and must
release all of them after either rollback or full acceptance; release remains idempotent for handles
that a rollback invalidated as discarded suffixes.

D32 creates a separate private checkpoint at `P` solely for operation cleanup. It releases that
handle before returning success. Any failure after its creation first attempts exact rollback to
`P`, then releases every operation-owned rollback handle and the private handle without resetting
the caller's sequence. Cleanup preserves pre-existing caller checkpoints and the original failure
when restoration succeeds. Checkpoint allocation IDs are deliberately not rewound, so a failed call
cannot make an old handle alias a later allocation.

The operation borrows one selector callable for its complete lifetime; it does not create, seed,
clone, or reset a selection session. A stateful selector therefore continues its caller-owned state,
while exact seeded replay uses backend rollback plus a fresh same-seed session. The result retains
only token IDs and opaque checkpoint capabilities: it contains no logit row, probability evidence,
selector or RNG, backend or model, target result, grammar state, tensor, or metric.

The deterministic fake proves row identity and alignment, exact token/cache transitions, every
rejection position, caller-owned replay and alternative selection, cleanup aggregation, monotonic
allocation identity, repeated epochs, bounded reuse, and optional-runtime-free import behavior.

D34 qualifies the existing pinned `TorchCUDATargetBackend`, loaded through
`load_torch_cuda_target()`, as a direct producer for this one isolated operation on `cuda:0`. No
draft-specific adapter, subclass, factory, lifecycle owner, or package export is added: `draft`
describes the backend's role for the call, not a selected release model. Every proposal uses the
ordinary one-token decode path with `logits_to_keep=1`; the selector receives the first `n` cropped
native FP16 CUDA rows directly, while the final decoded row is validated but not selected or
retained. The active `DynamicCache` object and D29 layout remain unchanged, and the returned
checkpoints map exactly to rejection after zero through `n-1` accepted proposal tokens.

Greedy replay borrows `select_cuda_argmax` directly. Seeded temperature/top-p replay rolls the
backend back to a caller-owned root and creates a fresh same-policy, same-seed CUDA sampler for the
new operation; an already-consumed sampler is not treated as replayable state. Unsupported profiles
fail at private-start checkpoint creation before decode or selection. A selector exception while
the backend remains healthy restores the private start, preserves an external root, releases all
D32-owned handles, and leaves the backend reusable. A terminal production decode failure instead
clears the backend epoch; D32 reports `DraftProposalCleanupError` containing the original typed
backend failure and the failed stale-checkpoint rollback rather than weakening that safe-empty
policy.

D30/D31 batched target verification and the D33 match/replace decision are regression boundaries,
not inputs to D34, and are never invoked by this path. There is still no selected two-model pair,
separate production draft engine, cache coordinator, acceptance loop, or grammar-state integration;
this isolated qualification is not speculative decoding.

### Batched target verification

`BatchedTargetVerificationBackend` is a separate optional capability layered beside the minimum
backend. Its positional `verify_proposal(current_token_id, proposal_token_ids)` operation evaluates
one uncached current token plus a nonempty proposal as one ordered backend batch. It returns a
frozen, slotted `BatchedTargetVerificationResult` containing an exact nonempty tuple of
backend-native logit rows and the resulting logical cache length. The result neither converts the
rows nor carries an acceptance decision, selected token, checkpoint, grammar state, sampling state,
metric, device, or framework metadata.

For starting cache length `P` and proposal `(d0, d1, ..., d[n-1])`, where `n >= 1`, the ordered input
is `(current_token, d0, d1, ..., d[n-1])`. A successful operation returns exactly `n + 1` rows and
advances the cache to `P + n + 1`:

- `r0`, produced after the current-token input, is the target distribution aligned to `d0`;
- for `1 <= i < n`, `ri`, produced after `d[i-1]`, is aligned to `di`; and
- `rn`, produced after `d[n-1]`, is the distribution after the complete proposal.

The result exposes `rn` in native order but assigns it no bonus-token or full-acceptance policy. The
current read-only Mac reference consumes only rows `r0` through `r[n-1]` while judging the proposal;
later Windows acceptance work may use or ignore the final row without changing this backend
contract.

The deterministic fake implements this capability by materializing and validating caller input,
staging all `n + 1` scripted rows, validating their count and vocabulary width, and constructing the
result before committing its exact token suffix, cache length, and script cursor. Invalid input,
inactive state, script exhaustion, malformed rows, and staging or result-construction failures leave
the cache, cursor, epoch, allocation counter, and checkpoint registry unchanged. Existing D28
checkpoints restore the exact pre-batch prefix and row cursor, so replay returns the same immutable
result and discarded-suffix handles retain their established invalidation behavior.

`TorchCUDATargetBackend` also implements the capability for exactly the pinned 0.5B profile on
`cuda:0`. It materializes the proposal once and runs one Transformers forward over the complete
`(current_token, *proposal)` suffix with `logits_to_keep=n+1`. The raw result must have shape
`(1, n+1, 151936)`, FP16 dtype, and `cuda:0` placement. Each returned native row view is cropped to
the 151,665 usable tokenizer IDs, and the same `DynamicCache` must advance from `P` to `P+n+1`
without changing its qualified layout. Exact Python token bookkeeping is committed only after all
rows, cache state, and the immutable result have been validated.

The production method fails closed before execution on every other profile or device. Caller-input
errors, a closed backend, unsupported scope, and verify-before-prefill leave a valid active sequence
unchanged. Corrupt active bookkeeping/layout and every failure after tensor execution begins clear
the cache and checkpoint registry and advance the sequence epoch, matching terminal decode failure
semantics. Returned rows and their parent logits tensor are caller-owned; the backend and checkpoint
registry retain no result, row, logits, or input-tensor reference.

Both `FakeAutoregressiveBackend` and the production backend satisfy this optional protocol. A
minimum backend without `verify_proposal()` remains valid. The contract itself imports no tensor
framework or optional runtime. D31 exposes the final row but does not select or consume it.

### Match/replace acceptance

`decide_match_replace_acceptance(proposal_token_ids, target_logit_rows, *, select_token)` is a pure,
framework-neutral decision over already-produced D32 proposal IDs and the complete D30/D31 target
row tuple. It accepts only those two exact built-in tuples rather than complete
`DraftProposalResult` or `BatchedTargetVerificationResult` objects. Cache lengths and rollback
checkpoints therefore cannot become decision inputs or accidental D33-owned resources.

The package exports `MatchReplaceAcceptanceError`, its
`MatchReplaceAcceptanceInvariantError` subclass, and the frozen, slotted
`MatchReplaceAcceptanceResult`. The result has exactly three fields: `proposal_token_ids`,
`accepted_count`, and `replacement_token_id`. It derives `fully_accepted`, `accepted_token_ids`,
`rejected_proposal_token_id`, `output_token_ids`, and `next_current_token_id` without retaining any
target row or selector.

The proposal must be an exact nonempty tuple of non-Boolean, nonnegative integers. For proposal
length `n`, `target_logit_rows` must be an exact tuple containing exactly `n + 1` opaque rows, and
`select_token` must be callable. All structural validation completes before the first selector
call. Primitive type/value failures use `TypeError` or `ValueError`; invalid row counts and
impossible cross-field result relationships use `MatchReplaceAcceptanceInvariantError`.

Only rows `r0` through `r[n-1]` are decision rows. For each position `i`, D33 passes `ri` directly
to the borrowed selector and accepts `proposal_token_ids[i]` when the selected target token matches.
The first mismatch at position `k` returns the accepted prefix plus that target-selected
replacement and stops immediately. Full acceptance returns the complete proposal after exactly `n`
selector calls:

| Outcome | Selected rows | Calls | Output |
|---|---|---:|---|
| Immediate mismatch | `r0` | 1 | target replacement |
| Middle mismatch at `k` | `r0..rk` | `k + 1` | `proposal[:k] + replacement` |
| Last-position mismatch | `r0..r[n-1]` | `n` | accepted prefix plus replacement |
| Full acceptance | `r0..r[n-1]` | `n` | complete proposal |

The required post-proposal row `rn` is never passed to the selector, including on full acceptance;
D33 defines no bonus-token policy. The previously generated current token is also not emitted
again. `next_current_token_id` is the replacement for a mismatch or the final proposal token after
full acceptance—the last token in `output_token_ids`.

D33 borrows one caller-owned selector/RNG session. It does not create, seed, clone, reset, snapshot,
or rewind that session. Selector exceptions propagate unchanged, and a Boolean, non-integer, or
negative selector return fails at its exact proposal position without a later row call. Any selector
state already consumed remains caller-owned; the operation is deliberately nontransactional with
respect to that external session.

The decision invokes no backend and performs no cache, checkpoint, grammar, metric, or lifecycle
operation. Its retained result contains only the exact proposal tuple and integer outcome metadata.
Deterministic fake integration proves that actual D30 and D32 tuple evidence composes while both
backend snapshots and every caller-owned D32 rollback handle remain unchanged. This boundary has no
production draft integration, cache coordinator, grammar-state composition, or iterative
speculative loop, and by itself is not speculative decoding.

### Tokenizer and text engine

`TokenizerAdapter` exposes `tokenizer_id`, `vocab_size`, `encode(text)`, and `decode(token_ids)`.
`onyx_cuda.testing.FakeCharacterTokenizer` maps one Unicode character to one token ID for exact
model-free tests.

`TargetTextEngine` composes a tokenizer, autoregressive backend, and backend-native selector. It
validates vocabulary compatibility, encodes plain prompt text, generates tokens, removes only the
user-visible terminal suffix, decodes the visible token IDs, and preserves complete sampled-token
metadata for diagnostics.

Prompts are currently plain text. The production tokenizer's stored chat template is fingerprinted
for compatibility checks but is not applied implicitly.

## Production target path

### Pinned default profile

The production default is pinned to:

`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`

`load_qwen_tokenizer()` requests only `tokenizer.json` and `tokenizer_config.json` at that immutable
revision, executes no remote code, and loads the tokenizer through the framework-free Tokenizers
runtime. Tokenizer-only use imports neither PyTorch nor Transformers.

The tokenizer exposes 151,665 usable token IDs. The model exposes 151,936 embedding/output rows, so
the backend crops every returned logits row to the tokenizer boundary and makes the 271 padded
model-only rows unselectable.

### Production grammar vocabulary

`build_qwen_grammar_vocabulary(tokenizer)` constructs an exact token-ID-indexed
`tuple[bytes, ...]` from the pinned `tokenizer.json`. It is lazy, deterministic, framework-neutral,
and uncached.

The builder validates a 151,643-piece BPE model, canonical ByteLevel decoding, disabled byte
fallback, 22 contiguous added-token IDs, and exact agreement between the asset and Tokenizers
runtime. Base pieces use the inverse ByteLevel byte-to-Unicode bijection so partial UTF-8 bytes are
preserved without lossy isolated token decoding.

The 14 special IDs 151643 through 151656 intentionally map to `b""`; the eight non-special added
IDs 151657 through 151664 emit their exact ByteLevel bytes. The canonical four-byte-length-prefixed
vocabulary fingerprint is:

`63ae520f9b74ae136cae96ce06470a10edfd3d5a3ae857d90b64ba8f870345f8`

A manually constructed `QwenTokenizerAdapter` remains valid for encoding and decoding but cannot
build the production grammar vocabulary because it has no authoritative pinned asset source.

### PyTorch CUDA target backend

`load_torch_cuda_target()` loads the pinned target with bitsandbytes 4-bit NF4 double quantization,
FP16 compute and dtype, one explicit CUDA device, and no implicit CPU offload.

`TorchCUDATargetBackend` creates a fresh Transformers `DynamicCache` for each prefill, consumes one
selected token per decode, requests only the final logits row, crops it to 151,665 usable IDs, and
returns an FP16 tensor on the configured CUDA device. It validates logits shape/device and logical
cache length after every forward. `reset()` discards active sequence state; `close()` also releases
backend-owned model and tokenizer references.

For the pinned 0.5B target on `cuda:0`, the same backend can verify a caller-supplied nonempty
proposal in one multi-token forward. It requests all `n+1` native rows, validates their exact raw
and usable shapes, FP16 CUDA placement, cache identity, cache-length transition, and D29 layout,
then records the full input suffix. This is a target primitive only: it does not produce a draft,
judge acceptance, choose a replacement or bonus token, or assign policy to the final row.

`load_production_target_engine()` composes this backend with the pinned tokenizer, CUDA greedy or
seeded sampling, CUDA peak-memory diagnostics, lifecycle-owned streaming, and lazy production
grammar support.

### Production DynamicCache rollback

For the pinned 0.5B profile on `cuda:0`, `TorchCUDATargetBackend` implements the optional checkpoint
contract using the internal mechanism named `transformers_dynamic_cache_native_crop`.

The internal adapter accepts only the measured Transformers 4.57.6 structure:

- one exact `transformers.cache_utils.DynamicCache`;
- 24 initialized `DynamicLayer` instances;
- no offloading, sliding layers, or replicated lazy layer class;
- FP16 PyTorch CUDA key/value tensors on `cuda:0`;
- exact shape `(1, 2, cache_length, 64)` on every layer;
- consistent cache and per-layer logical lengths;
- supported prefix-view strides and zero storage offset.

Rollback first validates the complete active layout and stages the exact cache dictionary,
layer-list reference, layer references, layer dictionaries, and key/value references. It calls
native `DynamicCache.crop(target_length)`, then validates every layer again. A crop exception,
unexpected return, or post-crop invariant failure restores and revalidates the exact original
references and contents before a typed backend error is raised.

Native crop retains the discarded suffix's backing allocation through prefix views. The next
ordinary decode compacts the active prefix through the existing `DynamicLayer` append path before
extension. Checkpoint creation and release themselves are allocation-neutral.

Production checkpoint handles and registry entries contain only CPU metadata: owner, epoch,
epoch-local allocation ID, immutable token prefix, cache length, and a tensor-free layout signature.
They retain no cache, layer, tensor, model, or logits object. Reset, successful replacement, close,
cache-creation failure, terminal forward failure, and terminal invariant failure invalidate the
affected epoch.

This qualification is deliberately narrow. Checkpoint and batched-verification support do not cover
the 3B candidate, another device, sliding/offloaded caches, arbitrary Transformers cache classes,
or a two-model speculative engine. D34 reuses the same narrow checkpoint support for an isolated
proposal role only.

## Selection, stops, and streaming

### Greedy and seeded sampling

`GreedySelection` is the default. `TemperatureTopPSelection(temperature, top_p, seed)` enables
explicit seeded temperature/top-p sampling without choosing release-default sampling values.

Each generation creates a private RNG session. The reference and CUDA implementations guarantee
replay within their own runtime but do not promise identical sequences across different RNG
implementations. CUDA scaling, FP32 softmax, stable nucleus filtering, and categorical selection
remain on-device; vocabulary-sized logits or probabilities are not copied to the CPU.

Negative infinity is valid masked input. All-negative-infinity and NaN support fail explicitly;
positive-infinity entries form equal-probability support before top-p filtering.

### Stop sequences

Stops are ordered token sequences. Matching examines generated-token suffixes only, and caller order
wins when several configured sequences complete together. The complete stop remains in sampled
metadata but is excluded from visible token IDs and text.

The removed `stop_token_ids` keyword is not accepted. A one-token stop is represented as a
one-element sequence, for example `stop_token_sequences=((eos_id,),)`.

### Streaming lifecycle

Target-only streaming emits immutable `TextGenerationDelta` events followed by one
`TextGenerationComplete` containing exactly the same result as equivalent non-streaming generation.
Possible stop prefixes are buffered, and trailing Unicode replacement characters are withheld until
cumulative token decoding becomes stable.

Only one production stream, constrained or unconstrained, may be active at a time. Exhaustion closes
the stream automatically. A consumer that stops early must call `close()` or use the stream as a
context manager so partial cache and grammar state are released. Cancellation resets unfinished
backend and timing state and leaves the production engine reusable.

## Grammar runtime

### Framework-neutral contract and native ABI

`GrammarConstraint` provides explicit independently branchable states with deterministic valid-token
enumeration, state advancement, match/dead queries, idempotent single/bulk release, and reset.
Framework-neutral scripted fakes cover ownership and lifecycle behavior without implementing a
second grammar engine.

The independent Windows Maturin/PyO3 extension is loaded only on explicit request. Runtime version
is `0.1.0`; grammar ABI is `3`. The loader atomically validates the complete regex and JSON symbol,
type, and exception surface before either factory compiles.

The public native factories are deliberately separate and honest:

```python
from onyx_cuda import compile_native_json_schema, compile_native_regex

regex = compile_native_regex((b"a", b"b", b"ab"), "ab")
json_constraint = compile_native_json_schema(
    (b"null", b"true", b""),
    '{"type":["null","boolean"]}',
)
```

No complete public `NativeGrammarCompiler` is exposed.

### Regex semantics

The Windows-owned regex core uses an anchored dense byte DFA from `regex-automata`. Every
well-typed in-range token consumes all its bytes and returns an independent child while preserving
the parent. A rejected continuation becomes a dead, nonmatching child; advancing a dead state
returns another dead child.

Empty-byte tokens are omitted from valid-token results to prevent non-progress sampling, but an
explicit empty-token advance creates an independent child at the same logical DFA state. Match
status is evaluated through the end-of-input transition.

### JSON semantics

The Windows JSON runtime implements the recorded subset used by Onyx: objects, arrays, strings,
numbers, integers, booleans, nulls, required and optional properties, enums, union types, string
patterns and length limits, array length limits and typed items, nested structures, and structural
whitespace.

It strictly rejects unknown or malformed constraints rather than silently weakening them. Parsing
follows RFC 8259, counts string lengths in Unicode code points, validates UTF-8 and surrogate pairs,
matches patterns against decoded Unicode values, accepts structural whitespace after a complete
root, and completes fully consumed enum candidates.

Unlike regex dead-child behavior, a well-formed in-range token that is invalid for a JSON parent
raises `GrammarStateError`, allocates no child, and preserves the parent.

### CUDA grammar-logit mask

`create_cuda_grammar_logit_mask()` creates a stateless, device-bound sparse-valid-index mask for one
exact tokenizer-sized CUDA logits row. The input must be a real floating-point CUDA tensor with
shape `(vocab_size,)`; valid IDs must be a nonempty, strictly increasing, unique tuple of in-range
Python integers.

The result is a distinct tensor on the same device with the same dtype and shape. Allowed values are
preserved bit-for-bit; every disallowed value becomes `-inf`; the input is never mutated. Empty
support is rejected without injecting EOS, and support whose allowed logits are all `-inf` is
rejected before selection.

The measured transport is `sparse_valid_indices`: each call materializes CPU `int64` indices,
transfers them to CUDA, gathers valid logits, creates a fresh `-inf` row, and restores valid values
with `index_copy_`. The mask retains no tensors, valid IDs, grammar state, or RNG.

### Constrained target generation

`generate_constrained()` and `stream_constrained()` compose a fresh native regex or JSON constraint,
the exact Qwen grammar vocabulary, the CUDA mask, existing selectors, ordered stops, and target
generation. Production grammar support is lazy: ordinary construction and unconstrained generation
do not build the byte vocabulary, load the extension, or create the mask.

At each token position, the engine validates the immutable native valid-ID tuple and adds EOS only
when the current grammar state already matches. It masks before selection, verifies membership,
advances exactly once to an independent nondead child, releases the parent, and retains only the
child. A live nonmatching state with no continuation raises `GrammarNoContinuationError`.

Matching alone is not terminal because regex and JSON prefixes may have valid continuations.
Selecting the injected EOS advances its empty-byte transition and finishes as `grammar_complete`.
The EOS stays in sampled metadata but is removed from visible output.

A configured stop is eligible for trimming only when the grammar state immediately before the
complete suffix was matching. Terminal precedence is eligible stop, grammar-completion EOS, then
token limit. A `length` result may end in either a matching or incomplete grammar state.

Successful calls release the final state and reset the fresh constraint. Failure and cancellation
attempt owned-state release, constraint reset, backend reset after model work starts, and metrics
abort; combined cleanup errors retain the original and cleanup failures.

## Metrics

Every successful target-only generation carries immutable `TargetGenerationMetrics`:

- `ttft`;
- `generation_time`;
- `tokens_per_second`;
- `cache_mode`;
- `peak_allocated_vram_bytes`;
- `peak_reserved_vram_bytes`;
- optional `grammar_timing`.

TTFT begins immediately before prefill and ends after the first validated sampled token. Generation
time accumulates active prefill, decode, selection, validation, and terminal work. It excludes model
loading, prompt encoding, final text decoding, event delivery, and time while a stream is suspended
waiting for its consumer. Throughput counts every sampled token, including a matched stop.

The production diagnostics session synchronizes the configured device and resets PyTorch peak
memory counters before prefill, then synchronizes and reads peak allocated/reserved bytes after the
terminal token. Framework-neutral backends report paired `None` values rather than synthetic zeros.
The production cache mode is `transformers_dynamic`.

Constrained results additionally expose `GrammarTimingMetrics` with:

- `compilation_time` for the fresh native factory call;
- `state_scan_time` for uncached valid-token scans;
- `valid_index_transfer_time` for CPU index materialization and completed host-to-device copies;
- `mask_application_time` for gather, fill, scatter, validation, and support checks after transfer.

Scan, transfer, and mask application are subsets of active generation time; they must not be added
to it. Compilation is outside generation time. Unconstrained results report
`grammar_timing is None`.

## Qualification evidence

### Development machine

The current acceptance machine is Windows with an NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB of
dedicated VRAM, compute capability 8.9, and 16 GiB of system RAM. Measurements are evidence for this
configuration, not portable guarantees.

The production 0.5B target has passed greedy and seeded generation, stops, streaming/cancellation,
regex/JSON constraints, metrics, and transactional cache rollback. The D29 rollback qualifier runs
two complete model lifecycles and 200 total rollback/replay cycles. Its synchronized rollback median
is 0.810050 ms; the higher observed peak is 553,419,776 allocated and 801,112,064 reserved CUDA
bytes. Final cleanup returns to the established 8,520,704 allocated / 497,025,024 reserved-byte
process envelope without second-lifecycle growth.

The D31 batched-verification qualifier also runs two complete model lifecycles and 200 total
batch/rollback/replay cycles. Each four-row batch uses one model forward, preserves the exact cache
object and qualified layout, aligns greedy decisions with sequential one-token characterization,
and replays bit-for-bit after rollback. The observed sequential-versus-batched FP16 maximum absolute
differences range from 0.0234375 to 0.025390625, so cross-kernel bitwise equality is not a contract.
The higher D31 peak is 553,419,776 allocated and 801,112,064 reserved bytes; final cleanup again
returns to 8,520,704 allocated / 497,025,024 reserved bytes without second-lifecycle growth.

The D34 production draft-proposal qualifier runs two complete single-model lifecycles and 200 total
proposal/root-rollback/release cycles with a qualification fixture length of three. The fixed
eight-token prompt selects current token `12890`; greedy proposal `(271, 785, 9960)` and fresh-seeded
proposal `(271, 2121, 949)` both replay exactly. Each operation uses four ordinary one-token
forwards and three selector calls, returns rejection checkpoint lengths `(9, 10, 11)`, and leaves
the full cache at length 12. Every rejection prefix, full-acceptance release, selector-failure
recovery, cache identity/layout, and all 24 physical layer prefixes pass exact checks. The unchanged
target-only baseline is `(12890, 271, 785, 9960)` before and after the matrix and across lifecycles.

After warmup, both D34 lifecycles stabilize at 467,202,560 allocated and 803,209,216 reserved bytes,
with 1,326 current allocations and 1,326 active allocations. Lifecycle allocated peaks are
544,899,072 and 553,419,776 bytes; reserved peak is 803,209,216 bytes in both. Post-close cleanup
returns to 8,520,704 allocated / 497,025,024 reserved bytes without second-lifecycle growth.

These measurements do not establish final release context, output, concurrency, or speculative
`gamma` limits.

### Separate 3B candidate

`QWEN_3B_CANDIDATE_PROFILE` pins
`Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1` separately from the
0.5B default. It remains a target-only qualification candidate and is not checkpoint-qualified by
D29.

The candidate tokenizer is exactly compatible with the default tokenizer. Two bounded offline NF4
lifecycles reach a 2,048-token cache ceiling with a 2,010,079,488-byte model footprint and a
2,969,567,232-byte peak VRAM measurement on the acceptance machine. This does not prove simultaneous
0.5B-draft plus 3B-target residency or select a release pair.

The pinned 0.5B model is Apache 2.0 licensed. The pinned 3B candidate uses the Qwen Research License
and is limited to the project's personal, noncommercial portfolio/evaluation scope unless separate
commercial permission is obtained. Model weights are downloaded separately and are not committed,
bundled, or redistributed by the package.

## Packaging and build boundary

`onyx_cuda/pyproject.toml` defines a Maturin mixed Python/Rust package. The native module is private
at `onyx_cuda._grammar_native`, and the independent crate lives under `onyx_cuda/rust`.

The CUDA extra pins the complete validated top-level runtime. PyTorch must be installed first from
the official CUDA 12.4 index because the default package index can otherwise resolve an unusable
CPU-only build or reject the exact local-version requirement.

Build a source distribution with the independent manifest explicit:

```powershell
python -m maturin sdist --manifest-path rust/Cargo.toml
```

Package rules exclude local planning records, implementation plans, root/Mac source, generated
caches, supplied build paths, and extra native binaries. The independent Windows extension has no
path, source, package, or import dependency on the root Rust crate.

## Deliberately not implemented yet

The current Windows package does not yet provide:

- a selected two-model draft/target pair or a separate production draft engine;
- a cache-coordinated iterative speculative loop, production evidence pairing, or final-row/bonus
  policy;
- grammar-state speculation;
- speculative streaming or acceptance metrics;
- fixed or adaptive `gamma`;
- final prompt, output, context, concurrency, or 6 GiB operating limits;
- CPU offload;
- implicit chat-template formatting;
- a FastAPI/OpenAI-compatible server;
- a complete public `NativeGrammarCompiler`;
- native valid-token caching or a persistent CUDA mask workspace.

Those capabilities remain separately sized roadmap work. Production rollback support, the
model-free draft-proposal primitive, the isolated production proposal-role qualification, the
batched-verification contract, and the pure match/replace acceptance decision do not form
speculative decoding without a selected pair and a separately owned cache-coordinated engine.
