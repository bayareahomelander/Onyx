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

### One-iteration speculative coordination

`coordinate_speculative_iteration(...)` composes D32 proposal generation, D30 target verification,
and D33 match/replace acceptance exactly once over two distinct, already-prefilled backend roles.
Its public signature requires a checkpointable draft, a separately checkpointable and
`BatchedTargetVerificationBackend` target, one common uncached current token, a positive proposal
length, borrowed draft and target selectors, and one caller-owned root checkpoint for each role:

```python
coordinate_speculative_iteration(
    draft_backend,
    target_backend,
    current_token_id,
    *,
    proposal_length,
    draft_select_token,
    target_select_token,
    draft_root_checkpoint,
    target_root_checkpoint,
)
```

Both backends must report the same positive initial cache length `P` and the same positive
vocabulary size. This layer proves numeric token-domain compatibility only; semantic tokenizer
identity remains a later production-pair responsibility. Each root must report `P`. Before D32
begins, same-position rollback validates the roots' actual owner, epoch, allocation, lifetime, and
canonical state without changing either logical prefix. The coordinator borrows the roots and never
releases them.

For proposal `(d0, ..., d[n-1])`, D32 runs once and leaves the draft at `P + n + 1`; D30 runs once
over the exact proposal tuple and leaves the target at the same length; D33 runs once over the exact
complete `n + 1` target-row tuple. The coordinator does not inspect, copy, convert, or retain any
row, and it never invokes the target selector independently of D33. The post-proposal row `rn`
remains unused.

On a mismatch after `k` accepted tokens, the draft rolls back through D32 checkpoint `k` to
`P + 1 + k`. The target rolls back to its root at `P`, then ordinary one-token `decode()` calls
rebuild exactly `(current_token_id, *proposal[:k])`. Replay rows are validated only as `ModelStep`
cache evidence and are never selected. Both roles finish with the exact common
`(prompt, current_token_id, *proposal[:k])` prefix; the target-selected replacement is returned as
the uncached continuation and is absent from both caches.

On full acceptance, neither role is rolled back or replayed. Both retain the complete
`(prompt, current_token_id, *proposal)` prefix at `P + n + 1`. No bonus token is selected, and the
result exposes no uncached continuation. This is a completed bounded transaction, not an iterative
handoff.

The frozen, slotted `SpeculativeIterationResult` has exactly five fields:

- `proposal_token_ids`;
- `accepted_count`;
- `replacement_token_id`;
- `initial_cache_length`; and
- `final_cache_length`.

It derives `fully_accepted`, `accepted_token_ids`, `rejected_proposal_token_id`,
`output_token_ids`, and `uncached_next_token_id`. The latter is the mismatch replacement or `None`
after full acceptance. There is deliberately no D35 `next_current_token_id` alias because the final
proposal token is already cached after full acceptance. The result retains no backend, checkpoint,
selector, row, model, tensor, grammar state, metric, or mutable collection.

D35 assumes ownership of every D32 rejection checkpoint as soon as the proposal returns. Success
releases those handles in proposal order while leaving both caller roots active. Any failure after
D32 begins attempts draft-root rollback, target-root rollback, and every D32-handle release in that
deterministic order. Healthy cleanup re-raises the original D28-D33, selector, or backend exception
unchanged. `SpeculativeIterationCleanupError` retains the original failure plus an immutable ordered
tuple of every cleanup failure. Terminal backend failure remains honest: a stale root is reported
instead of being hidden by `reset()`, while the healthy peer is still restored and remaining
cleanup continues.

The model-free D35 proof uses deterministic dual backends. It covers every mismatch position, full
acceptance, exact replay and cache equality, root/handle ownership, selector and row non-retention,
failure aggregation, bounded reuse, repeated epochs, and optional-runtime-free import. D35 adds no
production model pair, loader, iterative loop, grammar-state policy, stops, streaming, speculative
metrics, fixed `gamma`, bonus/final-row policy, offload, operating limit, or API behavior.

### Post-iteration continuation decision

`decide_post_iteration_continuation(...)` is the separate D37 decision that derives exactly one
uncached continuation token from an exact proposal, the complete D30 target-row tuple, and the
corresponding D33 acceptance result:

```python
decide_post_iteration_continuation(
    proposal_token_ids,
    target_logit_rows,
    acceptance_result,
    *,
    vocab_size,
    select_token,
)
```

The package exports `PostIterationContinuationError`, its
`PostIterationContinuationInvariantError` subclass, and the frozen, slotted
`PostIterationContinuationResult`. The result has exactly two fields:
`output_token_ids` and `uncached_next_token_id`. The output is the complete sequence newly emitted
for the accepted proposal outcome, and its final element is always the one uncached token handed to
a later iteration. The previously generated current token is never emitted again.

The result constructor requires an exact, nonempty built-in output tuple of non-Boolean,
nonnegative integers and a non-Boolean, nonnegative uncached token equal to the tuple's final
element. It retains no proposal, acceptance count, replacement/bonus distinction, vocabulary
bound, target row, acceptance object, selector or RNG, backend, checkpoint, cache length, model,
tensor, grammar state, metric, or mutable collection. Because the result deliberately retains no
vocabulary bound, direct construction can enforce nonnegativity but not an upper bound.

The decision accepts an explicit positive integer `vocab_size`. D30 rows are opaque and D33 results
do not carry a vocabulary upper bound, so D37 does not infer one from `len(row)` or change either
earlier contract. The scalar bound validates proposal IDs, a mismatch replacement, and a selected
full-acceptance bonus against `[0, vocab_size)`; it is not retained and proves numeric range only,
not semantic tokenizer compatibility.

Before any new selector call, D37 validates:

- an exact, nonempty built-in proposal tuple containing only non-Boolean integers in range;
- a positive, non-Boolean integer vocabulary size;
- an exact built-in target-row tuple containing exactly `n + 1` entries;
- a `MatchReplaceAcceptanceResult` whose raw proposal tuple equals the supplied proposal;
- a raw accepted count within `[0, n]` and the exact replacement relationship for that outcome; and
- a callable borrowed selector, including on a mismatch where it will not be invoked.

D37 recomputes those relationships from D33's three stored fields rather than trusting derived
properties. Full acceptance requires no replacement. A mismatch requires one in-range,
non-Boolean integer replacement different from the rejected proposal token. Primitive caller-shape
and scalar failures use `TypeError` or `ValueError`; impossible row-count, mixed-evidence,
acceptance, and result relationships use `PostIterationContinuationInvariantError`.

The outcome map is:

| Outcome | D33 output | D37 selected row | New calls | D37 output | Uncached token |
|---|---|---|---:|---|---|
| Mismatch at `k` | `proposal[:k] + (replacement,)` | none | 0 | unchanged D33 output | `replacement` |
| Full acceptance | `proposal` | only `r[n]` | 1 | `proposal + (bonus,)` | `bonus` |

On mismatch, D37 does not access, inspect, compare, copy, or pass any row element. It reuses the
validated D33 replacement and returns the exact accepted proposal prefix plus that token. On full
acceptance, it passes `target_logit_rows[-1]` directly and by identity to the borrowed selector
exactly once. Decision rows `r0` through `r[n-1]` are never selected again, and a bonus equal to the
last proposal token remains valid.

The caller owns one selector/RNG session across D33 and D37. D37 does not create, seed, clone,
snapshot, reset, or rewind it. A final-row selector exception propagates as the same exception
object. A Boolean, non-integer, negative, or out-of-range return fails after that one draw, and
result-construction failure likewise propagates after the required call. D33 deliberately retains
no selector identity, so supplying the same session to both operations is a caller responsibility
rather than a runtime-detectable relationship.

D37 rejects detectable evidence mixing: unequal proposal values, malformed acceptance fields,
invalid row count, and token evidence outside the configured numeric domain. It cannot distinguish
rows from a different D30 call with the same proposal length, an acceptance result from another
call with identical token values, or semantically incompatible tokenizers. Adding call provenance
would require a separate D30/D33 contract change.

The operation invokes no backend and performs no cache, checkpoint, grammar, metric, model, tensor,
or lifecycle work. Fake-backend composition proves that real D30 rows and D33 results produce every
mismatch and full-acceptance outcome while cache tokens, length, script cursor, epoch, checkpoint
registry, and allocation counter remain unchanged. The proof also covers selector state ownership,
failure consumption, weak-reference non-retention, bounded reuse, and isolated optional-runtime-
free imports.

D35 remains behaviorally unchanged: it still has its eight-parameter signature and five-field
result, reconciles mismatch caches exactly as before, leaves a mismatch replacement uncached, and
returns no uncached continuation after full acceptance while never selecting `r[n]`. D37 is not
called by D35, does not insert the bonus into either cache, and is not an iterative speculative
engine or user-visible speculative decoding.

### Continuation-aware one-transaction coordination

`coordinate_continuation_aware_speculative_iteration(...)` is the additive D38 integration of D32
proposal, D30 verification, D33 acceptance, the D35 cache-outcome rules, and the D37 continuation
decision. It operates once over two distinct, already-prefilled checkpointable roles:

```python
coordinate_continuation_aware_speculative_iteration(
    draft_backend,
    target_backend,
    current_token_id,
    *,
    proposal_length,
    draft_select_token,
    target_select_token,
    draft_root_checkpoint,
    target_root_checkpoint,
)
```

The inputs, root qualification, and shared numeric-vocabulary requirement are identical to D35.
Both roles must begin at the same positive cache length `P`, report the same positive vocabulary
size, and accept the current token in that range. The roots must identify the current state of
their respective backend. Same-position draft-then-target rollback proves their owner, epoch,
allocation, canonical state, and lifetime before proposal work begins. Semantic tokenizer identity
remains a later production-pair responsibility.

The transaction order is fixed:

1. D32 produces the exact proposal and transfers its rejection handles.
2. D30 verifies `(current_token_id, *proposal)` once and returns exactly `n + 1` opaque rows.
3. D33 selects decision rows in order through the borrowed target-selector session.
4. The coordinator completes and validates the D35 cache outcome.
5. D37 receives the exact proposal, row tuple, D33 result, shared vocabulary size, and the same
   target-selector object.
6. The coordinator validates the composed D37 evidence, constructs its minimal result, releases
   the D32 handles in proposal order, and revalidates both final cache lengths.

Cache reconciliation deliberately precedes D37. A cache rollback, replay, or final-length failure
therefore cannot consume the full-acceptance bonus draw. The coordinator never inspects, converts,
stacks, copies, infers a vocabulary from, or retains an individual target row. D33 and D37 alone
pass the applicable exact row objects to the selector.

The outcome map is:

| Outcome | Target-selector rows | Target replay | Final common cache | Newly emitted output | Only uncached token |
|---|---|---|---|---|---|
| Mismatch after `k` accepted | D33 uses `r0..rk`; D37 adds no call | `(current, *proposal[:k])` | `P + 1 + k` | `proposal[:k] + (replacement,)` | replacement |
| Full acceptance | D33 uses `r0..r[n-1]`; D37 then uses only `r[n]` | none | `P + n + 1` | `proposal + (bonus,)` | bonus |

The current token is consumed into both caches but is not emitted again. On mismatch, the draft
rolls through D32 checkpoint `k`; the target rolls to its caller root and replays the exact accepted
prefix. The replacement remains outside both caches. On full acceptance, both complete proposal
suffixes remain cached and the bonus remains outside both caches. A bonus equal to the final
proposal token is valid: only its newly emitted occurrence is uncached.

The frozen, slotted `ContinuationAwareSpeculativeIterationResult` has exactly six fields:

- `proposal_token_ids`;
- `accepted_count`;
- `replacement_token_id`;
- `initial_cache_length`;
- `final_cache_length`; and
- `uncached_next_token_id`.

It derives `fully_accepted`, `accepted_token_ids`, `rejected_proposal_token_id`, and
`output_token_ids`. The output always ends with `uncached_next_token_id`; on mismatch that token
must equal the replacement, while on full acceptance it is the bonus. Direct construction enforces
exact built-in tuple and scalar relationships, nonnegative token IDs, positive initial length, and
the exact outcome-specific final length. The result does not retain a vocabulary bound, so only the
coordinator enforces the operation-time upper bound. It retains no backend, checkpoint, target row,
verification/acceptance/continuation result, selector/RNG, model, tensor, grammar state, metric, or
mutable collection.

The caller continues to own both roots; D38 never releases them or creates the next iteration's
roots. It assumes ownership of every D32 rejection handle and releases those handles in proposal
order on success. Once D32 begins, verification, acceptance, reconciliation, D37 selection or
validation, result construction, handle release, and final cache validation all share one failure
domain. Cleanup attempts draft-root rollback, target-root rollback, and every acquired D32-handle
release in that deterministic order. Healthy cleanup re-raises the exact original exception.
`SpeculativeIterationCleanupError` preserves the original failure and every ordered cleanup
failure when restoration is incomplete. D38 never calls `reset()`, retries D37, or restores
selector/RNG state; a failed or invalid final-row draw remains consumed.

This coordinator and its tests are model-free and import without MLX, PyTorch, Transformers, the
native grammar extension, CUDA initialization, tokenizer assets, or network access. D38 does not
run a second speculative iteration, rotate roots, own prompt prefill or model lifecycles, select a
release pair, choose `gamma`, branch grammar state, apply stop/completion/length policy, stream,
cancel, add speculative metrics, or expose API behavior. Those remain separate production and
iterative-engine work.

### Bounded two-iteration handoff

`onyx_cuda.speculative_handoff` adds the five D39 public symbols
`SpeculativeHandoffError`, `SpeculativeHandoffInvariantError`,
`SpeculativeHandoffCleanupError`, `TwoIterationSpeculativeHandoffResult`, and:

```python
coordinate_two_iteration_speculative_handoff(
    draft_backend,
    target_backend,
    current_token_id,
    *,
    proposal_length,
    draft_select_token,
    target_select_token,
    draft_root_checkpoint,
    target_root_checkpoint,
)
```

The operation has the same eight-parameter shape as D38. It uses the exact same two already-
prefilled backend objects, caller-owned selector sessions, and positive caller-selected proposal
length for both calls. It is deliberately two calls rather than an iteration-count abstraction and
does not choose a release `gamma`.

Let `P` be the common caller-root length, `C0` the initial uncached current token, and `n` the
proposal length. For accepted counts `A1` and `A2`, define the first proposal and uncached output as
`D1` and `H1`, and the second as `D2` and `H2`:

```text
Q1 = P  + 1 + A1
Q2 = Q1 + 1 + A2

first output  = D1[:A1] + (H1,)
second output = D2[:A2] + (H2,)
combined      = first output + second output

final caches  = prompt + (C0,) + D1[:A1] + (H1,) + D2[:A2]
```

After the first D38 success, both caches end at `Q1` and `H1` is still uncached. D39 creates one
draft and then one target intermediate checkpoint at that exact prefix. Its second D38 call borrows
those roots and receives `H1` as its current token. D38 consumes that token into each cache but does
not emit a second current-token occurrence. `H1` therefore appears in the combined output exactly
where the first result emitted it; a later naturally selected token with the same numeric ID remains
valid. After the second success both caches end at `Q2`, and only `H2` remains uncached.

All four outcome categories follow the same formulas:

| First transaction | Second transaction | Intermediate length | Final length | Final uncached token |
|---|---|---:|---:|---|
| mismatch at `A1` | mismatch at `A2` | `P + 1 + A1` | `P + 2 + A1 + A2` | second replacement |
| mismatch at `A1` | full acceptance | `P + 1 + A1` | `P + 2 + A1 + n` | second bonus |
| full acceptance | mismatch at `A2` | `P + 1 + n` | `P + 2 + n + A2` | second replacement |
| full acceptance | full acceptance | `P + 1 + n` | `P + 2 + 2n` | second bonus |

One successful handoff makes exactly two D38 calls, `2n` draft-selector calls,
`2(n + 1)` draft decodes, two batched target verifications, and `A1 + A2 + 2`
target-selector calls. D39 directly creates and releases exactly one checkpoint per role and creates
no final or third-iteration root. Each D38 call continues to own and settle its own D32 checkpoint
group.

`TwoIterationSpeculativeHandoffResult` is frozen and slotted and stores exactly
`first_iteration` and `second_iteration`, both genuine immutable D38 results. It derives
`handoff_token_id`, exact concatenated `output_token_ids`, `uncached_next_token_id`,
`initial_cache_length`, `intermediate_cache_length`, and `final_cache_length`. Construction
requires equal positive proposal lengths, cache continuity, exact nonempty per-iteration outputs,
and each result's final output/uncached-token relationship. The result retains no backend,
checkpoint, selector, row, model, tensor, grammar state, metric session, or mutable collection.

The caller owns both initial roots throughout and D39 never releases them. D39 owns each
intermediate handle as soon as its creation call returns, before inspecting its metadata. On
success it releases the draft intermediate, validates both final cache lengths, releases the target
intermediate, and validates both lengths again. Any failure after the first D38 return starts outer
cleanup in this fixed order:

1. restore and validate the draft caller root;
2. restore and validate the target caller root;
3. settle the acquired draft intermediate root;
4. settle the acquired target intermediate root.

Failures inside the first D38 call remain solely in D38's transaction domain and propagate without
duplicate outer rollback. A later failure is re-raised as the exact object when outer cleanup is
healthy. If outer restoration or settlement also fails, `SpeculativeHandoffCleanupError` retains
the original failure as its cause plus the ordered immutable cleanup evidence. A nested
`SpeculativeIterationCleanupError` remains one unflattened original failure. D39 never calls
`reset()`, retries a transaction, or rewinds caller-owned selector/RNG state.

The module and deterministic proof are framework-neutral, model-free, and import without MLX,
PyTorch, Transformers, Tokenizers, Hugging Face Hub, bitsandbytes, Accelerate, ONNX Runtime,
psutil, the native grammar extension, CUDA initialization, assets, or network access. D39 is not a
variable-count or production engine and does not own prompt prefill, third-iteration roots,
termination or output budgets, grammar/stop/streaming/cancellation/metric policy, pair loading,
release models or quantization, fixed `gamma`, offload, operating limits, or API behavior.

### Caller-bounded multi-iteration root rotation

D40 extends `onyx_cuda.speculative_handoff` with
`MultiIterationSpeculativeHandoffResult` and:

```python
coordinate_multi_iteration_speculative_handoff(
    draft_backend,
    target_backend,
    current_token_id,
    *,
    iteration_count,
    proposal_length,
    draft_select_token,
    target_select_token,
    draft_root_checkpoint,
    target_root_checkpoint,
)
```

`iteration_count` is a required positive, non-Boolean integer. The coordinator calls D38 exactly
that many times using the exact same two already-prefilled backends, the exact same caller-owned
selector sessions, and one unchanged positive proposal length. It neither selects a release
`gamma` nor stops before the requested count.

Let `m` be the iteration count, `P` the common caller-root length, `n` the proposal length, `Ci`
the current token for call `i`, `Di` its proposal, `Ai` its accepted count, `Hi` its uncached next
token, `Oi` its output, and `Qi` the common cache length after that call. With one-based iteration
indices:

```text
Q0 = P
Qi = Q(i-1) + 1 + Ai

C1 = caller current token
Ci = H(i-1) for i > 1

Oi = Di[:Ai] + (Hi,)

combined output = O1 + O2 + ... + Om
final length    = P + m + sum(Ai for i in 1..m)
```

After all calls, both caches contain:

```text
prompt
+ (C1,) + D1[:A1]
+ (H1,) + D2[:A2]
+ ...
+ (H(m-1),) + Dm[:Am]
```

Every nonfinal `Hi` already appears once at the end of its producing output. The following D38
call consumes it into both caches as its current token; D40 does not append it again. A naturally
selected later token may have the same numeric ID without being coordinator duplication. Only
`Hm`, the last combined-output token, remains outside both final caches.

`MultiIterationSpeculativeHandoffResult` is frozen and slotted and stores exactly one field:

```python
iterations: tuple[ContinuationAwareSpeculativeIterationResult, ...]
```

The tuple must be an exact, nonempty built-in tuple of genuine structurally valid D38 results.
Every item uses the same positive proposal length, and adjacent final/initial cache lengths must be
continuous. The result derives only `output_token_ids`, `uncached_next_token_id`,
`initial_cache_length`, and `final_cache_length`. It retains the exact completed D38 result objects
in order, but no backend, selector, checkpoint, target row, model, tensor, grammar state, metric
session, or mutable collection.

Count one wraps one direct D38 outcome and allocates no D40 intermediate root. Count two calls D38
directly twice and is observationally equivalent to D39 for current-token handoff, selector
consumption, output, cache state, root ownership, release order, and cleanup. D39 remains an
unchanged independent regression boundary.

After every nonfinal success D40 creates a draft next root, validates its metadata and both cache
lengths, then creates and validates the target next root. If a prior D40 pair exists, it releases
the prior draft root and then the prior target root, validating both caches after each release,
before promoting the new pair. D40 tracks only one current pair and one transient next pair. Thus
stable state owns at most one D40 pair and rotation transiently owns at most two, regardless of
`m`. The caller's initial roots stay active and are never released. After the final result is
composed and validated, the last current pair is released draft then target; count one has no such
settlement.

For accepted counts `A1...Am`, one successful operation performs:

```text
D38 calls                  = m
draft selector calls       = m * n
target selector calls      = m + sum(Ai)
target batched verifies    = m
D40 roots created/released = m - 1 per role
draft allocation growth    = m * (n + 1) + (m - 1)
target allocation growth   = m - 1
```

Each target mismatch replays `Ai + 1` tokens; full acceptance requires no target replay. The
allocation formulas include D38's `n + 1` draft checkpoints per transaction but not the already
existing caller roots.

The first D38 call remains wholly in D38's transaction domain. If it fails, D40 owns no
intermediate root and performs no duplicate cleanup. The D40 outer domain begins as soon as that
first call returns, before trusting its result. Any later validation, creation, rotation, D38,
composition, final-release, or cache-validation failure attempts cleanup in this order:

1. restore and validate the draft caller root;
2. restore and validate the target caller root;
3. settle the current draft intermediate root;
4. settle the current target intermediate root;
5. settle a distinct transient next draft root;
6. settle a distinct transient next target root.

The corresponding cleanup labels are `draft initial root rollback`,
`target initial root rollback`, `draft intermediate root release`,
`target intermediate root release`, `draft next intermediate root release`, and
`target next intermediate root release`. A sole partially acquired first pair uses the two
non-`next` release labels. Cleanup continues after every failure. Healthy cleanup re-raises the
exact original exception; incomplete cleanup raises `SpeculativeHandoffCleanupError` with ordered
immutable evidence and the original as its cause. Nested D38 cleanup errors remain unflattened.
D40 never calls `reset()`, retries a transaction, or rewinds caller-owned selector/RNG state.

D40 is framework-neutral, model-free, and imports without optional model, CUDA, native, or Mac
runtimes. It is fixed-count mechanical orchestration, not a production speculative engine. It
does not own prompt or model lifecycles, select or load a release pair, choose fixed or adaptive
`gamma`, impose iteration or operating limits, terminate on EOS/grammar/stops/output budgets,
truncate output, branch grammar state, stream, cancel, add speculative metrics, enable offload, or
expose API behavior.

### One-iteration speculative grammar-state reconciliation

D41 adds the pure `onyx_cuda.speculative_grammar` module and exactly five public symbols:

```python
SpeculativeGrammarReconciliationCleanupError
SpeculativeGrammarReconciliationError
SpeculativeGrammarReconciliationInvariantError
SpeculativeGrammarReconciliationResult
reconcile_speculative_grammar_state
```

The base error derives from `GrammarError`; the invariant and cleanup errors derive from the D41
base. The operation has this exact signature:

```python
reconcile_speculative_grammar_state(
    constraint,
    starting_state,
    iteration_result,
    *,
    vocab_size,
)
```

`constraint`, `starting_state`, and the completed D38 `iteration_result` are borrowed. The required
positive, non-Boolean integer `vocab_size` must equal `constraint.vocab_size`; it supplies the
numeric range used to validate all proposal and output tokens because D38 results do not retain a
vocabulary bound. That equality does not prove tokenizer-byte compatibility or producer
provenance.

The starting state is already after D38's current token. D41 accepts no current-token parameter and
does not advance that token again. Let:

- `S` be the caller-owned starting state;
- `D = (d0, ..., d[n-1])` be the nonempty proposal;
- `A` be the accepted count;
- `H` be the uncached next token;
- `O = D[:A] + (H,)` be the exact committed output; and
- `m = len(O) = A + 1`.

The two branches are independently allocated from the same borrowed `S`:

```text
Sd[-1] = S
Sd[i]  = advance_state(Sd[i-1], D[i])   for 0 <= i < n

Sc[-1] = S
Sc[j]  = advance_state(Sc[j-1], O[j])   for 0 <= j < m
```

On mismatch, `A < n`, `H` equals the replacement and differs from `D[A]`; the committed branch is
the accepted prefix plus that replacement. On full acceptance, `A == n`, the committed branch
independently replays every proposal and then advances the bonus `H`. Equal numeric token IDs never
authorize state reuse. D41 compares opaque state handles only by object identity and rejects a
child that aliases `S`, its parent, an earlier same-branch child, or any child in the other branch.

A successful operation performs:

```text
draft advances               = n
committed advances           = m = A + 1
total advances               = n + A + 1
draft states released        = n
committed ancestors released = m - 1 = A
transferred states           = 1
D41-owned peak states        = n + m
```

Every child is D41-owned as soon as `advance_state(...)` returns. D41 validates every returned
child with `is_dead_state(...)`. A regex dead child is permitted only on the discard-only draft
branch and can continue producing dead children. A dead committed child is an invariant failure.
An invalid JSON transition instead propagates the original `GrammarStateError`; no child is
invented for a transition that raised.

Before releasing anything, D41 proves the borrowed start is unchanged and the final committed
state is live, then records the final match Boolean. Success releases every draft child in proposal
order, releases every nonfinal committed child in output order, and revalidates both retained
states. It then returns the frozen, slotted two-field result:

```python
@dataclass(frozen=True, slots=True)
class SpeculativeGrammarReconciliationResult:
    committed_state: StateT
    is_match: bool
```

The result transfers exactly the final committed state and retains no constraint, start, D38
evidence, draft child, committed ancestor, backend, selector, row, checkpoint, model, tensor,
metric session, cleanup history, or mutable collection. The caller retains ownership of `S` and
becomes responsible for releasing `result.committed_state`.

Any failure after acquisition attempts every still-owned unique state in this exact order:

1. remaining draft children in proposal order;
2. remaining committed children in output order, including the final child unless transferred.

The zero-based cleanup labels are `draft state release at position {i}` and
`committed state release at position {j}`. A success-path release that raises remains owned and is
retried once by failure cleanup, relying on the grammar contract's idempotent release behavior.
Cleanup continues after every exception. Healthy cleanup re-raises the exact original failure.
Incomplete cleanup raises `SpeculativeGrammarReconciliationCleanupError` with ordered immutable
evidence, exact exception identities, and the original failure as its cause. The borrowed start is
never released, and D41 never bulk-releases states, resets the constraint, or retries a transition.

D41 calls no valid-token scan, grammar mask, selector, backend, cache, checkpoint, model, metric, or
D38 operation. It is model-free and imports without optional native, CUDA, model, or Mac runtimes.
Live speculative grammar masks, grammar-driven early proposal termination, multi-iteration grammar
policy, EOS/grammar/stop/output-budget/streaming/cancellation integration, production pair
lifecycle, release model and `gamma` selection, operating limits, offload, metrics, and API behavior
remain separate work.

### One-row grammar-support masked selection

D42 adds the pure `onyx_cuda.grammar_selection` module and exactly four public symbols:

```python
GrammarMaskedSelectionError
GrammarMaskedSelectionInvariantError
GrammarMaskedSelectionResult
select_grammar_masked_token
```

Both errors derive from `GrammarError`, with the invariant error beneath the D42 base. The operation
has this exact signature:

```python
select_grammar_masked_token(
    constraint,
    state,
    logits,
    logit_mask,
    *,
    vocab_size,
    select_token,
)
```

The constraint, live state, backend-native logits row, stateless mask, and selector session are all
borrowed. The required `vocab_size` is a positive, non-Boolean built-in integer. It must equal the
positive built-in integer `vocab_size` reported by both the constraint and mask. This agreement
establishes only one numeric token domain; it does not prove tokenizer bytes, model provenance, or
release-pair compatibility. The constraint must report exactly `"regex"` or `"json_schema"`.

After validating all components and metadata, D42 calls `is_dead_state(state)`,
`is_match_state(state)`, and `get_valid_token_ids(state)` exactly once each in that order. The
borrowed state must be live, both status queries must return exact Booleans, and native support must
be an exact tuple of strictly increasing, unique, in-range built-in Python integers. D42 preserves
that tuple by identity: it does not materialize an iterable, sort, deduplicate, filter, or inject
EOS.

Empty support returns the explicit successful no-selection result without calling the mask or
selector:

```python
GrammarMaskedSelectionResult(
    valid_token_ids=valid_token_ids,
    is_match=is_match,
    selected_token_id=None,
)
```

This branch is the same for matching and nonmatching live states. It records facts only and does not
classify completion, failure, or another terminal reason.

For nonempty support, D42 calls `logit_mask.apply(logits, valid_token_ids)` exactly once, preserving
the input-row and support-tuple identities, then calls `select_token(masked_logits)` exactly once
with the exact returned row. It uses only the ordinary untimed mask operation and treats both rows
as opaque backend-native evidence. The selected token must be an exact built-in integer in
`[0, vocab_size)` and a member of the same native support tuple used for masking.

The frozen, slotted result contains exactly these fields in this order:

```python
valid_token_ids: tuple[int, ...]
is_match: bool
selected_token_id: int | None
```

Direct construction enforces exact tuple and scalar types, nonnegative strictly increasing unique
support, no selection for empty support, and one nonnegative in-support integer for nonempty
support. The operation enforces the upper vocabulary bound before construction. The result retains
no constraint, state, logits row, masked row, mask, selector or RNG session, backend, model, cache,
checkpoint, tensor, native runtime, timing metadata, or mutable history.

D42 never initializes, advances, releases, bulk-releases, or resets grammar state. Component query,
mask, and selector execution failures propagate as the exact original exceptions, without retry or
cleanup, because the operation acquires no resource. Selector/RNG consumption remains
caller-owned. The module and package-root exports remain usable without native, model, CUDA, or Mac
runtimes.

This isolated primitive is not integrated into proposal, acceptance, continuation, D38
coordination, D41 reconciliation, or multi-iteration handoff. State advancement, grammar-driven
proposal termination, EOS/completion/stop/output-budget policy, production pair lifecycle, release
model and `gamma` selection, live qualification, operating limits, streaming, cancellation,
metrics, offload, and API behavior remain separate work.

### One-step grammar-masked selection and child-state transfer

D43 adds the pure `onyx_cuda.grammar_transition` module and exactly five public symbols:

```python
GrammarMaskedTransitionCleanupError
GrammarMaskedTransitionError
GrammarMaskedTransitionInvariantError
GrammarMaskedTransitionResult
select_and_advance_grammar_state
```

The base error derives from `GrammarError`; the invariant and cleanup errors derive from the D43
base. The operation has the exact D42 argument shape:

```python
select_and_advance_grammar_state(
    constraint,
    state,
    logits,
    logit_mask,
    *,
    vocab_size,
    select_token,
)
```

All inputs are borrowed. D43 calls the unchanged public `select_grammar_masked_token(...)` exactly
once with the exact incoming objects and values. D42 remains solely responsible for component
protocol and metadata checks, the initial `is_dead_state`, `is_match_state`,
`get_valid_token_ids` query sequence, masking, and selector validation. D43 neither reads mask
metadata nor requests native support again.

Before advancing, D43 requires the returned value to be a genuine
`GrammarMaskedSelectionResult`; reads `valid_token_ids`, `is_match`, and `selected_token_id` in
that order; and revalidates their exact types, support ordering and uniqueness, the supplied upper
vocabulary bound, empty/nonempty consistency, and selected-token membership. It then revalidates
the borrowed parent with `is_dead_state` followed by `is_match_state`, requiring the parent to
remain live with D42's exact match flag.

Empty support returns without advancing or releasing any state:

```python
GrammarMaskedTransitionResult(
    selection=selection,
    child_state=None,
    child_is_match=None,
)
```

For a selected token, D43 calls `advance_state(state, selected_token_id)` exactly once. The returned
candidate is immediately D43-owned, including when its opaque value is `None`. A candidate that is
the borrowed parent by identity is rejected without releasing that alias. An independent child
must report exact `False` from `is_dead_state` and an exact Boolean from `is_match_state`. D43 then
revalidates the parent again in dead/match order before constructing and validating the result.
Only after exact selection identity, child identity, match evidence, and the derived transition
flag are confirmed is ownership of the child transferred to the caller.

The frozen, slotted result has exactly three stored fields:

```python
selection: GrammarMaskedSelectionResult
child_state: StateT | None
child_is_match: bool | None
```

Its derived `transitioned` property is true exactly when the nested D42 result contains a selected
token. Callers must use that property rather than `child_state is not None`, because `None` is a
legal opaque child-state value. The nested D42 result and its support tuple are retained directly,
so their identities and D42's parent-match evidence are preserved without flattening.

D42 failures, malformed D42 evidence, post-selection parent failures, and an
`advance_state(...)` exception occur before D43 owns a distinct child and therefore propagate
without cleanup. Once an independent child has returned, every later failure attempts exactly one
`release_state(child)`. Successful cleanup re-raises the exact original failure. If release also
fails, `GrammarMaskedTransitionCleanupError` retains the exact original exception and the immutable
single-entry cleanup evidence:

```python
(("child state release", cleanup_exception),)
```

The original failure is also the cleanup error's cause. D43 never releases the borrowed parent,
bulk-releases, resets, retries advancement, retries release, or owns/rewinds the selector session.
An advertised-valid regex token producing a dead child is an invariant failure and that child is
released. A JSON invalid-transition `GrammarStateError` propagates unchanged with no invented
child; a malformed JSON implementation that returns a dead child is rejected under the same
post-acquisition rule.

D43 records state facts only. A matching parent or child does not inject EOS or terminate
generation, and empty support is not classified as completion or failure. The operation does not
loop over tokens, integrate with proposal/acceptance/continuation/speculative coordination,
reconcile D41 branches, mutate caches, select or load models, choose `gamma`, add streaming,
cancellation, metrics, offload, operating limits, or API behavior. It remains model-free and
imports without optional native, CUDA, model, or Mac runtimes.

### Pinned dual-backend one-iteration qualification

D36 qualifies the unchanged D35 signature and transaction through two independent calls to
`load_torch_cuda_target(DEFAULT_TARGET_PROFILE, device_index=0, local_files_only=True)`. One loaded
backend is assigned the draft role and the other the target role for the fixture. They have separate
backend/model/tokenizer objects, tokenizer runtime objects, owner IDs, sequence epochs, checkpoint
allocation state and registries, caller roots, `DynamicCache` objects, layer lists, all 24 layer
objects, and every active key/value tensor. Cross-role root use fails before mutation.

The fixed prompt has `P = 9`, the common uncached current token is `4379`, and the genuine greedy
three-token proposal is `(25, 279, 7162)`. The exact transaction map is:

| Outcome | Accepted | Final cache | Target selector calls | Target forwards after prefill |
|---|---:|---:|---:|---:|
| forced mismatch at 0 | 0 | `P + 1 = 10` | 1 | one four-row batch + 1 replay |
| forced mismatch at 1 | 1 | `P + 2 = 11` | 2 | one four-row batch + 2 replays |
| forced mismatch at 2 | 2 | `P + 3 = 12` | 3 | one four-row batch + 3 replays |
| genuine full acceptance | 3 | `P + 4 = 13` | 3 | one four-row batch, no replay |

Every draft transaction uses four ordinary one-token forwards with `logits_to_keep=1`. Every target
transaction first uses one forward over `(4379, 25, 279, 7162)` with `logits_to_keep=4`. Mismatch
replay then uses only `(4379, *proposal[:k])` as ordinary one-token decodes. The full-acceptance
selector is unmodified `select_cuda_argmax` for both roles. At a forced mismatch, the target wrapper
first obtains the real greedy decision and requires it to match the proposal, then substitutes the
deterministic in-range successor only at the chosen position. Selection stops there. Row `r3` is
never selected, and the replacement remains uncached.

Physical cache validation is role-local. Draft mismatch states equal exact prefix slices of the
draft's sequential full-acceptance snapshot. Target mismatch states equal independent target-owned
sequential replay snapshots. Full target state equals its complete native batched result. Both roles
also retain their original D29 layout signature and exact Python prefix. No cross-role or
sequential-versus-batched bitwise equality is required because D31 established that those kernels
can differ numerically while producing the same greedy decisions.

For each transaction, D32 allocates a private checkpoint at `P` followed by rejection checkpoints at
`P + 1`, `P + 2`, and `P + 3`. Allocation IDs remain monotonic. D32 releases the private checkpoint;
D35 releases all rejection checkpoints in proposal order, including invalidated suffix handles.
Neither caller root is released, the target allocates no transaction checkpoint, and both final
registries contain only their caller root.

Production-backend composition tests use two real `TorchCUDATargetBackend` objects over the pinned
24-layer fake Torch/Transformers seam. Healthy draft-selector, target-selector, and pre-mutation
target-replay failures preserve exception identity, restore both roots exactly, settle every owned
handle, and permit immediate reuse. Post-cache-mutation draft and target failures preserve the typed
backend error, terminal safe-empty epoch, nested/outer cleanup composition and ordered stale-root
failure while leaving the healthy peer immediately usable. Weak-reference checks prove that results,
registries, and scalar-only observers retain no input, row, parent logits, model, cache, or
checkpoint tensor. A 100-transaction model-free production-backend loop and 100 transactions in
each live lifecycle keep epochs, roots, cache/layer identities, layouts, call counts and
outcome-grouped allocator state bounded.

The two live lifecycles close target then draft and draft then target, respectively. After the first
close, the closed role rejects work and its still-live peer successfully decodes and rolls back
through its unchanged root. This qualification is production-seam evidence only: using two
identical 0.5B instances does not select a release pair, select the 0.5B profile as a release draft,
or demonstrate useful speculative speedup.

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

D36 runs two complete dual-backend lifecycles and 200 total one-iteration transactions, reversing
close order between lifecycles. The fixed nine-token prompt is
`(35, 18, 16, 5670, 7162, 291, 2169, 22901, 45060)`, current token is `4379`, and both lifecycles
produce genuine greedy proposal `(25, 279, 7162)`. Full acceptance finishes at cache length 13.
Forced mismatch positions 0, 1, and 2 finish at lengths 10, 11, and 12 with uncached replacements
`26`, `280`, and `7163`. Draft/target selector counts are `3/3`, `3/1`, `3/2`, and `3/3`; forward
counts are `4/1`, `4/2`, `4/3`, and `4/4` for full, mismatch-0, mismatch-1, and mismatch-2.

First/second loader durations are 1.818356/1.119697 seconds in lifecycle 1 and
1.065408/1.071617 seconds in lifecycle 2. Simultaneous rooted active state is
924,817,408 allocated / 1,153,433,600 reserved bytes in lifecycle 1 and
924,653,568 / 1,157,627,904 bytes in lifecycle 2. The higher transaction peak is 937,167,360
allocated and 1,157,627,904 reserved bytes. After warmup, each outcome stabilizes as a bounded
periodic allocator pattern with 2,696 current and active allocations. Stable median transaction
times across the two lifecycles are 0.228409/0.235699 seconds for full acceptance,
0.286878/0.268526 for mismatch-0, 0.334574/0.317441 for mismatch-1, and
0.365808/0.372571 for mismatch-2.

Closing target first leaves the draft at 466,800,128 allocated / 803,209,216 reserved bytes.
Closing draft first leaves the target at 466,742,784 / 861,929,472 bytes. Each surviving peer is
immediately usable. First/second close durations are 0.095263/0.094136 seconds in lifecycle 1 and
0.086036/0.087953 seconds in lifecycle 2; cleanup takes 0.071058 and 0.066571 seconds. Both completed
lifecycles clean up to the identical 8,668,160 allocated / 501,219,328 reserved-byte state, with no
second-lifecycle retained growth. The figures are qualification evidence, not supported memory
limits or a speedup claim.

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
- a policy-driven production iterative engine or termination and output-budget policy;
- integration of the isolated one-row grammar-support masked-selection primitive into live
  speculative selection, grammar-driven proposal termination, or multi-iteration grammar-state
  policy;
- a production/user-visible grammar-aware speculative engine;
- speculative stops, streaming, cancellation, or acceptance metrics;
- fixed or adaptive `gamma`;
- final prompt, output, context, concurrency, or 6 GiB operating limits;
- CPU offload;
- implicit chat-template formatting;
- a FastAPI/OpenAI-compatible server;
- a complete public `NativeGrammarCompiler`;
- native valid-token caching or a persistent CUDA mask workspace.

Those capabilities remain separately sized roadmap work. D36 proves the unchanged one-iteration
coordinator through two independently owned pinned production backends. D37 defines the pure
post-iteration token handoff, D38 integrates it into one additive framework-neutral transaction,
and D39 proves one exact transition between two such transactions with a settled intermediate root
pair. D40 generalizes that mechanical handoff to a positive caller count with bounded root
rotation, and D41 independently reconciles one completed D38 result into a transferred committed
grammar state. D42 supplies one isolated borrowed-state, one-row grammar-support masked-selection
primitive. These deliverables still do not form user-visible speculative decoding without a
selected pair, live integration of that selection primitive, and a separately owned production
iterative engine.
