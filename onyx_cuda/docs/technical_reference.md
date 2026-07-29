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
- next-iteration root rotation, an iterative handoff owner, or a cache-coordinated multi-iteration
  speculative loop;
- grammar-state speculation;
- speculative stops, streaming, cancellation, or acceptance metrics;
- fixed or adaptive `gamma`;
- final prompt, output, context, concurrency, or 6 GiB operating limits;
- CPU offload;
- implicit chat-template formatting;
- a FastAPI/OpenAI-compatible server;
- a complete public `NativeGrammarCompiler`;
- native valid-token caching or a persistent CUDA mask workspace.

Those capabilities remain separately sized roadmap work. D36 proves the unchanged one-iteration
coordinator through two independently owned pinned production backends. D37 now defines the pure
post-iteration token handoff, and D38 integrates it into one additive framework-neutral
transaction. These deliverables still do not form user-visible speculative decoding without a
selected pair and a separately owned iterative engine.
