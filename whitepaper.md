# Grammar-Aware Speculative Decoding for Structured Output Generation

## Abstract

Large language models frequently fail to produce syntactically valid structured outputs, creating reliability challenges for applications that depend on parseable JSON or domain-specific formats. We present a prototype that combines grammar-constrained decoding with speculative execution. Our approach applies constraints—DFA-based regex enforcement and a stack-based JSON Schema subset—to both draft and target models before verification. The supported JSON subset includes nested objects, typed arrays, union types, enum restrictions, regex patterns on string fields, and length constraints. Recorded Apple Silicon experiments reported compliant output for the evaluated cases and workload-dependent performance: an 8B year-pattern run reached 18.9 tok/s versus a 17.4 tok/s constrained baseline (1.09x), while the 1.5B run reached 0.94x of baseline. These measurements describe the tested hardware, models, patterns, and benchmark scripts; they are not universal performance or compliance guarantees.

---

## 1. Problem Statement

### 1.1 The Reliability Gap in Agentic AI

Modern AI agents increasingly rely on language models to produce structured outputs—JSON objects for API calls, SQL queries for database access, or domain-specific formats for downstream processing. However, language models are fundamentally probabilistic text generators with no inherent understanding of syntactic constraints. This mismatch creates a reliability gap: the model may produce outputs that are semantically reasonable but syntactically invalid.

Consider an agent that needs to extract structured data:

```
Prompt: "Extract the user info as JSON: John Smith, age 32, engineer"
Expected: {"name": "John Smith", "age": 32, "occupation": "engineer"}
Actual: {"name": "John Smith", "age": 32, "occupation": "engineer"} (sometimes)
        {"name": "John Smith", age: 32, occupation: engineer} (invalid JSON)
        Here is the JSON: {"name": ... (extra text)
```

Production systems address this through retry loops, output validation, and error handling—all of which add latency and complexity. A more principled approach is to constrain the generation process itself.

### 1.2 The Latency Challenge

Grammar-constrained decoding solves reliability but introduces computational overhead. At each token step, the system must determine which tokens lead to valid grammar states and mask the others. For a vocabulary of 150,000+ tokens, this filtering operation is non-trivial.

Speculative decoding offers potential latency improvements by using a smaller draft model to propose multiple tokens that a larger target model verifies in a single forward pass. However, standard speculative decoding assumes unconstrained generation. When the draft model is "blind" to grammar constraints, it frequently proposes tokens that the constrained target model rejects, negating any speedup benefit.

### 1.3 Research Question

Can grammar constraints be integrated into speculative decoding in a way that preserves both reliability and performance? Specifically:

1. Does applying grammar constraints to the draft model improve acceptance rates?
2. Under what conditions does grammar-aware speculation outperform single-model constrained generation?

---

## 2. Architecture

### 2.1 System Overview

Our system, Onyx, consists of three primary components:

1. **Grammar Engine** (Rust): Compiles regex patterns into DFAs and JSON schemas into stack-based state machines, performs vocabulary filtering
2. **Speculative Engine** (Python/MLX): Coordinates draft-verify-rollback cycles with grammar state management
3. **API Server** (FastAPI): Provides OpenAI-compatible REST interface with grammar constraint extensions
4. **Adaptive Controller** (Python): Experimental controller that adjusts speculative draft length based on recent acceptance rate and timing signals

### 2.2 Grammar Constraint Engine

The grammar engine is implemented in Rust for performance and exposed to Python via PyO3. It supports two complementary constraint modes:

**Regex Mode (DFA-based):** For regular-language constraints (e.g., `[A-Z]{3}-[0-9]{4}`), the engine compiles the pattern into a deterministic finite automaton and maintains:

- A pre-computed vocabulary table mapping token IDs to UTF-8 byte sequences
- A compiled DFA representing the grammar constraint
- State traversal logic that advances through the DFA as tokens are generated

**JSON Schema Mode (Stack-based FSM):** For structured JSON generation, the engine uses a stack-based finite state machine that tracks nested context. The state machine maintains a scope stack where each frame represents the current structural context (object, array, string, number, boolean, null, or enum). It supports:

- **Typed properties**: `string`, `number`, `integer`, `boolean`, `null`, `object`, `array`
- **Nested objects**: Arbitrary nesting depth with per-property schema enforcement
- **Required fields**: Object closure is blocked until all required keys are emitted
- **Union types**: e.g., `["string", "null"]` for nullable fields
- **Enum values**: Byte-level prefix matching against a fixed set of allowed values
- **Regex patterns on strings**: DFA-compiled `pattern` validation on string content, integrated into the string scope
- **Length constraints**: `minLength`/`maxLength` for strings (blocking close quote or new characters), `minItems`/`maxItems` for arrays (blocking close bracket or new items)
- **Typed array items**: Schema enforcement applied to every array element via the item blueprint

In both modes, the key operation is `get_valid_token_ids(state)`, which returns token IDs that can continue from the current state. In v0.2.0, state cloning was refactored to share read-only schema blueprints and DFAs through `Arc` (Atomic Reference Counting) pointers instead of deep-cloning those structures. The recorded benchmark measured approximately 270-500 microseconds per full-vocabulary mask for a 151,000-token vocabulary; this figure is specific to that benchmark environment.

### 2.3 Speculative Decoding with Grammar Awareness

Standard speculative decoding proceeds as:

1. **Draft**: Generate γ tokens with a small, fast draft model
2. **Verify**: Run the target model on all draft tokens in one forward pass
3. **Accept**: Keep tokens up to the first mismatch between draft and target predictions
4. **Rollback**: Reset KV caches to the last accepted position

Grammar-aware speculation modifies this loop:

1. **Draft (Grammar-Constrained)**: Generate γ tokens using the draft model, applying grammar masks at each step
2. **Verify (Grammar-Constrained)**: Run the target model, applying grammar masks during verification
3. **Accept**: Compare draft and target predictions (both are now grammar-valid)
4. **Rollback**: Reset caches; grammar state is only updated after acceptance

We also implemented an experimental adaptive variant that treats γ as a control parameter rather than a fixed constant. The controller tracks a rolling window of draft acceptance rate, draft time, target verification time, and grammar-mask overhead. It increases γ when acceptance is high and target verification dominates, and decreases γ when acceptance is low or masking overhead becomes significant. This keeps the stable fixed-γ engine intact while making the benchmark harness more explicit about the performance tradeoff.

The following diagram compares the two approaches:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD SPECULATIVE DECODING (Blind Draft)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Draft Model          Target Model           Result                        │
│   (unconstrained)      (constrained)                                        │
│                                                                             │
│   ┌─────────┐          ┌─────────┐                                          │
│   │ "ABC"   │───────►  │ Grammar │──► Reject (invalid)                      │
│   │ "123"   │───────►  │  Mask   │──► Reject (invalid)                      │
│   │ "XY7"   │───────►  │ Applied │──► Reject (invalid)                      │
│   │ "2025"  │───────►  │         │──► Accept ✓                              │
│   └─────────┘          └─────────┘                                          │
│                                                                             │
│   Problem: Draft proposes many invalid tokens → Low acceptance rate         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAMMAR-AWARE SPECULATIVE DECODING                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Draft Model          Target Model           Result                        │
│   (constrained)        (constrained)                                        │
│                                                                             │
│   ┌─────────┐          ┌─────────┐                                          │
│   │ Grammar │          │ Grammar │                                          │
│   │  Mask   │          │  Mask   │                                          │
│   │ Applied │          │ Applied │                                          │
│   ├─────────┤          ├─────────┤                                          │
│   │ "2024"  │───────►  │ Verify  │──► Accept ✓                              │
│   │ "2025"  │───────►  │         │──► Accept ✓                              │
│   │ "1999"  │───────►  │         │──► Accept ✓                              │
│   └─────────┘          └─────────┘                                          │
│                                                                             │
│   Solution: Both models constrained → High acceptance rate                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The critical insight is that grammar state is maintained separately from model state. We keep a "main" grammar state that reflects accepted tokens, and temporary states for drafting and verification that are discarded after each iteration. This avoids the need for grammar state rollback.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GRAMMAR STATE MANAGEMENT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Main Grammar State (persistent)                                           │
│   ┌─────────────────────────────────────────┐                               │
│   │ State: S₃  (after "The year is ")       │                               │
│   └─────────────────────────────────────────┘                               │
│                          │                                                  │
│                          │ Clone for drafting                               │
│                          ▼                                                  │
│   ┌─────────────────────────────────────────┐                               │
│   │ Temp Draft State: S₃ → S₄ → S₅ → S₆     │  (propose "2025")             │
│   └─────────────────────────────────────────┘                               │
│                          │                                                  │
│                          │ Verify with target                               │
│                          ▼                                                  │
│   ┌─────────────────────────────────────────┐                               │
│   │ If accepted: Main State ← S₆            │                               │
│   │ If rejected: Discard temp, keep S₃      │                               │
│   └─────────────────────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Paged KV Cache

To support rollback, we implemented a paged KV cache that stores key-value tensors in fixed-size blocks (default 16 tokens). Rollback scans page-table metadata to find the retained boundary, updates counters, and drops trailing block references without copying retained KV tensor contents. Cache reads still concatenate active blocks into contiguous tensors, so the implementation should not be described as O(1) overall.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL KV CACHE (Monolithic)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │ Token 1 │ Token 2 │ Token 3 │ Token 4 │ Token 5 │ Token 6  │            │
│   └────────────────────────────────────────────────────────────┘            │
│                                              ▲                              │
│                                              │                              │
│   Rollback 2 tokens requires:                │                              │
│   • Allocate new tensor                      │                              │
│   • Copy tokens 1-4                    [MEMORY COPY]                        │
│   • Free old tensor                          │                              │
│                                              │                              │
│   Complexity: O(n) where n = valid tokens                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PAGED KV CACHE (Block-based)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Page Table                                                                │
│   ┌─────────┐     ┌──────────────────┐  ┌─────────────────┐                 │
│   │ Block 0 │────►│ T1 │ T2 │ T3 │ T4│  │ Block Size: 4   │                 │
│   ├─────────┤     └──────────────────┘  └─────────────────┘                 │
│   │ Block 1 │────►│ T5 │ T6 │    │  │                                       │
│   ├─────────┤     └─────────────────┘                                       │
│   │  NULL   │     valid_length = 6                                          │
│   └─────────┘                                                               │
│                                                                             │
│   Rollback 2 tokens:                                                        │
│   ┌─────────┐     ┌──────────────────┐                                      │
│   │ Block 0 │────►│ T1 │ T2 │ T3 │ T4│                                      │
│   ├─────────┤     └──────────────────┘                                      │
│   │  NULL   │     valid_length = 4  ◄── Just update counter!                │
│   └─────────┘                                                               │
│                                                                             │
│   Complexity: scan retained block metadata, discard trailing references     │
│   Retained KV tensor contents are not copied during rollback                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Methodology

### 3.1 Experimental Setup

All experiments were conducted on Apple Silicon hardware using the MLX framework. We used the following model configurations:

| Component | Model | Parameters | Quantization |
|-----------|-------|------------|--------------|
| Draft | Qwen2.5-0.5B-Instruct | 0.5B | 4-bit |
| Target (small) | Qwen2.5-1.5B-Instruct | 1.5B | 4-bit |
| Target (large) | Qwen3-8B | 8B | 4-bit |

The models share the same tokenizer family (vocabulary size ~152,000), enabling direct token comparison without remapping.

### 3.2 Test Patterns

We evaluated two grammar patterns of differing complexity:

1. **Year Pattern**: `[0-9]{4}` — A bounded pattern requiring exactly 4 digits
2. **Email Pattern**: `[a-z]+@[a-z]+\.com` — An unbounded pattern with multiple variable-length segments
3. **Forced Digits Pattern**: `[0-9]{32}` — A longer bounded pattern used to evaluate fixed versus adaptive γ over enough speculative iterations for the controller to react

### 3.3 Evaluation Metrics

- **Throughput**: Tokens generated per second (tok/s)
- **Acceptance Rate**: Percentage of draft tokens accepted by the target model
- **Speedup Factor**: Ratio of grammar-aware speculation throughput to baseline (single-model constrained generation)

### 3.4 Scenarios

We compared three generation strategies:

1. **Baseline**: Target model with grammar constraints, no speculation
2. **Blind Draft**: Draft model unconstrained, target model constrained
3. **Aware Draft**: Both draft and target models constrained to the same grammar

For the adaptive benchmark, we additionally compared fixed γ values (`1`, `2`, `4`, `8`) against an adaptive controller initialized at γ=4 with bounds `[1, 8]`. Each benchmark scenario includes one unreported warmup run so first-run compilation effects do not distort the reported average.

---

## 4. Results

### 4.1 The Blind Draft Problem

Our experiments confirm that naive speculative decoding fails for grammar-constrained generation. When the draft model is unaware of the grammar, it proposes tokens that have no chance of acceptance.

**Table 1: Acceptance Rates by Strategy (1.5B Target)**

| Pattern | Blind Draft | Aware Draft |
|---------|-------------|-------------|
| Year `[0-9]{4}` | 75.0% | 100.0% |
| Email `[a-z]+@[a-z]+\.com` | 0.0% | 18.2% |

The email pattern result is particularly striking: blind drafting achieved 0% acceptance because the unconstrained draft model never proposed the exact sequence of lowercase letters, `@` symbol, and `.com` suffix that the grammar-constrained target required. This renders speculative decoding completely ineffective—the system runs both models but gains nothing from speculation.

Grammar-aware drafting recovers acceptance rates by constraining the draft to the same valid token space as the target. For the year pattern, this achieves 100% acceptance; for the more complex email pattern, acceptance improves to 18.2%.

### 4.2 Throughput on the 1.5B Target

With the 1.5B target model, we observed that grammar-aware speculation does not provide a speedup over baseline:

**Table 2: Throughput Results (1.5B Target, Year Pattern)**

| Strategy | Throughput | vs Baseline |
|----------|------------|-------------|
| Baseline | 73.5 tok/s | 1.00x |
| Blind Draft | 57.2 tok/s | 0.78x |
| Aware Draft | 69.2 tok/s | 0.94x |

The aware draft achieves 94% of baseline performance—a 6% overhead. This is consistent with the hypothesis that speculation overhead dominates for this model and workload, but the benchmark did not directly measure the hardware bottleneck.

### 4.3 Throughput on the 8B Target

The recorded result changes with the larger target model:

**Table 3: Throughput Results (8B Target, Year Pattern)**

| Strategy | Throughput | vs Baseline |
|----------|------------|-------------|
| Baseline | 17.4 tok/s | 1.00x |
| Blind Draft | 13.4 tok/s | 0.77x |
| Aware Draft | 18.9 tok/s | 1.09x |

Grammar-aware speculation achieves a **1.09x speedup** (approximately 8.6% improvement) over baseline in this run. The result is consistent with the hypothesis that batched verification can help when model-weight loading dominates, but one run does not establish that explanation universally.

The blind draft result (0.77x) demonstrates that without grammar awareness, speculation is counterproductive—the system is slower than simply running the target model alone.

### 4.4 Crossover Analysis

**Table 4: Speedup Factor by Target Model Size**

| Target Model | Baseline Speed | Aware Draft Speed | Speedup Factor |
|--------------|----------------|-------------------|----------------|
| 1.5B | 73.5 tok/s | 69.2 tok/s | 0.94x |
| 8B | 17.4 tok/s | 18.9 tok/s | 1.09x |

The two measurements show a slowdown at 1.5B and a modest speedup at 8B on the tested system. They do not locate a general crossover point or establish that parameter count alone predicts whether speculation helps.

Additional model sizes and repeated runs would be required before estimating a crossover region.

### 4.5 Adaptive Gamma on Longer Constrained Outputs

Fixed γ is workload-dependent: too small a value underutilizes target-model verification, while too large a value wastes draft work when acceptance falls. We evaluated an experimental adaptive controller on the forced digits pattern (`[0-9]{32}`), comparing fixed γ settings against adaptive γ.

**Table 5: Adaptive Gamma Results (8B Target, Forced Digits Pattern)**

| Strategy | Throughput | vs Baseline | Acceptance |
|----------|------------|-------------|------------|
| Baseline | 21.9 tok/s | 1.00x | — |
| Fixed γ=2 | 25.3 tok/s | 1.16x | 93.8% |
| Fixed γ=4 | 29.1 tok/s | 1.33x | 88.2% |
| Fixed γ=8 | 27.6 tok/s | 1.26x | 78.9% |
| Adaptive γ | 29.2 tok/s | 1.34x | 88.2% |

The adaptive controller matched the best fixed setting in this run, reaching a 1.34x speedup over baseline. Its average γ was 4.2, final γ was 8, and it made 6 adjustments during generation. This result supports the intuition that larger memory-bound targets benefit from longer speculative batches when acceptance remains high, while still allowing the controller to react when conditions change.

On the 1.5B target, the same forced-digits task remained slower than baseline despite high acceptance. This is consistent with the compute-bound result above: adaptive γ can choose among speculative settings, but it cannot remove the fundamental overhead of running two models when speculation is not the right bottleneck match.

---

## 5. Limitations and Future Work

### 5.1 Model Size Dependency

The measured benefit is workload- and hardware-dependent. The 1.5B run was slower than baseline while the 8B run was modestly faster, but these two points do not isolate the bottleneck or predict other configurations.

### 5.2 Grammar Complexity

Our initial evaluation focused on regex patterns using the DFA-based engine. We have since extended the system with a stack-based JSON Schema engine that supports nested objects, typed arrays, union types, enum restrictions, regex patterns on string fields, and length constraints. This addresses the limitation of DFA-only approaches, which are restricted to regular languages. The JSON Schema engine handles recursive nesting through an explicit scope stack, avoiding the need for a full context-free grammar parser while supporting the structural complexity of real-world schemas.

Remaining limitations include: the engine does not yet support JSON Schema features such as `$ref` (recursive schema references), `additionalProperties`, `oneOf`/`anyOf`/`allOf` combinators, or conditional schemas (`if`/`then`/`else`). These extensions represent potential future work.

### 5.3 Acceptance Rate Variability

While grammar-aware drafting improves acceptance rates, the improvement varies by pattern complexity. Simple bounded patterns (digits) achieve near-perfect acceptance; complex unbounded patterns (email) show more modest gains. The draft model's predictions still diverge from the target, especially for larger target models where the capability gap is wider.

### 5.4 Hardware Specificity

All experiments were conducted on Apple Silicon using the MLX framework. Results may differ on other hardware (NVIDIA GPUs, CPUs) with different memory bandwidth characteristics and framework optimizations.

### 5.5 Single-Turn Evaluation

We evaluated single-turn generation, including both short bounded patterns and a 32-token forced-digits pattern. Long-context scenarios, multi-turn conversations, and batched inference may exhibit different performance characteristics that we did not investigate.

### 5.6 Adaptive Controller Scope

The adaptive γ controller is experimental and benchmark-oriented. It is implemented separately from the stable fixed-γ engine path and is not exposed through the OpenAI-compatible API. The current controller uses a simple heuristic rather than an offline tuner or statistically optimal bandit policy.

---

## 6. Related Work

This work builds on speculative decoding as introduced by Leviathan et al. [1], which demonstrated that draft-then-verify approaches can improve inference throughput on memory-bound models by amortizing memory bandwidth costs across multiple tokens verified in a single forward pass. Our contribution extends this framework to grammar-constrained generation scenarios.

The grammar constraint engine follows the DFA-based vocabulary filtering approach described by Willard and Louf [2], who showed that compiling grammar specifications into deterministic finite automata enables efficient token filtering during generation. We apply this technique to both draft and target models in a speculative pipeline, addressing the "blind draft" problem where unconstrained draft models propose invalid tokens.

The paged KV cache design draws from PagedAttention as introduced by Kwon et al. [4], which demonstrated that block-based memory management enables efficient memory utilization and supports operations like preemption and sharing. This prototype adapts block-based storage so rollback can discard trailing references without copying retained KV tensor contents; its metadata work scales with the retained page-table prefix.

All experiments use Apple's MLX framework [3], which provides efficient array operations optimized for Apple Silicon's unified memory architecture.

---

## 7. Conclusion

We presented grammar-aware speculative decoding, a technique that integrates grammar constraints into both draft and target models in a speculative decoding pipeline. Our experiments demonstrate that this approach:

1. **Improves evaluated output compliance**: Supported constraints filter invalid continuations, and completed grammar states matched the evaluated patterns and schema subset
2. **Addresses the blind draft problem**: Applying grammar constraints to the draft model improves acceptance rates from as low as 0% (blind) to 18-100% (aware)
3. **Produced a modest speedup in one 8B run**: The year-pattern benchmark measured 1.09x versus single-model constrained generation
4. **Supports a structured JSON subset**: The stack-based engine handles nested objects, typed arrays, union types, enum restrictions, regex patterns, and length constraints
5. **Adapts speculative batch size experimentally**: On the recorded 8B forced-digits benchmark, an adaptive γ controller reached 1.34x baseline throughput while adjusting γ during generation

The recorded 1.5B run was slower than baseline (0.94x). More hardware, model sizes, patterns, and repeated trials are needed before generalizing when the technique will help.

These prototype results motivate further evaluation of grammar-aware speculation, but they do not establish production reliability or general performance gains.

---

## Key Data Points

**FFI Overhead (Rust-Python boundary)**
- Python native regex: 0.261 µs/call
- Rust via FFI: 0.092 µs/call
- FFI overhead: Negative (Rust is 2.8x faster)

**Grammar Mask Computation**
- Vocabulary size: 151,643 tokens
- Average mask time: 271.70 µs
- At 200 tok/s: 5.43% overhead per token

**1.5B Target (Year Pattern)**
- Baseline: 72.6-73.5 tok/s
- Blind Draft: 55.6-57.2 tok/s (75% acceptance)
- Aware Draft: 68.1-69.2 tok/s (100% acceptance)

**8B Target (Year Pattern)**
- Baseline: 17.4 tok/s
- Blind Draft: 13.4 tok/s (12.5% acceptance)
- Aware Draft: 18.9 tok/s (25% acceptance)

**8B Target (Forced Digits Pattern, `[0-9]{32}`)**
- Baseline: 21.9 tok/s
- Fixed γ=4: 29.1 tok/s (88.2% acceptance, 1.33x)
- Adaptive γ: 29.2 tok/s (88.2% acceptance, 1.34x; avg γ=4.2, final γ=8)

**Email Pattern (1.5B Target)**
- Baseline: 80.7 tok/s
- Blind Draft: 23.8 tok/s (0% acceptance)
- Aware Draft: 42.1 tok/s (18.2% acceptance)

---

## Appendix A: API Usage

The system is packaged as an OpenAI-compatible REST API.

**Regex constraint:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-speculative",
    "messages": [{"role": "user", "content": "Generate an order ID:"}],
    "regex": "[A-Z]{3}-[0-9]{4}"
  }'
```

**JSON Schema constraint:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-speculative",
    "messages": [{"role": "user", "content": "Generate user data:"}],
    "json_schema": {
      "type": "object",
      "required": ["name", "age"],
      "properties": {
        "name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
        "age": {"type": "integer"},
        "tags": {
          "type": "array",
          "minItems": 1,
          "maxItems": 3,
          "items": {"type": "string", "maxLength": 10}
        }
      }
    }
  }'
```

Both `regex` and `json_schema` are extensions to the OpenAI API that enable grammar-constrained generation.

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML*.
2. Willard, B. & Louf, R. (2023). Efficient Guided Generation for Large Language Models. *arXiv:2307.09702*.
3. Apple Machine Learning Research. (2023). MLX: An Efficient Machine Learning Framework for Apple Silicon.
4. Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP*.
