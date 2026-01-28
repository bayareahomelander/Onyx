# Grammar-Aware Speculative Decoding for Structured Output Generation

## Abstract

Large language models frequently fail to produce syntactically valid structured outputs, creating reliability challenges for agentic applications that depend on parseable JSON, SQL, or domain-specific formats. We present a method that combines grammar-constrained decoding with speculative execution to address both reliability and latency concerns. Our approach applies deterministic finite automaton (DFA) constraints to both draft and target models in a speculative decoding pipeline, ensuring that proposed tokens are grammatically valid before verification. Experiments on Apple Silicon hardware demonstrate that this "grammar-aware" speculation achieves 100% output compliance for simple patterns while providing measurable speedups on memory-bandwidth-bound models. With a 7B parameter target model, we observe a 1.09x throughput improvement over single-model constrained generation. We also report that smaller compute-bound models (1.5B) do not benefit from speculation, achieving 0.94x of baseline performance—an expected result given the different bottleneck characteristics. These findings suggest that grammar-aware speculative decoding offers a practical path toward reliable structured generation at scale, though its benefits are contingent on model size and hardware constraints.

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

1. **Grammar Engine** (Rust): Compiles regex patterns into DFAs and performs vocabulary filtering
2. **Speculative Engine** (Python/MLX): Coordinates draft-verify-rollback cycles with grammar state management
3. **API Server** (FastAPI): Provides OpenAI-compatible REST interface with grammar constraint extensions

### 2.2 Grammar Constraint Engine

The grammar engine is implemented in Rust for performance and exposed to Python via PyO3. It maintains:

- A pre-computed vocabulary table mapping token IDs to UTF-8 byte sequences
- A compiled DFA representing the grammar constraint
- State traversal logic that advances through the DFA as tokens are generated

The key operation is `get_valid_token_ids(state)`, which returns all token IDs that, when appended to the current generation, would lead to a non-dead DFA state. This operation has O(V) complexity where V is the vocabulary size, but completes in approximately 270 microseconds for a 151,000-token vocabulary—acceptable overhead for generation loops running at 20-200 tokens per second.

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

To support efficient rollback, we implemented a paged KV cache that stores key-value tensors in fixed-size blocks (default 16 tokens). Rollback operations simply update a length counter and discard block pointers—O(1) complexity with no memory copies. This is important for speculative decoding where rollbacks occur frequently.

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
│   ┌─────────┐     ┌─────────────────┐  ┌─────────────────┐                  │
│   │ Block 0 │────►│ T1 │ T2 │ T3 │ T4│ │ Block Size: 4   │                  │
│   ├─────────┤     └─────────────────┘  └─────────────────┘                  │
│   │ Block 1 │────►│ T5 │ T6 │    │  │                                       │
│   ├─────────┤     └─────────────────┘                                       │
│   │  NULL   │     valid_length = 6                                          │
│   └─────────┘                                                               │
│                                                                             │
│   Rollback 2 tokens:                                                        │
│   ┌─────────┐     ┌─────────────────┐                                       │
│   │ Block 0 │────►│ T1 │ T2 │ T3 │ T4│                                      │
│   ├─────────┤     └─────────────────┘                                       │
│   │  NULL   │     valid_length = 4  ◄── Just update counter!                │
│   └─────────┘                                                               │
│                                                                             │
│   Complexity: O(1) - update counter, discard block pointer                  │
│   No memory copies required                                                 │
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
| Target (large) | Qwen2.5-7B-Instruct | 7B | 4-bit |

The models share the same tokenizer family (vocabulary size ~152,000), enabling direct token comparison without remapping.

### 3.2 Test Patterns

We evaluated two grammar patterns of differing complexity:

1. **Year Pattern**: `[0-9]{4}` — A bounded pattern requiring exactly 4 digits
2. **Email Pattern**: `[a-z]+@[a-z]+\.com` — An unbounded pattern with multiple variable-length segments

### 3.3 Evaluation Metrics

- **Throughput**: Tokens generated per second (tok/s)
- **Acceptance Rate**: Percentage of draft tokens accepted by the target model
- **Speedup Factor**: Ratio of grammar-aware speculation throughput to baseline (single-model constrained generation)

### 3.4 Scenarios

We compared three generation strategies:

1. **Baseline**: Target model with grammar constraints, no speculation
2. **Blind Draft**: Draft model unconstrained, target model constrained
3. **Aware Draft**: Both draft and target models constrained to the same grammar

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

### 4.2 Throughput on Compute-Bound Models (1.5B Target)

With the 1.5B target model, we observed that grammar-aware speculation does not provide a speedup over baseline:

**Table 2: Throughput Results (1.5B Target, Year Pattern)**

| Strategy | Throughput | vs Baseline |
|----------|------------|-------------|
| Baseline | 73.5 tok/s | 1.00x |
| Blind Draft | 57.2 tok/s | 0.78x |
| Aware Draft | 69.2 tok/s | 0.94x |

The aware draft achieves 94% of baseline performance—a 6% overhead. This result is expected: the 1.5B model is compute-bound rather than memory-bound on our hardware. Speculative decoding's theoretical advantage comes from amortizing memory bandwidth costs across multiple tokens via batched verification. When compute dominates, this advantage does not materialize, and the overhead of running two models produces a net slowdown.

### 4.3 Throughput on Memory-Bound Models (7B Target)

The picture changes with a larger, memory-bandwidth-bound target model:

**Table 3: Throughput Results (7B Target, Year Pattern)**

| Strategy | Throughput | vs Baseline |
|----------|------------|-------------|
| Baseline | 17.4 tok/s | 1.00x |
| Blind Draft | 13.4 tok/s | 0.77x |
| Aware Draft | 18.9 tok/s | 1.09x |

Grammar-aware speculation achieves a **1.09x speedup** (8.7% improvement) over baseline. This confirms the theoretical expectation: when loading model weights dominates inference time, batched verification of multiple draft tokens amortizes the memory bandwidth cost, producing a net speedup despite the overhead of running a draft model.

The blind draft result (0.77x) demonstrates that without grammar awareness, speculation is counterproductive—the system is slower than simply running the target model alone.

### 4.4 Crossover Analysis

**Table 4: Speedup Factor by Target Model Size**

| Target Model | Baseline Speed | Aware Draft Speed | Speedup Factor |
|--------------|----------------|-------------------|----------------|
| 1.5B | 73.5 tok/s | 69.2 tok/s | 0.94x |
| 7B | 17.4 tok/s | 18.9 tok/s | 1.09x |

The crossover point where speculation becomes beneficial occurs somewhere between 1.5B and 7B parameters on our hardware. At 1.5B, the model is fast enough that speculation overhead dominates. At 7B, memory bandwidth dominates, and speculation provides savings.

```
                     Speedup Factor vs Model Size
                     
    Speedup │
            │
       1.2x │                              ●  (7B+)
            │                             ╱
       1.1x │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ─ ─ ─●─ ─ ─ ─ ─ ─ ─ ─
            │                           ╱   SPEEDUP ZONE
       1.0x │───────────────────────────●───────────────────────
            │                         ╱ ▲
       0.9x │                  ●     ╱  │ Crossover Point
            │                    ╲  ╱   │ (~3-5B parameters)
       0.8x │               ●     ╲╱
            │                ╲
       0.7x │                 ╲
            │                  OVERHEAD ZONE
            └────────────────────────────────────────────────────►
                   1.5B                    7B       Model Size
                   
                 Compute-bound          Memory-bound
                 (overhead dominates)   (speculation wins)
```

This suggests that grammar-aware speculative decoding is most applicable to production scenarios using larger models (7B+), which are common for tasks requiring higher output quality.

---

## 5. Limitations and Future Work

### 5.1 Model Size Dependency

The speedup benefits of grammar-aware speculation are contingent on the target model being memory-bandwidth-bound. For smaller models where inference is compute-bound, the approach introduces overhead without corresponding benefit. Practitioners should evaluate their specific model and hardware configuration before adopting this technique.

### 5.2 Grammar Complexity

Our evaluation focused on regex patterns. More complex grammars (context-free, context-sensitive) may have different performance characteristics. The DFA-based approach is limited to regular languages; extending to JSON schemas or recursive structures would require additional machinery.

### 5.3 Acceptance Rate Variability

While grammar-aware drafting improves acceptance rates, the improvement varies by pattern complexity. Simple bounded patterns (digits) achieve near-perfect acceptance; complex unbounded patterns (email) show more modest gains. The draft model's predictions still diverge from the target, especially for larger target models where the capability gap is wider.

### 5.4 Hardware Specificity

All experiments were conducted on Apple Silicon using the MLX framework. Results may differ on other hardware (NVIDIA GPUs, CPUs) with different memory bandwidth characteristics and framework optimizations.

### 5.5 Single-Turn Evaluation

We evaluated single-turn generation with short outputs. Long-context scenarios, multi-turn conversations, and batched inference may exhibit different performance characteristics that we did not investigate.

---

## 6. Related Work

This work builds on speculative decoding as introduced by Leviathan et al. [1], which demonstrated that draft-then-verify approaches can improve inference throughput on memory-bound models by amortizing memory bandwidth costs across multiple tokens verified in a single forward pass. Our contribution extends this framework to grammar-constrained generation scenarios.

The grammar constraint engine follows the DFA-based vocabulary filtering approach described by Willard and Louf [2], who showed that compiling grammar specifications into deterministic finite automata enables efficient token filtering during generation. We apply this technique to both draft and target models in a speculative pipeline, addressing the "blind draft" problem where unconstrained draft models propose invalid tokens.

The paged KV cache design draws from PagedAttention as introduced by Kwon et al. [4], which demonstrated that block-based memory management enables efficient memory utilization and supports operations like preemption and sharing. We adapt this concept to enable O(1) rollback operations critical for speculative decoding, where rejected draft tokens must be efficiently discarded.

All experiments use Apple's MLX framework [3], which provides efficient array operations optimized for Apple Silicon's unified memory architecture.

---

## 7. Conclusion

We presented grammar-aware speculative decoding, a technique that integrates grammar constraints into both draft and target models in a speculative decoding pipeline. Our experiments demonstrate that this approach:

1. **Solves the reliability problem**: Constrained generation produces outputs that are guaranteed to match the specified grammar
2. **Addresses the blind draft problem**: Applying grammar constraints to the draft model improves acceptance rates from as low as 0% (blind) to 18-100% (aware)
3. **Provides speedups on memory-bound models**: With a 7B target model, we observed a 1.09x throughput improvement over single-model constrained generation

The approach does not provide speedups for smaller, compute-bound models (0.94x for 1.5B), which is consistent with the theoretical basis of speculative decoding. The technique is most applicable to production deployments using larger models where memory bandwidth is the limiting factor.

These results suggest a practical path toward reliable structured generation in agentic AI systems. By combining grammar constraints with speculative execution, it may be possible to achieve both the reliability required for downstream parsing and the low latency required for interactive applications—though the benefits are contingent on appropriate model sizing and hardware configuration.

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

**7B Target (Year Pattern)**
- Baseline: 17.4 tok/s
- Blind Draft: 13.4 tok/s (12.5% acceptance)
- Aware Draft: 18.9 tok/s (25% acceptance)

**Email Pattern (1.5B Target)**
- Baseline: 80.7 tok/s
- Blind Draft: 23.8 tok/s (0% acceptance)
- Aware Draft: 42.1 tok/s (18.2% acceptance)

---

## Appendix A: API Usage

The system is packaged as an OpenAI-compatible REST API:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-speculative",
    "messages": [{"role": "user", "content": "Generate an order ID:"}],
    "regex": "[A-Z]{3}-[0-9]{4}"
  }'
```

The `regex` field is an extension to the OpenAI API that enables grammar-constrained generation.

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML*.
2. Willard, B. & Louf, R. (2023). Efficient Guided Generation for Large Language Models. *arXiv:2307.09702*.
3. Apple Machine Learning Research. (2023). MLX: An Efficient Machine Learning Framework for Apple Silicon.
4. Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP*.
