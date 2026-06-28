# Onyx

![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![Python](https://img.shields.io/badge/python-3.12%20recommended-blue)

**Grammar-Aware Speculative Decoding for Structured LLM Outputs on Apple Silicon**

Onyx is a prototype inference engine for grammar-constrained structured output on Apple Silicon. In the recorded 8B year-pattern benchmark, grammar-aware speculation ran at 18.9 tok/s versus a 17.4 tok/s constrained baseline (**1.09x**); results are workload- and hardware-specific.

## What's New in v0.2.0
- **Shared Rust State Architecture:** Replaced deep cloning with `Arc` (Atomic Reference Counting) pointers for immutable schema blueprints and regex automata, reducing state-cloning overhead during vocabulary masking.
- **Enhanced Benchmark Accuracy:** Tightened benchmark regex scripts so forced long-generation runs are measured consistently.

---

## Key Results

| Configuration | Baseline | Onyx (Aware Draft) | Speedup |
|---------------|----------|---------------------|---------|
| 8B Target (year pattern) | 17.4 tok/s | 18.9 tok/s | **1.09x** |
| 1.5B Target (year pattern) | 73.5 tok/s | 69.2 tok/s | 0.94x |

- **Evaluated Grammar Compliance**: Recorded benchmark cases completed with grammar-compliant output; callers must check for `finish_reason="grammar_complete"`
- **Adaptive Speculation**: Experimental adaptive γ controller matches the best fixed γ setting on an 8B forced-digits benchmark (29.2 tok/s vs 21.9 tok/s baseline, **1.34x**)
- **OpenAI-Shaped API**: Prototype chat-completions and model-list endpoints for local demonstrations
- **JSON Schema Subset**: Nested objects, typed arrays, regex patterns, enums, unions, and length constraints

---

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M-Series)
- Python 3.12 recommended
- Rust toolchain
- Metal GPU access for MLX runtime execution

> The project metadata still allows Python 3.10+, but the tested setup path is Python 3.12 on Apple Silicon. If you use `pyenv`, the included `.python-version` selects Python 3.12 automatically. MLX may fail with `No Metal device available` in headless, sandboxed, or virtualized sessions where the GPU is not exposed.

### Installation

```bash
# Clone and enter directory
git clone https://github.com/bayareahomelander/Onyx.git
cd Onyx

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install build, runtime, and API dependencies
python -m pip install -U pip "maturin>=1.4,<2.0"
python -m pip install -e ".[server]"

# Build the Rust extension
python -m maturin develop --release
```

### Start the API Server

```bash
uvicorn onyx.server:app --host 0.0.0.0 --port 8000
```

### Make a Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-speculative",
    "messages": [{"role": "user", "content": "Generate user data:"}],
    "max_tokens": 100,
    "json_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      },
      "required": ["name"]
    },
    "compact_json": true
  }'
```

---

## JSON Schema Constraints

Onyx includes a structure-aware JSON generation engine built in Rust. It enforces the supported schema subset at token level. Output should be treated as schema-complete only when generation returns `finish_reason="grammar_complete"`; length-limited output can be incomplete.

Unsupported keywords—including `$ref`, `additionalProperties`, `oneOf`, `anyOf`, `allOf`, and conditional schemas—are not enforced. Invalid regex values in supported `pattern` fields are rejected during schema compilation.

### Supported Schema Features

| Feature | Schema Keyword | Description |
|---------|---------------|-------------|
| **Typed Properties** | `type` | `string`, `number`, `integer`, `boolean`, `null`, `object`, `array` |
| **Nested Objects** | `properties` | Arbitrary nesting depth with per-property schemas |
| **Required Fields** | `required` | Closing `}` blocked until all required keys are present |
| **Union Types** | `type: [...]` | e.g., `["string", "null"]` for nullable fields |
| **Enum Values** | `enum` | Restrict to a fixed set of allowed values |
| **Regex Patterns** | `pattern` | DFA-compiled regex validation on string content |
| **String Length** | `minLength`, `maxLength` | Closing `"` blocked below min; characters blocked above max |
| **Array Length** | `minItems`, `maxItems` | Closing `]` blocked below min; new items blocked above max |
| **Typed Arrays** | `items` | Schema enforcement on every array element |
| **Pretty Printing** | — | `\n`, `\t`, `\r` accepted in all structural positions |

### Basic Usage

```python
import json
from onyx._rust import GrammarConstraint

# Define your tokenizer vocabulary
vocab = [b'{', b'}', b'"name"', b':', b'"Alice"', b'"age"', b'25']

# Initialize with vocab
gc = GrammarConstraint(vocab)

# Compile a JSON schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name"]
}
gc.compile_json_schema(json.dumps(schema))

# During generation: get valid token mask at each step
state = gc.init_state()
valid_ids = gc.get_valid_token_ids(state)

# Advance with a token
state = gc.advance_state(state, token_id)
```

### Constrained Generation Loop

```python
import json
from onyx._rust import GrammarConstraint

def generate_constrained(model, tokenizer, schema_str, prompt, max_tokens=200):
    """Generate tokens with JSON schema enforcement."""
    # Build vocab from tokenizer
    vocab = [tokenizer.decode([i]).encode() for i in range(tokenizer.vocab_size)]

    gc = GrammarConstraint(vocab)
    gc.compile_json_schema(schema_str)

    state = gc.init_state()
    input_ids = tokenizer.encode(prompt)
    output_ids = []

    for _ in range(max_tokens):
        # Get logits from model
        logits = model(input_ids + output_ids)

        # Mask invalid tokens
        valid = gc.get_valid_token_ids(state)
        masked_logits = float('-inf') * torch.ones_like(logits)
        masked_logits[valid] = logits[valid]

        # Sample from masked distribution
        token_id = torch.argmax(masked_logits).item()
        output_ids.append(token_id)
        state = gc.advance_state(state, token_id)

        if gc.is_match_state(state):
            break

    return tokenizer.decode(output_ids)
```

### Advanced Schema Example

Onyx handles complex, real-world schemas with multiple constraint types applied simultaneously:

```python
schema = {
    "type": "object",
    "required": ["user_id", "profile"],
    "properties": {
        "user_id": {"type": "integer"},
        "profile": {
            "type": "object",
            "required": ["name", "tags"],
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[A-Z][a-z]+$"
                },
                "age": {
                    "type": ["number", "null"]
                },
                "status": {
                    "enum": ["active", "suspended"]
                },
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string", "maxLength": 5}
                }
            }
        }
    }
}
```

This schema enforces: integer-only IDs, capitalized names via regex, nullable age, enum-restricted status, and a bounded array of short string tags—all at the token level during generation.

> See [`examples/json_generate.py`](examples/json_generate.py) for a complete runnable example.

---

## Regex Constraints

Enforce regex patterns directly during generation. A result matches the pattern when generation finishes with `finish_reason="grammar_complete"`; `finish_reason="length"` indicates an incomplete result.

```python
from onyx.speculative import SpeculativeEngine

engine = SpeculativeEngine()

# Generate a valid email address
output, metrics = engine.generate(
    prompt="Contact email: ",
    max_tokens=50,
    regex=r"[a-z]+@[a-z]+\.com"
)
print(output)  # e.g., "support@example.com"
```

---

## OpenAI-Shaped API

The local prototype accepts the subset of the OpenAI chat-completions shape documented below:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="onyx-speculative",
    messages=[{"role": "user", "content": "Generate a product SKU:"}],
    extra_body={"regex": "[A-Z]{2}-[0-9]{6}"}
)
```

### Streaming Support

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-speculative",
    "messages": [{"role": "user", "content": "Generate ID:"}],
    "stream": true,
    "regex": "[A-Z]{3}-[0-9]{4}"
  }'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                           │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │   MLX-LM     │  │ SpeculativeEngine│  │  FastAPI      │  │
│  │  (Models)    │  │ (Draft + Target) │  │  (REST API)   │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────┬───────┘  │
│         │          ┌────────▼─────────┐          │          │
│         │          │ AdaptiveGamma    │          │          │
│         │          │ Controller (exp) │          │          │
│         │          └──────────────────┘          │          │
├─────────┴───────────────────┴────────────────────┴──────────┤
│                      Rust Backend (PyO3)                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              GrammarConstraint Engine                  │ │
│  │  • JSON Schema Engine (stack-based state machine)      │ │
│  │  • DFA Compilation (regex-automata)                    │ │
│  │  • Vocabulary Filtering (O(V) per token)               │ │
│  │  • Full-vocabulary state traversal                    │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Compile Grammar**: The JSON schema or regex pattern is compiled into internal state machines (stack-based FSM for JSON, DFA for regex)
2. **Filter Vocabulary**: At each generation step, only tokens leading to valid states are allowed
3. **Speculative Decoding**: A small draft model (0.5B) proposes tokens that a larger target model (1.5B/8B) verifies
4. **Grammar-Aware Drafting**: Both models are constrained to the grammar, which can improve draft acceptance
5. **Adaptive Draft Length (Experimental)**: An optional controller adjusts γ based on recent acceptance rate and timing signals

---

## Benchmarks

All benchmarks run on Apple Silicon with 4-bit quantized Qwen models.

### Acceptance Rate (1.5B Target)

| Pattern | Blind Draft | Aware Draft |
|---------|-------------|-------------|
| Year `[0-9]{4}` | 75.0% | 100.0% |
| Email `[a-z]+@[a-z]+\.com` | 0.0% | 18.2% |

### Throughput

| Configuration | Baseline | Aware Draft | vs Baseline |
|---------------|----------|-------------|-------------|
| 1.5B Target | 73.5 tok/s | 69.2 tok/s | 0.94x |
| 8B Target | 17.4 tok/s | 18.9 tok/s | **1.09x** |

These two recorded runs show a slowdown at 1.5B and a modest speedup at 8B. They do not establish a universal crossover model size.

### Adaptive Gamma Benchmark

The experimental adaptive path compares fixed speculative batch sizes against a controller initialized at γ=4 with bounds `[1, 8]`. Warmup runs are excluded from the reported averages.

| Configuration | Throughput | vs Baseline | Acceptance |
|---------------|------------|-------------|------------|
| 8B Baseline (`[0-9]{32}`) | 21.9 tok/s | 1.00x | — |
| Fixed γ=2 | 25.3 tok/s | 1.16x | 93.8% |
| Fixed γ=4 | 29.1 tok/s | **1.33x** | 88.2% |
| Fixed γ=8 | 27.6 tok/s | 1.26x | 78.9% |
| Adaptive γ | 29.2 tok/s | **1.34x** | 88.2% |

In this run, adaptive γ matched the best fixed setting while adjusting during generation (avg γ=4.2, final γ=8).

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-shaped chat completion endpoint for the supported prototype fields below.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier (default: `onyx-speculative`) |
| `messages` | array | Chat messages |
| `max_tokens` | integer | Maximum tokens to generate |
| `temperature` | number | Must be `0`; speculative generation is currently greedy-only |
| `top_p` | number | Must be `1`; speculative generation is currently greedy-only |
| `stream` | boolean | Enable streaming response |
| `stop` | array<string> | Stop strings. Multi-token stop strings are tokenized as full sequences and removed from returned text. |
| `regex` | string | **(Onyx extension)** Regex pattern to constrain output |
| `json_schema` | object | **(Onyx extension)** JSON Schema to constrain output |
| `compact_json` | boolean | **(Onyx extension)** Strip whitespace from JSON output (default: `true`) |

> Prototype note: stop strings are matched after tokenizer encoding. Streaming responses also buffer a small text suffix so stop strings split across token chunks are not emitted.

> Sampling contract: the speculative API models reject non-greedy settings with HTTP 400. Target-only `SpeculativeEngine.generate_baseline()` remains available for sampling experiments.

### `GET /v1/models`

List available models.

### `GET /`

Health check endpoint.

---

## Development

### Experimental CUDA Component

Onyx includes a separate `onyx_cuda` package for CUDA-oriented constrained
decoding experiments. The first component is a sparse masked-argmax CUDA kernel
that selects the highest-logit token from a grammar-valid token ID set.

This is not a full CUDA inference backend and does not replace the MLX runtime.
It is a focused kernel-level experiment for understanding decode-loop bottlenecks
around grammar masking, sparse token selection, and GPU launch overhead.

```bash
python -m pip install -U "maturin>=1.4,<2.0"
python -m pip install -e ".[cuda,dev]"
python -m pytest tests/test_cuda_masked_argmax.py -q
python -m pytest tests/test_cuda_decode_loop.py -q
python -m pytest tests/test_cuda_tokenizer_probe.py -q
python -m pytest tests/test_cuda_real_logits_handoff.py -q
python probe_cuda_tokenizer.py
python probe_cuda_real_logits.py --local-files-only
python benchmark_cuda_masked_argmax.py
python benchmark_cuda_grammar_handoff.py
python benchmark_cuda_decode_loop.py
```

On Windows, the benchmark also needs the NVIDIA CUDA Toolkit with `nvcc` on
`PATH`, compatible Microsoft C++ Build Tools, and a CUDA-enabled PyTorch build.
The current CUDA experiment is intended for editable-source-tree development.
The tokenizer probe loads only the `Qwen/Qwen2.5-0.5B-Instruct` tokenizer and
configuration metadata. The real-logits probe then loads the same pinned model
with Quanto INT4 weights, runs one CUDA forward pass, verifies the observed
logits width, and selects one grammar-valid token. It is intentionally not yet a
multi-token generation loop.

See [`onyx_cuda/README.md`](onyx_cuda/README.md) for scope and constraints.

### Verification Scripts

```bash
# Regex-constrained generation (single model, correctness focus)
python run_grammar.py

# Speculative decoding + grammar (benchmark, acceptance rate focus)
python run_speculative_grammar.py

# API server end-to-end (requires server running)
uvicorn onyx.server:app --port 8000
python test_api.py
```

```bash
# Build Rust extension in development mode
maturin develop

# Run all tests
pytest tests/

# Run API tests (requires server running)
python test_api.py

# Run adaptive speculative benchmark
python run_adaptive_speculative.py

# Build release
maturin develop --release
```

---

## Citation

If you use Onyx in your research, please cite:

```bibtex
@software{onyx2026,
  title = {Onyx: Grammar-Aware Speculative Decoding for Structured LLM Outputs},
  year = {2026},
  url = {https://github.com/bayareahomelander/Onyx}
}
```

---

## License

The package metadata declares the project under the MIT License.

---

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's machine learning framework
- [PyO3](https://github.com/PyO3/pyo3) — Rust bindings for Python
- [regex-automata](https://github.com/BurntSushi/regex-automata) — DFA construction
