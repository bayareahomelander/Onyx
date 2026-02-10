# Onyx

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**Grammar-Aware Speculative Decoding for Structured LLM Outputs on Apple Silicon**

Onyx is an inference engine that enforces structured output constraints (JSON Schema, regex patterns) on language models while maintaining—or exceeding—baseline generation speed. By applying grammar constraints to both draft and target models in a speculative decoding pipeline, Onyx achieves **100% output reliability** with a **1.09x speedup** on memory-bound models (7B+).

---

## Key Results

| Configuration | Baseline | Onyx (Aware Draft) | Speedup |
|---------------|----------|---------------------|---------|
| 7B Target (memory-bound) | 17.4 tok/s | 18.9 tok/s | **1.09x** |
| 1.5B Target (compute-bound) | 73.5 tok/s | 69.2 tok/s | 0.94x |

- **100% Grammar Compliance**: Output always matches the specified schema or pattern
- **OpenAI-Compatible API**: Drop-in replacement for existing agent frameworks
- **Full JSON Schema Support**: Nested objects, typed arrays, regex patterns, enums, unions, and length constraints

---

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M-Series)
- Python 3.10+
- Rust toolchain

### Installation

```bash
# Clone and enter directory
git clone https://github.com/bayareahomelander/Onyx.git
cd Onyx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Build the Rust extension
maturin develop --release

# Install API dependencies
pip install fastapi uvicorn
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
    }
  }'
```

---

## JSON Schema Constraints

Onyx includes a structure-aware JSON generation engine built in Rust. It enforces schemas at the token level during generation, guaranteeing that every output is valid JSON matching the specified schema.

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
from onyx_rust import GrammarConstraint

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
from onyx_rust import GrammarConstraint

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

Enforce regex patterns directly during generation. The output is guaranteed to match.

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

## OpenAI-Compatible API

Use Onyx with any framework that supports the OpenAI API:

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
│         │                   │                    │          │
├─────────┴───────────────────┴────────────────────┴──────────┤
│                      Rust Backend (PyO3)                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              GrammarConstraint Engine                  │ │
│  │  • JSON Schema Engine (stack-based state machine)     │ │
│  │  • DFA Compilation (regex-automata)                   │ │
│  │  • Vocabulary Filtering (O(V) per token)              │ │
│  │  • State Traversal (~270µs per mask)                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Compile Grammar**: The JSON schema or regex pattern is compiled into internal state machines (stack-based FSM for JSON, DFA for regex)
2. **Filter Vocabulary**: At each generation step, only tokens leading to valid states are allowed
3. **Speculative Decoding**: A small draft model (0.5B) proposes tokens that a larger target model (1.5B/7B) verifies
4. **Grammar-Aware Drafting**: Both models are constrained to the grammar, ensuring high acceptance rates

---

## Benchmarks

All benchmarks run on Apple Silicon with 4-bit quantized Qwen2.5 models.

### Acceptance Rate (Grammar: `[0-9]{4}`)

| Method | 1.5B Target | 7B Target |
|--------|-------------|-----------|
| Blind Draft | 75% | 12.5% |
| **Aware Draft** | **100%** | **25%** |

### Throughput

| Configuration | Baseline | Aware Draft | vs Baseline |
|---------------|----------|-------------|-------------|
| 1.5B Target | 73.5 tok/s | 69.2 tok/s | 0.94x |
| 7B Target | 17.4 tok/s | 18.9 tok/s | **1.09x** |

The crossover point where speculation beats baseline occurs when the target model becomes memory-bandwidth-bound (typically 7B+ parameters).

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completion endpoint.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier (default: `onyx-speculative`) |
| `messages` | array | Chat messages |
| `max_tokens` | integer | Maximum tokens to generate |
| `stream` | boolean | Enable streaming response |
| `regex` | string | **(Onyx extension)** Regex pattern to constrain output |
| `json_schema` | object | **(Onyx extension)** JSON Schema to constrain output |

### `GET /v1/models`

List available models.

### `GET /`

Health check endpoint.

---

## Development

```bash
# Build Rust extension in development mode
maturin develop

# Run all tests
pytest tests/

# Run API tests (requires server running)
python test_api.py

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

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) — Apple's machine learning framework
- [PyO3](https://github.com/PyO3/pyo3) — Rust bindings for Python
- [regex-automata](https://github.com/BurntSushi/regex-automata) — DFA construction
