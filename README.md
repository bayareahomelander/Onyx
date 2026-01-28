# Onyx

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**Grammar-Aware Speculative Decoding for Structured LLM Outputs on Apple Silicon**

Onyx is an inference engine that enforces structured output constraints (JSON, regex patterns) on language models while maintaining—or exceeding—baseline generation speed. By applying grammar constraints to both draft and target models in a speculative decoding pipeline, Onyx achieves **100% output reliability** with a **1.09x speedup** on memory-bound models (7B+).

---

## Key Results

| Configuration | Baseline | Onyx (Aware Draft) | Speedup |
|---------------|----------|---------------------|---------|
| 7B Target (memory-bound) | 17.4 tok/s | 18.9 tok/s | **1.09x** |
| 1.5B Target (compute-bound) | 73.5 tok/s | 69.2 tok/s | 0.94x |

- **100% Grammar Compliance**: Output always matches the specified pattern
- **OpenAI-Compatible API**: Drop-in replacement for existing agent frameworks

---

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M-Series)
- Python 3.10+
- Rust toolchain

### Installation

```bash
# Clone and enter directory
git clone https://github.com/bayareahomelander/onyx.git
cd onyx

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
    "messages": [{"role": "user", "content": "Generate an order ID:"}],
    "max_tokens": 20,
    "regex": "[A-Z]{3}-[0-9]{4}"
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1769218707,
  "model": "onyx-speculative",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "ORD-1234"},
    "finish_reason": "grammar_complete"
  }],
  "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
  "onyx_metrics": {
    "tokens_per_second": 41.4,
    "acceptance_rate": 100.0,
    "ttft_ms": 78.8,
    "grammar_constrained": true,
    "speculative_iterations": 2
  }
}
```

---

## Features

### Grammar-Constrained Generation

Enforce regex patterns during generation. The output is guaranteed to match.

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

### OpenAI-Compatible API

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
│  │  • DFA Compilation (regex-automata)                    │ │
│  │  • Vocabulary Filtering (O(V) per token)               │ │
│  │  • State Traversal (~270µs per mask)                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

1. **Compile Grammar**: The regex pattern is compiled into a Deterministic Finite Automaton (DFA)
2. **Filter Vocabulary**: At each generation step, only tokens leading to valid DFA states are allowed
3. **Speculative Decoding**: A small draft model (0.5B) proposes tokens that a larger target model (1.5B/7B) verifies
4. **Grammar-Aware Drafting**: Both models are constrained to the grammar, ensuring high acceptance rates

This approach solves the "Blind Draft" problem where unconstrained draft models propose invalid tokens that the target rejects.

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

### `GET /v1/models`

List available models.

### `GET /`

Health check endpoint.

---

## Development

```bash
# Build Rust extension in development mode
maturin develop

# Run tests
pytest tests/

# Run API tests (requires server running)
python test_api.py

# Build release
maturin build --release
```

---

## Citation

If you use Onyx in your research, please cite:

```bibtex
@software{onyx2026,
  title = {Onyx: Grammar-Aware Speculative Decoding for Structured LLM Outputs},
  year = {2026},
  url = {https://github.com/bayareahomelander/onyx}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [PyO3](https://github.com/PyO3/pyo3) - Rust bindings for Python
- [regex-automata](https://github.com/BurntSushi/regex-automata) - DFA construction
