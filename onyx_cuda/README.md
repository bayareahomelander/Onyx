# Onyx CUDA

Onyx CUDA is the Windows and NVIDIA CUDA edition of Onyx, currently under development.

## Current capability

The package can load the `Qwen/Qwen2.5-0.5B-Instruct` tokenizer and FP16 causal model on `cuda:0` without CPU offload:

```python
from onyx_cuda.generation import generate_tokens
from onyx_cuda.model import load_model
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt

loaded = load_model()
print(loaded.revision)

prompt = format_prompt(
    loaded.tokenizer,
    [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Reply with CUDA ready."},
    ],
)
print(prompt.text)
print(prompt.token_ids)

result = prefill(loaded.model, prompt.token_ids)
print(loaded.tokenizer.decode(result.token_id.tolist()))
print(result.past_key_values.get_seq_length())

stop_sequences = [loaded.tokenizer.encode(" Ready", add_special_tokens=False)]
generated = generate_tokens(
    loaded.model,
    prompt.token_ids,
    max_tokens=16,
    eos_token_ids=loaded.tokenizer.eos_token_id,
    stop_sequences=stop_sequences,
    temperature=0.8,
    top_p=0.9,
    seed=1234,
)
print(loaded.tokenizer.decode(generated.token_ids, skip_special_tokens=True))
print(generated.finish_reason)
```

The Qwen chat template uses `<|im_end|>` (ID 151645) as EOS, `<|endoftext|>` (ID 151643) as padding, and no BOS token. A formatted prompt ends with `<|im_start|>assistant\n` rather than EOS so generation can begin. API request models and fallback prompt formatting are not implemented yet.

The single prefill returns last-position vocabulary logits, a Transformers dynamic KV cache on CUDA, and one greedy CUDA token. Generation reuses that cache and supports greedy decoding at temperature zero or seeded temperature/top-p sampling. It reports `eos`, `stop`, or `length`; explicit token stop sequences can span tokens, and the longest matching suffix is removed before decode. Beam search, batching, repetition penalties, text-fragment buffering, and SSE are not implemented yet.

Pass `measure=True` to `generate_tokens()` to include synchronized time to first token, decode tokens per second, and total generation time in `result.timings`.

The model is downloaded to the external Hugging Face cache. Loading fails instead of falling back to CPU when CUDA is unavailable.

The native extension also exposes `onyx_cuda._rust.GrammarConstraint` for model-free regex and JSON-schema compilation with branchable opaque state handles. `onyx_cuda.vocabulary.build_token_byte_vocabulary()` maps the complete model-logit ID space to standalone UTF-8 bytes and reports special/empty-token counts. `onyx_cuda.masking.apply_grammar_mask()` returns a new CUDA logits tensor with invalid token IDs set to negative infinity. Constrained `generate_tokens()` integration is not implemented yet.

## Windows development setup

Prerequisites are 64-bit Windows, Python 3.12, an NVIDIA GPU and driver, the Rust MSVC toolchain, and Visual Studio Build Tools with the C++ workload.

Create the environment from this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
```

Install the CUDA wheel before the project dependency so pip cannot silently select a CPU-only PyTorch build. The validated baseline is:

| Component | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.6 |
| Maturin | 1.14.1 |
| Pytest | 9.1.1 |
| Rust/Cargo | 1.96.1 |
| NVIDIA driver | 610.88 |
| GPU | GeForce RTX 4050 Laptop GPU (6 GB) |

## Baseline benchmark

After installing the development environment, run the fixed greedy benchmark from this directory:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark
```

It warms up three fixed prompts, measures each three times, verifies stable token IDs, termination reasons, and post-run CUDA allocation, then writes raw results to the ignored `benchmarks/results/phase2_baseline.json` file.
