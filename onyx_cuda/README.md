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

`onyx_cuda.model.load_model_pair()` loads that model as the draft together with `Qwen/Qwen2.5-1.5B-Instruct` as the FP16 target. It rejects the pair unless logits widths, every token ID's decoded bytes, special/EOS IDs, and chat-template output are identical. Both models fit and complete cached forwards together on the validated 6 GB GPU. The same `generate_tokens()` path is verified against Transformers greedy output for the 1.5B target, including regex and JSON constraints.

`onyx_cuda.speculative.generate_speculative()` runs greedy fixed-gamma decoding with `gamma >= 1`, exact `max_tokens`, EOS, and overlapping token-stop handling and returns the same token/cache/finish-reason result shape as `generate_tokens()`. It accepts the same `regex`, `json_schema`, and `token_byte_vocabulary` inputs, masks both draft and target choices, verifies from the canonical grammar branch, and releases every temporary state. Positive-temperature requests, including constrained requests, are routed unchanged through the trustworthy target-only temperature/top-p sampler; sampled speculative acceptance is not approximated. Streaming is not implemented yet.

The Qwen chat template uses `<|im_end|>` (ID 151645) as EOS, `<|endoftext|>` (ID 151643) as padding, and no BOS token. A formatted prompt ends with `<|im_start|>assistant\n` rather than EOS so generation can begin. `onyx_cuda.server.create_app()` builds a FastAPI app that loads the configured 0.5B/1.5B pair once at startup, serves `GET /`, `GET /v1/models`, and non-streaming `POST /v1/chat/completions` from the engine registry, and drops registry plus CUDA cache memory at shutdown. Request preparation uses the tokenizer chat template with a generation prompt and the documented `System:`/`User:`/`Assistant:` fallback when that template is missing. HTTP generation enables synchronized timings; `n > 1` reports Onyx metrics from the final choice and sums completion tokens. Streaming is not implemented yet.

The HTTP API preserves request-validation responses as 422, returns 400 for unknown models and invalid constraints, returns 503 when model or CUDA execution is unavailable, and returns a generic 500 response for unexpected failures without exposing traceback details.

The single prefill returns last-position vocabulary logits, a Transformers dynamic KV cache on CUDA, and one greedy CUDA token. Generation reuses that cache and supports greedy decoding at temperature zero or seeded temperature/top-p sampling. It reports `eos`, `stop`, or `length`; explicit token stop sequences can span tokens, and the longest matching suffix is removed before decode. Beam search, batching, repetition penalties, text-fragment buffering, and SSE are not implemented yet.

Pass `measure=True` to `generate_tokens()` to include synchronized time to first token, decode tokens per second, and total generation time in `result.timings`. Constrained calls also report grammar setup, valid-token enumeration, and CUDA mask/ID-transfer time. Greedy `generate_speculative(..., measure=True)` uses the same result field and additionally reports proposed and accepted token counts, the accepted/proposed rate as a value from zero to one, iteration count, and synchronized draft, verify, and combined grammar-mask seconds. Draft and verify seconds exclude the separately reported mask time.

The models are downloaded to the external Hugging Face cache. Loading fails instead of falling back to CPU when CUDA is unavailable.

The native extension also exposes `onyx_cuda._rust.GrammarConstraint` for model-free regex and JSON-schema compilation with branchable opaque state handles. `onyx_cuda.vocabulary.build_token_byte_vocabulary()` maps the complete model-logit ID space to standalone UTF-8 bytes and reports special/empty-token counts. `onyx_cuda.masking.apply_grammar_mask()` returns a new CUDA logits tensor with invalid token IDs set to negative infinity. Pass `regex=...` or `json_schema=...` and the matching `token_byte_vocabulary` to `generate_tokens()` for constrained decoding; JSON Schema takes precedence when both constraints are supplied. Request preparation serializes JSON Schema once and forwards regex, schema, temperature, top-p, and stop sequences to the Phase 4 generation options; JSON Schema still takes precedence over regex. Non-streaming chat completions compact schema output only after `json.loads` succeeds. Streaming is not implemented yet.

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

The optional `server` extra pins FastAPI 0.141.1, Pydantic 2.13.5, Uvicorn 0.52.4, and httpx 0.28.1. `.[dev]` includes that extra so request/response models, `create_app()`, and non-streaming chat completions import. Streaming is not implemented yet.

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

## Benchmark gates

After installing the development environment, run the fixed greedy benchmark from this directory:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark
```

It warms up three fixed prompts, measures each three times, verifies stable token IDs, termination reasons, and post-run CUDA allocation, then writes raw results to the ignored `benchmarks/results/phase2_baseline.json` file.

After retaining that baseline, run the Phase 3 constraint gate:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --constraints
```

It reruns and compares the unconstrained outputs with the Phase 2 file, then measures deterministic regex and JSON-schema generation. The ignored `benchmarks/results/phase3_constraint_gate.json` records grammar setup, valid-token enumeration, mask/transfer, throughput, and peak allocated VRAM.

Run the 1.5B target-only correctness baseline before speculative work:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --target
```

It uses the same prompts, generation loop, warmups, repetitions, timing, and memory checks and writes `benchmarks/results/phase4_target_baseline.json`. Validate target constraints against that retained baseline with:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --target --constraints
```

That command writes `benchmarks/results/phase4_target_constraint_gate.json`. Retain it, then run the complete fixed-gamma gate:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --speculative
```

The speculative gate loads the compatible pair; measures all three unconstrained prompts plus the regex and JSON cases at `gamma` 1, 2, and 4 after warmup; checks every output and termination reason against the 1.5B target-only file; rejects post-run allocation drift; and records TTFT, total/output throughput, proposal acceptance, iteration and stage timings, and peak allocated VRAM in ignored `benchmarks/results/phase4_speculative_gate.json`. It reports the gamma with the highest median per-prompt output throughput across the five cases and prints the measured target-time ratio even when speculation is slower.
