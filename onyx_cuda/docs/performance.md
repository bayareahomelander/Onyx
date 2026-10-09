# Onyx CUDA performance

The [README](../README.md#measured-performance) summarizes these results. This
page gives the method, both recovery modes, earlier measurements, and validation.

## Latest comparison

On a Linux RTX 2080 Ti reporting 22 GiB, the latest September 28, 2026 comparison
covered 48 cases with one warmup and three measured runs per mode. Aggregates sum
per-case median generation times, excluding model loading and graph setup:

| Mode | Aggregate speedup over target-only |
| --- | ---: |
| Fixed gamma 3, draft graphs, scalar recovery (`ONYX_REPLAY_BACKEND=scalar`) | 1.370x |
| Fixed gamma 3, draft graphs, graph recovery (default on this GPU) | **1.492x** |

The same comparison by workload category, as labeled in
[benchmark_corpus.py](../src/onyx_cuda/benchmark_corpus.py):

| Category | Cases | Prompts | Graph recovery | Scalar recovery |
| --- | ---: | --- | ---: | ---: |
| Code | 6 | Python, TypeScript, and JavaScript functions and a SQL query | 2.439x | 2.438x |
| JSON | 4 | JSON Schema: a boolean, an enum, an object, a 12-integer array | 2.112x | 2.110x |
| Regex | 4 | Regex: a year, 32 digits, a product code, repeated words | 2.071x | 2.070x |
| Extraction | 6 | Emails, dates, names, quantities, and CSV from given text | 1.852x | 1.849x |
| Text | 9 | The original nine prompts, one regex- and one JSON-constrained | 1.621x | 1.619x |
| Changing | 3 | Repetitive output that switches to free text, or the reverse | 1.392x | 1.213x |
| Prose | 8 | Explanations, stories, and other short prose, plus a long-context summary | 1.220x | 1.097x |
| Short | 8 | Answers of a number or a few words | 1.133x | 1.132x |

Each baseline uses the same target model, prompt, precision, and output budget.
Regex and JSON baselines enforce the same constraints; these figures measure
the benefit of speculation over already-constrained target-only generation.
The two recovery modes differ only where numerical recovery ran: four times in
the prose prompts and once in the changing prompts. Category results describe
the tested cases and do not guarantee a speedup for every request.

Every output in every mode matched target-only tokens and finish reasons. Nine
cases remain slower than target-only: six with very short outputs (at most 21 ms
slower) and three longer requests that needed numerical recovery.

## Earlier measurements

Earlier on September 28, before draft catch-up steps were folded into the next
draft step (2.1% less time), gamma 3 measured 1.346x / 1.462x and the previous
default, gamma 2, 1.292x / 1.397x. Gamma 3 improved or held every workload
category over gamma 2 and needed no additional numerical recoveries. The
September 25 comparison, which also timed the ordinary draft forward, measured
1.130x and 1.211x at gamma 2 without draft graphs; draft graphs reduced total
generation time by 12.6% and 13.4%.

## Time to first token

Speculative requests reach their first token as quickly as target-only requests:
the draft prefills the prompt only after the target has selected the first token,
and not at all when generation ends there. Across the 48 cases, speculation adds
0.3 ms in total to time to first token, down from 637 ms on September 27.

## Draft graphs and memory

Draft graphs cut draft cost from about 7.5-9.8 ms to 4.1-4.4 ms per token and
add about 4 seconds of startup; graph recovery preparation adds about 45 seconds.
Peak memory in the comparison was 17.31 GiB allocated. On September 27, forced
recoveries with 5,000 to 8,000 prompt tokens and both graph sets loaded matched
target-only output and peaked at 19.54 GiB allocated.

## Long constrained outputs

Constrained generation caches each grammar state's valid tokens and reuses them
as GPU masks, so long text inside a JSON string no longer rescans the vocabulary
at every token. In a separate September 27 check of four long-text requests (a
JSON summary, JSON records with descriptions, code in a JSON string, and a broad
regex), speculation measured 0.74x-2.01x against target-only, up from
0.49x-0.99x before the cache, with identical output. The JSON summary stays
slower because its one numerical recovery runs in scalar steps: graph recovery
does not apply to constrained requests.

## Validation

On October 8 the full CUDA suite passed all 858 tests with gamma 3, draft graphs,
and default graph recovery. September 22 checks matched full logits and KV
caches bitwise through 8192 tokens with graph recovery, and the API completed a
4096-prompt/4096-output capacity test. If a graph block still exhausts GPU
memory, that recovery continues with scalar steps and later requests keep the
graphs. Draft graphs have not yet been validated on a Windows GPU.

## Reproducing

To reproduce the comparison on your GPU, use a new output filename for each run:

```sh
python -m onyx_cuda.benchmark --output validation/speedups.json
```

One interleaved run times target-only and fixed gamma 3 generation, and adds a
graph-recovery mode when the GPU supports it (compute capability 7.5).
`--gamma 2 3` compares several gammas in the same run.
Every mode uses the configured draft backend, draft graphs by default. Every
speculative output must match target-only generation token for token. The report
records which modes ran and why a graph backend was unavailable. Each speculative
run also records its proposal, acceptance, and recovery counts, and a graph mode
fails if any of its recoveries fell back to scalar steps.
