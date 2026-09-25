# CUDA benchmarks

[Setup](../README.md) · [Configuration](configuration.md) · [Validation](validation.md)

## Latest comparison: draft graphs (September 25, 2026)

On the Linux RTX 2080 Ti reporting 22,528 MiB VRAM, one interleaved run compared
the ordinary draft forward with draft graphs, each with scalar and graph
recovery, across all **48 corpus cases**. It used the pinned Qwen3-8B FP16 target
and Qwen2.5-0.5B-Instruct draft, fixed gamma 2, non-thinking greedy generation,
the Torch selector, one warmup and three measured runs per case and mode,
rotating measured order, and synchronized full-generation wall time. All 960
measured and warmup runs matched their case's target-only tokens and finish
reason, and the draft graph cache was released after every generation.

| Mode | Sum of per-case median latency | Speedup versus target-only |
| --- | ---: | ---: |
| Target-only | 92.75 s | 1.000x |
| Fixed gamma 2, ordinary draft, scalar recovery | 82.11 s | 1.130x |
| Fixed gamma 2, ordinary draft, graph recovery | 76.62 s | 1.211x |
| Fixed gamma 2, draft graphs, scalar recovery (default) | 71.79 s | 1.292x |
| Fixed gamma 2, draft graphs, graph recovery (opt-in) | **66.35 s** | **1.398x** |

Draft graphs reduced total latency by 12.6% with scalar recovery and 13.4% with
graph recovery, and no case was slower than with the ordinary draft. With draft
graphs and graph recovery, code averaged 2.02x target-only speed, regex 1.79x,
extraction 1.61x, JSON 1.58x, prose 1.20x, and short replies 1.06x. Nine cases
remain slower than target-only: six very short requests by at most 23 ms, and
the recovery-heavy `cache_long`, `count_then_explain`, and `comparison`.

A preceding stage profile explained the gain. The ordinary draft forward launched
about 1,350 GPU kernels per token and took about 7.6 ms per proposed token, over
a fifth of a 35 ms target decode step, although the draft reads about 1 GB of
weights per token. A draft graph step replays 712 kernels, about half as many,
and took 4.1-4.4 ms per token. Verifying three target positions cost 39 ms,
close to a single target decode step, leaving little launch overhead to remove there.

Startup is excluded from these timings: draft graphs took **2.0 seconds** and
graph recovery **44.8 seconds** to prepare. Peak benchmark memory was
**17.29 GiB allocated**, compared with 17.32 GiB for the ordinary draft. With draft
graphs enabled by default, the full CUDA suite passed **872 tests** with three
Windows-specific skips, and all 51 GPU tests restored their starting allocation;
the highest test peak was 18.69 GiB.

## Eight-token graph recovery (September 22, 2026)

Graph recovery replays unconstrained history in eight-token blocks, with
two/three-token blocks and scalar steps for remainders. An independent
ten-repetition comparison confirmed 1.92% lower fixed-mode latency and 2.56%
lower adaptive-mode latency than the previous two/three-token backend. No case
crossed the predeclared regression threshold of both 5% and 5 ms. Graph
preparation took 45.3 seconds versus 32.1 seconds for the previous backend.
Preallocating graph outputs reduced prepared reserved memory from 18.15 to
17.22 GiB; allocated memory increased slightly, from 16.47 to 16.52 GiB. Separate
checks matched full logits and KV caches bitwise through 8192 tokens.

## Graph-recovery comparison (September 17, 2026)

The graph-recovery implementation passed **578 Python tests**, with three
Windows-specific tests skipped and no failures, on a Linux RTX 2080 Ti reporting
22,528 MiB VRAM. All 49 GPU tests restored their starting CUDA allocation after
cleanup. This validates the Linux source implementation; Windows wheel delivery
still requires the separate [Windows delivery gate](validation.md).

A fresh run of all **48 corpus cases** compared the pinned Qwen3-8B FP16 target
and Qwen2.5-0.5B-Instruct draft using non-thinking greedy generation and the Torch
selector. Each mode received one warmup and three measured runs per case, with
rotating measured order and synchronized full-generation wall time. All **768
generation runs** matched their case's target-only reference tokens and finish
reason, including all 576 speculative runs.

| Mode | Sum of per-case median latency | Speedup versus target-only |
| --- | ---: | ---: |
| Target-only | 93.03 s | 1.000x |
| Fixed gamma 2, scalar recovery (default) | 83.63 s | 1.112x |
| Fixed gamma 2, graph recovery (opt-in) | **79.66 s** | **1.168x** |
| Adaptive speculation, graph recovery (opt-in) | 82.62 s | 1.126x |

Graph recovery reduced total fixed-speculation latency by **4.76%**; fixed graph
mode used **14.38% less time than target-only** and won on 34 of 48 cases.
Code averaged 1.65x target-only speed, extraction 1.39x, regex 1.48x, and JSON
1.13x. Prose was nearly tied at 1.02x, while short replies were slightly slower
at 0.96x. Recovery-heavy requests improved by 12–15% but some still trailed
target-only. Adaptive was slower overall than fixed graph mode and remains
experimental.

Startup is excluded from these timings: model loading took 9.6 seconds and
graph preparation added **28.2 seconds**. Graph storage remained resident across
all comparison modes; scalar and target-only runs detached the recovery backend.
Peak benchmark memory was **17.25 GiB allocated / 18.81 GiB reserved**, and final
allocated memory returned to zero. These corpus peaks do not represent worst-case
8192-token capacity; see the separate [context-limit checks](#graph-recovery-capacity-checks). Graph recovery
remains opt-in with `ONYX_REPLAY_BACKEND=graph` on its supported configuration.

## Graph-recovery capacity checks

Separate integration checks completed a 4096-token prompt plus 4096-token
constrained API completion with graphs retained (20.69 GiB peak reserved).
An additional live-cache stress case at the context limit exhausted graph
workspace; scalar fallback completed with exact logits and KV contents after
graphs were released. These checks are separate from the 48-case timing run.

## Earlier measurements

An earlier default revision passed **511 Python tests and 44 Rust tests** on the Linux
RTX 2080 Ti reporting 22 GiB VRAM, with three Windows-only tests skipped. The
Windows CPU suite passed 468 tests. An additional regression for an omitted
output budget under a smaller configured limit passed on both platforms.

The API completed a **4096-token prompt plus 4096 generated tokens** with gamma
2, peaking at **17.55 GiB allocated / 18.56 GiB reserved**. Streaming, sampled,
and thinking requests also succeeded after that long generation. Separate
runtime validation passed all 20 checks, including both 8192-token caches.
These checks cover the working-tree implementation; a new Windows GPU wheel
release still requires the [Windows delivery gate](validation.md).

With the production loader and **fixed gamma 2**, the bundled five-case corpus
ran **1.29x faster in aggregate** than target-only (sum of median full-generation
wall times). Median reported output throughput was 31.48 versus 28.17 tokens/s;
the median per-case speedup was 1.09x, and some short cases regressed.

Four additional prompts outside that corpus used the same fixed setting:

| Workload | Speedup versus target-only |
| --- | --- |
| Two-sentence cache explanation | 1.17x |
| Python function | 1.48x |
| City-name extraction | 1.47x |
| Customer-support reply | 1.17x |

Their aggregate speedup was **1.27x**. All outputs completed and matched the
same-run target-only baseline token-for-token. Both comparisons used FP16,
non-thinking target chat formatting, one warmup, three measured repetitions,
a 256-token budget, Torch selection, and synchronized full-generation wall
time with timing instrumentation enabled. These small corpora establish gains
for the tested workloads, not a general speedup guarantee.

Historically, Qwen2.5 0.5B draft + 1.5B target speculation was slower on a 6 GB
RTX 4050 laptop, so that configuration used target-only generation. It is no
longer the default. A larger target makes speculation worthwhile when enough
draft tokens are accepted; short replies can still be slower.

On September 14, 2026 (UTC), commit `6cf435f` passed 497 Python tests and 43 Rust
tests on a Linux RTX 2080 Ti reporting 22 GiB VRAM. Three Windows-only tests were
skipped. All 44 GPU tests returned to their starting allocation after cleanup.

An exploratory FP16 retest paired Qwen3 8B with Qwen2.5 0.5B and 1.5B drafts:

| Workload | 0.5B draft speedup (gamma) | 1.5B draft speedup (gamma) |
| --- | --- | --- |
| Four-digit year | 1.27x (2) | 1.19x (2) |
| 32 constrained digits | 1.91x (8) | 2.07x (8) |
| Counting 1 through 10 | 2.41x (8) | 2.14x (8) |
| Short JSON response | 1.41x (2) | 1.32x (2) |
| One-sentence GPU explanation | 1.09x (2) | 1.02x (2) |

These entries select the best observed gamma per case, not one fixed setting.
The numeric cases reproduced the Mac's raw prompts and regex constraints; the
other cases used non-thinking target chat formatting. Each used one warmup,
three measured repetitions, synchronized full-call wall time, and disabled
internal timing instrumentation. Every output matched the same-run target-only
baseline token-for-token and completed successfully. Counting reached 69.7
output tokens/s with the 0.5B draft versus 28.9 target-only.

The historical retest used an isolated loader because production then rejected
the cross-family pair. The current loader validates shared token meanings and
uses the target tokenizer throughout. The weights still differ from the Mac's
MLX 4-bit artifacts; the comparison does not isolate operating-system or
quantization effects. These are workload-specific gains, not a general 2x claim.

## Running comparisons

Run commands from `onyx_cuda` with the development environment activated.
On Linux, use the activated `python` in place of the Windows executable.

To compare proposal lengths 0, 1, 2, 4, and 8:

```powershell
.\.venv\Scripts\python.exe -m onyx_cuda.benchmark --compare --require-complete --output benchmarks/results/custom-comparison.json
```

The benchmark defaults to non-thinking answers and a 256-token budget. Use
`--enable-thinking` or `--max-tokens` to change those settings. `--disable-thinking`
remains accepted. It reports individual regressions and selects the mode with
the lowest total generation wall time over its corpus, not a universal winner.

### Adaptive controller evaluation

The versioned 48-case corpus includes the original nine measured cases and
16 held-out cases. The paired benchmark warms every mode, rotates execution
order, compares every output token and finish reason, and records full-call
latency and source hashes. It stops on any mismatch or truncated answer:

```sh
python -m onyx_cuda.benchmark_adaptive --split development --repetitions 10 --output validation/adaptive-development.json
python -m onyx_cuda.benchmark_adaptive --split all --repetitions 10 --output validation/adaptive-final.json
python -m onyx_cuda.validate_model --speculative-mode adaptive --output validation/adaptive-runtime.json
```

Use a new output filename for each run. The default comparison uses normal
runtime timing; `--measure` requests a separate detailed-profile comparison.
Promotion requires exact greedy outputs, retained gains, reduced regressions,
and a completed repeated comparison. Partial reports cannot pass promotion.

The first ten-repetition candidate passed exact output checks on all 48 cases
and retained 97% of the original gains, but failed the regression-reduction
and aggregate-latency gates. The retained controller also passed all 48 exact
output checks in a subsequent single-repetition screening run, but still
failed both performance gates. Adaptive remains opt-in: the broader performance
objective is not yet met. Model weights, FP16 precision, and attention settings
remain unchanged.

`benchmark_adaptive` prepares graph recovery and adds graph-recovery modes when
the target supports them. Every mode uses the configured draft backend, draft
graphs by default; set `ONYX_DRAFT_BACKEND=eager` to time the ordinary draft
forward. Other programmatic comparisons must prepare either backend explicitly
as shown in the [configuration guide](configuration.md).
