"""Paired summaries retain regressions and cannot claim gains from their average."""

import json
from types import SimpleNamespace

import pytest

from onyx_cuda.benchmark_adaptive import paired_latency_interval, select_modes, summarize


def row(name, baseline, fixed, adaptive, original=False, recovery=0):
    return {"name": name, "original": original,
            "median_seconds": {"gamma0": baseline, "gamma2": fixed, "adaptive": adaptive},
            "runs": {"adaptive": [{"adaptive_stats": {"recovery_count": recovery}}]}}


def test_large_win_cannot_hide_short_response_regressions():
    report = summarize([row("winner", 10, 5, 4, True, 1), row("loser", 0.1, 0.2, 0.3)])
    assert report["gates"]["retained_gains"]
    assert report["gates"]["aggregate"]
    assert not report["gates"]["reduced_regressions"]
    assert report["gates"]["recovery_observed"]


def test_recovery_is_required_separately_from_latency_gains():
    report = summarize([row("winner", 10, 5, 4, True), row("loser", 0.1, 0.2, 0.11)])
    assert report["gates"]["reduced_regressions"]
    assert not report["gates"]["recovery_observed"]


def test_paired_interval_is_reproducible_and_requires_repetitions():
    runs = {"gamma2": [{"seconds": 1}], "adaptive": [{"seconds": 0.5}]}
    assert paired_latency_interval(runs) is None
    runs["gamma2"] *= 10
    runs["adaptive"] *= 10
    assert paired_latency_interval(runs) == [0.5, 0.5]


def test_graph_modes_run_only_when_graphs_are_active():
    assert select_modes({"active": "graph"}) == [
        "gamma0", "gamma2", "adaptive", "gamma2_graph", "adaptive_graph"]
    assert select_modes({"active": "scalar", "reason": "unsupported"}) == ["gamma0", "gamma2", "adaptive"]


def test_graph_summary_reports_every_mode_and_graph_saving():
    case = row("winner", 10, 5, 4, True, 1)
    case["median_seconds"].update(gamma2_graph=4.5, adaptive_graph=3.6)
    report = summarize([case])
    assert set(report["speedup_vs_target"]) == {"gamma0", "gamma2", "adaptive", "gamma2_graph", "adaptive_graph"}
    assert report["graph_latency_reduction_vs_scalar"] == pytest.approx(0.1)
    assert "graph_latency_reduction_vs_scalar" not in summarize([row("winner", 10, 5, 4, True, 1)])


@pytest.fixture
def benchmark(monkeypatch):
    import torch
    import onyx_cuda.benchmark_adaptive as module
    from onyx_cuda.benchmark_corpus import CORPUS
    from onyx_cuda.generation import AcceptedTokenEvent, GenerationFinishedEvent, GenerationResult

    model = SimpleNamespace(config=SimpleNamespace(vocab_size=8))
    tokenizer = SimpleNamespace(eos_token_id=7)
    graphs = SimpleNamespace(closed=False, fallback_reason=None)
    state = SimpleNamespace(graph=True, calls=[], closes=0, close_during_run=False)

    def prepare(target, mode):
        assert target is model and mode == "graph"
        if not state.graph:
            return {"requested": "graph", "active": "scalar", "reason": "unsupported", "setup_seconds": 0.0}
        target._onyx_replay_backend = graphs
        return {"requested": "graph", "active": "graph", "setup_seconds": 1.5}

    def close(target):
        state.closes += 1
        if hasattr(target, "_onyx_replay_backend"):
            del target._onyx_replay_backend

    def generate(*, gamma, adaptive, measure, **_):
        attached = getattr(model, "_onyx_replay_backend", None)
        state.calls.append((gamma, adaptive, attached is graphs))
        if attached is not None and state.close_during_run:
            graphs.closed, graphs.fallback_reason = True, "CUDA memory exhausted during graph recovery"
        yield AcceptedTokenEvent(1)
        yield GenerationFinishedEvent(GenerationResult([1], None, "eos"))

    pair = SimpleNamespace(target=SimpleNamespace(model=model, tokenizer=tokenizer, model_id="t", revision="r"),
                           draft=SimpleNamespace(model=object(), model_id="d", revision="r"))
    monkeypatch.setattr(module, "CORPUS", [CORPUS[0], next(c for c in CORPUS if c["regex"])])
    monkeypatch.setattr(module, "load_model_pair", lambda **_: pair)
    monkeypatch.setattr(module, "prepare_replay_backend", prepare)
    monkeypatch.setattr(module, "close_replay_backend", close)
    monkeypatch.setattr(module, "generate_speculative_events", generate)
    monkeypatch.setattr(module, "format_prompt", lambda *_, **__: SimpleNamespace(token_ids=[1, 2]))
    monkeypatch.setattr(module, "get_token_byte_vocabulary", lambda *_: object())
    for name, value in (("synchronize", None), ("reset_peak_memory_stats", None),
                        ("max_memory_allocated", 0), ("memory_allocated", 0), ("get_device_name", "fake")):
        monkeypatch.setattr(torch.cuda, name, lambda *_, value=value: value)
    state.model, state.run = model, module.run
    return state


@pytest.mark.parametrize("graph", [True, False])
def test_one_run_interleaves_every_mode_and_attaches_graphs_only_to_graph_modes(benchmark, tmp_path, graph):
    benchmark.graph = graph
    report = benchmark.run(tmp_path / "report.json", repetitions=2)
    modes = [(0, False, False), (2, False, False), (2, True, False)]
    if graph:
        modes += [(2, False, True), (2, True, True)]
    assert set(benchmark.calls) == set(modes)
    assert len(benchmark.calls) == 2 * len(modes) * 3  # Two cases; one warmup and two measured runs.
    assert list(report["settings"]["modes"]) == list(report["cases"][0]["median_seconds"])
    assert report["settings"]["replay_backend"]["active"] == ("graph" if graph else "scalar")
    assert report["settings"]["draft_backend"]["requested"] == "graph"
    assert report["complete"] and report["correctness_passed"]
    assert benchmark.closes == 1 and not hasattr(benchmark.model, "_onyx_replay_backend")


def test_graph_shutdown_during_the_run_fails_instead_of_timing_scalar_recovery(benchmark, tmp_path):
    benchmark.close_during_run = True
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="Graph recovery closed"):
        benchmark.run(output, repetitions=1)
    saved = json.loads(output.read_text())
    assert not saved["complete"] and "CUDA memory exhausted" in saved["error"]
    assert benchmark.closes == 1
