"""Paired summaries report every mode and the cases slower than target-only."""

import json
from types import SimpleNamespace

import pytest

from onyx_cuda.benchmark import select_modes, summarize


def row(name, baseline, fixed, graph=None):
    medians = {"gamma0": baseline, "gamma2": fixed}
    if graph is not None:
        medians["gamma2_graph"] = graph
    return {"name": name, "median_seconds": medians}


def test_summary_keeps_regressions_visible_beside_the_aggregate():
    report = summarize([row("winner", 10, 5), row("loser", 0.1, 0.2)])
    assert report["speedup_vs_target"]["gamma2"] == pytest.approx(10.1 / 5.2)
    assert report["slower_than_target"] == {"gamma2": ["loser"]}


def test_graph_modes_run_only_when_graphs_are_active():
    assert select_modes({"active": "graph"}) == {"gamma0": (0, False), "gamma2": (2, False),
                                                 "gamma2_graph": (2, True)}
    assert list(select_modes({"active": "scalar", "reason": "unsupported"})) == ["gamma0", "gamma2"]
    assert list(select_modes({"active": "graph"}, (2, 3))) == [
        "gamma0", "gamma2", "gamma2_graph", "gamma3", "gamma3_graph"]


def test_graph_summary_reports_every_mode_and_graph_saving():
    report = summarize([row("winner", 10, 5, 4.5)])
    assert set(report["speedup_vs_target"]) == {"gamma0", "gamma2", "gamma2_graph"}
    assert report["graph_latency_reduction_vs_scalar"] == {"gamma2": pytest.approx(0.1)}
    assert "graph_latency_reduction_vs_scalar" not in summarize([row("winner", 10, 5)])
    extra = row("winner", 10, 5, 4.5)
    extra["median_seconds"].update(gamma3=4, gamma3_graph=3)
    report = summarize([extra])
    assert list(report["speedup_vs_target"]) == ["gamma0", "gamma2", "gamma2_graph", "gamma3", "gamma3_graph"]
    assert report["graph_latency_reduction_vs_scalar"] == {"gamma2": pytest.approx(0.1),
                                                           "gamma3": pytest.approx(0.25)}


@pytest.fixture
def benchmark(monkeypatch):
    import torch
    import onyx_cuda.benchmark as module
    from onyx_cuda.benchmark_corpus import CORPUS
    from onyx_cuda.generation import (AcceptedTokenEvent, GenerationFinishedEvent, GenerationResult,
                                      SpeculationStats)

    model = SimpleNamespace(config=SimpleNamespace(vocab_size=8))
    tokenizer = SimpleNamespace(eos_token_id=7)
    graphs = SimpleNamespace(closed=False, fallback_reason=None)
    state = SimpleNamespace(graph=True, calls=[], closes=0, close_during_run=False, fallback_during_run=False)

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

    def generate(*, gamma, measure, **_):
        attached = getattr(model, "_onyx_replay_backend", None)
        state.calls.append((gamma, attached is graphs))
        if attached is not None and state.close_during_run:
            graphs.closed, graphs.fallback_reason = True, "CUDA memory exhausted during graph recovery"
        # Target-only generation has no speculative counters.
        speculation = SpeculationStats(2, 1, 1, {"graph_replay_fallbacks": int(
            attached is not None and state.fallback_during_run)}) if gamma else None
        yield AcceptedTokenEvent(1)
        yield GenerationFinishedEvent(GenerationResult([1], None, "eos", speculation=speculation))

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
    state.model, state.graphs, state.run = model, graphs, module.run
    return state


@pytest.mark.parametrize("graph", [True, False])
def test_one_run_interleaves_every_mode_and_attaches_graphs_only_to_graph_modes(benchmark, tmp_path, graph):
    benchmark.graph = graph
    report = benchmark.run(tmp_path / "report.json", repetitions=2)
    modes = [(0, False), (2, False)]
    if graph:
        modes += [(2, True)]
    assert set(benchmark.calls) == set(modes)
    assert len(benchmark.calls) == 2 * len(modes) * 3  # Two cases; one warmup and two measured runs.
    assert list(report["settings"]["modes"]) == list(report["cases"][0]["median_seconds"])
    assert report["settings"]["replay_backend"]["active"] == ("graph" if graph else "scalar")
    assert report["settings"]["draft_backend"]["requested"] == "graph"
    assert report["complete"] and report["correctness_passed"]
    assert benchmark.closes == 1 and not hasattr(benchmark.model, "_onyx_replay_backend")
    runs = report["cases"][0]["runs"]
    assert runs["gamma0"][0]["speculation"] is None
    assert runs["gamma2"][0]["speculation"]["accepted_proposal_count"] == 1


def test_extra_gammas_join_the_same_interleaved_run(benchmark, tmp_path):
    report = benchmark.run(tmp_path / "report.json", repetitions=1, gammas=(2, 3))
    assert set(benchmark.calls) == {(0, False), (2, False), (2, True), (3, False), (3, True)}
    assert report["settings"]["modes"]["gamma3_graph"] == {"gamma": 3, "graph_recovery": True}
    assert list(report["summary"]["speedup_vs_target"]) == list(report["settings"]["modes"])


@pytest.mark.parametrize("gammas", [(), (2, 2), (0,), (True,)])
def test_invalid_gammas_fail_before_loading_models(benchmark, tmp_path, monkeypatch, gammas):
    import onyx_cuda.benchmark as module
    monkeypatch.setattr(module, "load_model_pair", lambda **_: pytest.fail("models must not load"))
    with pytest.raises(ValueError, match="gammas"):
        benchmark.run(tmp_path / "report.json", gammas=gammas)


def test_per_request_graph_fallback_fails_without_measurement(benchmark, tmp_path):
    benchmark.fallback_during_run = True
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="Graph recovery fell back"):
        benchmark.run(output, repetitions=1)
    assert not json.loads(output.read_text())["complete"]
    # Graphs stay open after a per-request fallback; only the counters reveal it.
    assert not benchmark.graphs.closed


def test_graph_shutdown_during_the_run_fails_instead_of_timing_scalar_recovery(benchmark, tmp_path):
    benchmark.close_during_run = True
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="Graph recovery closed"):
        benchmark.run(output, repetitions=1)
    saved = json.loads(output.read_text())
    assert not saved["complete"] and "CUDA memory exhausted" in saved["error"]
    assert benchmark.closes == 1
