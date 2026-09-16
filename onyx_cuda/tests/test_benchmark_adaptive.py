"""Paired summaries retain regressions and cannot claim gains from their average."""

from onyx_cuda.benchmark_adaptive import paired_latency_interval, summarize


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
