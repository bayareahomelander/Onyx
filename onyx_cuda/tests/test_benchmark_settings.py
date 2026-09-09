"""Baselines cannot silently mix backends or old timing definitions."""

import pytest

from onyx_cuda import benchmark


def test_baseline_requires_same_backend_and_timing_contract(monkeypatch):
    monkeypatch.setenv("ONYX_GREEDY_BACKEND", "torch")
    baseline = {"settings": benchmark._comparison_settings()}
    benchmark._require_matching_settings(baseline)
    monkeypatch.setattr(benchmark, "version", lambda _: "13.6.0")
    monkeypatch.setenv("ONYX_GREEDY_BACKEND", "cuda")
    with pytest.raises(RuntimeError, match="regenerate"):
        benchmark._require_matching_settings(baseline)
    assert benchmark._comparison_settings()["cupy"] == "13.6.0"
    monkeypatch.setenv("ONYX_GREEDY_BACKEND", "torch")
    del baseline["settings"]["timing_contract"]
    with pytest.raises(RuntimeError, match="regenerate"):
        benchmark._require_matching_settings(baseline)
