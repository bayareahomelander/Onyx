"""Serial CUDA ownership, hardware selection, and CUDA memory cleanup checks."""

import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_model_selection(monkeypatch):
    """The regression suite always starts with bundled, pinned models."""
    from onyx_cuda.config import MODEL_ENVIRONMENT_VARIABLES

    for name in (*MODEL_ENVIRONMENT_VARIABLES, "ONYX_MAX_CONTEXT_TOKENS", "ONYX_MAX_OUTPUT_TOKENS",
                 "ONYX_MAX_ACTIVE_REQUESTS", "ONYX_STREAM_BUFFER_CHUNKS", "ONYX_SPECULATIVE_MODE", "ONYX_GREEDY_BACKEND",
                 "ONYX_REPLAY_BACKEND", "ONYX_DRAFT_BACKEND"):
        monkeypatch.delenv(name, raising=False)


def pytest_addoption(parser):
    parser.addoption("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")


def pytest_configure(config):
    if config.getoption("--require-cuda") and not torch.cuda.is_available():
        raise pytest.UsageError("--require-cuda needs an NVIDIA GPU and a CUDA PyTorch build")


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="requires NVIDIA CUDA; use --require-cuda to enforce the GPU gate")
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip)


def _collect_cuda():
    # empty_cache alone cannot release tensors retained by Python reference cycles.
    gc.collect()
    torch.cuda.synchronize()
    # PyTorch 2.6 retains an 8.125 MiB cuBLAS workspace per handle even after
    # every tensor is gone. Release this test-only cache before checking zero
    # growth; production generation keeps its normal workspace caching.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu") and torch.cuda.is_available():
        _collect_cuda()
        item._onyx_allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    try:
        return (yield)
    finally:
        if hasattr(item, "_onyx_allocated_before"):
            # Run AFTER fixture finalizers, including monkeypatch undo, so their
            # closures no longer keep models alive when the next test loads a pair.
            _collect_cuda()
            after = torch.cuda.memory_allocated()
            before = item._onyx_allocated_before
            # Failed tests can retain tensors in pytest's traceback; preserve the
            # original failure instead of adding a misleading cleanup failure.
            if getattr(item, "_onyx_call_passed", False):
                assert after == before, f"CUDA tensors survived test teardown: {before} -> {after} bytes"


@pytest.fixture
def record_model_revision(request):
    from onyx_cuda.revisions import MODEL_REVISIONS

    def record(model_id, loaded):
        assert loaded.revision == MODEL_REVISIONS[model_id]
        assert loaded.model.config._commit_hash == loaded.revision

    return record


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "call":
        item._onyx_call_passed = report.passed
    return report
