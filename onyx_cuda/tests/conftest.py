"""Serial CUDA ownership, hardware selection, and machine-readable validation evidence."""

import gc
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_model_selection(monkeypatch):
    """The regression suite always starts with bundled, pinned models."""
    from onyx_cuda.config import MODEL_ENVIRONMENT_VARIABLES

    for name in MODEL_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def pytest_addoption(parser):
    parser.addoption("--require-kernels", action="store_true", help="Require CUDA and initialize custom kernels")
    parser.addoption("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")
    parser.addoption("--validation-report", help="Write validation evidence as JSON")


def pytest_configure(config):
    config._onyx_validation = {"models": {}, "tests": [], "gpu_memory": {}}
    if config.getoption("--require-cuda") and not torch.cuda.is_available():
        raise pytest.UsageError("--require-cuda needs an NVIDIA GPU and a CUDA PyTorch build")
    if config.getoption("--require-kernels"):
        from onyx_cuda.config import initialize_greedy_backend

        try:
            initialize_greedy_backend("cuda")
        except Exception as error:
            raise pytest.UsageError(f"--require-kernels needs working custom CUDA kernels: {error}") from error


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
            before_collection = torch.cuda.memory_allocated()
            _collect_cuda()
            after = torch.cuda.memory_allocated()
            before = item._onyx_allocated_before
            item.config._onyx_validation["gpu_memory"][item.nodeid] = {
                "allocated_before_bytes": before,
                "allocated_before_collection_bytes": before_collection,
                "allocated_after_bytes": after,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
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
        request.config._onyx_validation["models"][model_id] = loaded.revision

    return record


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "call":
        item._onyx_call_passed = report.passed
    if report.when == "call" or report.failed or report.skipped:
        item.config._onyx_validation["tests"].append({
            "nodeid": report.nodeid, "phase": report.when,
            "outcome": report.outcome, "seconds": report.duration,
        })
    return report


def pytest_sessionfinish(session, exitstatus):
    destination = session.config.getoption("--validation-report")
    if not destination:
        return
    result = session.config._onyx_validation
    from onyx_cuda.config import resolve_greedy_backend

    result["greedy_backend"] = resolve_greedy_backend()
    result["kernels_required"] = session.config.getoption("--require-kernels")
    if result["kernels_required"]:
        result["cupy"] = version("cupy-cuda12x")
    result.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": int(exitstatus), "collected": session.testscollected,
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: version(name) for name in (
            "onyx-cuda", "torch", "transformers", "maturin", "pytest", "fastapi", "pydantic", "uvicorn"
        )},
        "cuda_runtime": torch.version.cuda,
        "cuda_required": session.config.getoption("--require-cuda"),
    })
    if torch.cuda.is_available():
        result["device"] = {
            "name": torch.cuda.get_device_name(0),
            "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
