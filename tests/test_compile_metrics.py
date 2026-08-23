import importlib
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def clean_imported_modules():
    yield
    sys.modules.pop("onyx.speculative", None)
    sys.modules.pop("onyx.server", None)


def import_speculative_with_fake_mlx(monkeypatch, compile_fn=None):
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    nn = types.ModuleType("mlx.nn")

    core.array = object
    core.bool_ = bool
    core.argmax = lambda logits, axis=-1: ("argmax", logits, axis)
    if compile_fn is not None:
        core.compile = compile_fn

    mlx.core = core
    mlx.nn = nn

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = lambda *_args, **_kwargs: None

    mlx_lm_models = types.ModuleType("mlx_lm.models")
    mlx_lm_cache = types.ModuleType("mlx_lm.models.cache")
    mlx_lm_cache.KVCache = type("KVCache", (), {})
    mlx_lm_cache.make_prompt_cache = lambda *_args, **_kwargs: []

    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", mlx_lm_cache)
    monkeypatch.delitem(sys.modules, "onyx.speculative", raising=False)

    import onyx.speculative

    return importlib.reload(onyx.speculative)


def test_compile_requested_without_mx_compile_reports_inactive(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)

    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=True)

    assert engine.compile_requested is True
    assert engine.compile_active is False
    assert engine.compile_reason == "mlx_compile_unavailable"
    assert engine._compile_metrics() == {
        "jit_compiled": False,
        "compile_requested": True,
        "compile_reason": "mlx_compile_unavailable",
    }


def test_compile_disabled_reports_not_requested_or_active(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)

    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=False)

    assert engine.compile_requested is False
    assert engine.compile_active is False
    assert engine.compile_reason == "disabled"
    assert engine._compile_metrics() == {
        "jit_compiled": False,
        "compile_requested": False,
        "compile_reason": "disabled",
    }


def test_fake_mx_compile_marks_helper_compilation_active(monkeypatch):
    compiled = []

    def fake_compile(fn):
        compiled.append(fn.__name__)

        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    speculative = import_speculative_with_fake_mlx(monkeypatch, compile_fn=fake_compile)

    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=True)

    assert engine.compile_requested is True
    assert engine.compile_active is True
    assert engine.compile_reason == "sampling_helpers_compiled"
    assert compiled == ["_greedy_argmax", "_mask_logits_with_indices"]
    assert engine._compile_metrics()["jit_compiled"] is True


def test_api_metrics_do_not_report_jit_active_for_request_only(monkeypatch):
    fastapi = types.ModuleType("fastapi")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

    class FakeHTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi.FastAPI = FakeFastAPI
    fastapi.HTTPException = FakeHTTPException

    responses = types.ModuleType("fastapi.responses")
    responses.StreamingResponse = type("StreamingResponse", (), {})

    pydantic = types.ModuleType("pydantic")

    class FakeBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump_json(self):
            return "{}"

    pydantic.BaseModel = FakeBaseModel
    pydantic.Field = lambda default=None, **_kwargs: default

    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses)
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)
    monkeypatch.delitem(sys.modules, "onyx.server", raising=False)

    try:
        from onyx.server import build_onyx_metrics

        metrics = build_onyx_metrics(
            {
                "tokens_per_second": 12.5,
                "acceptance_rate": 50.0,
                "ttft": 0.01,
                "speculative_iterations": 2,
                "jit_compiled": False,
                "compile_requested": True,
                "compile_reason": "mlx_compile_unavailable",
            },
            grammar_active=False,
        )

        assert metrics["jit_compiled"] is False
        assert metrics["compile_requested"] is True
        assert metrics["compile_reason"] == "mlx_compile_unavailable"
    finally:
        sys.modules.pop("onyx.server", None)
