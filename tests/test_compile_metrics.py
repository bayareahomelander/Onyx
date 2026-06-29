import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def clean_imported_modules():
    yield
    sys.modules.pop("onyx.speculative", None)
    sys.modules.pop("onyx.server", None)
    sys.modules.pop("onyx.adaptive", None)
    sys.modules.pop("onyx.engine", None)


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


class GreedyFakeModel:
    def __call__(self, input_ids, cache):
        logits = np.zeros((1, input_ids.shape[1], 2), dtype=np.float32)
        logits[..., 1] = 1.0
        return logits


class TrackingCache:
    def __init__(self):
        self.offset = 0


class CacheAwareGreedyFakeModel(GreedyFakeModel):
    def __call__(self, input_ids, cache):
        for layer_cache in cache:
            layer_cache.offset += input_ids.shape[1]
        return super().__call__(input_ids, cache)


class GreedyFakeTokenizer:
    vocab_size = 2
    eos_token_id = None

    def encode(self, _text):
        return [0]

    def decode(self, token_ids):
        return "x" * len(token_ids)


def configure_numpy_decode_runtime(module):
    module.mx.array = np.array
    module.mx.argmax = lambda logits, axis=-1: np.argmax(logits, axis=axis)
    module.mx.eval = lambda *_args: None
    module.mx.random = types.SimpleNamespace(
        categorical=lambda logits: np.argmax(logits, axis=-1),
    )


def configure_tracking_caches(engine):
    def reset_caches():
        engine.draft_cache = [TrackingCache()]
        engine.target_cache = [TrackingCache()]

    engine._reset_caches = reset_caches


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


def test_sampling_masks_ids_outside_paired_tokenizer_vocabulary(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    configure_numpy_decode_runtime(speculative)
    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=False)
    engine._tokenizer_vocab_mask = np.array([0.0, float("-inf")])

    selected = engine._sample_token(np.array([[0.0, 100.0]]))

    assert selected.item() == 0


def test_draft_token_budget_never_exceeds_remaining_limit(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)

    assert speculative._draft_token_budget(max_tokens=5, generated_count=4, gamma=4) == 1
    assert speculative._draft_token_budget(max_tokens=8, generated_count=2, gamma=4) == 4


def test_speculative_generation_honors_hard_token_limit(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    configure_numpy_decode_runtime(speculative)

    engine = speculative.SpeculativeEngine(
        cache_mode="naive",
        lazy_load=True,
        use_compile=False,
    )
    engine.draft_model = GreedyFakeModel()
    engine.target_model = GreedyFakeModel()
    engine.tokenizer = GreedyFakeTokenizer()

    output, metrics = engine.generate("prompt", max_tokens=5, gamma=4)

    assert output == "xxxxx"
    assert metrics["generated_tokens"] == 5
    assert metrics["finish_reason"] == "length"

    baseline_output, baseline_metrics = engine.generate_baseline("prompt", max_tokens=5)

    assert baseline_output == "xxxxx"
    assert baseline_metrics["generated_tokens"] == 5
    assert baseline_metrics["finish_reason"] == "length"

    sampled_output, sampled_metrics = engine.generate_baseline(
        "prompt",
        max_tokens=2,
        temperature=0.5,
    )

    assert sampled_output == "xx"
    assert sampled_metrics["finish_reason"] == "length"

    streamed = list(engine.stream_generate("prompt", max_tokens=5, gamma=4))
    streamed_text = "".join(text for text, metrics in streamed if metrics is None)
    final_metrics = streamed[-1][1]

    assert streamed_text == "xxxxx"
    assert final_metrics["generated_tokens"] == 5
    assert final_metrics["finish_reason"] == "length"

    stopped_output, stopped_metrics = engine.generate(
        "prompt",
        max_tokens=5,
        gamma=4,
        stop_tokens=[1],
    )

    assert stopped_output == ""
    assert stopped_metrics["generated_tokens"] == 0
    assert stopped_metrics["finish_reason"] == "stop"


def test_complete_draft_acceptance_keeps_caches_aligned(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    configure_numpy_decode_runtime(speculative)

    engine = speculative.SpeculativeEngine(
        cache_mode="naive",
        lazy_load=True,
        use_compile=False,
    )
    engine.draft_model = CacheAwareGreedyFakeModel()
    engine.target_model = CacheAwareGreedyFakeModel()
    engine.tokenizer = GreedyFakeTokenizer()
    configure_tracking_caches(engine)

    output, metrics = engine.generate("prompt", max_tokens=9, gamma=4)

    assert output == "x" * 9
    assert metrics["acceptance_rate"] == 100.0
    assert engine.draft_cache[0].offset == 9
    assert engine.target_cache[0].offset == 9

    streamed = list(engine.stream_generate("prompt", max_tokens=9, gamma=4))
    streamed_text = "".join(text for text, chunk_metrics in streamed if chunk_metrics is None)

    assert streamed_text == "x" * 9
    assert streamed[-1][1]["acceptance_rate"] == 100.0
    assert engine.draft_cache[0].offset == 9
    assert engine.target_cache[0].offset == 9


def test_adaptive_generation_honors_hard_token_limit(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    configure_numpy_decode_runtime(speculative)
    sys.modules.pop("onyx.adaptive", None)
    adaptive = importlib.import_module("onyx.adaptive")

    engine = adaptive.AdaptiveSpeculativeEngine(
        cache_mode="naive",
        lazy_load=True,
        use_compile=False,
    )
    engine.draft_model = GreedyFakeModel()
    engine.target_model = GreedyFakeModel()
    engine.tokenizer = GreedyFakeTokenizer()

    output, metrics = engine.generate_adaptive("prompt", max_tokens=5)

    assert output == "xxxxx"
    assert metrics["generated_tokens"] == 5
    assert metrics["finish_reason"] == "length"


def test_adaptive_complete_draft_acceptance_keeps_caches_aligned(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    configure_numpy_decode_runtime(speculative)
    sys.modules.pop("onyx.adaptive", None)
    adaptive = importlib.import_module("onyx.adaptive")

    engine = adaptive.AdaptiveSpeculativeEngine(
        cache_mode="naive",
        lazy_load=True,
        use_compile=False,
    )
    engine.draft_model = CacheAwareGreedyFakeModel()
    engine.target_model = CacheAwareGreedyFakeModel()
    engine.tokenizer = GreedyFakeTokenizer()
    configure_tracking_caches(engine)

    output, metrics = engine.generate_adaptive("prompt", max_tokens=9)

    assert output == "x" * 9
    assert metrics["acceptance_rate"] == 100.0
    assert engine.draft_cache[0].offset == 9
    assert engine.target_cache[0].offset == 9


def test_single_model_engine_reports_length_and_rejects_invalid_limit(monkeypatch):
    import_speculative_with_fake_mlx(monkeypatch)
    sys.modules.pop("onyx.engine", None)
    engine_module = importlib.import_module("onyx.engine")
    configure_numpy_decode_runtime(engine_module)

    class FakeCacheManager:
        def reset(self):
            pass

        def as_list(self):
            return []

    engine = engine_module.OnyxEngine(lazy_load=True)
    engine.model = GreedyFakeModel()
    engine.tokenizer = GreedyFakeTokenizer()
    engine.cache_manager = FakeCacheManager()

    output, metrics = engine.generate("prompt", max_tokens=5)

    assert output == "xxxxx"
    assert metrics["generated_tokens"] == 5
    assert metrics["finish_reason"] == "length"

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        engine.generate("prompt", max_tokens=0)


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_generation_rejects_invalid_max_tokens_before_model_loading(monkeypatch, max_tokens):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=False)

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        engine.generate("prompt", max_tokens=max_tokens)

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        engine.generate_baseline("prompt", max_tokens=max_tokens)

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        next(engine.stream_generate("prompt", max_tokens=max_tokens))


def test_speculative_generation_rejects_invalid_gamma_before_model_loading(monkeypatch):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=False)

    with pytest.raises(ValueError, match="gamma must be a positive integer"):
        engine.generate("prompt", gamma=0)

    with pytest.raises(ValueError, match="gamma must be a positive integer"):
        next(engine.stream_generate("prompt", gamma=0))


@pytest.mark.parametrize(
    "sampling_kwargs",
    [
        {"temperature": 0.1, "top_p": 1.0},
        {"temperature": 0.0, "top_p": 0.9},
    ],
)
def test_speculative_generation_rejects_non_greedy_sampling_before_model_loading(
    monkeypatch,
    sampling_kwargs,
):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=False)

    with pytest.raises(ValueError, match="supports greedy sampling only"):
        engine.generate("prompt", **sampling_kwargs)

    with pytest.raises(ValueError, match="supports greedy sampling only"):
        next(engine.stream_generate("prompt", **sampling_kwargs))

    assert engine.draft_model is None
    assert engine.target_model is None


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
