from types import SimpleNamespace
import importlib
import sys

import numpy as np
import pytest

from onyx._rust import GrammarConstraint
from onyx.adaptive_controller import AdaptiveGammaConfig
from tests.test_compile_metrics import import_speculative_with_fake_mlx


PATTERN = "[A-Z]{3}-[0-9]{4}"
VOCAB = [b"", b"ABC-123", b"5.", b"5", b"."]
# Every position prefers "5.", which completes the match and then continues.
PREFERENCE = [0.0, 3.0, 4.0, 2.0, 1.0]


class PreferenceModel:
    def __call__(self, ids, cache):
        cache[0].length += ids.shape[1]
        return np.tile(PREFERENCE, (1, ids.shape[1], 1))


def mask(logits, valid):
    allowed = np.full(logits.shape[-1], -np.inf)
    allowed[valid] = 0
    return logits + allowed


@pytest.fixture(autouse=True)
def release_simulated_mlx_modules():
    yield
    sys.modules.pop("onyx.adaptive", None)
    sys.modules.pop("onyx.speculative", None)
    sys.modules.pop("onyx.cache", None)


def make_engine(monkeypatch, method):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    monkeypatch.setattr(speculative.mx, "array", np.array)
    monkeypatch.setattr(speculative.mx, "argmax", np.argmax)
    monkeypatch.setattr(speculative.mx, "eval", lambda *args: None, raising=False)
    monkeypatch.setattr(speculative, "_GrammarConstraint", GrammarConstraint)
    monkeypatch.delitem(sys.modules, "onyx.adaptive", raising=False)
    adaptive = importlib.import_module("onyx.adaptive")
    monkeypatch.setattr(adaptive, "_GrammarConstraint", GrammarConstraint)

    module = adaptive.AdaptiveSpeculativeEngine if method == "generate_adaptive" else (
        speculative.SpeculativeEngine
    )
    engine = module(lazy_load=True, use_compile=False, cache_mode="naive")
    engine.target_model = PreferenceModel()
    engine.draft_model = PreferenceModel()
    engine.vocab_bytes = VOCAB
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: [0],
        decode=lambda ids: b"".join(VOCAB[i] for i in ids).decode(),
        eos_token_id=0,
    )

    def reset():
        engine.draft_cache = [SimpleNamespace(length=0)]
        engine.target_cache = [SimpleNamespace(length=0)]

    def rollback(cache, length):
        assert length <= cache[0].length
        cache[0].length = length

    engine._reset_caches = reset
    engine._get_cache_size = lambda cache: cache[0].length
    engine._rollback_cache = rollback
    engine._apply_grammar_mask = mask
    return engine


@pytest.mark.parametrize("gamma", [1, 4])
@pytest.mark.parametrize("method", ["generate", "stream_generate", "generate_adaptive"])
def test_regex_rejects_token_that_runs_past_the_match(monkeypatch, method, gamma):
    engine = make_engine(monkeypatch, method)

    if method == "generate":
        text, metrics = engine.generate("prompt", max_tokens=10, gamma=gamma, regex=PATTERN)
    elif method == "stream_generate":
        events = list(engine.stream_generate("prompt", max_tokens=10, gamma=gamma, regex=PATTERN))
        text, metrics = "".join(text for text, _ in events), events[-1][1]
    else:
        config = AdaptiveGammaConfig(min_gamma=gamma, max_gamma=gamma, initial_gamma=gamma)
        text, metrics = engine.generate_adaptive(
            "prompt", max_tokens=10, controller_config=config, regex=PATTERN
        )

    assert text == "ABC-1235"
    # generate_adaptive does not report a finish reason.
    if method != "generate_adaptive":
        assert metrics["finish_reason"] == "grammar_complete"
