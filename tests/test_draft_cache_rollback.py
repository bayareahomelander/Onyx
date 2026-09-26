from types import SimpleNamespace
import importlib
import sys

import numpy as np
import pytest

from onyx.adaptive_controller import AdaptiveGammaConfig
from tests.test_compile_metrics import import_speculative_with_fake_mlx


PROMPT = [1, 2, 3]
BUDGET = 20
VOCAB = 16
MISS = 12  # only ever proposed by the draft
STOP = 4
GRAMMAR_LENGTH = 11  # the grammar matches after this many generated tokens
STOP_LENGTH = 13  # the target emits STOP as this generated token
EARLY_STOP = len(PROMPT) + 4  # the draft proposes STOP here, the target does not


def truth(position):
    """Token after absolute `position`; depends on the position alone."""
    return 5 + position % 7


def stop_target(position):
    return STOP if position == len(PROMPT) + STOP_LENGTH - 2 else truth(position)


# name: (target, draft, generated tokens or None for the budget, extra kwargs)
SCENARIOS = {
    "full": (truth, truth, None, {}),
    "partial": (truth, lambda p: MISS if p % 5 == 0 else truth(p), None, {}),
    "grammar": (truth, truth, GRAMMAR_LENGTH, {"regex": "counted"}),
    "stop": (
        stop_target,
        lambda p: STOP if p == EARLY_STOP else stop_target(p),
        STOP_LENGTH,
        {"stop_tokens": [STOP]},
    ),
}


class PositionModel:
    """Records the tokens fed into its cache and predicts by position."""

    def __init__(self, predict):
        self.predict = predict

    def __call__(self, ids, cache):
        tokens = cache[0].tokens
        logits = np.zeros((1, ids.shape[1], VOCAB))
        for i, token in enumerate(ids[0].tolist()):
            logits[0, i, self.predict(len(tokens))] = 1
            tokens.append(token)
        return logits


class CountingGrammar:
    def __init__(self, vocab):
        pass

    def compile_regex(self, regex):
        pass

    def init_state(self):
        return 0

    def advance_state(self, state, token):
        return state + 1

    def get_valid_token_ids(self, state):
        return list(range(VOCAB))

    def is_match_state(self, state):
        return state >= GRAMMAR_LENGTH

    def release_state(self, state):
        pass

    def release_states(self, states):
        pass


@pytest.fixture(autouse=True)
def release_simulated_mlx_modules():
    yield
    sys.modules.pop("onyx.adaptive", None)
    sys.modules.pop("onyx.speculative", None)
    sys.modules.pop("onyx.cache", None)


def make_engine(monkeypatch, method, target, draft):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    monkeypatch.setattr(speculative.mx, "array", np.array)
    monkeypatch.setattr(speculative.mx, "argmax", np.argmax)
    monkeypatch.setattr(speculative.mx, "eval", lambda *args: None, raising=False)
    monkeypatch.setattr(speculative, "_GrammarConstraint", CountingGrammar)
    monkeypatch.delitem(sys.modules, "onyx.adaptive", raising=False)
    adaptive = importlib.import_module("onyx.adaptive")
    monkeypatch.setattr(adaptive, "_GrammarConstraint", CountingGrammar)

    if method == "generate_adaptive":
        engine = adaptive.AdaptiveSpeculativeEngine(
            lazy_load=True, use_compile=False, cache_mode="naive"
        )
    else:
        engine = speculative.SpeculativeEngine(
            lazy_load=True, use_compile=False, cache_mode="naive"
        )
    engine.target_model = PositionModel(target)
    engine.draft_model = PositionModel(draft)
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: list(PROMPT),
        decode=lambda ids: " ".join(map(str, ids)),
        eos_token_id=0,
    )

    def reset():
        engine.draft_cache = [SimpleNamespace(tokens=[], rollbacks=[])]
        engine.target_cache = [SimpleNamespace(tokens=[], rollbacks=[])]

    def rollback(cache, length):
        assert length <= len(cache[0].tokens)
        del cache[0].tokens[length:]
        cache[0].rollbacks.append(list(cache[0].tokens))

    engine._reset_caches = reset
    engine._get_cache_size = lambda cache: len(cache[0].tokens)
    engine._rollback_cache = rollback
    engine._apply_grammar_mask = lambda logits, valid: logits
    return engine


def run(engine, method, gamma, kwargs):
    if method == "generate":
        return engine.generate("prompt", max_tokens=BUDGET, gamma=gamma, **kwargs)
    if method == "stream_generate":
        events = list(engine.stream_generate("prompt", max_tokens=BUDGET, gamma=gamma, **kwargs))
        return "".join(text for text, _ in events), events[-1][1]
    config = AdaptiveGammaConfig(min_gamma=gamma, max_gamma=gamma, initial_gamma=gamma)
    return engine.generate_adaptive(
        "prompt", max_tokens=BUDGET, controller_config=config, **kwargs
    )


@pytest.mark.parametrize("gamma", [1, 2, 4])
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("method", ["generate", "stream_generate", "generate_adaptive"])
def test_draft_cache_holds_accepted_history(monkeypatch, method, scenario, gamma):
    target, draft, n_generated, kwargs = SCENARIOS[scenario]
    engine = make_engine(monkeypatch, method, target, draft)

    text, metrics = run(engine, method, gamma, kwargs)

    if n_generated is None:
        n_generated = BUDGET
        if method == "generate_adaptive":
            # generate_adaptive does not cap its last proposal at the budget.
            n_generated = metrics["generated_tokens"]
            assert n_generated >= BUDGET
    generated = [target(p) for p in range(len(PROMPT) - 1, len(PROMPT) - 1 + n_generated)]
    visible = generated
    if scenario == "stop" and method != "generate_adaptive":
        visible = generated[:-1]
    assert text == " ".join(map(str, visible))

    # After every round both caches hold the accepted history except the new
    # current token, which the next round feeds again.
    history = PROMPT + generated
    draft_rollbacks = engine.draft_cache[0].rollbacks
    assert draft_rollbacks == engine.target_cache[0].rollbacks
    assert all(snapshot == history[: len(snapshot)] for snapshot in draft_rollbacks)
    assert draft_rollbacks[-1] == history[:-1]

    # With aligned caches, a proposal is rejected only where the draft's
    # position rule differs from the target's.
    misses = sum(
        draft(p) != target(p) for p in range(len(PROMPT), len(PROMPT) + n_generated - 1)
    )
    assert metrics["draft_tokens_accepted"] == n_generated - 1 - misses
    if not misses:
        assert metrics["acceptance_rate"] == 100.0
