from types import SimpleNamespace
import sys

import numpy as np
import pytest

from onyx.output import TokenTextStream, finish_reason
from tests.test_compile_metrics import import_speculative_with_fake_mlx


@pytest.fixture(autouse=True)
def release_simulated_mlx_modules():
    yield
    sys.modules.pop("onyx.speculative", None)
    sys.modules.pop("onyx.cache", None)


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("reject", [False, True])
def test_generation_paths_obey_budget_and_match(monkeypatch, budget, reject):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    monkeypatch.setattr(speculative.mx, "array", np.array)
    monkeypatch.setattr(speculative.mx, "argmax", np.argmax)
    monkeypatch.setattr(speculative.mx, "eval", lambda *args: None, raising=False)

    class Model:
        def __init__(self, token):
            self.token = token

        def __call__(self, ids, cache):
            cache[0].length += ids.shape[1]
            logits = np.zeros((1, ids.shape[1], 10))
            logits[:, :, self.token] = 1
            return logits

    engine = speculative.SpeculativeEngine(lazy_load=True, use_compile=False, cache_mode="naive")
    engine.target_model = Model(1)
    engine.draft_model = Model(2 if reject else 1)
    engine.tokenizer = SimpleNamespace(encode=lambda text: [3, 4],
                                       decode=lambda ids: "".join(str(i) for i in ids), eos_token_id=9)
    def reset():
        engine.draft_cache = [SimpleNamespace(length=0)]
        engine.target_cache = [SimpleNamespace(length=0)]
    engine._reset_caches = reset
    engine._get_cache_size = lambda cache: cache[0].length
    def rollback(cache, length):
        assert length <= cache[0].length
        cache[0].length = length
    engine._rollback_cache = rollback
    text, metrics = engine.generate("prompt", max_tokens=budget, gamma=4)
    assert text == "1" * budget
    assert metrics["generated_tokens"] == budget
    assert metrics["finish_reason"] == "length"
    events = list(engine.stream_generate("prompt", max_tokens=budget, gamma=4))
    assert "".join(text for text, _ in events) == text
    assert events[-1][1]["generated_tokens"] == budget
    assert events[-1][1]["finish_reason"] == "length"
    # EOS at the budget boundary takes precedence over length and is hidden.
    engine.target_model = Model(9)
    engine.draft_model = Model(9)
    assert engine.generate("prompt", max_tokens=1)[0] == ""
    events = list(engine.stream_generate("prompt", max_tokens=1))
    assert "".join(text for text, _ in events) == ""
    assert events[-1][1]["finish_reason"] == "stop"
    engine.target_model = Model(1)
    engine.draft_model = Model(1)
    class Grammar:
        def __init__(self, vocab):
            pass
        def compile_regex(self, regex):
            pass
        def init_state(self):
            return 0
        def advance_state(self, state, token):
            return state + 1
        def get_valid_token_ids(self, state):
            return [1]
        def is_match_state(self, state):
            return state >= 2
        def release_state(self, state):
            pass
        def release_states(self, states):
            pass
    monkeypatch.setattr(speculative, "_GrammarConstraint", Grammar)
    engine._apply_grammar_mask = lambda logits, valid: logits
    text, metrics = engine.generate("prompt", max_tokens=budget, regex="11")
    assert text == "1" * min(budget, 2)
    expected = "length" if budget < 2 else "grammar_complete"
    assert metrics["finish_reason"] == expected
    events = list(engine.stream_generate("prompt", max_tokens=budget, regex="11"))
    assert "".join(text for text, _ in events) == text
    assert events[-1][1]["finish_reason"] == expected


def test_token_stop_prefix_is_withheld_and_unmatched_prefix_flushed():
    tokenizer = SimpleNamespace(decode=lambda ids: "".join(map(str, ids)))
    stream = TokenTextStream(tokenizer, [[2, 3], [9]])
    assert stream.update([1, 2]) == "1"
    assert stream.update([1, 2, 3]) == ""
    assert stream.update([1, 2, 3], final=True) == ""
    stream = TokenTextStream(tokenizer, [[2, 3]])
    assert stream.update([1, 2]) == "1"
    assert stream.update([1, 2], final=True) == "2"
    stream = TokenTextStream(tokenizer, [[3], [2, 3]])
    assert stream.update([1, 2]) == "1"
    assert stream.update([1, 2, 3], final=True) == ""


def test_split_unicode_is_only_emitted_once_stable():
    tokenizer = SimpleNamespace(decode=lambda ids: {1: "\ufffd", 2: "é"}[len(ids)])
    stream = TokenTextStream(tokenizer, [])
    assert stream.update([1]) == ""
    assert stream.update([1, 2]) == "é"
    assert stream.update([1, 2], final=True) == ""


def test_finish_reason_requires_real_completion():
    assert finish_reason([1], [], True, 1) == "grammar_complete"
    assert finish_reason([1], [[1]], False, 1) == "stop"
    assert finish_reason([1], [], False, 1) == "length"
    with pytest.raises(RuntimeError):
        finish_reason([1], [], False, 2)
