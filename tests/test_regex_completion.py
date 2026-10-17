from types import SimpleNamespace
import importlib
import sys

import numpy as np
import pytest

from onyx._rust import GrammarConstraint
from onyx.adaptive_controller import AdaptiveGammaConfig
from tests.test_compile_metrics import import_speculative_with_fake_mlx


SPECULATIVE = ["generate", "stream_generate", "generate_adaptive"]
MAX_MODEL_CALLS = 100  # turns a generation loop that never advances into a failure


class PreferenceModel:
    """Scores every position with the same fixed logits."""

    def __init__(self, preference):
        self.preference = preference
        self.calls = 0

    def __call__(self, ids, cache):
        self.calls += 1
        assert self.calls <= MAX_MODEL_CALLS, "generation stopped making progress"
        cache[0].length += ids.shape[1]
        return np.tile(self.preference, (1, ids.shape[1], 1))


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


def make_engine(monkeypatch, method, vocab, target, draft=None):
    speculative = import_speculative_with_fake_mlx(monkeypatch)
    monkeypatch.setattr(speculative.mx, "array", np.array)
    monkeypatch.setattr(speculative.mx, "argmax", np.argmax)
    monkeypatch.setattr(speculative.mx, "eval", lambda *args: None, raising=False)
    monkeypatch.setattr(speculative, "_GrammarConstraint", GrammarConstraint)
    monkeypatch.setattr(
        speculative, "make_prompt_cache", lambda model: [SimpleNamespace(length=0)]
    )
    monkeypatch.delitem(sys.modules, "onyx.adaptive", raising=False)
    adaptive = importlib.import_module("onyx.adaptive")
    monkeypatch.setattr(adaptive, "_GrammarConstraint", GrammarConstraint)

    module = adaptive.AdaptiveSpeculativeEngine if method == "generate_adaptive" else (
        speculative.SpeculativeEngine
    )
    engine = module(lazy_load=True, use_compile=False, cache_mode="naive")
    engine.target_model = PreferenceModel(target)
    engine.draft_model = PreferenceModel(target if draft is None else draft)
    engine.vocab_bytes = vocab
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: [0],
        decode=lambda ids: b"".join(vocab[i] for i in ids).decode(),
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


def run(engine, method, regex, gamma=4, **kwargs):
    if method == "generate_baseline":
        return engine.generate_baseline("prompt", max_tokens=10, regex=regex)
    if method == "generate":
        return engine.generate("prompt", max_tokens=10, gamma=gamma, regex=regex, **kwargs)
    if method == "stream_generate":
        events = list(engine.stream_generate(
            "prompt", max_tokens=10, gamma=gamma, regex=regex, **kwargs
        ))
        return "".join(text for text, _ in events), events[-1][1]
    config = AdaptiveGammaConfig(min_gamma=gamma, max_gamma=gamma, initial_gamma=gamma)
    return engine.generate_adaptive(
        "prompt", max_tokens=10, controller_config=config, regex=regex, **kwargs
    )


def assert_grammar_complete(method, metrics):
    # generate_adaptive and generate_baseline do not report a finish reason.
    if method in ("generate", "stream_generate"):
        assert metrics["finish_reason"] == "grammar_complete"


@pytest.mark.parametrize("gamma", [1, 4])
@pytest.mark.parametrize("method", SPECULATIVE)
def test_regex_rejects_token_that_runs_past_the_match(monkeypatch, method, gamma):
    # Every position prefers "5.", which completes the match and then continues.
    vocab = [b"", b"ABC-123", b"5.", b"5", b"."]
    engine = make_engine(monkeypatch, method, vocab, target=[0.0, 3.0, 4.0, 2.0, 1.0])

    text, metrics = run(engine, method, "[A-Z]{3}-[0-9]{4}", gamma)

    assert text == "ABC-1235"
    assert_grammar_complete(method, metrics)


@pytest.mark.parametrize("draft_grammar_aware", [True, False])
@pytest.mark.parametrize("method", SPECULATIVE + ["generate_baseline"])
def test_grammar_without_continuation_fails_the_request(monkeypatch, method, draft_grammar_aware):
    if method == "generate_baseline" and not draft_grammar_aware:
        pytest.skip("generate_baseline has no draft")
    # After "a" the pattern needs "b", which the vocabulary only has as "abc".
    vocab = [b"", b"a", b"abc"]
    engine = make_engine(monkeypatch, method, vocab, target=[0.0, 2.0, 1.0])
    kwargs = {} if method == "generate_baseline" else {"draft_grammar_aware": draft_grammar_aware}

    with pytest.raises(ValueError, match="no valid token continuation"):
        run(engine, method, "ab", **kwargs)


@pytest.mark.parametrize("method", SPECULATIVE + ["generate_baseline"])
def test_regex_that_can_never_match_fails_before_generation(monkeypatch, method):
    vocab = [b"", b"a", b"b"]
    engine = make_engine(monkeypatch, method, vocab, target=[0.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="can never match"):
        run(engine, method, "a$b")
    assert engine.target_model.calls == 0


@pytest.mark.parametrize("gamma", [2, 4])
@pytest.mark.parametrize("draft_grammar_aware", [True, False])
@pytest.mark.parametrize("method", SPECULATIVE)
def test_draft_dead_end_is_left_to_the_target(monkeypatch, method, draft_grammar_aware, gamma):
    # The draft prefers "ab", after which "cd" is unreachable; the target picks
    # "a" and completes the pattern with "bcd".
    vocab = [b"", b"x", b"a", b"ab", b"bcd", b"cde"]
    engine = make_engine(
        monkeypatch, method, vocab,
        target=[0.0, 1.0, 3.0, 2.0, 1.0, 1.0],
        draft=[0.0, 1.0, 2.0, 3.0, 1.0, 1.0],
    )

    text, metrics = run(engine, method, "xabcd", gamma, draft_grammar_aware=draft_grammar_aware)

    assert text == "xabcd"
    assert_grammar_complete(method, metrics)


def assert_finish(method, metrics, reason):
    # generate_adaptive and generate_baseline do not report a finish reason.
    if method in ("generate", "stream_generate"):
        assert metrics["finish_reason"] == reason


# Token 0 is EOS: it has no grammar bytes and stops generation.
DIGITS = [b"", b"5", b"x"]
PREFER_DIGIT = [0.0, 3.0, 1.0]
PREFER_EOS = [5.0, 3.0, 1.0]


@pytest.mark.parametrize("method", SPECULATIVE + ["generate_baseline"])
def test_match_that_can_grow_continues_until_it_cannot(monkeypatch, method):
    # "5" already matches, but the model prefers more digits up to the limit.
    engine = make_engine(monkeypatch, method, DIGITS, target=PREFER_DIGIT)

    text, metrics = run(engine, method, "[0-9]{1,3}")

    assert text == "555"
    assert_finish(method, metrics, "grammar_complete")


@pytest.mark.parametrize("method", SPECULATIVE + ["generate_baseline"])
def test_model_ends_a_match_that_can_grow_with_eos(monkeypatch, method):
    # EOS is not selectable before the first digit; afterwards the model picks it.
    engine = make_engine(monkeypatch, method, DIGITS, target=PREFER_EOS)

    text, metrics = run(engine, method, "[0-9]{1,3}")

    assert text == "5"
    assert_finish(method, metrics, "stop")


@pytest.mark.parametrize("draft_grammar_aware", [True, False])
@pytest.mark.parametrize("method", SPECULATIVE)
@pytest.mark.parametrize(
    "target, draft, expected",
    [(PREFER_DIGIT, PREFER_EOS, "555"), (PREFER_EOS, PREFER_DIGIT, "5")],
    ids=["target-continues", "target-ends"],
)
def test_target_decides_eos_when_the_draft_disagrees(
    monkeypatch, method, draft_grammar_aware, target, draft, expected
):
    engine = make_engine(monkeypatch, method, DIGITS, target=target, draft=draft)

    text, metrics = run(
        engine, method, "[0-9]{1,3}", draft_grammar_aware=draft_grammar_aware
    )

    assert text == expected
    assert_finish(method, metrics, "grammar_complete" if expected == "555" else "stop")


@pytest.mark.parametrize("method", SPECULATIVE)
@pytest.mark.parametrize(
    "target, expected, reason",
    [(PREFER_DIGIT, "5" * 10, "length"), (PREFER_EOS, "5", "stop")],
    ids=["continues", "ends"],
)
def test_root_number_is_not_cut_at_its_first_digit(monkeypatch, method, target, expected, reason):
    engine = make_engine(monkeypatch, method, DIGITS, target=target)

    text, metrics = run(engine, method, None, json_schema='{"type": "integer"}')

    if method == "generate_adaptive" and reason == "length":
        # generate_adaptive does not cap its last proposal at the budget.
        assert set(text) == {"5"} and len(text) >= len(expected)
    else:
        assert text == expected
    assert_finish(method, metrics, reason)


@pytest.mark.parametrize("method", SPECULATIVE + ["generate_baseline"])
def test_empty_match_may_end_with_eos_at_once(monkeypatch, method):
    engine = make_engine(monkeypatch, method, DIGITS, target=PREFER_EOS)

    text, metrics = run(engine, method, "[0-9]*")

    assert text == ""
    assert_finish(method, metrics, "stop")


@pytest.mark.parametrize("method", SPECULATIVE + ["generate_baseline"])
def test_pattern_matching_only_empty_output_generates_nothing(monkeypatch, method):
    engine = make_engine(monkeypatch, method, DIGITS, target=PREFER_DIGIT)

    text, metrics = run(engine, method, "")

    assert text == ""
    assert_finish(method, metrics, "grammar_complete")
    assert engine.target_model.calls == 0
