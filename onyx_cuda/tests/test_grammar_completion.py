"""A complete match that can still grow lets the model, not the first match, end output."""

import json
from types import SimpleNamespace

import pytest
import torch

import onyx_cuda.generation as generation
import onyx_cuda.numerics as numerics
import onyx_cuda.speculative as speculative
from onyx_cuda import _rust
from onyx_cuda.vocabulary import TokenByteVocabulary

NATIVE_CONSTRAINT = _rust.GrammarConstraint
EOS = 0  # Lowest ID, so a spurious FP16 tie with a digit selects EOS.
LETTER = 11
TOKEN_BYTES = [b"", *(str(digit).encode() for digit in range(10)), b"a"]
VOCABULARY = TokenByteVocabulary(TOKEN_BYTES, 1, 1)
PROMPT = [LETTER] * 3


def ids(text):
    return [int(character) + 1 for character in text]


def scripted_model(role, text):
    """Prefer the scripted digits, then EOS; every grammar here masks LETTER."""
    script = ids(text)

    def scores(history, dirty):
        position = len(history) - len(PROMPT)
        preferred = script[position] if position < len(script) else EOS
        values = torch.full((len(TOKEN_BYTES),), -4.0, dtype=torch.float16)
        values[LETTER] = 5
        values[EOS] = 1
        values[preferred] = 3
        if dirty and position > 0 and preferred != EOS:
            values[EOS] = 3  # Batched verification rounding; single-token forwards stay exact.
        return values

    return SimpleNamespace(role=role, scores=scores, parameters=lambda: iter([torch.zeros(1)]))


class Cache:
    device = torch.device("cpu")
    def __init__(self, kv):
        self.past_key_values = kv
        self.attention_mask = torch.ones((1, len(kv.history)), dtype=torch.long)
        self.cache_position = torch.arange(len(kv.history))

    @classmethod
    def from_prefill(cls, kv, device):
        return cls(kv)

    @property
    def length(self):
        return len(self.past_key_values.history)

    def extend(self, model, input_ids):
        kv = self.past_key_values
        if model.role == "target":
            kv.forwards.append(input_ids.shape[1])
            if input_ids.shape[1] > 1:
                kv.dirty = kv.dirty or kv.rounding
        rows = []
        for token in input_ids[0].tolist():
            kv.history.append(token)
            rows.append(model.scores(kv.history, kv.dirty))
        self.attention_mask = torch.ones((1, self.length), dtype=torch.long)
        self.cache_position = torch.arange(self.length)
        return torch.stack(rows)[None]

    def crop(self, length):
        del self.past_key_values.history[length:]
        self.attention_mask = self.attention_mask[:, :length]
        self.cache_position = self.cache_position[:length]


class TrackedConstraint:
    """The native grammar, with every live state handle recorded."""

    instances = []

    def __init__(self, token_bytes):
        self.inner = NATIVE_CONSTRAINT(token_bytes)
        self.live = set()
        TrackedConstraint.instances.append(self)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def init_state(self):
        state = self.inner.init_state()
        self.live.add(state)
        return state

    def advance_state(self, state, token_id):
        next_state = self.inner.advance_state(state, token_id)
        self.live.add(next_state)
        return next_state

    def release_state(self, state):
        self.release_states([state])

    def release_states(self, states):
        self.inner.release_states(states)
        self.live.difference_update(states)


def masked(scores, valid, **_):
    result = torch.full_like(scores, -torch.inf)
    result[..., valid] = scores[..., valid]
    return result


@pytest.fixture
def run(monkeypatch):
    forwards = []
    TrackedConstraint.instances = []
    monkeypatch.setattr(_rust, "GrammarConstraint", TrackedConstraint)
    for module in (generation, speculative, numerics):
        monkeypatch.setattr(module, "CacheState", Cache)
        monkeypatch.setattr(module, "grammar_argmax", lambda *args, **kwargs: masked(*args).argmax(-1))
    monkeypatch.setattr(generation, "apply_grammar_mask", masked)

    def execute(mode, target_text, draft_text=None, *, max_tokens=8, rounding=False,
                measure=False, **constraint):
        target = scripted_model("target", target_text)
        draft = scripted_model("draft", target_text if draft_text is None else draft_text)

        def prefill(model, prompt_ids):
            kv = SimpleNamespace(history=list(prompt_ids), dirty=False, rounding=rounding,
                                 forwards=forwards if model.role == "target" else [])
            logits = model.scores(kv.history, False)[None]
            return SimpleNamespace(logits=logits, past_key_values=kv, token_id=logits.argmax(-1))

        for module in (generation, speculative, numerics):
            monkeypatch.setattr(module, "prefill", prefill)
        forwards.clear()
        arguments = dict(prompt_token_ids=PROMPT, max_tokens=max_tokens, eos_token_ids=[EOS],
                         token_byte_vocabulary=VOCABULARY, **constraint)
        if mode == "target":
            return generation.generate_tokens(target, **arguments)
        if mode == "sampled":
            return generation.generate_tokens(target, temperature=0.01, seed=0, **arguments)
        return speculative.generate_speculative(
            draft, target, gamma=2, adaptive=mode == "adaptive",
            measure=measure or mode == "measured", **arguments)

    execute.forwards = forwards
    yield execute
    assert all(not constraint.live for constraint in TrackedConstraint.instances)


MODES = ["target", "sampled", "fixed", "adaptive", "measured"]
OPEN_ENDED = [{"regex": "[0-9]+"}, {"json_schema": json.dumps({"type": "integer"})}]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("constraint", OPEN_ENDED)
@pytest.mark.parametrize("draft_text", [None, "4271", "4", "9999"])
def test_open_ended_constraint_ends_where_the_model_ends(run, mode, constraint, draft_text):
    result = run(mode, "427", draft_text, **constraint)
    assert (result.token_ids, result.finish_reason) == ([*ids("427"), EOS], "eos")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("constraint", OPEN_ENDED)
@pytest.mark.parametrize("budget,expected,reason", [
    (2, ids("42"), "length"),
    (3, ids("427"), "length"),
    (4, [*ids("427"), EOS], "eos"),
])
def test_budget_can_end_a_complete_but_growing_match(run, mode, constraint, budget, expected, reason):
    result = run(mode, "427", max_tokens=budget, **constraint)
    assert (result.token_ids, result.finish_reason) == (expected, reason)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pattern,target_text,expected,reason", [
    ("[0-9]{2}", "427", ids("42"), "stop"),
    ("[0-9]{2,4}", "427", [*ids("427"), EOS], "eos"),
    ("[0-9]{2,4}", "42715", ids("4271"), "stop"),
])
def test_bounded_regex_stops_only_when_it_cannot_grow(run, mode, pattern, target_text, expected, reason):
    result = run(mode, target_text, regex=pattern)
    assert (result.token_ids, result.finish_reason) == (expected, reason)


def test_complete_match_that_cannot_grow_costs_no_extra_forward(run):
    result = run("target", "427", regex="[0-9]{2}")
    assert result.finish_reason == "stop"
    assert run.forwards == [1]  # Prefill chose 4; one forward chose 2, then generation ended.


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target_text,expected,reason", [
    ("123", ids("123"), "stop"),
    ("12", [*ids("12"), EOS], "eos"),
])
def test_enum_value_prefix_does_not_end_a_longer_value(run, mode, target_text, expected, reason):
    result = run(mode, target_text, json_schema=json.dumps({"enum": [12, 123]}))
    assert (result.token_ids, result.finish_reason) == (expected, reason)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("constraint", OPEN_ENDED)
@pytest.mark.parametrize("draft_text", [None, "4271", "4"])
def test_eos_rounding_ties_replay_to_the_target_only_result(run, adaptive, constraint, draft_text):
    oracle = run("target", "427", **constraint)
    result = run("adaptive" if adaptive else "fixed", "427", draft_text, rounding=True,
                 measure=True, **constraint)
    assert (result.token_ids, result.finish_reason) == (oracle.token_ids, oracle.finish_reason)
    assert oracle.token_ids == [*ids("427"), EOS]
    # The guard must see EOS as eligible, or the tie would silently end output early.
    assert result.timings.verification_replays > 0
