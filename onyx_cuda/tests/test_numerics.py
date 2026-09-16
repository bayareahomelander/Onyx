"""Rounding guards, independent canonical caches, and grammar ownership."""

from types import SimpleNamespace

import pytest
import torch

from onyx_cuda.numerics import ambiguous_logits


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("clean_steps", [0, 2])
def test_replay_stops_before_rejected_suffix_and_keeps_checkpoint_prefix(
    constrained, clean_steps, monkeypatch
):
    from onyx_cuda.numerics import GreedyCheckpoint
    from test_speculative import TrackingGrammar

    def select(logits, valid, **kwargs):
        masked = torch.full_like(logits, -torch.inf)
        masked[:, valid] = logits[:, valid]
        return masked.argmax(-1)

    monkeypatch.setattr("onyx_cuda.numerics.grammar_argmax", select)

    class Cache:
        def __init__(self):
            self.past_key_values = SimpleNamespace(history=[8, 8])
            self.attention_mask = torch.ones(1, 2)
            self.cache_position = torch.arange(2)

        @property
        def length(self):
            return len(self.past_key_values.history)

        def extend(self, model, ids):
            self.past_key_values.history.extend(ids[0].tolist())
            self.attention_mask = torch.ones(1, self.length)
            self.cache_position = torch.arange(self.length)
            logits = torch.full((1, 1, 9), -10.0)
            logits[0, 0, 5] = 1
            return logits

    class Grammar(TrackingGrammar):
        def get_valid_token_ids(self, state):
            assert state in self.states
            return [4, 5]

    grammar = Grammar({}, set()) if constrained else None
    live = set()
    checkpoint = GreedyCheckpoint(None, [8, 8], grammar, live, "torch")
    target = Cache()
    initial = torch.full((1, 9), -10.0)
    initial[0, 5] = 1
    checkpoint.seed(target, initial)
    for _ in range(clean_steps):
        checkpoint.after_scalar(target.extend(None, torch.tensor([[5]])))
    batch = torch.tensor([[5, 4, 4, 4]])
    generated = [5] * (clean_steps + 1)
    checkpoint.before_forward(target, batch, generated)
    logits = checkpoint.replay(target, batch, generated)
    assert logits.shape[1] == 1
    assert target.past_key_values.history == [8, 8, *generated]
    checkpoint.commit(target, batch, 0, logits)
    assert checkpoint.report()["skipped_proposal_tokens"] == 3
    assert checkpoint.report()["proposal_replay_tokens"] == 1
    assert checkpoint.report()["history_replay_tokens"] == 0
    # Mutating the working cache cannot mutate the retained checkpoint.
    target.extend(None, torch.tensor([[4]]))
    assert checkpoint.cache.past_key_values.history == [8, 8, *generated]
    if grammar:
        assert grammar.states[checkpoint.state] == tuple(generated)
        grammar.release_states(list(live))
        assert not grammar.active_states


def test_replay_rejects_noncanonical_emitted_prefix():
    from onyx_cuda.numerics import GreedyCheckpoint

    checkpoint = GreedyCheckpoint(None, [8], None, set(), "torch")
    checkpoint.cache = SimpleNamespace(length=1)
    checkpoint.logits = torch.tensor([[0., 2.]])
    with pytest.raises(RuntimeError, match="noncanonical emitted prefix"):
        checkpoint.replay(None, torch.tensor([[1]]), [0, 1])


def test_rounding_guard_checks_only_relevant_eligible_tokens():
    logits = torch.tensor([[[1, 0, -5], [1, 1, -5]]], dtype=torch.float16)
    assert not ambiguous_logits(logits, 1)
    assert ambiguous_logits(logits, 2)
    constraint = SimpleNamespace(get_valid_token_ids=lambda state: [0, 2])
    assert not ambiguous_logits(logits, 2, constraint, [0, 1])
    constraint.get_valid_token_ids = lambda state: [1]
    assert not ambiguous_logits(logits, 2, constraint, [0, 1])


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_replay_uses_clean_prefix_and_releases_grammar(monkeypatch, adaptive, constrained):
    import onyx_cuda.generation as generation
    import onyx_cuda.speculative as speculative
    import onyx_cuda.numerics as numerics
    from onyx_cuda.vocabulary import TokenByteVocabulary
    from test_speculative import TrackingGrammar

    prefills = []
    grammars = []

    class Cache:
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

        def extend(self, model, ids):
            if model.role == "target" and ids.shape[1] > 1:
                self.past_key_values.dirty = True
            logits = torch.full((1, ids.shape[1], 9), -10, dtype=torch.float16)
            for i, token in enumerate(ids[0].tolist()):
                self.past_key_values.history.append(token)
                logits[0, i, 5] = 1
                if model.role == "target" and self.past_key_values.dirty and self.length >= 5:
                    logits[0, i, 4] = 1  # Lower token ID wins a spurious FP16 tie.
            self.attention_mask = torch.ones((1, self.length), dtype=torch.long)
            self.cache_position = torch.arange(self.length)
            return logits

        def crop(self, length):
            assert length <= self.length
            del self.past_key_values.history[length:]
            self.attention_mask = self.attention_mask[:, :length]
            self.cache_position = self.cache_position[:length]

    def prefill(model, ids):
        prefills.append(model.role)
        logits = torch.full((1, 9), -10, dtype=torch.float16)
        logits[0, 5] = 1
        return SimpleNamespace(logits=logits, token_id=torch.tensor([5]),
                               past_key_values=SimpleNamespace(history=list(ids), dirty=False))

    for module in (generation, speculative, numerics):
        monkeypatch.setattr(module, "prefill", prefill)
        monkeypatch.setattr(module, "CacheState", Cache)

    class Grammar(TrackingGrammar):
        def get_valid_token_ids(self, state):
            assert state in self.states
            return [4, 5]

        def is_match_state(self, state):
            return len(self.states[state]) == 10

        def release_state(self, state):
            self.release_states([state])

    def initialize(*args):
        grammar = Grammar({}, set())
        grammars.append(grammar)
        return grammar, grammar.init_state()

    def select(logits, valid, **kwargs):
        masked = torch.full_like(logits, -torch.inf)
        masked[:, valid] = logits[:, valid]
        return masked.argmax(-1)

    for module in (generation, speculative):
        monkeypatch.setattr(module, "_initialize_grammar_constraint", initialize)
        monkeypatch.setattr(module, "grammar_argmax", select)
    monkeypatch.setattr(numerics, "grammar_argmax", select)
    target = SimpleNamespace(role="target", parameters=lambda: iter([torch.zeros(1)]))
    draft = SimpleNamespace(role="draft", parameters=lambda: iter([torch.zeros(1)]))
    args = dict(prompt_token_ids=[8] * 3, max_tokens=18, eos_token_ids=[])
    if constrained:
        args.update(regex="[45]{10}", token_byte_vocabulary=TokenByteVocabulary(
            [str(i).encode() for i in range(9)], 0, 0))
    oracle = generation.generate_tokens(target, **args)
    prefills.clear()
    result = speculative.generate_speculative(draft, target, gamma=2, adaptive=adaptive, **args)
    assert (result.token_ids, result.finish_reason) == (oracle.token_ids, oracle.finish_reason)
    assert prefills.count("target") == 1  # Reuse the original clean prompt cache.
    assert all(not grammar.active_states for grammar in grammars)
    if adaptive:
        assert result.timings.verification_replays > 0
    # Closing a stream after replay releases checkpoint-owned grammar handles.
    events = speculative.generate_speculative_events(draft, target, gamma=2, adaptive=adaptive, **args)
    for _ in range(5):
        next(events)
    events.close()
    assert all(not grammar.active_states for grammar in grammars)


@pytest.mark.gpu
def test_real_rounding_regressions_match_target_only_with_and_without_profiling():
    from onyx_cuda.model import load_model_pair
    from onyx_cuda.prompt import format_prompt
    from onyx_cuda.benchmark_corpus import CORPUS
    from onyx_cuda.speculative import generate_speculative

    pair = load_model_pair()
    replay_count = 0
    for name in ("apology", "comparison"):
        case = next(c for c in CORPUS if c["name"] == name)
        ids = format_prompt(pair.target.tokenizer, case["messages"], enable_thinking=False).token_ids
        args = dict(draft_model=pair.draft.model, target_model=pair.target.model,
                    prompt_token_ids=ids, max_tokens=512,
                    eos_token_ids=pair.target.tokenizer.eos_token_id)
        oracle = generate_speculative(**args, gamma=0)
        expected = (oracle.token_ids.copy(), oracle.finish_reason)
        del oracle
        for adaptive, measure in ((False, True), (True, False), (True, True)):
            result = generate_speculative(**args, gamma=2, adaptive=adaptive, measure=measure)
            assert (result.token_ids, result.finish_reason) == expected
            replay_count += result.timings.verification_replays
            del result
    assert replay_count > 0
