"""Deterministic scheduling and cache-transition invariants."""

import pytest
import torch

from onyx_cuda.adaptive import AdaptiveController
from onyx_cuda.speculative import _catch_up_draft


def calibrated(cost=0.04):
    controller = AdaptiveController()
    assert controller.choose(100) == 2
    controller.observe(2, 2, 2, 3, cost)
    assert controller.choose(100) == 0
    controller.observe(0, 0, 0, 1, 0.03)
    return controller


def test_cost_and_acceptance_expand_profitable_proposals():
    controller = calibrated()
    for _ in range(2):
        controller.observe(2, 2, 2, 3, 0.04)
    assert controller.choose(100) == 4
    for _ in range(2):
        controller.observe(4, 4, 4, 5, 0.055)
    assert controller.choose(100) == 8
    assert controller.choose(3) == 2
    assert controller.choose(1) == 0


def test_expansion_must_beat_the_measured_shorter_proposal():
    controller = calibrated()
    for _ in range(2):
        controller.observe(2, 2, 2, 3, 0.04)
    assert controller.gamma == 4
    for _ in range(2):
        controller.observe(4, 4, 4, 5, 0.10)
    # Gamma 4 still beats target-only (0.02 vs 0.03 seconds/token), but loses
    # to gamma 2 (0.013 seconds/token).
    assert controller.gamma == 2


def test_large_execution_loss_switches_to_target_without_more_trials():
    controller = calibrated()
    controller.observe(2, 2, 2, 3, 0.5)
    assert controller.choose(100) == 0
    for _ in range(8):
        controller.observe(0, 0, 0, 1, 0.03)
    assert controller.choose(100) == 2


def test_bad_expansion_returns_to_a_profitable_shorter_proposal():
    controller = calibrated()
    for _ in range(2):
        controller.observe(2, 2, 2, 3, 0.04)
    assert controller.gamma == 4
    controller.observe(4, 4, 0, 1, 0.08)
    assert controller.gamma == 2


def test_measured_draft_cost_and_acceptance_prevent_unprofitable_expansion():
    controller = calibrated()
    # Intermittent rejection followed by a full-acceptance streak must not
    # trigger an expansion whose extra drafting has negative expected value.
    for _ in range(4):
        controller.observe(2, 2, 0, 1, 0.05, proposal_seconds=0.02)
        controller.observe(2, 2, 2, 3, 0.05, proposal_seconds=0.02)
    for _ in range(2):
        controller.observe(2, 2, 2, 3, 0.05, proposal_seconds=0.02)
    assert controller.gamma == 2


def test_expansion_estimate_includes_full_acceptance_cache_catchup():
    controller = AdaptiveController(target_cost=0.035)
    controller.acceptance.append((6, 10))
    controller.draft_costs.append((0.008, 1))
    # With p=0.6, gamma 2 saves over 5% versus gamma 1 after accounting
    # for the extra draft step owed by a fully accepted round.
    assert controller._expansion_pays(1)


def test_full_acceptance_does_not_excuse_expensive_drafting():
    controller = calibrated(0.15)
    for _ in range(2):
        controller.observe(2, 2, 2, 3, 0.15)
    assert controller.choose(100) == 1
    for _ in range(3):
        controller.observe(1, 1, 1, 2, 0.09)
    assert controller.gamma == 0
    assert controller.choose(100) == 0


def test_failed_retries_back_off_but_success_restores_speculation():
    controller = calibrated(0.15)
    for _ in range(2):
        controller.observe(2, 2, 0, 1, 0.09)
    for _ in range(3):
        controller.observe(1, 1, 0, 1, 0.09)
    for _ in range(8):
        assert controller.choose(100) == 0
        controller.observe(0, 0, 0, 1, 0.03)
    assert controller.choose(100) == 2
    controller.observe(2, 2, 2, 3, 0.15, catchup_seconds=0.1)
    for _ in range(2):
        controller.observe(2, 2, 2, 3, 0.15)
    assert controller.gamma == 0
    assert controller.retry_interval == 16
    for _ in range(16):
        controller.observe(0, 0, 0, 1, 0.03)
    assert controller.choose(100) == 2
    controller.observe(2, 2, 2, 3, 0.05, catchup_seconds=0.01)
    for _ in range(2):
        controller.observe(2, 2, 2, 3, 0.05)
    assert controller.gamma == 2
    assert controller.recoveries == 1
    assert controller.retries == 2
    assert controller.catchup_seconds == pytest.approx(0.11)


def test_small_timing_noise_does_not_toggle_modes():
    controller = calibrated(0.09)
    for seconds in (0.089, 0.091, 0.092, 0.088):
        controller.observe(2, 2, 1, 3, seconds)
    assert controller.gamma == 2


def test_catchup_consumes_only_missing_accepted_prefix_in_bounded_chunks():
    class Cache:
        length = 5
        attention_mask = torch.ones((1, 5))
        inputs = []

        def extend(self, model, ids):
            self.inputs.append(ids[0].tolist())
            self.length += ids.shape[1]

    cache = Cache()
    generated = list(range(100))
    _catch_up_draft(None, cache, [10] * 4, generated)
    assert sum(cache.inputs, []) == generated[1:-1]
    assert max(map(len, cache.inputs)) <= 32
    assert cache.length == 4 + len(generated) - 1
    cache.length += 1
    with pytest.raises(RuntimeError, match="prefix"):
        _catch_up_draft(None, cache, [10] * 4, generated)


def test_corpus_is_fixed_and_has_disjoint_heldout_cases():
    from onyx_cuda.benchmark_corpus import CORPUS
    assert len(CORPUS) == len({c["name"] for c in CORPUS}) == 48
    assert sum(c["original"] for c in CORPUS) == 9
    assert sum(c["split"] == "heldout" for c in CORPUS) == 16


@pytest.mark.parametrize("stop", [[], [[6, 7]]])
@pytest.mark.parametrize("budget", [1, 2, 3, 12, 45])
@pytest.mark.parametrize("constrained", [False, True])
def test_mode_transitions_preserve_target_tokens_and_cache(monkeypatch, stop, budget, constrained):
    import onyx_cuda.speculative as speculative
    import onyx_cuda.generation as generation
    from types import SimpleNamespace

    forwards = []
    grammars = []

    class Cache:
        def __init__(self, history):
            self.history = list(history)
            self.attention_mask = torch.ones((1, len(history)), dtype=torch.long)
            self.past_key_values = self

        @property
        def length(self):
            return len(self.history)

        def extend(self, model, ids):
            forwards.append((model.role, ids[0].tolist()))
            logits = torch.full((1, ids.shape[1], 9), -100.0)
            for i, token in enumerate(ids[0].tolist()):
                self.history.append(token)
                next_token = len(self.history) % 7 + 1
                # Both acceptance and rejection, including after catch-up.
                if model.role == "draft" and len(self.history) % 5 == 0:
                    next_token = next_token % 7 + 1
                logits[0, i, next_token] = 1
            return logits

        def crop(self, length):
            assert length <= self.length
            del self.history[length:]

        @classmethod
        def from_prefill(cls, cache, device):
            return cache

    def prefill(model, ids):
        forwards.append((model.role, list(ids)))
        token = len(ids) % 7 + 1
        logits = torch.full((1, 9), -100.0)
        logits[0, token] = 1
        return SimpleNamespace(past_key_values=Cache(ids), logits=logits, token_id=torch.tensor([token]))

    class Scheduled(AdaptiveController):
        def choose(self, remaining):
            schedule = [2, 0, 0, 0, 2, 1, 0, 0, 4, 8]
            return min(schedule[self.rounds % len(schedule)], max(remaining - 1, 0))

    for module in (generation, speculative):
        monkeypatch.setattr(module, "prefill", prefill)
        monkeypatch.setattr(module, "CacheState", Cache)
    if constrained:
        from test_speculative import TrackingGrammar
        from onyx_cuda.vocabulary import TokenByteVocabulary

        class Grammar(TrackingGrammar):
            def release_state(self, state):
                self.release_states([state])

            def get_valid_token_ids(self, state):
                assert state in self.states
                return list(range(1, 8))

            def is_match_state(self, state):
                return len(self.states[state]) == 13

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
    monkeypatch.setattr(speculative, "AdaptiveController", Scheduled)
    target = SimpleNamespace(role="target", parameters=lambda: iter([torch.zeros(1)]))
    draft = SimpleNamespace(role="draft", parameters=lambda: iter([torch.zeros(1)]))
    kwargs = dict(prompt_token_ids=[8, 8, 8, 8], max_tokens=budget, eos_token_ids=[], stop_sequences=stop)
    if constrained:
        kwargs.update(regex="[1-7]{13}", token_byte_vocabulary=TokenByteVocabulary(
            [str(i).encode() for i in range(9)], 8, 8))
    oracle = generation.generate_tokens(target, **kwargs)
    forwards.clear()
    result = speculative.generate_speculative(draft, target, gamma=2, adaptive=True, **kwargs)
    assert (result.token_ids, result.finish_reason) == (oracle.token_ids, oracle.finish_reason)
    if not stop and not constrained:
        assert result.past_key_values.history == oracle.past_key_values.history
    if budget <= 2:
        assert all(role != "draft" for role, _ in forwards)
    if budget == 45 and not stop:
        assert result.timings.adaptive_stats["gamma_histogram"][0] >= 3
        assert any(role == "draft" and len(ids) > 1 for role, ids in forwards[2:])
    assert all(not grammar.active_states for grammar in grammars)
    events = speculative.generate_speculative_events(draft, target, gamma=2, adaptive=True, **kwargs)
    next(events)
    events.close()
    assert all(not grammar.active_states for grammar in grammars)


def test_elapsed_cost_weights_each_useful_token_once():
    controller = calibrated()
    controller.observe(2, 2, 0, 1, 0.04)
    controller.observe(2, 2, 2, 3, 0.04)
    assert controller.costs[2] == pytest.approx(0.12 / 7)
    assert controller.gamma == 2


@pytest.mark.gpu
def test_adaptive_default_pair_matches_fixed_modes_and_streams():
    from onyx_cuda.model import load_model_pair
    from onyx_cuda.prompt import format_prompt
    from onyx_cuda.speculative import generate_speculative, generate_speculative_events, decode_speculative_events
    from onyx_cuda.generation import GenerationFinishedEvent
    from onyx_cuda.vocabulary import get_token_byte_vocabulary

    pair = load_model_pair()
    tokenizer = pair.target.tokenizer
    vocabulary = get_token_byte_vocabulary(tokenizer, pair.target.model.config.vocab_size)
    for prompt, constraint in [
        ("List the integers 1 through 25 separated by commas.", {}),
        ("Give a short explanation of why a cache speeds up a program.", {}),
        ("Return 32 digits.", {"regex": "[0-9]{32}", "token_byte_vocabulary": vocabulary}),
        ("Return JSON true.", {"json_schema": '{"type":"boolean"}', "token_byte_vocabulary": vocabulary}),
    ]:
        ids = format_prompt(tokenizer, [{"role": "user", "content": prompt}], enable_thinking=False).token_ids
        args = dict(draft_model=pair.draft.model, target_model=pair.target.model, prompt_token_ids=ids,
                    max_tokens=256, eos_token_ids=tokenizer.eos_token_id, **constraint)
        oracle = generate_speculative(**args, gamma=0)
        expected = (oracle.token_ids.copy(), oracle.finish_reason)
        del oracle
        fixed = generate_speculative(**args, gamma=2)
        assert (fixed.token_ids, fixed.finish_reason) == expected
        del fixed
        for measure in (False, True):
            parts = []
            for event in decode_speculative_events(
                generate_speculative_events(**args, gamma=2, adaptive=True, measure=measure), tokenizer
            ):
                if isinstance(event, GenerationFinishedEvent):
                    assert (event.result.token_ids, event.result.finish_reason) == expected
                    assert event.result.timings.adaptive_stats is not None
                else:
                    parts.append(event.text)
            assert "".join(parts) == tokenizer.decode(expected[0], skip_special_tokens=True)
            del event
