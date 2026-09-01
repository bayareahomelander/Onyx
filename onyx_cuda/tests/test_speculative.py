import json
from types import SimpleNamespace

import pytest
import torch

import onyx_cuda.speculative as speculative_module
from onyx_cuda.benchmark import MAX_TOKENS, PROMPTS
from onyx_cuda.cache import CacheState
from onyx_cuda.generation import GenerationResult, generate_tokens
from onyx_cuda.model import load_model_pair
from onyx_cuda.prefill import prefill
from onyx_cuda.prompt import format_prompt
from onyx_cuda.speculative import (
    AcceptedTokenEvent,
    GenerationFinishedEvent,
    ProposalResult,
    TextDeltaEvent,
    VerificationResult,
    decode_speculative_events,
    generate_speculative,
    generate_speculative_events,
    propose_tokens,
    verify_proposal,
)
from onyx_cuda.vocabulary import (
    TokenByteVocabulary,
    build_token_byte_vocabulary,
)


class ScriptedCache:
    def __init__(self, token_ids):
        self.token_ids = iter(token_ids)
        self.inputs = []
        self.length = 4
        self.attention_mask = torch.ones((1, 4), dtype=torch.long)
        self.past_key_values = self

    def extend(self, model, input_ids):
        self.inputs.extend(input_ids.flatten().tolist())
        self.length += input_ids.shape[1]
        logits = torch.full((1, input_ids.shape[1], 16), -1.0)
        for position in range(input_ids.shape[1]):
            logits[0, position, next(self.token_ids)] = 0
        return logits

    def crop(self, length):
        self.length = length


class TrackingGrammar:
    def __init__(self, valid_tokens, matching_histories):
        self.valid_tokens = valid_tokens
        self.matching_histories = set(matching_histories)
        self.states = {}
        self.next_state = 1
        self.advance_calls = []

    @property
    def active_states(self):
        return set(self.states)

    def init_state(self):
        state = self.next_state
        self.next_state += 1
        self.states[state] = ()
        return state

    def get_valid_token_ids(self, state):
        return self.valid_tokens.get(self.states[state], [])

    def advance_state(self, state, token_id):
        history = self.states[state]
        self.advance_calls.append((history, token_id))
        next_state = self.next_state
        self.next_state += 1
        self.states[next_state] = (*history, token_id)
        return next_state

    def is_match_state(self, state):
        return self.states[state] in self.matching_histories

    def release_states(self, states):
        for state in states:
            if state not in self.states:
                raise ValueError(f"unknown grammar state {state}")
        for state in states:
            del self.states[state]


def _cpu_grammar_mask(logits, valid_token_ids):
    masked = torch.full_like(logits, -torch.inf)
    masked[..., valid_token_ids] = logits[..., valid_token_ids]
    return masked


def _run_scripted_constrained_speculation(
    monkeypatch,
    grammar,
    *,
    draft_tokens,
    target_tokens,
    max_tokens=3,
    regex="pattern",
    json_schema=None,
    vocabulary=None,
    events=False,
    collect=True,
):
    draft_model = object()
    target_model = object()
    draft_cache = ScriptedCache(draft_tokens)
    target_cache = ScriptedCache(target_tokens)

    def scripted_prefill(model, prompt_token_ids):
        cache = draft_cache if model is draft_model else target_cache
        logits = torch.full((1, 16), -1.0)
        logits[0, 1] = 0
        return SimpleNamespace(
            logits=logits,
            past_key_values=cache,
            token_id=torch.tensor([1]),
        )

    monkeypatch.setattr(speculative_module, "prefill", scripted_prefill)
    monkeypatch.setattr(
        speculative_module.CacheState,
        "from_prefill",
        classmethod(lambda cls, cache, device: cache),
    )
    monkeypatch.setattr(
        speculative_module,
        "_initialize_grammar_constraint",
        lambda *args: (grammar, grammar.init_state()),
    )
    monkeypatch.setattr(speculative_module, "apply_grammar_mask", _cpu_grammar_mask)
    vocabulary = vocabulary or TokenByteVocabulary([b""] * 16, 0, 16)
    generate = generate_speculative_events if events else generate_speculative
    result = generate(
        draft_model,
        target_model,
        [0],
        max_tokens=max_tokens,
        gamma=4,
        eos_token_ids=15,
        regex=regex,
        token_byte_vocabulary=vocabulary,
        json_schema=json_schema,
    )
    if events and collect:
        result = list(result)
    return result, draft_cache, target_cache


def _run_scripted_speculation(
    monkeypatch,
    *,
    first_token,
    draft_tokens,
    target_tokens,
    max_tokens,
    gamma=4,
    eos=15,
    stops=None,
    measure=False,
    events=False,
):
    draft_model = torch.nn.Linear(1, 1) if measure else object()
    target_model = torch.nn.Linear(1, 1) if measure else object()
    draft_cache = ScriptedCache(draft_tokens)
    target_cache = ScriptedCache(target_tokens)

    def scripted_prefill(model, prompt_token_ids):
        cache = draft_cache if model is draft_model else target_cache
        return SimpleNamespace(
            logits=torch.zeros((1, 16)),
            past_key_values=cache,
            token_id=torch.tensor([first_token]),
        )

    monkeypatch.setattr(speculative_module, "prefill", scripted_prefill)
    monkeypatch.setattr(
        speculative_module.CacheState,
        "from_prefill",
        classmethod(lambda cls, cache, device: cache),
    )
    generate = generate_speculative_events if events else generate_speculative
    result = generate(
        draft_model,
        target_model,
        [0],
        max_tokens=max_tokens,
        gamma=gamma,
        eos_token_ids=eos,
        stop_sequences=stops,
        measure=measure,
    )
    if events:
        result = list(result)
    return result, draft_cache, target_cache


def _result_from_events(events, tokenizer, stop=None):
    events = list(events)
    accepted = [event.token_id for event in events if isinstance(event, AcceptedTokenEvent)]
    decoded = list(decode_speculative_events(events, tokenizer, stop=stop))
    text = "".join(event.text for event in decoded if isinstance(event, TextDeltaEvent))
    finished = [event for event in decoded if isinstance(event, GenerationFinishedEvent)]
    assert len(finished) == 1
    result = finished[0].result
    assert accepted == result.token_ids
    expected = tokenizer.decode(result.token_ids, skip_special_tokens=True)
    if stop:
        positions = [expected.find(sequence) for sequence in stop if sequence]
        positions = [position for position in positions if position >= 0]
        if positions:
            expected = expected[: min(positions)]
    assert text == expected
    return result, text


class _MappedTokenizer:
    def __init__(self, decoded):
        self.decoded = decoded

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is True
        key = tuple(token_ids)
        if key in self.decoded:
            return self.decoded[key]
        return "".join(self.decoded[token] for token in token_ids)


def test_speculative_options_fail_before_prefill(monkeypatch):
    monkeypatch.setattr(
        speculative_module,
        "prefill",
        lambda *args: pytest.fail("prefill must not run"),
    )

    with pytest.raises(ValueError, match="gamma"):
        generate_speculative(object(), object(), [0], 1, 0, [])
    with pytest.raises(ValueError, match="max_tokens"):
        generate_speculative(object(), object(), [0], 0, 1, [])
    with pytest.raises(ValueError, match="token_byte_vocabulary"):
        generate_speculative(object(), object(), [0], 1, 1, [], regex="CUDA")


def test_constrained_rejection_verifies_from_canonical_state_and_releases(
    monkeypatch,
):
    grammar = TrackingGrammar(
        {(): [1], (1,): [2, 3]},
        {(1, 3)},
    )
    result, _, _ = _run_scripted_constrained_speculation(
        monkeypatch,
        grammar,
        draft_tokens=[2],
        target_tokens=[3, 0],
    )

    assert result == GenerationResult([1, 3], result.past_key_values, "stop")
    assert ((1,), 2) in grammar.advance_calls
    assert ((1,), 3) in grammar.advance_calls
    assert ((1, 2), 3) not in grammar.advance_calls
    assert not grammar.active_states


def test_constrained_event_stream_matches_result_and_releases_states(
    monkeypatch,
):
    grammar = TrackingGrammar(
        {(): [1], (1,): [2, 3]},
        {(1, 3)},
    )
    events, _, _ = _run_scripted_constrained_speculation(
        monkeypatch,
        grammar,
        draft_tokens=[2],
        target_tokens=[3, 0],
        events=True,
    )

    result, text = _result_from_events(events, _MappedTokenizer({1: "A", 3: "B"}))
    assert result.token_ids == [1, 3]
    assert result.finish_reason == "stop"
    assert text == "AB"
    assert not grammar.active_states


def test_constrained_draft_masks_invalid_token_and_validates_json(monkeypatch):
    grammar = TrackingGrammar(
        {(): [1], (1,): [2]},
        {(1, 2)},
    )
    token_bytes = [b""] * 16
    token_bytes[1] = b"{"
    token_bytes[2] = b"}"
    vocabulary = TokenByteVocabulary(token_bytes, 0, 14)
    result, _, _ = _run_scripted_constrained_speculation(
        monkeypatch,
        grammar,
        draft_tokens=[7, 0],
        target_tokens=[2, 9],
        regex="ignored",
        json_schema='{"type":"object"}',
        vocabulary=vocabulary,
    )

    assert result.token_ids == [1, 2]
    assert result.finish_reason == "stop"
    assert ((1,), 2) in grammar.advance_calls
    assert all(token_id != 7 for _, token_id in grammar.advance_calls)
    assert not grammar.active_states


def test_constrained_error_releases_every_state(monkeypatch):
    grammar = TrackingGrammar({(): [1], (1,): []}, set())

    with pytest.raises(ValueError, match="no valid token continuation"):
        _run_scripted_constrained_speculation(
            monkeypatch,
            grammar,
            draft_tokens=[2],
            target_tokens=[2],
        )

    assert not grammar.active_states


def test_closing_event_iterator_releases_grammar(monkeypatch):
    grammar = TrackingGrammar(
        {(): [1], (1,): [2, 3]},
        {(1, 3)},
    )
    events, _, _ = _run_scripted_constrained_speculation(
        monkeypatch,
        grammar,
        draft_tokens=[2],
        target_tokens=[3, 0],
        events=True,
        collect=False,
    )

    try:
        event = next(events)
        assert isinstance(event, AcceptedTokenEvent)
        assert event.token_id == 1
        assert grammar.active_states
    finally:
        events.close()

    assert not grammar.active_states


def test_sampled_speculation_forwards_grammar_to_target(monkeypatch):
    vocabulary = TokenByteVocabulary([b"x"], 0, 0)
    expected = GenerationResult([0], object(), "length")
    captured = {}

    def fake_generate_tokens(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(speculative_module, "generate_tokens", fake_generate_tokens)
    result = generate_speculative(
        object(),
        object(),
        [0],
        max_tokens=1,
        gamma=1,
        eos_token_ids=[],
        temperature=0.8,
        regex="x",
        token_byte_vocabulary=vocabulary,
        json_schema='{"type":"string"}',
    )

    assert result is expected
    assert captured["regex"] == "x"
    assert captured["token_byte_vocabulary"] is vocabulary
    assert captured["json_schema"] == '{"type":"string"}'
    assert captured["measure"] is False

    captured.clear()
    events = list(
        generate_speculative_events(
            object(),
            object(),
            [0],
            max_tokens=1,
            gamma=1,
            eos_token_ids=[],
            temperature=0.8,
            regex="x",
            token_byte_vocabulary=vocabulary,
            json_schema='{"type":"string"}',
        )
    )
    replayed, text = _result_from_events(events, _MappedTokenizer({0: "x"}))
    assert captured["regex"] == "x"
    assert captured["json_schema"] == '{"type":"string"}'
    assert replayed is expected
    assert [event.token_id for event in events if isinstance(event, AcceptedTokenEvent)] == [0]
    assert text == "x"


def test_speculative_metrics_count_proposals_and_stages(monkeypatch):
    result, _, _ = _run_scripted_speculation(
        monkeypatch,
        first_token=1,
        draft_tokens=[2, 3, 0],
        target_tokens=[2, 4, 9, 5],
        max_tokens=4,
        gamma=2,
        measure=True,
    )

    assert result.token_ids == [1, 2, 4, 5]
    assert result.finish_reason == "length"
    assert result.timings is not None
    assert result.timings.proposed_token_count == 2
    assert result.timings.accepted_proposal_count == 1
    assert result.timings.acceptance_rate == 0.5
    assert result.timings.speculative_iteration_count == 2
    assert result.timings.draft_seconds >= 0
    assert result.timings.verify_seconds >= 0
    assert result.timings.mask_seconds == 0
    assert result.timings.total_seconds >= (result.timings.time_to_first_token_seconds)


@pytest.mark.parametrize(
    (
        "max_tokens",
        "draft_tokens",
        "target_tokens",
        "expected_tokens",
    ),
    [
        (1, [], [], [1]),
        (2, [0], [2], [1, 2]),
        (3, [2, 0], [2, 3], [1, 2, 3]),
    ],
)
def test_speculative_final_capacity_is_exact(
    monkeypatch,
    max_tokens,
    draft_tokens,
    target_tokens,
    expected_tokens,
):
    result, draft_cache, target_cache = _run_scripted_speculation(
        monkeypatch,
        first_token=1,
        draft_tokens=draft_tokens,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )

    assert result.token_ids == expected_tokens
    assert result.finish_reason == "length"
    assert len(result.token_ids) == max_tokens
    if max_tokens == 1:
        assert draft_cache.inputs == target_cache.inputs == []


def test_text_events_match_collected_greedy_output(monkeypatch):
    events, _, _ = _run_scripted_speculation(
        monkeypatch,
        first_token=1,
        draft_tokens=[2, 3, 0],
        target_tokens=[2, 4, 9, 5],
        max_tokens=4,
        gamma=2,
        events=True,
    )

    result, text = _result_from_events(
        events,
        _MappedTokenizer({1: "A", 2: "B", 4: "C", 5: "D"}),
    )
    assert text == "ABCD"
    assert result.token_ids == [1, 2, 4, 5]
    assert result.finish_reason == "length"


def test_text_events_decode_split_utf8_and_hide_fragmented_stop(monkeypatch):
    events, _, _ = _run_scripted_speculation(
        monkeypatch,
        first_token=1,
        draft_tokens=[2, 3, 0],
        target_tokens=[2, 3, 4],
        max_tokens=4,
        gamma=2,
        events=True,
    )
    tokenizer = _MappedTokenizer(
        {
            (1,): "\ufffd",
            (1, 2): "caf\u00e9 ",
            (1, 2, 3): "caf\u00e9 EN",
            (1, 2, 3, 4): "caf\u00e9 END",
        }
    )
    result, text = _result_from_events(events, tokenizer, stop=["END"])
    assert result.token_ids == [1, 2, 3, 4]
    assert text == "caf\u00e9 "
    assert "\ufffd" not in text
    assert "END" not in text


@pytest.mark.parametrize(
    (
        "draft_tokens",
        "target_tokens",
        "eos",
        "stops",
        "expected_tokens",
        "expected_reason",
    ),
    [
        ([15, 0], [15, 9], 15, [], [1, 15], "eos"),
        ([8, 0], [8, 9], 15, [[8], [7, 8]], [], "stop"),
    ],
)
def test_speculative_eos_and_overlapping_stops_ignore_plus_one(
    monkeypatch,
    draft_tokens,
    target_tokens,
    eos,
    stops,
    expected_tokens,
    expected_reason,
):
    events, _, _ = _run_scripted_speculation(
        monkeypatch,
        first_token=7 if stops else 1,
        draft_tokens=draft_tokens,
        target_tokens=target_tokens,
        max_tokens=4,
        eos=eos,
        stops=stops,
        events=True,
    )
    accepted = [event.token_id for event in events if isinstance(event, AcceptedTokenEvent)]
    finished = [event for event in events if isinstance(event, GenerationFinishedEvent)]
    assert accepted == expected_tokens
    assert len(finished) == 1
    assert finished[0].result.token_ids == accepted
    assert finished[0].result.finish_reason == expected_reason


@pytest.mark.parametrize(
    (
        "generated",
        "scripted",
        "gamma",
        "remaining",
        "eos",
        "stops",
        "expected_tokens",
        "expected_inputs",
    ),
    [
        ([1], [2, 3, 4], 2, 5, 15, [], [2, 3], [1, 2]),
        ([1], [2, 3], 4, 1, 15, [], [2], [1]),
        ([1], [15, 3], 4, 4, 15, [], [15], [1]),
        ([7], [8, 9], 4, 4, 15, [[7, 8]], [8], [7]),
        ([15], [2], 4, 4, 15, [], [], []),
    ],
)
def test_scripted_draft_proposal_bounds_order_and_termination(
    generated,
    scripted,
    gamma,
    remaining,
    eos,
    stops,
    expected_tokens,
    expected_inputs,
):
    cache = ScriptedCache(scripted)
    result = propose_tokens(object(), cache, generated, gamma, remaining, eos, stops)

    assert result == ProposalResult(expected_tokens, 4, 4 + len(expected_tokens))
    assert cache.inputs == expected_inputs


def test_real_draft_proposal_matches_direct_greedy_prefix():
    pair = load_model_pair()
    prompt = format_prompt(
        pair.draft.tokenizer,
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with CUDA ready."},
        ],
    )
    draft_prefill = prefill(pair.draft.model, prompt.token_ids)
    target_prefill = prefill(pair.target.model, prompt.token_ids)
    draft_cache = CacheState.from_prefill(
        draft_prefill.past_key_values, draft_prefill.logits.device
    )
    target_token = target_prefill.token_id.item()
    proposal = propose_tokens(
        pair.draft.model,
        draft_cache,
        [target_token],
        gamma=4,
        remaining_tokens=4,
        eos_token_ids=pair.draft.tokenizer.eos_token_id,
    )
    direct = generate_tokens(
        pair.draft.model,
        prompt.token_ids,
        max_tokens=5,
        eos_token_ids=pair.draft.tokenizer.eos_token_id,
    )

    assert direct.token_ids[0] == target_token
    assert proposal.token_ids == direct.token_ids[1:]
    assert proposal == ProposalResult(
        [30982, 151645], len(prompt.token_ids), len(prompt.token_ids) + 2
    )
    assert draft_cache.attention_mask.shape == (1, len(prompt.token_ids) + 2)
    print(f"target_first_token_id={target_token}")
    print(f"draft_proposal_token_ids={proposal.token_ids}")


@pytest.mark.parametrize(
    ("target_tokens", "expected_tokens", "accepted"),
    [
        ([2, 3, 4, 5], [2, 3, 4, 5], 3),
        ([9, 3, 4, 5], [9], 0),
        ([2, 9, 4, 5], [2, 9], 1),
        ([2, 3, 9, 5], [2, 3, 9], 2),
    ],
)
def test_scripted_target_verification_acceptance_and_cache_lengths(
    target_tokens, expected_tokens, accepted
):
    proposal = ProposalResult([2, 3, 4], 4, 7)
    draft_cache = ScriptedCache([0])
    draft_cache.length = proposal.draft_cache_length_after
    target_cache = ScriptedCache(target_tokens)
    target_cache.length = 10

    result = verify_proposal(object(), draft_cache, object(), target_cache, 1, proposal)

    assert result == VerificationResult(expected_tokens, accepted)
    assert target_cache.inputs == [1, 2, 3, 4]
    assert target_cache.length == 10 + accepted + 1
    assert draft_cache.length == 4 + accepted + 1
    assert draft_cache.inputs == ([4] if accepted == 3 else [])


def test_real_speculation_matches_target_oracle_and_rejects_cleanly():
    pair = load_model_pair()
    prompt = format_prompt(
        pair.draft.tokenizer,
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with CUDA ready."},
        ],
    )
    draft_prefill = prefill(pair.draft.model, prompt.token_ids)
    target_prefill = prefill(pair.target.model, prompt.token_ids)
    draft_cache = CacheState.from_prefill(
        draft_prefill.past_key_values, draft_prefill.logits.device
    )
    target_cache = CacheState.from_prefill(
        target_prefill.past_key_values, target_prefill.logits.device
    )
    target_token = target_prefill.token_id.item()
    proposal = propose_tokens(
        pair.draft.model,
        draft_cache,
        [target_token],
        gamma=4,
        remaining_tokens=4,
        eos_token_ids=pair.draft.tokenizer.eos_token_id,
    )
    verified = verify_proposal(
        pair.draft.model,
        draft_cache,
        pair.target.model,
        target_cache,
        target_token,
        proposal,
    )
    assert verified == VerificationResult([5527], 0)

    replay_token = verified.token_ids[-1]
    max_replay_differences = {}
    for name, loaded, dirty_cache in (
        ("draft", pair.draft, draft_cache),
        ("target", pair.target, target_cache),
    ):
        with torch.inference_mode():
            dirty_logits = dirty_cache.extend(
                loaded.model,
                torch.tensor([[replay_token]], device=draft_prefill.logits.device),
            )[:, -1, :]
        clean_prefill = prefill(loaded.model, prompt.token_ids)
        clean_cache = CacheState.from_prefill(
            clean_prefill.past_key_values, clean_prefill.logits.device
        )
        with torch.inference_mode():
            clean_cache.extend(
                loaded.model,
                torch.tensor([[target_token]], device=clean_prefill.logits.device),
            )
            clean_logits = clean_cache.extend(
                loaded.model,
                torch.tensor([[replay_token]], device=clean_prefill.logits.device),
            )[:, -1, :]
        torch.testing.assert_close(dirty_logits, clean_logits, rtol=1e-2, atol=5e-2)
        assert dirty_logits.argmax(dim=-1).item() == (clean_logits.argmax(dim=-1).item())
        max_replay_differences[name] = (dirty_logits - clean_logits).abs().max().item()

    eos_token_id = pair.target.tokenizer.eos_token_id
    oracle_ids = {}
    for name, user_prompt in PROMPTS.items():
        formatted = format_prompt(
            pair.target.tokenizer,
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": user_prompt},
            ],
        )
        oracle = generate_tokens(
            pair.target.model,
            formatted.token_ids,
            max_tokens=MAX_TOKENS,
            eos_token_ids=eos_token_id,
        )
        oracle_ids[name] = oracle.token_ids
        for gamma in (1, 2, 4):
            measure = name == "cuda_ready" and gamma == 2
            speculative, streamed_text = _result_from_events(
                generate_speculative_events(
                    pair.draft.model,
                    pair.target.model,
                    formatted.token_ids,
                    max_tokens=MAX_TOKENS,
                    gamma=gamma,
                    eos_token_ids=eos_token_id,
                    measure=measure,
                ),
                pair.target.tokenizer,
            )
            assert speculative.token_ids == oracle.token_ids
            assert speculative.finish_reason == oracle.finish_reason
            assert streamed_text == pair.target.tokenizer.decode(
                oracle.token_ids, skip_special_tokens=True
            )
            if measure:
                assert speculative.timings is not None
                assert speculative.timings.proposed_token_count > 0
                assert speculative.timings.accepted_proposal_count >= 0
                assert 0 <= speculative.timings.acceptance_rate <= 1
                assert speculative.timings.speculative_iteration_count > 0
                assert speculative.timings.draft_seconds > 0
                assert speculative.timings.verify_seconds > 0
                assert speculative.timings.mask_seconds == 0

    sampled_oracle = generate_tokens(
        pair.target.model,
        prompt.token_ids,
        max_tokens=8,
        eos_token_ids=eos_token_id,
        temperature=0.8,
        top_p=0.9,
        seed=1234,
    )
    sampled_speculative, sampled_text = _result_from_events(
        generate_speculative_events(
            pair.draft.model,
            pair.target.model,
            prompt.token_ids,
            max_tokens=8,
            gamma=4,
            eos_token_ids=eos_token_id,
            temperature=0.8,
            top_p=0.9,
            seed=1234,
        ),
        pair.target.tokenizer,
    )
    assert sampled_speculative.token_ids == sampled_oracle.token_ids
    assert sampled_speculative.finish_reason == sampled_oracle.finish_reason
    assert sampled_text == pair.target.tokenizer.decode(
        sampled_oracle.token_ids, skip_special_tokens=True
    )

    vocabulary = build_token_byte_vocabulary(
        pair.target.tokenizer, pair.target.model.config.vocab_size
    )
    regex_oracle = generate_tokens(
        pair.target.model,
        prompt.token_ids,
        max_tokens=MAX_TOKENS,
        eos_token_ids=eos_token_id,
        regex="CUDA Ready",
        token_byte_vocabulary=vocabulary,
    )
    for gamma in (1, 2, 4):
        constrained, constrained_text = _result_from_events(
            generate_speculative_events(
                pair.draft.model,
                pair.target.model,
                prompt.token_ids,
                max_tokens=MAX_TOKENS,
                gamma=gamma,
                eos_token_ids=eos_token_id,
                regex="CUDA Ready",
                token_byte_vocabulary=vocabulary,
                measure=gamma == 2,
            ),
            pair.target.tokenizer,
        )
        assert constrained.token_ids == regex_oracle.token_ids
        assert constrained.finish_reason == regex_oracle.finish_reason
        assert constrained_text == pair.target.tokenizer.decode(
            regex_oracle.token_ids, skip_special_tokens=True
        )
        if gamma == 2:
            assert constrained.timings is not None
            assert constrained.timings.grammar_compile_seconds > 0
            assert constrained.timings.mask_seconds > 0

    schema = json.dumps(
        {
            "type": "object",
            "properties": {"content": {"enum": ["CUDA ready", "Ready"]}},
            "required": ["content"],
        },
        separators=(",", ":"),
    )
    json_prompt = format_prompt(
        pair.target.tokenizer,
        [
            {"role": "system", "content": "Return compact JSON only."},
            {
                "role": "user",
                "content": "Use no spaces or newlines in the JSON response.",
            },
        ],
    )
    json_oracle = generate_tokens(
        pair.target.model,
        json_prompt.token_ids,
        max_tokens=64,
        eos_token_ids=eos_token_id,
        json_schema=schema,
        token_byte_vocabulary=vocabulary,
    )
    for gamma in (1, 2, 4):
        constrained, constrained_text = _result_from_events(
            generate_speculative_events(
                pair.draft.model,
                pair.target.model,
                json_prompt.token_ids,
                max_tokens=64,
                gamma=gamma,
                eos_token_ids=eos_token_id,
                token_byte_vocabulary=vocabulary,
                json_schema=schema,
            ),
            pair.target.tokenizer,
        )
        assert constrained.token_ids == json_oracle.token_ids
        assert constrained.finish_reason == json_oracle.finish_reason
        assert constrained_text == pair.target.tokenizer.decode(
            json_oracle.token_ids, skip_special_tokens=True
        )
        assert json.loads(pair.target.tokenizer.decode(constrained.token_ids))

    print(f"speculative_oracle_token_ids={oracle_ids}")
    print(f"rejected_replay_max_abs_differences={max_replay_differences}")
