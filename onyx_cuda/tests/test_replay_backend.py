"""Recovery isolation, safe fallback, and explicit graph lifecycle."""
from types import SimpleNamespace

import pytest
import torch

from onyx_cuda.replay_backend import prepare_replay_backend, close_replay_backend, resolve_replay_backend
from onyx_cuda.numerics import GreedyCheckpoint


def test_backend_selection_and_unsupported_model(monkeypatch):
    monkeypatch.setenv("ONYX_REPLAY_BACKEND", "graph")
    assert resolve_replay_backend() == "graph"
    assert resolve_replay_backend("scalar") == "scalar"
    with pytest.raises(ValueError, match="scalar or graph"):
        resolve_replay_backend("automatic")
    model = SimpleNamespace()
    assert prepare_replay_backend(model)["active"] == "scalar"
    assert not hasattr(model, "_onyx_replay_backend")
    close_replay_backend(model)


class HistoryCache:
    def __init__(self):
        self.past_key_values = SimpleNamespace(tokens=[8])
        self.attention_mask = torch.ones(1, 1)
        self.cache_position = torch.arange(1)

    @property
    def length(self):
        return len(self.past_key_values.tokens)

    def extend(self, model, ids):
        self.past_key_values.tokens.extend(ids[0].tolist())
        logits = torch.zeros(1, ids.shape[1], 9)
        logits[:, :, 5] = 1
        return logits


@pytest.mark.parametrize("behavior", ["exact", "mismatch", "unsupported"])
def test_chunks_check_prefix_and_rollback_before_scalar_fallback(behavior):
    calls = []
    def extend(cache, ids):
        calls.append(ids.shape[1])
        if behavior == "unsupported":
            return None
        logits = cache.extend(None, ids)
        if behavior == "mismatch":
            logits[:, :, 4] = 2
        return logits
    model = SimpleNamespace(_onyx_replay_backend=SimpleNamespace(extend=extend))
    checkpoint = GreedyCheckpoint(model, [8], None, set(), "torch")
    cache = HistoryCache()
    initial = torch.zeros(1, 9); initial[:, 5] = 1
    checkpoint.seed(cache, initial)
    output = checkpoint.replay(cache, torch.tensor([[5]]), [5] * 6)
    assert cache.past_key_values.tokens == [8] + [5] * 6
    assert output.argmax(-1).item() == 5
    assert checkpoint.history_tokens == 5
    assert checkpoint.chunked_history_tokens == (5 if behavior == "exact" else 0)
    assert checkpoint.graph_replay_fallbacks == (0 if behavior == "exact" else 1)
    assert calls == ([3, 2] if behavior == "exact" else [3])


def test_bad_prefix_still_raises_after_chunk_fallback():
    model = SimpleNamespace(_onyx_replay_backend=SimpleNamespace(extend=lambda cache, ids: cache.extend(None, ids)))
    checkpoint = GreedyCheckpoint(model, [8], None, set(), "torch")
    cache = HistoryCache(); initial = torch.zeros(1, 9); initial[:, 5] = 1
    checkpoint.seed(cache, initial)
    with pytest.raises(RuntimeError, match="noncanonical emitted prefix"):
        checkpoint.replay(cache, torch.tensor([[5]]), [5, 4, 5, 5])
    assert checkpoint.graph_replay_fallbacks == 1
    assert checkpoint.cache.past_key_values.tokens == [8, 5]


@pytest.mark.parametrize("history, widths", [
    (0, []), (1, []), (2, [2]), (3, [3]), (4, [3]), (5, [3, 2]),
    (6, [3, 3]), (7, [3, 3]), (8, [8]), (9, [8]), (10, [8, 2]),
    (11, [8, 3]), (12, [8, 3]), (13, [8, 3, 2]), (14, [8, 3, 3]),
    (15, [8, 3, 3]), (16, [8, 8]), (24, [8, 8, 8]),
])
def test_history_blocks_preserve_cache_and_proposal_boundary(history, widths):
    calls = []
    def extend(cache, ids):
        calls.append(ids.shape[1])
        return cache.extend(None, ids)
    model = SimpleNamespace(_onyx_replay_backend=SimpleNamespace(extend=extend))
    checkpoint = GreedyCheckpoint(model, [8], None, set(), "torch")
    cache = HistoryCache()
    checkpoint.seed(cache, torch.nn.functional.one_hot(torch.tensor([5]), 9).float())
    # The rejected suffix must never enter either the history or proposal cache.
    output = checkpoint.replay(cache, torch.tensor([[5, 4, 4]]), [5] * (history + 1))
    assert calls == widths
    assert cache.past_key_values.tokens == [8] + [5] * (history + 1)
    assert output.shape == (1, 1, 9)
    assert checkpoint.history_tokens == history
    assert checkpoint.chunked_history_tokens == sum(widths)
    assert checkpoint.proposal_tokens == 1
    assert checkpoint.skipped_proposal_tokens == 2
    assert checkpoint.replayed_tokens == history + 1
    assert checkpoint.graph_replay_fallbacks == 0


@pytest.mark.parametrize("committed", [0, 8])
@pytest.mark.parametrize("bad_row", [None, *range(8)])
def test_wide_block_failure_discards_trial_and_preserves_committed_prefix(committed, bad_row):
    calls = []
    def extend(cache, ids):
        calls.append((cache.length, ids.shape[1]))
        logits = cache.extend(None, ids)
        if cache.length == 1 + committed + 8:
            # Even a backend returning None after tentative mutation cannot
            # publish the trial. A mismatch in any row also rejects the block.
            if bad_row is None:
                return None
            logits[:, bad_row, 4] = 2
        return logits
    model = SimpleNamespace(_onyx_replay_backend=SimpleNamespace(extend=extend))
    checkpoint = GreedyCheckpoint(model, [8], None, set(), "torch")
    cache = HistoryCache()
    checkpoint.seed(cache, torch.nn.functional.one_hot(torch.tensor([5]), 9).float())
    history = committed + 16
    checkpoint.replay(cache, torch.tensor([[5]]), [5] * (history + 1))
    assert calls == ([(1, 8), (9, 8)] if committed else [(1, 8)])
    assert cache.past_key_values.tokens == [8] + [5] * (history + 1)
    assert checkpoint.history_tokens == history
    assert checkpoint.chunked_history_tokens == committed
    assert checkpoint.graph_replay_fallbacks == 1
    assert checkpoint.replay_backend is None


@pytest.mark.parametrize("bad_index", range(18))
def test_wide_history_rejects_corrupt_emitted_and_current_tokens(bad_index):
    model = SimpleNamespace(_onyx_replay_backend=SimpleNamespace(
        extend=lambda cache, ids: cache.extend(None, ids)))
    checkpoint = GreedyCheckpoint(model, [8], None, set(), "torch")
    cache = HistoryCache()
    checkpoint.seed(cache, torch.nn.functional.one_hot(torch.tensor([5]), 9).float())
    generated = [5] * 18
    generated[bad_index] = 4
    kind = "current token" if bad_index == 17 else "emitted prefix"
    with pytest.raises(RuntimeError, match=f"noncanonical {kind}"):
        checkpoint.replay(cache, torch.tensor([[generated[-1]]]), generated)
    assert checkpoint.cache.past_key_values.tokens == [8] + [5] * bad_index
    assert cache.past_key_values.tokens == [8]


@pytest.mark.parametrize("constrained", [False, True])
def test_scalar_and_constrained_history_never_use_graphs(constrained, monkeypatch):
    from test_speculative import TrackingGrammar
    class Grammar(TrackingGrammar):
        def get_valid_token_ids(self, state):
            return [4, 5]
    grammar = Grammar({}, set()) if constrained else None
    live = set()
    def unexpected(*args):
        pytest.fail("Constrained history used graph recovery")
    model = SimpleNamespace(_onyx_replay_backend=SimpleNamespace(extend=unexpected)) if constrained else None
    monkeypatch.setattr("onyx_cuda.numerics.grammar_argmax", lambda logits, valid, **kw: logits.argmax(-1))
    checkpoint = GreedyCheckpoint(model, [8], grammar, live, "torch")
    cache = HistoryCache()
    checkpoint.seed(cache, torch.nn.functional.one_hot(torch.tensor([5]), 9).float())
    output = checkpoint.replay(cache, torch.tensor([[5, 4]]), [5] * 18)
    assert cache.past_key_values.tokens == [8] + [5] * 18
    assert checkpoint.history_tokens == 17 and checkpoint.chunked_history_tokens == 0
    assert checkpoint.proposal_tokens == 1 and checkpoint.skipped_proposal_tokens == 1
    checkpoint.commit(cache, torch.tensor([[5, 4]]), 0, output)
    if grammar:
        assert grammar.states[checkpoint.state] == tuple([5] * 18)
        grammar.release_states(list(live))
        assert not grammar.active_states


@pytest.mark.parametrize("width, length, supported", [
    (1, 1, False), (2, 1, True), (3, 1, True), (4, 1, False),
    (7, 1, False), (8, 1, True), (9, 1, False),
    (8, 8184, True), (8, 8185, False), (3, 8189, True), (2, 8190, True),
])
def test_graph_support_checks_actual_width_and_context(width, length, supported):
    from onyx_cuda.cache import CacheState
    from onyx_cuda.replay_backend import GraphReplayBackend
    from transformers.cache_utils import DynamicCache
    kv = DynamicCache()
    keys = torch.zeros(1, 1, length, 1)
    kv.update(keys, keys.clone(), 0)
    cache = CacheState.from_prefill(kv, torch.device("cpu"))
    backend = GraphReplayBackend.__new__(GraphReplayBackend)
    backend._model = lambda: SimpleNamespace(training=False)
    backend.closed = False
    backend.device = torch.device("cpu")
    ids = torch.ones((1, width), dtype=torch.long)
    assert backend.supports(cache, ids) is supported
    assert not backend.supports(cache, ids.float())
    assert not backend.supports(cache, ids.flatten())
    assert not backend.supports(cache, ids.expand(2, -1))
    backend.device = torch.device("cuda")
    assert not backend.supports(cache, ids)
    backend.device = torch.device("cpu")
    cache.attention_mask[0, 0] = 0
    assert not backend.supports(cache, ids)
    cache.attention_mask[0, 0] = 1
    backend.closed = True
    assert not backend.supports(cache, ids)
    assert cache.length == length and torch.equal(kv.layers[0].keys, keys)


def test_setup_reuse_and_release(monkeypatch):
    import onyx_cuda.replay_backend as replay
    instances = []
    class Backend:
        def __init__(self, model):
            self.closed = False; self.setup_seconds = 3
            instances.append(self)
        def close(self):
            self.closed = True
    monkeypatch.setattr(replay, "_unsupported_reason", lambda model: None)
    monkeypatch.setattr(replay, "GraphReplayBackend", Backend)
    model = SimpleNamespace()
    assert prepare_replay_backend(model)["active"] == "graph"
    assert prepare_replay_backend(model)["setup_seconds"] == 3
    assert len(instances) == 1
    assert prepare_replay_backend(model, "scalar")["active"] == "scalar"
    assert instances[0].closed
    assert not hasattr(model, "_onyx_replay_backend")
    assert prepare_replay_backend(model)["active"] == "graph"
    assert len(instances) == 2 and not instances[-1].closed
    close_replay_backend(model)


def test_concurrent_setup_and_close_are_idempotent(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import onyx_cuda.replay_backend as replay
    calls = []
    class Backend:
        def __init__(self, model):
            calls.append("build")
            time.sleep(0.01)
            self.closed = False
            self.setup_seconds = 1
        def close(self):
            calls.append("close")
            time.sleep(0.01)
            self.closed = True
    monkeypatch.setattr(replay, "_unsupported_reason", lambda model: None)
    monkeypatch.setattr(replay, "GraphReplayBackend", Backend)
    model = SimpleNamespace()
    barrier = Barrier(4)
    def prepare(_):
        barrier.wait()
        return prepare_replay_backend(model)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r["active"] == "graph" for r in pool.map(prepare, range(4)))
        list(pool.map(lambda _: close_replay_backend(model), range(4)))
    assert calls == ["build", "close"]


@pytest.mark.parametrize("width", [2, 3, 8])
def test_memory_failure_discards_partial_cache_and_releases_graphs(monkeypatch, width):
    from contextlib import nullcontext
    from threading import RLock
    from onyx_cuda.replay_backend import GraphReplayBackend
    class Event:
        def record(self, stream): pass
        def synchronize(self): pass
    stream = SimpleNamespace(wait_event=lambda event: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    released = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(True))
    class Cache(HistoryCache):
        def extend(self, model, ids):
            self.past_key_values.tokens.extend(ids[0].tolist())
            raise torch.OutOfMemoryError("injected allocation failure")
    backend = GraphReplayBackend.__new__(GraphReplayBackend)
    backend._lock = RLock(); backend.device = "cpu"; backend._event = None
    backend.closed = False; backend.fallback_reason = None
    backend._graphs = {"allocated": object()}
    backend.supports = lambda cache, ids: not backend.closed
    cache = Cache()
    assert backend.extend(cache, torch.full((1, width), 5)) is None
    assert cache.past_key_values.tokens == [8]
    assert backend.closed and not backend._graphs and released == [True]
    assert "memory exhausted" in backend.fallback_reason


@pytest.mark.gpu
def test_graph_recovery_exact_concurrent_streams_and_cleanup(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from transformers.models.qwen3 import modeling_qwen3 as qwen
    from onyx_cuda.model import load_model_pair
    from onyx_cuda.cache import CacheState, snapshot_cache
    from onyx_cuda.prefill import prefill
    from onyx_cuda.prompt import format_prompt
    from onyx_cuda.speculative import generate_speculative, generate_speculative_events
    from onyx_cuda.benchmark_corpus import CORPUS

    pair = load_model_pair()
    original_forward = pair.target.model.forward
    attention = qwen.ALL_ATTENTION_FUNCTIONS["sdpa"]
    configuration = prepare_replay_backend(pair.target.model)
    if configuration["active"] != "graph":
        pytest.skip(configuration["reason"])
    backend = pair.target.model._onyx_replay_backend
    assert prepare_replay_backend(pair.target.model)["setup_seconds"] == configuration["setup_seconds"]
    try:
        ids = format_prompt(pair.target.tokenizer, [{"role": "user", "content": "Explain caching."}],
                            enable_thinking=False).token_ids
        with torch.inference_mode():
            initial = prefill(pair.target.model, ids)
            seed = CacheState.from_prefill(initial.past_key_values, initial.logits.device)
            scalar = snapshot_cache(seed)
            token = initial.token_id.item()
            inputs, rows, references = [], [], {}
            for i in range(8):
                inputs.append(token)
                row = scalar.extend(pair.target.model, torch.tensor([[token]], device="cuda"))
                rows.append(row)
                token = row[:, -1].argmax(-1).item()
                if i + 1 in (2, 3, 8):
                    references[i + 1] = (snapshot_cache(scalar), torch.cat(rows, 1))

        def run(width):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream), torch.inference_mode():
                cache = snapshot_cache(seed)
                actual = backend.extend(cache, torch.tensor([inputs[:width]], device="cuda"))
                stream.synchronize()
                reference, expected = references[width]
                return (torch.equal(actual, expected) and
                        torch.equal(cache.attention_mask, reference.attention_mask) and
                        torch.equal(cache.cache_position, reference.cache_position) and
                        all(torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
                            for a, b in zip(cache.past_key_values.layers, reference.past_key_values.layers)))

        torch.cuda.synchronize()
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert all(executor.map(run, (8, 2, 3, 8)))
        # A later graph replay must not overwrite a previous caller's logits.
        with torch.inference_mode():
            actual = backend.extend(snapshot_cache(seed), torch.tensor([inputs], device="cuda"))
            saved = actual.clone()
            backend.extend(snapshot_cache(seed), torch.tensor([[inputs[0]] * 8], device="cuda"))
            assert torch.equal(actual, saved)
        del initial, seed, scalar, references, row, rows, actual, saved
        for name in ("cache_long", "apology"):
            case = next(c for c in CORPUS if c["name"] == name)
            ids = format_prompt(pair.target.tokenizer, case["messages"], enable_thinking=False).token_ids
            args = dict(draft_model=pair.draft.model, target_model=pair.target.model,
                        prompt_token_ids=ids, max_tokens=512,
                        eos_token_ids=pair.target.tokenizer.eos_token_id)
            oracle = generate_speculative(**args, gamma=0)
            result = generate_speculative(**args, gamma=2, measure=True)
            assert (result.token_ids, result.finish_reason) == (oracle.token_ids, oracle.finish_reason)
            assert result.timings.replay_stats["chunked_history_tokens"] > 0
            assert result.timings.replay_stats["graph_replay_fallbacks"] == 0
            events = generate_speculative_events(**args, gamma=2)
            for _ in range(min(190, len(oracle.token_ids))):
                next(events)
            events.close()
            del events, result, oracle
        assert pair.target.model.forward == original_forward
        assert qwen.ALL_ATTENTION_FUNCTIONS["sdpa"] is attention
        from onyx_cuda.replay_backend import GraphReplayBackend
        original_call = GraphReplayBackend.__call__
        def exhaust_after_forward(self, **kwargs):
            original_call(self, **kwargs)
            raise torch.OutOfMemoryError("injected post-forward memory pressure")
        initial = prefill(pair.target.model, ids)
        cache = CacheState.from_prefill(initial.past_key_values, initial.logits.device)
        length = cache.length
        with monkeypatch.context() as patch:
            patch.setattr(GraphReplayBackend, "__call__", exhaust_after_forward)
            assert backend.extend(cache, initial.token_id.reshape(1, 1).expand(1, 8)) is None
        assert cache.length == length
        assert all(layer.keys.shape[-2] == length for layer in cache.past_key_values.layers)
        assert backend.closed and not backend._graphs
        del cache, initial
    finally:
        close_replay_backend(pair.target.model)
    assert backend.closed and not backend._graphs
    close_replay_backend(pair.target.model)
    from onyx_cuda.replay_backend import GraphReplayBackend
    original_build = GraphReplayBackend._build
    attempts = []
    def fail_during_setup(self, module, width):
        if width == 8:
            raise RuntimeError("injected graph setup failure")
        attempts.append(True)
        original_build(self, module, width)
    with monkeypatch.context() as patch:
        patch.setattr(GraphReplayBackend, "_build", fail_during_setup)
        with pytest.raises(RuntimeError, match="injected graph setup failure"):
            prepare_replay_backend(pair.target.model)
    assert not hasattr(pair.target.model, "_onyx_replay_backend")
