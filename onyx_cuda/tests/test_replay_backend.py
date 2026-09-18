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


def test_memory_failure_discards_partial_cache_and_releases_graphs(monkeypatch):
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
    assert backend.extend(cache, torch.tensor([[5, 5]])) is None
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
            inputs, rows = [], []
            for _ in range(3):
                inputs.append(token)
                row = scalar.extend(pair.target.model, torch.tensor([[token]], device="cuda"))
                rows.append(row)
                token = row[:, -1].argmax(-1).item()
            expected = torch.cat(rows, 1)

        def run():
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream), torch.inference_mode():
                cache = snapshot_cache(seed)
                actual = backend.extend(cache, torch.tensor([inputs], device="cuda"))
                stream.synchronize()
                return (torch.equal(actual, expected) and
                        all(torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
                            for a, b in zip(cache.past_key_values.layers, scalar.past_key_values.layers)))

        torch.cuda.synchronize()
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert all(executor.map(lambda _: run(), range(4)))
        del initial, seed, scalar, expected, row, rows
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
            assert backend.extend(cache, initial.token_id.reshape(1, 1).expand(1, 2)) is None
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
        if attempts:
            raise RuntimeError("injected graph setup failure")
        attempts.append(True)
        original_build(self, module, width)
    with monkeypatch.context() as patch:
        patch.setattr(GraphReplayBackend, "_build", fail_during_setup)
        with pytest.raises(RuntimeError, match="injected graph setup failure"):
            prepare_replay_backend(pair.target.model)
    assert not hasattr(pair.target.model, "_onyx_replay_backend")
