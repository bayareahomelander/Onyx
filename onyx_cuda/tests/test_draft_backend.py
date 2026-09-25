"""Draft graph decoding: selection, static-cache ownership, fallback, and exact output."""
import gc
import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

import onyx_cuda.draft_backend as draft_backend
import onyx_cuda.speculative as speculative_module
from onyx_cuda.draft_backend import (GraphDraftBackend, close_draft_backend, draft_backend_status,
                                     prepare_draft_backend, resolve_draft_backend)


def test_backend_selection_and_unsupported_model(monkeypatch):
    assert resolve_draft_backend() == "graph"
    monkeypatch.setenv("ONYX_DRAFT_BACKEND", "eager")
    assert resolve_draft_backend() == "eager"
    assert resolve_draft_backend("graph") == "graph"
    with pytest.raises(ValueError, match="eager or graph"):
        resolve_draft_backend("automatic")
    model = SimpleNamespace()
    configuration = prepare_draft_backend(model)
    assert configuration["active"] == "eager" and "Qwen2.5-0.5B" in configuration["reason"]
    assert not hasattr(model, "_onyx_draft_backend")
    assert prepare_draft_backend(None, "eager")["active"] == "eager"
    close_draft_backend(model)


def test_setup_reuse_capacity_and_status(monkeypatch):
    instances = []
    class Backend:
        def __init__(self, model, capacity):
            self.capacity, self.closed, self.setup_seconds = capacity, False, 2
            instances.append(self)
        def close(self):
            self.closed = True
    monkeypatch.setattr(draft_backend, "_unsupported_reason", lambda model: None)
    monkeypatch.setattr(draft_backend, "GraphDraftBackend", Backend)
    model = SimpleNamespace()
    configuration = prepare_draft_backend(model, capacity=4096)
    assert configuration == {"requested": "graph", "active": "graph", "capacity": 4096, "setup_seconds": 2}
    assert prepare_draft_backend(model, capacity=2048)["capacity"] == 4096 and len(instances) == 1
    assert prepare_draft_backend(model, capacity=8192)["capacity"] == 8192
    assert instances[0].closed and len(instances) == 2
    assert draft_backend_status(model, configuration)["active"] == "graph"
    instances[1].closed = True
    assert draft_backend_status(model, configuration)["active"] == "eager"
    assert prepare_draft_backend(model, "eager")["active"] == "eager"
    assert not hasattr(model, "_onyx_draft_backend")


def _cpu_backend(monkeypatch, capacity=512):
    """A backend whose graphs record their inputs, for CPU checks of cache bookkeeping."""
    class Event:
        def record(self, stream): pass
        def synchronize(self): pass
    stream = SimpleNamespace(wait_event=lambda event: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    backend = GraphDraftBackend.__new__(GraphDraftBackend)
    backend.device = torch.device("cpu")
    backend.capacity, backend.kv_heads, backend.head_dim, backend.layer_count = capacity, 2, 4, 1
    backend._lock, backend._owner, backend._event, backend.closed = threading.RLock(), None, None, False
    chunks = capacity // 256
    backend._layers = [{"keys": torch.zeros((chunks, 2, 256, 4), dtype=torch.float16),
                        "values": torch.zeros((chunks, 2, 256, 4), dtype=torch.float16)}]
    backend._ids = torch.zeros((1,), dtype=torch.long)
    backend._position = torch.zeros((1,), dtype=torch.long)
    replays = []
    class Graph:
        def __init__(self, chunks):
            self.chunks = chunks
        def replay(self):
            replays.append((self.chunks, backend._ids.item(), backend._position.item()))
            backend._graphs[self.chunks][1].fill_(backend._position.item())
    backend._graphs = {c: (Graph(c), torch.zeros((1, 8))) for c in range(1, chunks + 1)}
    return backend, replays


def test_static_cache_copy_positions_ownership_and_capacity(monkeypatch):
    backend, replays = _cpu_backend(monkeypatch)
    keys = torch.arange(300 * 8, dtype=torch.float16).view(1, 2, 300, 4)
    kv = DynamicCache()
    kv.update(keys, -keys, 0)
    cache = backend.start(kv, 400)
    assert cache is not None and cache.length == 300
    assert torch.equal(backend._layers[0]["keys"][1, :, :44], keys[0, :, 256:300])
    assert torch.equal(backend._layers[0]["values"][0], -keys[0, :, :256])
    assert backend.start(kv, 400) is None
    logits = cache.extend(None, torch.tensor([[7, 8]]))
    assert logits.shape == (1, 2, 8) and cache.length == 302
    assert replays == [(2, 7, 300), (2, 8, 301)]
    assert logits[0, :, 0].tolist() == [300, 301]
    cache.crop(256)
    cache.extend(None, torch.tensor([[9]]))
    cache.crop(10)
    cache.extend(None, torch.tensor([[5]]))
    assert replays[-2:] == [(2, 9, 256), (1, 5, 10)]
    with pytest.raises(ValueError, match="between 0 and"):
        cache.crop(12)
    cache.extend(None, torch.zeros((1, 500), dtype=torch.long))
    assert cache.length == 511 and replays[-1] == (2, 0, 510)
    with pytest.raises(ValueError, match="capacity"):
        cache.extend(None, torch.tensor([[1, 2]]))
    cache.release()
    with pytest.raises(RuntimeError, match="no longer active"):
        cache.extend(None, torch.tensor([[1]]))
    assert backend.start(kv, backend.capacity + 1) is None
    abandoned = backend.start(kv, 512)
    assert abandoned is not None
    del abandoned
    gc.collect()
    assert backend.start(kv, 512) is not None


@pytest.mark.parametrize("available", [True, False])
def test_speculation_uses_graph_cache_and_always_releases_it(monkeypatch, available):
    from test_speculative import ScriptedCache
    released = []
    graph_cache = ScriptedCache([2, 9, 9, 9])
    graph_cache.release = lambda: released.append(True)
    fallback_cache = ScriptedCache([2, 9, 9, 9])
    requests = []
    class Backend:
        def start(self, past_key_values, required_length):
            requests.append((past_key_values, required_length))
            return graph_cache if available else None
    draft_model = SimpleNamespace(_onyx_draft_backend=Backend())
    target_model = object()

    def scripted_prefill(model, prompt_token_ids):
        cache = fallback_cache if model is draft_model else ScriptedCache([2, 3, 3, 3])
        logits = torch.full((1, 16), -1.0)
        logits[0, 1] = 0
        return SimpleNamespace(logits=logits, past_key_values=cache, token_id=torch.tensor([1]))

    monkeypatch.setattr(speculative_module, "prefill", scripted_prefill)
    monkeypatch.setattr(speculative_module.CacheState, "from_prefill",
                        classmethod(lambda cls, cache, device: cache))
    result = speculative_module.generate_speculative(draft_model, target_model, [0] * 4, 3, 2, [15])
    assert result.token_ids == [1, 2, 3]
    assert requests == [(fallback_cache, 7)]
    assert released == ([True] if available else [])
    assert (graph_cache if available else fallback_cache).inputs[:1] == [1]

    events = speculative_module.generate_speculative_events(draft_model, target_model, [0] * 4, 3, 2, [15])
    next(events)
    events.close()
    assert released == ([True, True] if available else [])


@pytest.mark.gpu
def test_graph_draft_tracks_eager_draft_with_exact_rollback(record_model_revision):
    from onyx_cuda.cache import CacheState
    from onyx_cuda.model import load_model
    from onyx_cuda.prefill import prefill
    from onyx_cuda.revisions import MODEL_REVISIONS

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    loaded = load_model(model_id, revision=MODEL_REVISIONS[model_id])
    record_model_revision(model_id, loaded)
    model = loaded.model
    configuration = prepare_draft_backend(model, capacity=1024)
    if configuration["active"] != "graph":
        pytest.skip(configuration["reason"])
    assert configuration["capacity"] == 1024
    backend = model._onyx_draft_backend
    try:
        # Longer than one 256-token chunk, so attention spans chunks.
        prompt = loaded.tokenizer("Explain how a KV cache speeds up decoding. " * 30)["input_ids"]
        assert 256 < len(prompt) < 512
        with torch.inference_mode():
            first = prefill(model, prompt)
            eager = CacheState.from_prefill(first.past_key_values, first.logits.device)
            tokens, reference = [first.token_id.item()], []
            for _ in range(48):
                row = eager.extend(model, torch.tensor([[tokens[-1]]], device="cuda"))[0]
                reference.append(row)
                tokens.append(row.argmax(-1).item())
            reference = torch.cat(reference)
            del eager, first
            cache = backend.start(prefill(model, prompt).past_key_values, len(prompt) + 64)
            assert cache is not None and cache.length == len(prompt)
            assert backend.start(prefill(model, prompt).past_key_values, len(prompt) + 64) is None
            rows = cache.extend(model, torch.tensor([tokens[:48]], device="cuda"))[0]
            assert rows.shape == reference.shape and cache.length == len(prompt) + 48
            # Draft arithmetic differs slightly from the ordinary forward; it only proposes.
            assert (rows.argmax(-1) == reference.argmax(-1)).float().mean().item() >= 0.95
            assert (rows.float() - reference.float()).abs().max().item() < 1.0
            cache.crop(len(prompt) + 8)
            again = torch.cat([cache.extend(model, torch.tensor([[t]], device="cuda"))[0] for t in tokens[8:20]])
            assert torch.equal(again, rows[8:20])
            cache.release()
            assert backend.start(prefill(model, prompt).past_key_values, 1025) is None
            replacement = backend.start(prefill(model, prompt).past_key_values, 1024)
            assert replacement is not None
            del reference, rows, again
    finally:
        close_draft_backend(model)
    assert backend.closed and not hasattr(model, "_onyx_draft_backend")
    with pytest.raises(RuntimeError, match="no longer active"):
        replacement.extend(model, torch.tensor([[tokens[0]]], device="cuda"))


@pytest.mark.gpu
def test_speculation_with_graph_draft_matches_target_only():
    from onyx_cuda.benchmark_corpus import CORPUS
    from onyx_cuda.model import load_model_pair
    from onyx_cuda.prompt import format_prompt
    from onyx_cuda.speculative import generate_speculative
    from onyx_cuda.vocabulary import get_token_byte_vocabulary

    pair = load_model_pair()
    configuration = prepare_draft_backend(pair.draft.model, capacity=4096)
    if configuration["active"] != "graph":
        pytest.skip(configuration["reason"])
    backend = pair.draft.model._onyx_draft_backend
    started = []
    original_start = backend.start
    def start(past_key_values, required_length):
        cache = original_start(past_key_values, required_length)
        started.append(cache is not None)
        return cache
    backend.start = start
    try:
        vocabulary = get_token_byte_vocabulary(pair.target.tokenizer, pair.target.model.config.vocab_size)
        cases = {case["name"]: case for case in CORPUS}
        for name in ("python_function", "extract_cities", "long_context", "apology", "product_code", "object_json"):
            case = cases[name]
            args = dict(draft_model=pair.draft.model, target_model=pair.target.model,
                        prompt_token_ids=format_prompt(pair.target.tokenizer, case["messages"],
                                                       enable_thinking=False).token_ids,
                        max_tokens=min(case["max_tokens"], 160),
                        eos_token_ids=pair.target.tokenizer.eos_token_id, greedy_backend="torch")
            if case["regex"] or case["json_schema"]:
                args.update(regex=case["regex"], token_byte_vocabulary=vocabulary,
                            json_schema=json.dumps(case["json_schema"]) if case["json_schema"] else None)
            oracle = generate_speculative(**args, gamma=0)
            for adaptive in (False, True):
                started.clear()
                result = generate_speculative(**args, gamma=2, adaptive=adaptive)
                assert (result.token_ids, result.finish_reason) == (oracle.token_ids, oracle.finish_reason), name
                assert started and all(started), name
                assert backend._owner is None, name
        # Beyond the static capacity, generation keeps the ordinary draft cache.
        started.clear()
        args["max_tokens"] = 4096
        oracle = generate_speculative(**args, gamma=0)
        result = generate_speculative(**args, gamma=2)
        assert started == [False]
        assert (result.token_ids, result.finish_reason) == (oracle.token_ids, oracle.finish_reason)
    finally:
        del backend.start
        close_draft_backend(pair.draft.model)
