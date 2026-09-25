import sys
import gc
import weakref
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from onyx_cuda.server import MODEL_ID, create_app, get_engine
import onyx_cuda.server as server


def test_replay_backend_startup_status_and_shutdown(monkeypatch):
    import onyx_cuda.replay_backend as replay
    model = SimpleNamespace()
    engine = SimpleNamespace(target=SimpleNamespace(model=model), draft=None)
    calls = []
    def prepare(target, mode):
        assert target is model
        calls.append(("prepare", mode))
        model._onyx_replay_backend = SimpleNamespace(closed=False)
        return {"requested": mode, "active": "graph", "setup_seconds": 1.0}
    def close(target):
        assert target is model
        calls.append(("close",))
    monkeypatch.setattr(replay, "prepare_replay_backend", prepare)
    monkeypatch.setattr(replay, "close_replay_backend", close)
    monkeypatch.setenv("ONYX_REPLAY_BACKEND", "graph")
    app = create_app(engine=engine)
    monkeypatch.setenv("ONYX_REPLAY_BACKEND", "scalar")
    with TestClient(app) as client:
        assert client.get("/").json()["replay_backend"]["active"] == "graph"
        model._onyx_replay_backend.closed = True
        model._onyx_replay_backend.fallback_reason = "memory pressure"
        assert client.get("/").json()["replay_backend"]["reason"] == "memory pressure"
    assert calls == [("prepare", "graph"), ("close",)]


@pytest.mark.parametrize("gamma, setting, active", [
    (2, None, "graph"), (2, "eager", "eager"), (0, None, "eager")])
def test_draft_backend_startup_status_and_shutdown(monkeypatch, gamma, setting, active):
    import onyx_cuda.draft_backend as draft
    model = SimpleNamespace()
    engine = SimpleNamespace(target=SimpleNamespace(model=None), draft=SimpleNamespace(model=model))
    calls = []
    def prepare(target, mode, *, capacity):
        assert target is model
        calls.append(("prepare", mode, capacity))
        if mode == "graph":
            model._onyx_draft_backend = SimpleNamespace(closed=False)
        return {"requested": mode, "active": mode, "setup_seconds": 2.0}
    def close(target):
        calls.append(("close", target is model))
    monkeypatch.setattr(draft, "prepare_draft_backend", prepare)
    monkeypatch.setattr(draft, "close_draft_backend", close)
    if setting is not None:
        monkeypatch.setenv("ONYX_DRAFT_BACKEND", setting)
    with TestClient(create_app(engine=engine, gamma=gamma)) as client:
        status = client.get("/").json()["draft_backend"]
        assert status["active"] == active
        if gamma == 0:
            assert status == {"requested": "graph", "active": "eager", "setup_seconds": 2.0,
                              "reason": "speculation is disabled"}
        if active == "graph":
            model._onyx_draft_backend.closed = True
            assert client.get("/").json()["draft_backend"]["active"] == "eager"
    assert calls == [("prepare", active, 8192), ("close", True)]


def test_replay_setup_failure_releases_loaded_engine(monkeypatch):
    import onyx_cuda.replay_backend as replay
    references = []
    class Engine:
        target = SimpleNamespace(model=None)
    def load():
        engine = Engine()
        references.append(weakref.ref(engine))
        return engine
    def fail(*args):
        raise RuntimeError("setup failed")
    monkeypatch.setattr(replay, "prepare_replay_backend", fail)
    app = create_app(load_engine=load, replay_backend="graph")
    with pytest.raises(RuntimeError, match="setup failed"):
        with TestClient(app):
            pass
    gc.collect()
    assert references[0]() is None
    assert not app.state.engines


def test_target_only_does_not_prepare_graphs():
    with TestClient(create_app(engine=object(), gamma=0, replay_backend="graph")) as client:
        configuration = client.get("/").json()["replay_backend"]
        assert configuration["requested"] == "graph"
        assert configuration["active"] == "scalar"
        assert configuration["reason"] == "speculation is disabled"


def test_shutdown_releases_injected_engine_even_if_app_is_retained():
    class Engine:
        pass

    engine = Engine()
    reference = weakref.ref(engine)
    app = create_app(engine=engine)
    del engine
    with TestClient(app):
        assert reference() is app.state.engines[MODEL_ID]
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match="use load_engine to restart"):
        with TestClient(app):
            pass


def test_loader_can_restart_app_with_a_new_engine_each_time():
    references = []

    class Engine:
        pass

    def load():
        engine = Engine()
        references.append(weakref.ref(engine))
        return engine

    app = create_app(load_engine=load)
    for _ in range(2):
        with TestClient(app):
            assert references[-1]() is app.state.engines[MODEL_ID]
        gc.collect()
        assert references[-1]() is None
    assert len(references) == 2


def test_create_app_does_not_load_until_lifespan():
    starts = []

    def load_engine():
        starts.append(object())
        return starts[-1]

    app = create_app(load_engine=load_engine)
    assert starts == []

    with TestClient(app) as client:
        assert len(starts) == 1
        assert client.app.state.engines[MODEL_ID] is starts[0]
        assert list(client.app.state.engine_locks) == [MODEL_ID]
        root = client.get("/")
        assert root.status_code == 200
        body = root.json()
        assert body["status"] == "ok"
        assert body["service"] == "Onyx CUDA API"
        assert body["endpoints"] == [
            "/",
            "/v1/models",
            "/v1/chat/completions",
        ]
        models = client.get("/v1/models")
        assert models.status_code == 200
        ids = [item["id"] for item in models.json()["data"]]
        assert ids == list(client.app.state.engines)
        assert ids == [MODEL_ID]

    assert len(starts) == 1
    assert client.app.state.engine_locks == {}
    assert client.app.state.engines == {}


def test_unknown_model_fails_and_known_id_matches_registry():
    engine = object()
    app = create_app(engine=engine)
    with TestClient(app) as client:
        assert get_engine(client.app, MODEL_ID) is engine
        with pytest.raises(HTTPException) as exc:
            get_engine(client.app, "missing")
        assert exc.value.status_code == 400
        assert "missing" in exc.value.detail
        assert MODEL_ID in exc.value.detail

    with pytest.raises(HTTPException) as exc:
        get_engine(app, MODEL_ID)
    assert exc.value.status_code == 400


def test_shutdown_releases_cuda_cache(monkeypatch):
    calls = []

    class FakeCuda:
        def is_available(self):
            return True

        def empty_cache(self):
            calls.append(1)

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    app = create_app(engine=object())
    with TestClient(app) as client:
        assert calls == []
        assert MODEL_ID in client.app.state.engines
    assert calls == [1]
    assert app.state.engines == {}


def test_server_defaults_to_speculation_and_allows_target_only(monkeypatch):
    calls = []
    monkeypatch.delenv("ONYX_SPECULATIVE_GAMMA", raising=False)
    monkeypatch.setattr(
        server, "_load_configured_engine", lambda gamma, selection: calls.append(gamma) or object()
    )
    with TestClient(create_app()) as client:
        assert client.app.state.speculative_gamma == server.GAMMA
        assert client.get("/").json()["speculative_gamma"] == server.GAMMA
    monkeypatch.setenv("ONYX_SPECULATIVE_GAMMA", "2")
    with TestClient(create_app()) as client:
        assert client.app.state.speculative_gamma == 2
        assert client.get("/").json()["speculative_gamma"] == 2
    with TestClient(create_app(gamma=0)) as client:
        assert client.app.state.speculative_gamma == 0
    assert calls == [server.GAMMA, 2, 0]
    monkeypatch.setenv("ONYX_SPECULATIVE_GAMMA", "auto")
    with pytest.raises(ValueError, match="ONYX_SPECULATIVE_GAMMA"):
        create_app()
    with pytest.raises(ValueError, match="nonnegative"):
        create_app(gamma=-1)


def test_adaptive_mode_is_explicit_and_frozen_at_creation(monkeypatch):
    monkeypatch.setenv("ONYX_SPECULATIVE_MODE", "adaptive")
    app = create_app(load_engine=lambda: object())
    monkeypatch.setenv("ONYX_SPECULATIVE_MODE", "fixed")
    with TestClient(app) as client:
        assert client.get("/").json()["speculative_mode"] == "adaptive"
    assert create_app(speculative_mode="fixed").state.speculative_mode == "fixed"
    with pytest.raises(ValueError, match="SPECULATIVE_MODE"):
        create_app(speculative_mode="typo")


def test_backend_is_validated_and_frozen_before_startup(monkeypatch):
    monkeypatch.setenv("ONYX_GREEDY_BACKEND", "typo")
    with pytest.raises(ValueError, match="ONYX_GREEDY_BACKEND"):
        create_app(engine=object())
    calls = []
    monkeypatch.setattr(server, "initialize_greedy_backend", calls.append)
    app = create_app(engine=object(), greedy_backend="cuda")
    monkeypatch.setenv("ONYX_GREEDY_BACKEND", "torch")
    assert calls == []
    with TestClient(app) as client:
        assert calls == ["cuda"]
        assert client.get("/").json()["greedy_backend"] == "cuda"


def test_backend_startup_failure_prevents_model_loading(monkeypatch):
    def unavailable(_):
        raise RuntimeError("kernel unavailable")

    monkeypatch.setattr(server, "initialize_greedy_backend", unavailable)
    app = create_app(
        load_engine=lambda: pytest.fail("Do not load models after kernel initialization fails"),
        greedy_backend="cuda",
    )
    with pytest.raises(RuntimeError, match="kernel unavailable"):
        with TestClient(app):
            pass
