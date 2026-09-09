import sys
import gc
import weakref

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from onyx_cuda.server import MODEL_ID, create_app, get_engine
import onyx_cuda.server as server


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


def test_server_defaults_to_target_only_and_allows_explicit_speculation(monkeypatch):
    calls = []
    monkeypatch.delenv("ONYX_SPECULATIVE_GAMMA", raising=False)
    monkeypatch.setattr(
        server, "_load_configured_engine", lambda gamma: calls.append(gamma) or object()
    )
    with TestClient(create_app()) as client:
        assert client.app.state.speculative_gamma == 0
        assert client.get("/").json()["speculative_gamma"] == 0
    monkeypatch.setenv("ONYX_SPECULATIVE_GAMMA", "2")
    with TestClient(create_app()) as client:
        assert client.app.state.speculative_gamma == 2
        assert client.get("/").json()["speculative_gamma"] == 2
    with TestClient(create_app(gamma=0)) as client:
        assert client.app.state.speculative_gamma == 0
    assert calls == [0, 2, 0]
    monkeypatch.setenv("ONYX_SPECULATIVE_GAMMA", "auto")
    with pytest.raises(ValueError, match="ONYX_SPECULATIVE_GAMMA"):
        create_app()
    with pytest.raises(ValueError, match="nonnegative"):
        create_app(gamma=-1)


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
