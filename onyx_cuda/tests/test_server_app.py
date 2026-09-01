import sys

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from onyx_cuda.server import MODEL_ID, create_app, get_engine


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
