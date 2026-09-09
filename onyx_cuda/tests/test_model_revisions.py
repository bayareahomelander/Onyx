from types import SimpleNamespace

import pytest

import onyx_cuda.model as loader
from onyx_cuda.revisions import MODEL_REVISIONS


@pytest.mark.parametrize("model_id", MODEL_REVISIONS)
def test_loading_uses_one_pinned_revision_for_config_tokenizer_and_weights(monkeypatch, model_id):
    calls = []
    revision = MODEL_REVISIONS[model_id]
    config = SimpleNamespace(_commit_hash=revision)

    class FakeModel:
        def to(self, device):
            assert device == "cuda:0"
            return self

        def eval(self):
            calls.append("eval")

    def get_config(name, **options):
        assert name == model_id
        assert options == {"revision": revision}
        calls.append("config")
        return config

    def get_tokenizer(name, **options):
        assert name == model_id
        assert options == {"revision": revision}
        calls.append("tokenizer")
        return object()

    def get_model(name, **options):
        assert name == model_id
        assert options["revision"] == revision
        assert options["config"] is config
        calls.append("weights")
        return FakeModel()

    monkeypatch.setattr(loader, "require_cuda", lambda: "cuda:0")
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", get_config)
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", get_tokenizer)
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", get_model)
    assert loader.load_model(model_id).revision == revision
    assert calls == ["config", "tokenizer", "weights", "eval"]


def test_mismatched_pin_fails_before_loading_weights(monkeypatch):
    monkeypatch.setattr(loader, "require_cuda", lambda: "cuda:0")
    monkeypatch.setattr(
        loader.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(_commit_hash="wrong")
    )
    with pytest.raises(RuntimeError, match="differs from the validation pin"):
        loader.load_model()
