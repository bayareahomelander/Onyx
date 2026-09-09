"""Startup selection, snapshot identity, and model compatibility boundaries."""

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import onyx_cuda.model as model
import onyx_cuda.server as server
from onyx_cuda.config import DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL, resolve_model_selection
from onyx_cuda.revisions import MODEL_REVISIONS


def test_selection_preserves_defaults_and_explicit_fields_override_environment(monkeypatch):
    defaults = resolve_model_selection()
    assert defaults.target_model == DEFAULT_TARGET_MODEL
    assert defaults.draft_model == DEFAULT_DRAFT_MODEL
    assert defaults.target_revision == MODEL_REVISIONS[DEFAULT_TARGET_MODEL]
    assert defaults.draft_revision == MODEL_REVISIONS[DEFAULT_DRAFT_MODEL]
    monkeypatch.setenv("ONYX_TARGET_MODEL", "example/environment")
    monkeypatch.setenv("ONYX_TARGET_REVISION", "environment-branch")
    monkeypatch.setenv("ONYX_DRAFT_MODEL", "example/draft")
    chosen = resolve_model_selection(target_model="example/explicit", target_revision="release")
    assert (chosen.target_model, chosen.target_revision) == ("example/explicit", "release")
    assert (chosen.draft_model, chosen.draft_revision) == ("example/draft", None)
    assert resolve_model_selection().target_revision == "environment-branch"


@pytest.mark.parametrize("name,value", [
    ("ONYX_TARGET_MODEL", ""), ("ONYX_TARGET_MODEL", "../weights"),
    ("ONYX_DRAFT_MODEL", "https://example.com/model"), ("ONYX_TARGET_REVISION", " "),
    ("ONYX_DRAFT_REVISION", "main "), ("ONYX_TARGET_MODEL", "owner/repo/extra"),
])
def test_invalid_selection_fails_at_app_creation(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        server.create_app()


def test_local_directory_is_not_interpreted_as_a_remote_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights").mkdir()
    with pytest.raises(ValueError, match="repository IDs"):
        resolve_model_selection(target_model="weights")


@pytest.mark.parametrize("gamma", [0, 2])
def test_startup_selection_is_frozen_and_metadata_is_observed(monkeypatch, gamma):
    calls = []
    monkeypatch.setenv("ONYX_TARGET_MODEL", "example/target")
    monkeypatch.setenv("ONYX_DRAFT_MODEL", "example/draft")

    def load(name, *, revision):
        calls.append((name, revision))
        return model.LoadedModel(SimpleNamespace(dtype=torch.float16, _onyx_model_id=name), object(), "a" * 40)

    monkeypatch.setattr(model, "load_model", load)
    monkeypatch.setattr(model, "_require_compatible_models", lambda *_: None)
    app = server.create_app(gamma=gamma, target_revision="release")
    monkeypatch.setenv("ONYX_TARGET_MODEL", "example/changed-after-creation")
    assert calls == []
    with TestClient(app) as client:
        health = client.get("/").json()
        configuration = client.get("/v1/models").json()["data"][0]["configuration"]
        assert health["models"] == configuration
        assert configuration["target"] == {
            "id": "example/target", "revision": "a" * 40,
            "precision": "torch.float16", "validated_snapshot": False,
        }
        assert (configuration["draft"] is not None) == (gamma > 0)
    assert calls == [("example/target", "release")] + ([("example/draft", None)] if gamma else [])


def test_model_options_are_not_silently_ignored_with_injected_engines():
    with pytest.raises(ValueError, match="injected engine"):
        server.create_app(engine=object(), target_model="example/target")
    with pytest.raises(ValueError, match="injected engine"):
        server.create_app(load_engine=lambda: object(), draft_revision="release")


@pytest.mark.parametrize("requested", [None, "branch", "b" * 40])
def test_custom_revision_resolves_once_and_pins_tokenizer_and_weights(monkeypatch, requested):
    observed = []
    resolved = "b" * 40
    config = SimpleNamespace(_commit_hash=resolved)
    monkeypatch.setattr(model, "require_cuda", lambda: "cuda:0")
    monkeypatch.setattr(model.AutoConfig, "from_pretrained", lambda name, **kw:
                        observed.append(("config", name, kw["revision"])) or config)
    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", lambda name, **kw:
                        observed.append(("tokenizer", name, kw["revision"])) or object())
    class Weights:
        def to(self, device):
            assert device == "cuda:0"
            return self
        def eval(self):
            return self
    def weights(name, **kw):
        assert kw["dtype"] == torch.float16
        observed.append(("weights", name, kw["revision"]))
        return Weights()
    monkeypatch.setattr(model.AutoModelForCausalLM, "from_pretrained", weights)
    monkeypatch.setattr(model, "_validate_tokenizer", lambda *_: None)
    monkeypatch.setattr(model, "_validate_runtime", lambda *_: None)
    loaded = model.load_model("example/custom", revision=requested)
    assert loaded.model_id == "example/custom"
    assert len(loaded) == 3  # Preserve the existing unpacking contract.
    assert loaded.revision == resolved
    assert observed == [("config", "example/custom", requested),
                        ("tokenizer", "example/custom", resolved), ("weights", "example/custom", resolved)]


@pytest.mark.parametrize("bad_config,message", [
    (SimpleNamespace(_commit_hash=None), "resolve a revision"),
    (SimpleNamespace(_commit_hash="c" * 40), "differs from the validation pin"),
    (SimpleNamespace(_commit_hash="b" * 40, quantization_config={}), "Prequantized"),
])
def test_resolution_and_quantization_errors_prevent_weight_loading(monkeypatch, bad_config, message):
    monkeypatch.setattr(model, "require_cuda", lambda: "cuda:0")
    monkeypatch.setattr(model.AutoConfig, "from_pretrained", lambda *a, **kw: bad_config)
    monkeypatch.setattr(model.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: pytest.fail("Loaded weights"))
    with pytest.raises((ValueError, RuntimeError), match=message):
        model.load_model("example/custom", revision="b" * 40)


def test_unsupported_target_tokenizer_fails_before_weights(monkeypatch):
    monkeypatch.setattr(model, "require_cuda", lambda: "cuda:0")
    monkeypatch.setattr(model.AutoConfig, "from_pretrained", lambda *a, **kw:
                        SimpleNamespace(_commit_hash="b" * 40, vocab_size=2))
    class UnsupportedTokenizer:
        all_special_ids = []
        def get_vocab(self):
            return {"a": 0}
    tokenizer = UnsupportedTokenizer()
    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(model.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: pytest.fail("Loaded weights"))
    with pytest.raises(ValueError, match="ByteLevel"):
        model.load_model("example/custom")


@pytest.mark.parametrize("broken", [False, True])
def test_runtime_probe_exercises_cache_replay(broken):
    from transformers.cache_utils import DynamicCache

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(vocab_size=4)
            self.calls = 0
        def forward(self, input_ids, past_key_values=None, **kwargs):
            self.calls += 1
            cache = past_key_values if past_key_values is not None else DynamicCache()
            values = torch.zeros((1, 1, input_ids.shape[1], 2))
            cache.update(values, values, 0)
            logits = torch.zeros((1, input_ids.shape[1], 4))
            if broken and self.calls == 3:
                logits += 1
            return SimpleNamespace(logits=logits, past_key_values=cache)
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, tokenize, **kw: [1, 2, 3] if tokenize else "Hi")
    toy = ToyModel()
    toy._onyx_model_id = "example/toy"
    loaded = model.LoadedModel(toy, tokenizer, "snapshot")
    if broken:
        with pytest.raises(RuntimeError, match="Cache replay changed"):
            model._validate_runtime(loaded)
    else:
        model._validate_runtime(loaded)
    assert toy.calls == 3


def test_failed_model_load_prevents_application_startup(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("Selected model does not fit on CUDA")
    monkeypatch.setattr(model, "load_model", fail)
    with pytest.raises(RuntimeError, match="does not fit"):
        with TestClient(server.create_app(target_model="example/custom")):
            pytest.fail("Application became ready")


@pytest.mark.parametrize("is_target", [False, True])
def test_benchmark_records_selected_identity_and_rejects_other_model_baselines(monkeypatch, tmp_path, is_target):
    import sys
    import onyx_cuda.benchmark as benchmark

    monkeypatch.setenv("ONYX_TARGET_MODEL", "example/target")
    monkeypatch.setenv("ONYX_TARGET_REVISION", "target-branch")
    monkeypatch.setenv("ONYX_DRAFT_MODEL", "example/draft")
    monkeypatch.setenv("ONYX_DRAFT_REVISION", "draft-branch")
    calls = []
    def load(name, *, revision):
        calls.append((name, revision))
        return model.LoadedModel(SimpleNamespace(_onyx_model_id=name), object(), "b" * 40)
    monkeypatch.setattr(benchmark, "load_model", load)
    monkeypatch.setattr(benchmark, "PROMPTS", {})
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "test-device")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(total_memory=100))
    output = tmp_path / "baseline.json"
    arguments = ["benchmark", "--output", str(output)] + (["--target"] if is_target else [])
    monkeypatch.setattr(sys, "argv", arguments)
    benchmark.main()
    name = "example/target" if is_target else "example/draft"
    assert calls == [(name, "target-branch" if is_target else "draft-branch")]
    baseline = json.loads(output.read_text())
    assert baseline["model"] == {"id": name, "revision": "b" * 40}
    for wrong in ({"id": "example/other", "revision": "b" * 40},
                  {"id": name, "revision": "c" * 40}):
        baseline["model"] = wrong
        output.write_text(json.dumps(baseline))
        monkeypatch.setattr(sys, "argv", [*arguments, "--constraints", "--baseline", str(output)])
        with pytest.raises(RuntimeError, match="model or revision does not match"):
            benchmark.main()


def test_comparison_cli_uses_same_model_selection(monkeypatch, tmp_path):
    import sys
    import onyx_cuda.benchmark as benchmark

    monkeypatch.setenv("ONYX_TARGET_MODEL", "example/target")
    monkeypatch.setenv("ONYX_DRAFT_MODEL", "example/draft")
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    calls = []
    monkeypatch.setattr(benchmark, "_run_speculative_gate", lambda *args: calls.append(args[-1]))
    for mode in ("--compare", "--speculative"):
        monkeypatch.setattr(sys, "argv", ["benchmark", mode, "--output", str(tmp_path / "result.json")])
        benchmark.main()
    assert calls == [resolve_model_selection(), resolve_model_selection()]


@pytest.mark.gpu
def test_real_nondefault_pair_selection_and_target_baseline(monkeypatch, record_model_revision):
    from onyx_cuda.speculative import generate_speculative

    # Reuse downloaded snapshots while genuinely changing the selected target.
    monkeypatch.setenv("ONYX_TARGET_MODEL", DEFAULT_DRAFT_MODEL)
    monkeypatch.setenv("ONYX_DRAFT_MODEL", DEFAULT_DRAFT_MODEL)
    with TestClient(server.create_app(gamma=2)) as client:
        pair = client.app.state.engines[server.MODEL_ID]
        record_model_revision(DEFAULT_DRAFT_MODEL, pair.target)
        assert pair.target.model_id == pair.draft.model_id == DEFAULT_DRAFT_MODEL
        payload = {"messages": [{"role": "user", "content": "Give four digits"}],
                   "regex": "[0-9]{4}", "max_tokens": 8}
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        arguments = server.prepare_generation(server.ChatCompletionRequest(**payload), pair, gamma=0)
        expected = generate_speculative(**arguments)
        assert response.json()["choices"][0]["message"]["content"] == pair.target.tokenizer.decode(expected.token_ids, skip_special_tokens=True)
        assert response.json()["choices"][0]["finish_reason"] == expected.finish_reason == "stop"
        assert client.get("/").json()["models"]["target"]["id"] == DEFAULT_DRAFT_MODEL
