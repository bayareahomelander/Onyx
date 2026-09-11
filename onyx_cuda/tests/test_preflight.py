"""Metadata-only eligibility must never become a claim of runtime validation."""

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config

from onyx_cuda.config import ModelSelection
from onyx_cuda.model import ModelMetadata
import onyx_cuda.preflight as preflight


SELECTION = ModelSelection("example/target", "main", "example/draft", "draft-branch")


def tiny_config():
    return Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)


def stub_metadata(monkeypatch):
    calls = []
    def inspect(name, *, revision):
        calls.append((name, revision))
        return ModelMetadata(tiny_config(), object(), ("a" if name.endswith("target") else "b") * 40)
    monkeypatch.setattr(preflight, "inspect_model", inspect)
    return calls


def test_meta_estimation_uses_no_weights_or_cuda(monkeypatch):
    monkeypatch.setattr(preflight.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: pytest.fail("Downloaded weights"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("Initialized CUDA"))
    estimate = preflight.estimate_memory(tiny_config(), 2048)
    assert estimate["fp16_weight_bytes"] == estimate["parameter_count"] * 2
    assert estimate["full_attention_kv_bytes"] == 2 * 2 * 2 * 4 * 2048 * 2
    assert estimate["parameter_count"] > 0


def test_target_only_precheck_pins_revision_and_reports_insufficient_capacity(monkeypatch):
    calls = stub_metadata(monkeypatch)
    monkeypatch.setattr(preflight, "require_compatible_metadata", lambda *a: pytest.fail("Checked draft"))
    report = {"checks": []}
    pinned = preflight.precheck(SELECTION, vram_gib=0.000001, report=report)
    assert calls == [("example/target", "main")]
    assert pinned.target_revision == "a" * 40
    assert pinned.draft_revision == "draft-branch"
    assert report["support_level"] == "prechecked"
    assert report["memory"]["capacity_assessment"] == "weights_exceed_capacity"
    assert set(report["models"]) == {"target"}


def test_pair_precheck_checks_compatibility_and_freezes_both_revisions(monkeypatch):
    calls = stub_metadata(monkeypatch)
    pairs = []
    monkeypatch.setattr(preflight, "require_compatible_metadata", lambda draft, target: pairs.append((draft, target)))
    report = {"checks": []}
    pinned = preflight.precheck(SELECTION, gamma=2, vram_gib=32, report=report)
    assert calls == [("example/target", "main"), ("example/draft", "draft-branch")]
    assert pinned.target_revision == "a" * 40 and pinned.draft_revision == "b" * 40
    assert len(pairs) == 1
    assert report["memory"]["capacity_assessment"] == "fit_not_established"


def test_pair_incompatibility_prevents_pass_and_preserves_resolved_metadata(monkeypatch, tmp_path):
    stub_metadata(monkeypatch)
    def incompatible(*args):
        raise RuntimeError("private-path-in-error")
    monkeypatch.setattr(preflight, "require_compatible_metadata", incompatible)
    output = tmp_path / "failed.json"
    assert preflight.main(["--target-model", "example/target", "--gamma", "2", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["support_level"] == "unverified"
    assert report["checks"][-1] == {"name": "pair.tokenizer_compatibility", "status": "failed", "error_type": "RuntimeError"}
    assert report["models"]["target"]["revision"] == "a" * 40
    assert "private-path-in-error" not in output.read_text()


def test_existing_report_never_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    output.write_text("previous evidence")
    monkeypatch.setattr(preflight, "inspect_model", lambda *a, **k: pytest.fail("Started download"))
    with pytest.raises(FileExistsError):
        preflight.main(["--output", str(output)])
    assert output.read_text() == "previous evidence"


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_invalid_capacity_rejected_before_output(value, tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        preflight.main(["--vram-gib", value, "--output", str(output)])
    assert not output.exists()


def test_unknown_cache_layout_has_no_kv_estimate(monkeypatch):
    class Skeleton:
        def parameters(self):
            return iter([torch.empty(10, device="meta")])
        def buffers(self):
            return iter([])
    monkeypatch.setattr(preflight.AutoModelForCausalLM, "from_config", lambda *a, **k: Skeleton())
    assert preflight.estimate_memory(SimpleNamespace(model_type="unknown"), 2048)["full_attention_kv_bytes"] is None
