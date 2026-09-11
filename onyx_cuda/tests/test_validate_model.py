"""Failures and partial streams must never produce tested-model evidence."""

import json
from types import SimpleNamespace

import pytest

import onyx_cuda.validate_model as validation
from onyx_cuda._validation_report import check


def chunk(content=None, finish=None):
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": finish}]})


@pytest.mark.parametrize("lines", [
    ["data: [DONE]"],
    [chunk("hello")],
    [chunk(finish="stop")],
    ['data: {"error": {"message": "failed"}}', "data: [DONE]"],
    [chunk(finish="stop"), chunk(finish="stop"), "data: [DONE]"],
    [chunk(finish="stop"), "data: [DONE]", chunk("late")],
    [chunk(finish="stop"), chunk("late"), "data: [DONE]"],
])
def test_sse_requires_successful_complete_protocol(lines):
    with pytest.raises(RuntimeError):
        validation.parse_sse(lines)


def test_sse_collects_unicode_and_one_finish():
    assert validation.parse_sse([chunk("hé"), chunk("llo"), chunk(finish="stop"), "data: [DONE]"]) == ("héllo", "stop")


@pytest.mark.parametrize("payload,text", [({"regex": "abc"}, "abc"), ({"json_schema": {"type": "object"}}, "{}")])
def test_budget_limited_constraints_do_not_pass(payload, text):
    with pytest.raises(RuntimeError, match="incomplete"):
        validation.validate_output(payload, text, "length")


def test_validation_failure_records_stage_and_does_not_claim_tested(monkeypatch, tmp_path):
    seen = []
    def precheck(selection, **kwargs):
        return selection._replace() if hasattr(selection, "_replace") else selection
    def fail(selection, *, gamma, backend, context_tokens, report):
        seen.append((selection, gamma, backend))
        report["support_level"] = "startup-verified"
        raise RuntimeError("secret local path")
    monkeypatch.setattr(validation, "precheck", precheck)
    monkeypatch.setattr(validation, "validate_selected", fail)
    output = tmp_path / "report.json"
    assert validation.main(["--target-model", "example/custom", "--gamma", "2", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert seen[0][0].target_model == "example/custom"
    assert seen[0][1:] == (2, "torch")
    assert report["status"] == "failed" and report["support_level"] == "startup-verified"
    assert report["checks"][-1]["status"] == "failed"
    assert "secret local path" not in output.read_text()


def test_generation_gate_detects_wrong_tokens_even_when_text_matches(monkeypatch):
    import onyx_cuda.server as server
    import onyx_cuda.speculative as speculative
    from onyx_cuda.generation import GenerationFinishedEvent, GenerationResult

    monkeypatch.setattr(server, "prepare_generation", lambda *a, **k: {})
    monkeypatch.setattr(speculative, "generate_speculative", lambda **k: GenerationResult([1], None, "stop", None))
    monkeypatch.setattr(speculative, "generate_speculative_events", lambda **k: iter([]))
    monkeypatch.setattr(speculative, "decode_speculative_events", lambda *a: iter([
        GenerationFinishedEvent(GenerationResult([2], None, "stop", None))]))
    pair = SimpleNamespace(target=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **k: "same")))
    with pytest.raises(RuntimeError, match="token oracle"):
        validation.generation_check(pair, {"messages": [{"role": "user", "content": "hi"}]}, 2, "torch")


def test_check_preserves_nested_stage_failure():
    report = {"checks": []}
    def fail():
        raise ValueError("failure")
    with pytest.raises(ValueError):
        check(report, "outer", lambda: check(report, "inner", fail))
    assert [entry["status"] for entry in report["checks"]] == ["failed", "failed"]


def test_wheel_inside_checkout_does_not_inherit_its_git_identity(monkeypatch, tmp_path):
    import onyx_cuda._validation_report as reporting

    (tmp_path / ".git").mkdir()
    package = tmp_path / ".venv" / "Lib" / "site-packages" / "onyx_cuda"
    package.mkdir(parents=True)
    module = package / "_validation_report.py"
    module.write_text("# installed wheel")
    monkeypatch.setattr(reporting, "__file__", str(module))
    monkeypatch.setattr(reporting.subprocess, "check_output", lambda *a, **k: pytest.fail("Attributed unrelated Git history"))
    result = reporting.evidence("test")
    assert result["source"]["commit"] is None
    assert result["source"]["dirty"] is None
    assert "_validation_report.py" in result["source"]["package_sha256"]
