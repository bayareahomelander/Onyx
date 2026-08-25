import torch

from onyx_cuda import diagnostics


def test_diagnostics_run_on_cuda():
    result = diagnostics.collect_diagnostics()

    assert result["selected_device"] == "cuda:0"
    assert result["device_name"]
    assert result["vram_mib"] > 0
    assert result["cuda_operation"] == "pass"


def test_diagnostics_fail_without_cuda(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert diagnostics.main() == 1
    assert "requires an NVIDIA GPU" in capsys.readouterr().err
