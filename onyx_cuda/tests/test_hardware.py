import subprocess

import pytest

from onyx_cuda.hardware import (
    NvidiaDevice,
    NvidiaSmiExecutionError,
    NvidiaSmiNotFoundError,
    NvidiaSmiOutputError,
    discover_nvidia_devices,
)


class RecordingRunner:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def completed_process(*, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run_probe(monkeypatch, runner):
    monkeypatch.setattr("onyx_cuda.hardware.subprocess.run", runner)
    return discover_nvidia_devices()


def test_discovers_one_or_more_devices_without_using_a_shell(monkeypatch):
    runner = RecordingRunner(
        result=completed_process(
            stdout=(
                "0, NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 610.47, 8.9\n"
                '1, "NVIDIA Test, Secondary GPU", 8192, 610.47, 9.0\n'
            )
        )
    )

    devices = run_probe(monkeypatch, runner)

    assert devices == (
        NvidiaDevice(0, "NVIDIA GeForce RTX 4050 Laptop GPU", 6141, "610.47", "8.9"),
        NvidiaDevice(1, "NVIDIA Test, Secondary GPU", 8192, "610.47", "9.0"),
    )
    command, kwargs = runner.calls[0]
    assert command == [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": 10,
    }


def test_empty_successful_output_reports_no_devices(monkeypatch):
    runner = RecordingRunner(result=completed_process(stdout="\n"))

    assert run_probe(monkeypatch, runner) == ()


@pytest.mark.parametrize(
    "output",
    [
        "0, incomplete\n",
        "gpu-zero, NVIDIA GPU, 6141, 610.47, 8.9\n",
        "0, NVIDIA GPU, unknown, 610.47, 8.9\n",
        "-1, NVIDIA GPU, 6141, 610.47, 8.9\n",
        "0, NVIDIA GPU, 0, 610.47, 8.9\n",
        "0, , 6141, 610.47, 8.9\n",
    ],
)
def test_rejects_malformed_driver_output(monkeypatch, output):
    runner = RecordingRunner(result=completed_process(stdout=output))

    with pytest.raises(NvidiaSmiOutputError, match="row 1"):
        run_probe(monkeypatch, runner)


def test_reports_missing_nvidia_smi_separately(monkeypatch):
    runner = RecordingRunner(error=FileNotFoundError())

    with pytest.raises(NvidiaSmiNotFoundError, match="not found"):
        run_probe(monkeypatch, runner)


def test_reports_timeout_as_an_execution_error(monkeypatch):
    runner = RecordingRunner(
        error=subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=10)
    )

    with pytest.raises(NvidiaSmiExecutionError, match="10 seconds"):
        run_probe(monkeypatch, runner)


def test_reports_operating_system_execution_errors(monkeypatch):
    runner = RecordingRunner(error=PermissionError("access denied"))

    with pytest.raises(NvidiaSmiExecutionError, match="could not be executed: access denied"):
        run_probe(monkeypatch, runner)


def test_reports_nonzero_exit_with_driver_details(monkeypatch):
    runner = RecordingRunner(
        result=completed_process(returncode=9, stderr="driver communication failed")
    )

    with pytest.raises(NvidiaSmiExecutionError, match="status 9: driver communication failed"):
        run_probe(monkeypatch, runner)
