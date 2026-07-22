"""Read-only NVIDIA hardware discovery without initializing a CUDA runtime."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass


class HardwareProbeError(RuntimeError):
    """Base error raised when NVIDIA hardware discovery cannot complete."""


class NvidiaSmiNotFoundError(HardwareProbeError):
    """Raised when the NVIDIA driver utility is not installed or not on PATH."""


class NvidiaSmiExecutionError(HardwareProbeError):
    """Raised when the NVIDIA driver utility cannot execute successfully."""


class NvidiaSmiOutputError(HardwareProbeError):
    """Raised when the NVIDIA driver utility returns an unsupported response."""


@dataclass(frozen=True, slots=True)
class NvidiaDevice:
    """NVIDIA GPU properties relevant to the Onyx CUDA memory budget."""

    index: int
    name: str
    memory_total_mib: int
    driver_version: str
    compute_capability: str


_QUERY_FIELDS = ("index", "name", "memory.total", "driver_version", "compute_cap")


def discover_nvidia_devices() -> tuple[NvidiaDevice, ...]:
    """Return installed NVIDIA devices reported by ``nvidia-smi``.

    The command is executed directly, without a shell. Discovery does not import an inference
    framework, create a CUDA context, or allocate GPU memory. An empty successful response means
    no devices were reported.
    """

    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise NvidiaSmiNotFoundError(
            "nvidia-smi was not found; install an NVIDIA driver and ensure nvidia-smi is on PATH"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise NvidiaSmiExecutionError("nvidia-smi did not respond within 10 seconds") from exc
    except OSError as exc:
        raise NvidiaSmiExecutionError(f"nvidia-smi could not be executed: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or "no error details were returned"
        raise NvidiaSmiExecutionError(
            f"nvidia-smi exited with status {result.returncode}: {detail}"
        )

    return _parse_devices(result.stdout)


def _parse_devices(output: str) -> tuple[NvidiaDevice, ...]:
    devices = []

    rows = csv.reader(output.splitlines(), skipinitialspace=True)
    for row_number, row in enumerate(rows, start=1):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != len(_QUERY_FIELDS):
            raise NvidiaSmiOutputError(
                f"nvidia-smi row {row_number} contained {len(row)} fields; "
                f"expected {len(_QUERY_FIELDS)}"
            )

        index_text, name, memory_text, driver_version, compute_capability = (
            value.strip() for value in row
        )
        try:
            index = int(index_text)
            memory_total_mib = int(memory_text)
        except ValueError as exc:
            raise NvidiaSmiOutputError(
                f"nvidia-smi row {row_number} contained a non-integer index or memory value"
            ) from exc

        if index < 0 or memory_total_mib <= 0:
            raise NvidiaSmiOutputError(
                f"nvidia-smi row {row_number} contained an invalid index or memory value"
            )
        if not name or not driver_version or not compute_capability:
            raise NvidiaSmiOutputError(
                f"nvidia-smi row {row_number} contained an empty device property"
            )

        devices.append(
            NvidiaDevice(
                index=index,
                name=name,
                memory_total_mib=memory_total_mib,
                driver_version=driver_version,
                compute_capability=compute_capability,
            )
        )

    return tuple(devices)
