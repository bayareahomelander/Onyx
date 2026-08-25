"""Minimal Windows CUDA environment check."""

import sys

import torch

from onyx_cuda.device import require_cuda


def collect_diagnostics() -> dict[str, str | int]:
    """Collect versions and prove one synchronized GPU operation."""
    device = require_cuda()
    probe = torch.ones(1, device=device) + 1
    torch.cuda.synchronize(device)
    if probe.item() != 2:
        raise RuntimeError("CUDA probe returned an unexpected result")

    properties = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "bundled_cuda": torch.version.cuda or "none",
        "device_count": torch.cuda.device_count(),
        "selected_device": str(device),
        "device_name": properties.name,
        "vram_mib": properties.total_memory // (1024 * 1024),
        "cuda_operation": "pass",
    }


def main() -> int:
    try:
        diagnostics = collect_diagnostics()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for name, value in diagnostics.items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
