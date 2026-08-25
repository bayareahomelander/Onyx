# Onyx CUDA

Independent Windows and NVIDIA CUDA implementation of Onyx.

This project must not import from or modify the macOS `onyx` runtime.

## Windows development setup

Prerequisites are 64-bit Windows, Python 3.12, an NVIDIA GPU and driver, the Rust MSVC toolchain, and Visual Studio Build Tools with the C++ workload.

Create the environment from this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
```

Install the CUDA wheel before the project dependency so pip cannot silently select a CPU-only PyTorch build. The validated baseline is:

| Component | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.6 |
| Maturin | 1.14.1 |
| Pytest | 9.1.1 |
| Rust/Cargo | 1.96.1 |
| NVIDIA driver | 610.88 |
| GPU | GeForce RTX 4050 Laptop GPU (6 GB) |
