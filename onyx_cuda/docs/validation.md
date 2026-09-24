# CUDA validation

[Setup](../README.md) · [Configuration](configuration.md) · [Benchmarks](benchmarks.md)

From a development install in `onyx_cuda` (on Linux, use the activated `python`):

```powershell
.\.venv\Scripts\python.exe -m pytest --require-cuda
.\.venv\Scripts\python.exe -m onyx_cuda.preflight --vram-gib 22 --output validation/default-preflight.json
.\.venv\Scripts\python.exe -m onyx_cuda.validate_model --output validation/default-runtime.json
```

Preflight checks metadata without loading weights; it does not establish GPU fit.
Both tools default to the configured pair, gamma 2, and 8192 context tokens.
Use `--gamma 0` for target-only validation or `--context-tokens` to probe another
capacity. Runtime validation checks generation, constraints, API/SSE, and caches.
Neither tool guarantees speedup. Reports require a new output filename.

For a fresh Windows build/install check, run
`./scripts/validate_windows.ps1 -Mode Cuda -Profile Core -Python ./.venv/Scripts/python.exe`.
Add `-Benchmarks` for performance comparisons or `-Profile Full` for the optional
CUDA kernels. Routine CI builds the package and runs Rust/CPU tests. The manual
[Windows release workflow](../../.github/workflows/windows-release.yml) verifies the
candidate wheel on Windows CPU and GPU. Linux validation does not replace that
Windows delivery check.
