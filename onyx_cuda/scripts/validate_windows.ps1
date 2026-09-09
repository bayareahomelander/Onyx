param(
    [ValidateSet("Cpu", "Cuda")][string]$Mode = "Cpu",
    [string]$Python = "python",
    [string]$OutputDirectory,
    [switch]$Benchmarks
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ($env:OS -ne "Windows_NT") { throw "This validation entry point requires Windows." }
if ($Benchmarks -and $Mode -ne "Cuda") { throw "Benchmarks require -Mode Cuda." }
$project = Split-Path -Parent $PSScriptRoot
$constraints = Join-Path $project "requirements-validation.txt"
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $project ("validation/results/" + [guid]::NewGuid().ToString("N"))
}
$runRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $runRoot) { throw "Use a new output directory: $runRoot" }
New-Item -ItemType Directory -Path $runRoot | Out-Null
$log = Join-Path $runRoot "validation.log"

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    "$Executable $($Arguments -join ' ')" | Tee-Object -FilePath $log -Append | Out-Host
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}

$savedEnvironment = @{}
foreach ($name in @("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN", "ONYX_SPECULATIVE_GAMMA", "ONYX_GREEDY_BACKEND", "CUPY_CACHE_IN_MEMORY", "ONYX_TARGET_MODEL", "ONYX_TARGET_REVISION", "ONYX_DRAFT_MODEL", "ONYX_DRAFT_REVISION")) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
Push-Location $project
try {
    $env:PYTHONPATH = $null
    $env:PYTHONHOME = $null
    $env:PYTHONNOUSERSITE = "1"
    $env:ONYX_SPECULATIVE_GAMMA = $null
    $env:ONYX_TARGET_MODEL = $null
    $env:ONYX_TARGET_REVISION = $null
    $env:ONYX_DRAFT_MODEL = $null
    $env:ONYX_DRAFT_REVISION = $null
    $env:ONYX_GREEDY_BACKEND = "torch"
    $env:CUPY_CACHE_IN_MEMORY = "1"
    $env:CARGO_TARGET_DIR = Join-Path $runRoot "cargo-target"
    $env:RUSTUP_TOOLCHAIN = "1.96.1"
    Invoke-Checked $Python @("-c", "import sys; assert sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32, sys.version")
    Invoke-Checked "rustup" @("toolchain", "install", "1.96.1", "--profile", "minimal", "--target", "x86_64-pc-windows-msvc")
    Invoke-Checked "rustc" @("--version")
    Invoke-Checked $Python @("-m", "venv", (Join-Path $runRoot "build-env"))
    $buildPython = Join-Path $runRoot "build-env/Scripts/python.exe"
    Invoke-Checked $buildPython @("-m", "pip", "install", "-c", $constraints, "pip==26.1.2", "maturin==1.14.1")
    Invoke-Checked "cargo" @("test", "--locked", "--manifest-path", "rust/Cargo.toml")
    $artifacts = Join-Path $runRoot "artifacts"
    Invoke-Checked $buildPython @("-m", "maturin", "sdist", "--out", $artifacts)
    $sdist = @(Get-ChildItem -LiteralPath $artifacts -Filter "*.tar.gz")
    if ($sdist.Count -ne 1) { throw "Expected exactly one source distribution." }
    # Rebuild from the source distribution, not from an editable checkout.
    Invoke-Checked $buildPython @("-m", "pip", "wheel", "--no-deps", "--wheel-dir", $artifacts, $sdist[0].FullName)
    $wheels = @(Get-ChildItem -LiteralPath $artifacts -Filter "*.whl")
    if ($wheels.Count -ne 1) { throw "Expected exactly one wheel." }
    Get-FileHash -Algorithm SHA256 -LiteralPath $sdist[0].FullName, $wheels[0].FullName |
        Select-Object Path, Hash | ConvertTo-Json | Set-Content (Join-Path $runRoot "artifact-hashes.json")

    Invoke-Checked $Python @("-m", "venv", (Join-Path $runRoot "test-env"))
    $testPython = Join-Path $runRoot "test-env/Scripts/python.exe"
    Invoke-Checked $testPython @("-m", "pip", "install", "pip==26.1.2")
    $torchIndex = if ($Mode -eq "Cuda") { "cu124" } else { "cpu" }
    Invoke-Checked $testPython @("-m", "pip", "install", "-c", $constraints, "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/$torchIndex")
    $extras = if ($Mode -eq "Cuda") { "[dev,kernels]" } else { "[dev]" }
    Invoke-Checked $testPython @("-m", "pip", "install", "-c", $constraints, ($wheels[0].FullName + $extras))
    Invoke-Checked $testPython @("-m", "pip", "check")
    & $testPython -m pip freeze --all | Set-Content (Join-Path $runRoot "requirements-resolved.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not record installed dependencies." }
    Copy-Item -LiteralPath (Join-Path $project "tests") -Destination (Join-Path $runRoot "tests") -Recurse
    Copy-Item -LiteralPath (Join-Path $project "pyproject.toml") -Destination $runRoot
    Copy-Item -LiteralPath $constraints -Destination $runRoot
    Push-Location $runRoot
    try {
        Invoke-Checked $testPython @("-I", "-c", "import pathlib, sys, onyx_cuda; from onyx_cuda import _rust; root=pathlib.Path(sys.prefix).resolve(); assert pathlib.Path(onyx_cuda.__file__).resolve().is_relative_to(root); assert pathlib.Path(_rust.__file__).resolve().is_relative_to(root); print(onyx_cuda.__file__); print(_rust.__file__)")
        $pytestArgs = @("-I", "-m", "pytest", "tests", "-ra", "--strict-markers", "--junitxml=pytest.xml", "--validation-report=validation.json")
        if ($Mode -eq "Cuda") { $pytestArgs += "--require-cuda" }
        else { $pytestArgs += @("-m", "not gpu") }
        Invoke-Checked $testPython $pytestArgs
        if ($Mode -eq "Cuda") {
            $env:ONYX_GREEDY_BACKEND = "cuda"
            Invoke-Checked $testPython @("-I", "-m", "pytest", "tests", "-m", "gpu", "-ra", "--require-cuda", "--require-kernels", "--strict-markers", "--junitxml=pytest-kernels.xml", "--validation-report=validation-kernels.json")
            $env:ONYX_GREEDY_BACKEND = "torch"
        }
        if ($Benchmarks) {
            Invoke-Checked $testPython @("-I", "-m", "onyx_cuda.benchmark", "--target")
            Invoke-Checked $testPython @("-I", "-m", "onyx_cuda.benchmark", "--target", "--constraints")
            Invoke-Checked $testPython @("-I", "-m", "onyx_cuda.benchmark", "--speculative")
            Invoke-Checked $testPython @("-I", "-m", "onyx_cuda.benchmark_masking", "--models")
        }
    } finally { Pop-Location }
    Write-Host "Validation passed. Evidence and fresh environments: $runRoot"
} finally {
    Pop-Location
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }
}
