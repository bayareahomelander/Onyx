"""Build, bind, and verify Windows release evidence. Uses only the standard library."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def local_file(root, name):
    require(isinstance(name, str) and name == Path(name).name and "/" not in name and "\\" not in name,
            "Artifact/report names must be simple filenames")
    path = Path(root) / name
    require(path.is_file(), f"Missing file: {name}")
    return path


def package_hashes(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return {Path(name).name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if name.startswith("onyx_cuda/") and len(Path(name).parts) == 2
                and Path(name).suffix in (".py", ".pyd", ".so")}


def literal(path, name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing source setting: {name}")


def source_state(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "--",
                "onyx_cuda", ".github/workflows/windows-cuda.yml", ".github/workflows/windows-release.yml")
    hashes = {name: digest(root / name) for name in sorted(set(paths.splitlines())) if (root / name).is_file()}
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "inputs_sha256": hashes}


def create_candidate(wheel, sdist, root, output):
    wheel, sdist, root = Path(wheel), Path(sdist), Path(root).resolve()
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        require(len(metadata_files) == 1, "Expected one wheel metadata file")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        require(metadata["Name"] == "onyx-cuda", "Wrong distribution")
        wheel_metadata = archive.read(metadata_files[0].replace("METADATA", "WHEEL")).decode()
        require("Tag: cp312-cp312-win_amd64" in wheel_metadata, "Expected a CPython 3.12 Windows x64 wheel")
    project = tomllib.loads((root / "onyx_cuda/pyproject.toml").read_text(encoding="utf-8"))["project"]
    require(metadata["Version"] == project["version"], "Wheel version differs from the source package")
    package = root / "onyx_cuda/src/onyx_cuda"
    pins = literal(package / "revisions.py", "MODEL_REVISIONS")
    models = {}
    for role in ("target", "draft"):
        model_id = literal(package / "config.py", f"DEFAULT_{role.upper()}_MODEL")
        models[role] = {"id": model_id, "revision": pins[model_id]}
    candidate = {"format": 1, "version": metadata["Version"], "source": source_state(root),
                 "tag": "cp312-cp312-win_amd64", "models": models,
                 "artifacts": {role: {"name": path.name, "sha256": digest(path)}
                               for role, path in (("wheel", wheel), ("sdist", sdist))},
                 "package_sha256": package_hashes(wheel),
                 "build": {"python": platform.python_version(), "os": platform.system(),
                           "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}}
    require(re.fullmatch(r"\d+\.\d+\.\d+", candidate["version"]), "Expected a three-part package version")
    write(output, candidate)


def verify_candidate(path, source_root=None):
    path = Path(path)
    candidate = read(path)
    require(candidate.get("format") == 1 and candidate.get("tag") == "cp312-cp312-win_amd64", "Unsupported candidate format/platform")
    require(set(candidate.get("artifacts", {})) == {"wheel", "sdist"}, "Unexpected candidate artifacts")
    for item in candidate["artifacts"].values():
        require(digest(local_file(path.parent, item["name"])) == item["sha256"], "Candidate artifact checksum mismatch")
    wheel = local_file(path.parent, candidate["artifacts"]["wheel"]["name"])
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        require(len(names) == 1, "Expected one distribution in the wheel")
        metadata = BytesParser().parsebytes(archive.read(names[0]))
        require(metadata["Name"] == "onyx-cuda" and metadata["Version"] == candidate["version"], "Candidate distribution/version mismatch")
        require("Tag: cp312-cp312-win_amd64" in archive.read(names[0].replace("METADATA", "WHEEL")).decode(), "Wheel platform tag mismatch")
    require(package_hashes(wheel) == candidate["package_sha256"], "Candidate package fingerprint mismatch")
    require("preflight.py" in candidate["package_sha256"] and any(n.endswith(".pyd") for n in candidate["package_sha256"]),
            "Candidate lacks CLI/native files")
    if source_root is not None:
        actual = source_state(Path(source_root).resolve())
        require(actual["commit"] == candidate["source"]["commit"] and
                actual["inputs_sha256"] == candidate["source"]["inputs_sha256"],
                "Validation source differs from the candidate source")
    return candidate


def audit_install(candidate_path, output):
    import onyx_cuda
    from onyx_cuda import _rust

    candidate = verify_candidate(candidate_path)
    package = Path(onyx_cuda.__file__).resolve().parent
    prefix = Path(sys.prefix).resolve()
    require(package.is_relative_to(prefix) and Path(_rust.__file__).resolve().is_relative_to(prefix),
            "An editable/source package was imported instead of the installed wheel")
    actual = {name: digest(package / name) for name in candidate["package_sha256"]}
    require(actual == candidate["package_sha256"], "Installed files differ from the candidate wheel")
    write(output, {"status": "passed", "wheel_sha256": candidate["artifacts"]["wheel"]["sha256"],
                   "package_sha256": actual, "os": platform.system(), "python": platform.python_version(),
                   "dependencies": {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}})


def passed_checks(report, required_names):
    checks = report.get("checks", [])
    names = [c.get("name") for c in checks]
    require(len(names) == len(set(names)), "Duplicate validation checks")
    require(set(required_names) <= set(names), "Required validation checks are missing")
    require(all(c.get("status") == "passed" for c in checks), "A required validation check failed/skipped")


def check_model(actual, expected):
    require(actual is not None and all(actual.get(key) == value for key, value in expected.items()),
            "Model identity or revision differs from the candidate")


def validate_run(candidate, directory, mode, profile):
    """Validate report contents, not just the exit code of an outer script."""
    directory = Path(directory)
    reports = {}
    def get(name):
        reports[name] = read(local_file(directory, name))
        return reports[name]
    wheel_hash = candidate["artifacts"]["wheel"]["sha256"]
    audit = get("installed.json")
    require(audit.get("status") == "passed" and audit.get("wheel_sha256") == wheel_hash and
            audit.get("package_sha256") == candidate["package_sha256"], "Installation is not bound to this wheel")
    require(audit.get("os") == "Windows" and audit.get("python", "").startswith("3.12."), "Wrong validation OS/Python")
    require(audit["dependencies"].get("onyx-cuda") == candidate["version"], "Installed version mismatch")

    def pytest_report(name, kernels=False):
        report = get(name)
        tests = report.get("tests", [])
        require(report.get("exit_code") == 0 and tests and report.get("collected", 0) > 0, "Pytest did not pass/collect tests")
        require(report.get("cuda_required") is (mode == "Cuda"), "CUDA requirement mismatch")
        require(report.get("kernels_required") is kernels, "Kernel requirement mismatch")
        require(report.get("greedy_backend") == ("cuda" if kernels else "torch"), "Selector mismatch")
        for test in tests:
            optional_skip = (mode == "Cuda" and profile == "Core" and not kernels and
                             test.get("outcome") == "skipped" and test.get("nodeid", "").startswith("tests/test_sparse_argmax.py::"))
            require(test.get("outcome") == "passed" or optional_skip, "Required pytest case failed or skipped")
        if mode == "Cuda":
            require(report.get("device", {}).get("total_vram_bytes", 0) > 0 and report.get("cuda_runtime") == "12.4", "Missing CUDA 12.4 hardware evidence")
            for model in candidate["models"].values():
                require(report.get("models", {}).get(model["id"]) == model["revision"], "Pytest model revision mismatch")
    pytest_report("validation.json")
    if mode == "Cuda" and profile == "Full":
        pytest_report("validation-kernels.json", kernels=True)

    preflight = get("preflight.json")
    require(preflight.get("status") == "passed" and preflight.get("support_level") == "prechecked", "Preflight did not pass")
    require(preflight.get("source", {}).get("package_sha256") == candidate["package_sha256"], "Preflight used another package")
    require(preflight.get("settings", {}).get("gamma") == 0, "Unexpected preflight mode")
    passed_checks(preflight, ["selection", "target.metadata", "target.architecture_and_memory"])
    check_model(preflight.get("models", {}).get("target"), candidate["models"]["target"])
    if mode == "Cpu":
        rejection = get("cpu-runtime-rejection.json")
        require(rejection.get("status") == "failed" and rejection.get("support_level") == "prechecked", "CPU runtime refusal was not verified")
        require(any(c.get("name") == "cuda.available" and c.get("status") == "failed" for c in rejection.get("checks", [])), "CPU failed for a reason other than absent CUDA")
        require(rejection.get("source", {}).get("package_sha256") == candidate["package_sha256"], "CPU refusal used another package")
    else:
        for name, gamma in (("selected-target.json", 0), ("selected-pair.json", 2)):
            selected = get(name)
            require(selected.get("status") == "passed" and selected.get("support_level") == "tested", "Selected-model validation failed")
            require(selected.get("source", {}).get("package_sha256") == candidate["package_sha256"], "Selected validation used another package")
            settings = selected.get("settings", {})
            require(settings.get("gamma") == gamma and settings.get("greedy_backend") == "torch" and
                    settings.get("context_tokens") == 2048 and settings.get("corpus") == "onyx-selected-model-v1", "Selected validation settings mismatch")
            required = ["selection", "preflight", "cuda.validation", "cuda.available", "selector.startup", "target.context_cache"]
            required += [f"{case}.{kind}" for case in ("cuda_ready", "gpu_summary", "number_sequence", "regex", "json_schema", "sampled")
                         for kind in ("generation", "api_and_sse")]
            if gamma:
                required += ["draft.context_cache", "speculation.forced_rejection_and_replay"]
            passed_checks(selected, required)
            check_model(selected.get("models", {}).get("target"), candidate["models"]["target"])
            if gamma:
                check_model(selected.get("models", {}).get("draft"), candidate["models"]["draft"])
            else:
                require(selected.get("models", {}).get("draft") is None, "Target-only validation loaded a draft")
        consumer = get("consumer-installed.json")
        require(consumer.get("status") == "passed" and consumer.get("wheel_sha256") == wheel_hash and
                consumer.get("package_sha256") == candidate["package_sha256"], "Consumer installed a different wheel")
        require(not {"pytest", "maturin"} & set(consumer["dependencies"]), "Consumer environment contains development dependencies")
        smoke = get("consumer-smoke.json")
        require(smoke.get("status") == "passed" and smoke.get("source", {}).get("package_sha256") == candidate["package_sha256"], "Consumer smoke used another package or failed")
        passed_checks(smoke, ["readiness", "text", "regex", "json_schema", "stream", "shutdown"])
        check_model(smoke.get("models", {}).get("target"), candidate["models"]["target"])
        require(smoke.get("models", {}).get("draft") is None, "Consumer smoke changed the default draft setting")
    return reports


def seal(candidate_path, directory, mode, profile):
    candidate = verify_candidate(candidate_path)
    reports = validate_run(candidate, directory, mode, profile)
    write(Path(directory) / "release-run.json", {"status": "passed", "mode": mode, "profile": profile,
          "candidate_sha256": digest(candidate_path), "wheel_sha256": candidate["artifacts"]["wheel"]["sha256"],
          "reports_sha256": {name: digest(Path(directory) / name) for name in reports}})


def verify_release(candidate_path, cpu, cuda, output, expected_commit, require_kernels=True):
    candidate = verify_candidate(candidate_path)
    require(candidate["source"].get("dirty") is False, "Release candidate was built from a dirty checkout")
    require(candidate["source"].get("commit") == expected_commit and re.fullmatch(r"[0-9a-f]{40}", expected_commit), "Source commit mismatch")
    summary = {"status": "passed", "candidate": candidate, "validation": {}}
    for directory, mode in ((Path(cpu), "Cpu"), (Path(cuda), "Cuda")):
        receipt = read(local_file(directory, "release-run.json"))
        require(receipt.get("status") == "passed" and receipt.get("mode") == mode, "Missing/wrong validation receipt")
        profile = receipt.get("profile")
        require(profile in ("Core", "Full"), "Unknown validation profile")
        require(not (mode == "Cuda" and require_kernels and profile != "Full"), "Required kernel validation is missing")
        require(receipt.get("candidate_sha256") == digest(candidate_path) and receipt.get("wheel_sha256") == candidate["artifacts"]["wheel"]["sha256"], "Evidence belongs to another candidate")
        reports = validate_run(candidate, directory, mode, profile)
        require(receipt.get("reports_sha256") == {name: digest(directory / name) for name in reports}, "Validation reports changed after sealing")
        summary["validation"][mode] = {"profile": profile, "reports": reports}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for artifact in candidate["artifacts"].values():
        shutil.copyfile(local_file(Path(candidate_path).parent, artifact["name"]), output / artifact["name"])
    write(output / "release-evidence.json", summary)
    (output / "SHA256SUMS.txt").write_text("".join(f"{digest(path)}  {path.name}\n" for path in sorted(output.iterdir())), encoding="utf-8")
    (output / "release-notes.md").write_text(
        f"Windows Onyx CUDA {candidate['version']} prerelease\n\n"
        f"Source: {expected_commit}\n\n"
        "CPython 3.12 x64 and CUDA 12.4 PyTorch are required. Install the wheel with its [server] extra. "
        "The default is the pinned Qwen2.5 1.5B target, gamma 0, FP16, Torch selection. "
        "See the Windows README for installation and the attached evidence for tested configurations. "
        "Larger models remain experimental; no general speedup is claimed.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("candidate")
    for name in ("wheel", "sdist", "source-root", "output"):
        build.add_argument(f"--{name}", required=True, type=Path)
    verify = commands.add_parser("candidate-check")
    verify.add_argument("--candidate", type=Path, required=True)
    verify.add_argument("--source-root", type=Path)
    installed = commands.add_parser("installed")
    installed.add_argument("--candidate", type=Path, required=True)
    installed.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("seal")
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--directory", type=Path, required=True)
    run.add_argument("--mode", choices=("Cpu", "Cuda"), required=True)
    run.add_argument("--profile", choices=("Core", "Full"), default="Full")
    release = commands.add_parser("release")
    for name in ("candidate", "cpu", "cuda", "output"):
        release.add_argument(f"--{name}", type=Path, required=True)
    release.add_argument("--expected-commit", required=True)
    release.add_argument("--allow-core-only", action="store_true")
    args = parser.parse_args()
    if args.command == "candidate":
        create_candidate(args.wheel, args.sdist, args.source_root, args.output)
    elif args.command == "candidate-check":
        verify_candidate(args.candidate, args.source_root)
    elif args.command == "installed":
        audit_install(args.candidate, args.output)
    elif args.command == "seal":
        seal(args.candidate, args.directory, args.mode, args.profile)
    else:
        verify_release(args.candidate, args.cpu, args.cuda, args.output, args.expected_commit, not args.allow_core_only)


if __name__ == "__main__":
    main()
