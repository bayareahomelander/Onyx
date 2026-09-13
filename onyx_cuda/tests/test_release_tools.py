"""Release promotion must be tied to the tested artifact and complete evidence."""

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = script("verify_release")
smoke = script("smoke_windows_release")


def save(root, name, data):
    (root / name).write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def reports(tmp_path):
    candidate = {"version": "0.1.0", "artifacts": {"wheel": {"sha256": "w"}},
                 "package_sha256": {"preflight.py": "p", "_rust.pyd": "r"},
                 "models": {"target": {"id": "target", "revision": "a" * 40},
                            "draft": {"id": "draft", "revision": "b" * 40}}}
    package = candidate["package_sha256"]
    audit = {"status": "passed", "wheel_sha256": "w", "package_sha256": package,
             "os": "Windows", "python": "3.12.10", "dependencies": {"onyx-cuda": "0.1.0"}}
    save(tmp_path, "installed.json", audit)
    save(tmp_path, "consumer-installed.json", audit)
    pytest_result = {"exit_code": 0, "collected": 1, "tests": [{"nodeid": "tests/test_example.py::test_a", "outcome": "passed"}],
                     "cuda_required": True, "kernels_required": False, "greedy_backend": "torch",
                     "device": {"total_vram_bytes": 6 * 2**30}, "cuda_runtime": "12.4",
                     "models": {"target": "a" * 40, "draft": "b" * 40}}
    save(tmp_path, "validation.json", pytest_result)
    save(tmp_path, "validation-kernels.json", {**pytest_result, "kernels_required": True, "greedy_backend": "cuda"})
    def passed(names):
        return [{"name": name, "status": "passed"} for name in names]
    save(tmp_path, "preflight.json", {"status": "passed", "support_level": "prechecked", "source": {"package_sha256": package},
         "settings": {"gamma": 0}, "models": {"target": candidate["models"]["target"]},
         "checks": passed(["selection", "target.metadata", "target.architecture_and_memory"])})
    for filename, gamma in (("selected-target.json", 0), ("selected-pair.json", 2)):
        names = ["selection", "preflight", "cuda.validation", "cuda.available", "selector.startup", "target.context_cache"]
        names += [f"{case}.{kind}" for case in ("cuda_ready", "gpu_summary", "number_sequence", "regex", "json_schema", "sampled")
                  for kind in ("generation", "api_and_sse")]
        if gamma:
            names += ["draft.context_cache", "speculation.forced_rejection_and_replay"]
        save(tmp_path, filename, {"status": "passed", "support_level": "tested", "source": {"package_sha256": package},
             "settings": {"gamma": gamma, "greedy_backend": "torch", "context_tokens": 2048, "corpus": "onyx-selected-model-v1"},
             "models": {"target": candidate["models"]["target"], "draft": candidate["models"]["draft"] if gamma else None},
             "checks": passed(names)})
    save(tmp_path, "consumer-smoke.json", {"status": "passed", "source": {"package_sha256": package},
         "models": {"target": candidate["models"]["target"], "draft": None},
         "checks": passed(["readiness", "text", "regex", "json_schema", "stream", "shutdown"])})
    return candidate, tmp_path


def test_complete_gpu_evidence_passes(reports):
    candidate, root = reports
    assert len(release.validate_run(candidate, root, "Cuda", "Full")) == 8


@pytest.mark.parametrize("filename,mutation", [
    ("installed.json", lambda r: r.update(wheel_sha256="other")),
    ("validation.json", lambda r: r.update(exit_code=1)),
    ("validation.json", lambda r: r["tests"][0].update(outcome="skipped")),
    ("validation.json", lambda r: r.update(cuda_required=False)),
    ("validation-kernels.json", lambda r: r.update(kernels_required=False)),
    ("selected-pair.json", lambda r: r["models"]["draft"].update(revision="c" * 40)),
    ("selected-pair.json", lambda r: r["settings"].update(gamma=0)),
    ("selected-pair.json", lambda r: r["checks"].pop()),
    ("selected-target.json", lambda r: r["source"].update(package_sha256={})),
    ("consumer-installed.json", lambda r: r["dependencies"].update(pytest="9")),
    ("consumer-smoke.json", lambda r: r["checks"][-1].update(status="failed")),
])
def test_mismatched_or_incomplete_evidence_fails(reports, filename, mutation):
    candidate, root = reports
    data = release.read(root / filename)
    mutation(data)
    save(root, filename, data)
    with pytest.raises(ValueError):
        release.validate_run(candidate, root, "Cuda", "Full")


def test_missing_gpu_report_fails(reports):
    candidate, root = reports
    (root / "selected-pair.json").unlink()
    with pytest.raises(ValueError, match="Missing file"):
        release.validate_run(candidate, root, "Cuda", "Full")


def test_core_allows_only_optional_kernel_skips(reports):
    candidate, root = reports
    data = release.read(root / "validation.json")
    data["tests"].append({"nodeid": "tests/test_sparse_argmax.py::test_optional", "outcome": "skipped"})
    save(root, "validation.json", data)
    release.validate_run(candidate, root, "Cuda", "Core")
    with pytest.raises(ValueError, match="skipped"):
        release.validate_run(candidate, root, "Cuda", "Full")


def test_manifest_cannot_escape_artifact_directory(tmp_path):
    with pytest.raises(ValueError, match="simple filenames"):
        release.local_file(tmp_path, "../somewhere.whl")


def test_dirty_source_never_promotes(monkeypatch, tmp_path):
    monkeypatch.setattr(release, "verify_candidate", lambda p: {"source": {"dirty": True}})
    with pytest.raises(ValueError, match="dirty checkout"):
        release.verify_release(tmp_path / "candidate.json", tmp_path, tmp_path, tmp_path / "bundle", "a" * 40)
    assert not (tmp_path / "bundle").exists()


@pytest.fixture
def candidate_files(tmp_path):
    wheel = tmp_path / "onyx_cuda-0.1.0-cp312-cp312-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("onyx_cuda/preflight.py", "# test")
        archive.writestr("onyx_cuda/_rust.pyd", b"test native bytes")
        archive.writestr("onyx_cuda-0.1.0.dist-info/METADATA", "Name: onyx-cuda\nVersion: 0.1.0\n")
        archive.writestr("onyx_cuda-0.1.0.dist-info/WHEEL", "Tag: cp312-cp312-win_amd64\n")
    sdist = tmp_path / "onyx_cuda-0.1.0.tar.gz"
    sdist.write_bytes(b"test archive")
    manifest = tmp_path / "candidate.json"
    save(tmp_path, manifest.name, {"format": 1, "tag": "cp312-cp312-win_amd64", "version": "0.1.0",
         "source": {"commit": "a" * 40, "dirty": False},
         "package_sha256": release.package_hashes(wheel), "artifacts": {
             "wheel": {"name": wheel.name, "sha256": release.digest(wheel)},
             "sdist": {"name": sdist.name, "sha256": release.digest(sdist)}}})
    return manifest, wheel


def test_corrupted_wheel_is_rejected(candidate_files):
    manifest, wheel = candidate_files
    release.verify_candidate(manifest)
    with wheel.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        release.verify_candidate(manifest)


def test_report_tampering_and_core_profile_block_promotion(candidate_files, reports, tmp_path):
    manifest, wheel = candidate_files
    candidate, root = reports
    disk = release.read(manifest)
    disk.update(models=candidate["models"])
    save(manifest.parent, manifest.name, disk)
    # Bind the realistic fixture reports to this specific wheel.
    for path in root.glob("*.json"):
        if path == manifest:
            continue
        data = release.read(path)
        if "package_sha256" in data:
            data["package_sha256"] = disk["package_sha256"]
            data["wheel_sha256"] = release.digest(wheel)
        if "source" in data:
            data["source"]["package_sha256"] = disk["package_sha256"]
        save(root, path.name, data)
    release.seal(manifest, root, "Cuda", "Core")
    # Check profile rejection independently of CPU validation by supplying a valid
    # CPU receipt/report set in its own directory.
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    for name in ("installed.json", "validation.json", "preflight.json"):
        save(cpu, name, release.read(root / name))
    data = release.read(cpu / "validation.json")
    data["cuda_required"] = False
    save(cpu, "validation.json", data)
    save(cpu, "cpu-runtime-rejection.json", {"status": "failed", "support_level": "prechecked",
         "source": {"package_sha256": disk["package_sha256"]},
         "checks": [{"name": "cuda.available", "status": "failed"}]})
    release.seal(manifest, cpu, "Cpu", "Full")
    with pytest.raises(ValueError, match="kernel validation"):
        release.verify_release(manifest, cpu, root, tmp_path / "bundle", "a" * 40)
    bundle = tmp_path / "development-bundle"
    release.verify_release(manifest, cpu, root, bundle, "a" * 40, require_kernels=False)
    assert release.digest(bundle / wheel.name) == release.digest(wheel)
    assert release.read(bundle / "release-evidence.json")["status"] == "passed"
    for line in (bundle / "SHA256SUMS.txt").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        assert release.digest(bundle / name) == checksum
    data = release.read(root / "selected-target.json")
    data["extra"] = "changed after sealing"
    save(root, "selected-target.json", data)
    with pytest.raises(ValueError, match="changed after sealing"):
        release.verify_release(manifest, cpu, root, tmp_path / "bundle", "a" * 40, require_kernels=False)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console exit contract")
@pytest.mark.parametrize("exit_code", [3, 3221225786])
def test_windows_control_exit_requires_completed_shutdown_log(tmp_path, exit_code):
    child = SimpleNamespace(poll=lambda: exit_code, returncode=exit_code, pid=123,
                            wait=lambda **k: exit_code)
    log = tmp_path / "server.log"
    log.write_text("Started server process [456]\nApplication shutdown complete.\nFinished server process [456]\n")
    assert smoke.stop_server(child, log=log)["application_shutdown_complete"]
    log.write_text("Shutting down\n")
    with pytest.raises(RuntimeError, match="unexpected code"):
        smoke.stop_server(child, log=log)


def test_forced_shutdown_reaps_owned_child(monkeypatch):
    calls = []
    monkeypatch.setattr(smoke.subprocess, "run", lambda args, **kwargs: calls.append(args))
    class Child:
        pid = 789
        returncode = None
        killed = False
        def poll(self):
            return self.returncode
        def send_signal(self, value):
            raise OSError("cannot signal")
        def kill(self):
            self.killed = True
            self.returncode = -1
        def wait(self, timeout):
            return self.returncode
    child = Child()
    with pytest.raises(OSError):
        smoke.stop_server(child)
    if sys.platform == "win32":
        assert calls == [["taskkill", "/PID", "789", "/T", "/F"]]
    assert child.killed and child.poll() is not None


def test_wrong_process_cannot_satisfy_readiness(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("Loading model...", encoding="utf-8")
    process = SimpleNamespace(poll=lambda: None)
    client = SimpleNamespace(get=lambda *a, **k: pytest.fail("Contacted an unowned listener"))
    with pytest.raises(TimeoutError):
        smoke.wait_ready(process, client, log, 12345, 0.01)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows delivery entry point")
def test_bad_wheel_digest_fails_before_creating_environment(tmp_path):
    wheel = tmp_path / "fake.whl"
    wheel.write_bytes(b"not a wheel")
    manifest = tmp_path / "candidate.json"
    manifest.write_text("{}")
    destination = tmp_path / "must-not-exist"
    command = ["pwsh", "-NoProfile", "-File", str(Path(__file__).resolve().parents[1] / "scripts/validate_windows.ps1"),
               "-Python", sys.executable, "-WheelPath", str(wheel), "-WheelSha256", "0" * 64,
               "-CandidateManifest", str(manifest), "-OutputDirectory", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "checksum mismatch" in result.stderr
    assert not destination.exists()
