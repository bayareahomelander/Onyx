"""Shared CLI configuration and portable evidence (no host paths or user prompts)."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from onyx_cuda.config import resolve_model_selection


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_gib(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def add_selection_arguments(parser):
    for setting in ("target-model", "target-revision", "draft-model", "draft-revision"):
        parser.add_argument(f"--{setting}", help="Overrides the corresponding ONYX environment setting")
    parser.add_argument("--gamma", type=nonnegative_int, default=0,
                        help="Draft proposal length; default 0 inspects/loads only the target")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path (never overwritten)")


def selected_models(args):
    return resolve_model_selection(**{name: getattr(args, name) for name in (
        "target_model", "target_revision", "draft_model", "draft_revision",
    )})


def reserve_report(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fail before downloads/model loading if the report already exists.
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "interrupted", "message": "Run has not completed."}, stream)


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evidence(kind):
    dependencies = {}
    for name in ("onyx-cuda", "torch", "transformers", "tokenizers", "huggingface-hub",
                 "fastapi", "pydantic", "httpx", "cupy-cuda12x"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    package = Path(__file__).resolve().parent
    hashes = {}
    for path in sorted(package.iterdir()):
        if path.suffix in (".py", ".pyd", ".so"):
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                hashes[path.name] = digest.hexdigest()
    source = {"commit": None, "dirty": None, "package_sha256": hashes}
    # A virtualenv can live inside a checkout while containing an unrelated wheel.
    # Attribute Git history only to the actual source layout, never its ancestors.
    root = next((parent for parent in package.parents
                 if (parent / ".git").exists() and package in (
                     parent / "onyx_cuda" / "src" / "onyx_cuda", parent / "src" / "onyx_cuda")), None)
    if root is not None:
        try:
            def git(*arguments):
                return subprocess.check_output(["git", "-C", str(root), *arguments],
                                               stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
            source.update(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))
        except (OSError, subprocess.SubprocessError):
            pass
    return {"report_version": 1, "kind": kind, "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed", "support_level": "unverified", "source": source,
            "environment": {"os": platform.system(), "os_release": platform.release(),
                            "machine": platform.machine(), "python": platform.python_version(),
                            "dependencies": dependencies}, "checks": []}


def check(report, name, operation):
    entry = {"name": name, "status": "failed"}
    report["checks"].append(entry)
    try:
        result = operation()
    except Exception as error:
        # Raw exceptions may contain cache paths, credentials, or response bodies.
        # The terminal has details; the shareable report has the failing stage/type.
        entry["error_type"] = type(error).__name__
        raise
    entry["status"] = "passed"
    return result
