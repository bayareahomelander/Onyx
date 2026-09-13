"""Exercise an installed consumer wheel through the real Uvicorn command."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from onyx_cuda._validation_report import check, evidence, reserve_report, write_report
from onyx_cuda.config import DEFAULT_TARGET_MODEL, MODEL_ENVIRONMENT_VARIABLES
from onyx_cuda.revisions import MODEL_REVISIONS
from onyx_cuda.validate_model import parse_sse, require


def stop_server(process, timeout=30, log=None):
    """Always reap the owned child; forced cleanup is a failed graceful-shutdown check."""
    try:
        if process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            process.wait(timeout=timeout)
        contents = log.read_text(encoding="utf-8", errors="replace") if log is not None else ""
        # The Windows venv launcher can have a different PID from its interpreter.
        # Match startup and shutdown in the log owned exclusively by this child.
        started = re.findall(r"Started server process \[(\d+)\]", contents)
        finished = re.findall(r"Finished server process \[(\d+)\]", contents)
        completed = len(started) == 1 and finished == started and "Application shutdown complete." in contents
        # Uvicorn re-raises SIGBREAK after cleanup; the CRT can return 3 or
        # STATUS_CONTROL_C_EXIT. Both require the matching cleanup log above.
        windows_control_exit = os.name == "nt" and process.returncode % (2**32) in (3, 0xC000013A) and completed
        require(process.returncode in (0, -signal.SIGINT, 130) or windows_control_exit,
                f"Server shutdown returned an unexpected code: {process.returncode}")
        require(log is None or completed, "Server did not complete application shutdown")
        return {"exit_code": process.returncode, "application_shutdown_complete": bool(completed)}
    except BaseException:
        if process.poll() is None:
            if os.name == "nt":
                # Include the interpreter behind Windows' venv launcher.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
                               check=False)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=timeout)
        raise


def wait_ready(process, client, log, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, "Server exited during startup; inspect the local log")
        # Uvicorn binds after model startup. Its own ready line prevents a process
        # that races for the port from accidentally satisfying our readiness check.
        if f"Uvicorn running on http://127.0.0.1:{port}" in log.read_text(encoding="utf-8", errors="replace"):
            try:
                response = client.get("/", timeout=2)
                if response.status_code == 200:
                    health = response.json()
                    require(health["speculative_gamma"] == 0 and health["greedy_backend"] == "torch", "Server defaults changed")
                    target = health["models"]["target"]
                    require(target["id"] == DEFAULT_TARGET_MODEL and target["revision"] == MODEL_REVISIONS[DEFAULT_TARGET_MODEL], "Wrong startup model/revision")
                    require(health["models"]["draft"] is None, "Default server loaded a draft")
                    return health["models"]
            except httpx.TransportError:
                pass
        time.sleep(0.1)
    raise TimeoutError("Server readiness deadline expired; inspect the local log")


def completion(client, options):
    payload = {"messages": [{"role": "user", "content": "Return a short response."}], "max_tokens": 32, **options}
    response = client.post("/v1/chat/completions", json=payload)
    require(response.status_code == 200, f"Completion returned HTTP {response.status_code}")
    choice = response.json()["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]


def exercise(client, report):
    def text():
        content, reason = completion(client, {})
        require(bool(content) and reason in ("stop", "length"), "Text completion failed")
    def regex():
        require(completion(client, {"regex": "CUDA Ready"}) == ("CUDA Ready", "stop"), "Regex completion failed")
    def schema():
        content, reason = completion(client, {"json_schema": {"type": "string", "enum": ["ready"]}})
        require(reason == "stop" and json.loads(content) == "ready", "Schema completion failed")
    def stream():
        payload = {"messages": [{"role": "user", "content": "Reply with CUDA ready."}],
                   "regex": "CUDA Ready", "max_tokens": 32, "stream": True}
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            require(response.status_code == 200 and response.headers.get("content-type", "").startswith("text/event-stream"), "SSE HTTP response failed")
            require(parse_sse(response.iter_lines()) == ("CUDA Ready", "stop"), "SSE completion failed")
    for name, operation in (("text", text), ("regex", regex), ("json_schema", schema), ("stream", stream)):
        check(report, name, operation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args(argv)
    require(args.startup_timeout > 0, "Startup timeout must be positive")
    reserve_report(args.output)
    report = evidence("consumer-release-smoke")
    process = None
    try:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        environment = os.environ.copy()
        for key in (*MODEL_ENVIRONMENT_VARIABLES, "ONYX_SPECULATIVE_GAMMA", "ONYX_GREEDY_BACKEND", "PYTHONPATH", "PYTHONHOME"):
            environment.pop(key, None)
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
        log = args.output.with_suffix(".log")
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-I", "-m", "uvicorn", "onyx_cuda.server:create_app", "--factory",
                                        "--host", "127.0.0.1", "--port", str(port), "--no-use-colors"],
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, **options)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180, trust_env=False) as client:
                report["models"] = check(report, "readiness", lambda: wait_ready(process, client, log, port, args.startup_timeout))
                exercise(client, report)
            report["shutdown"] = check(report, "shutdown", lambda: stop_server(process, log=log))
        report["status"] = "passed"
    except Exception as error:
        print(f"Consumer smoke failed: {error}", file=sys.stderr)
        return 1
    finally:
        if process is not None and process.poll() is None:
            try:
                stop_server(process, log=log)
            except Exception:
                report["status"] = "failed"
        write_report(args.output, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
