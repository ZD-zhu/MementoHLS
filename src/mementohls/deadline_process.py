from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import psutil


@dataclass(frozen=True)
class DeadlineProcessResult:
    return_code: int
    stdout: str
    stderr: str
    t0: float
    t1: float
    execution_time: float
    timed_out: bool


def _signal_process_tree(
    process: subprocess.Popen[Any],
    signal_number: int,
) -> None:
    """Signal an isolated group and every descendant visible in procfs."""

    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []

    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass

    for descendant in reversed(descendants):
        try:
            descendant.send_signal(signal_number)
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass


def _wait_until_exit(
    process: subprocess.Popen[Any],
    *,
    deadline: float,
    poll_seconds: float,
) -> bool:
    while True:
        if process.poll() is not None:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(poll_seconds, max(0.01, remaining)))


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def run_process_file_backed(
    command: Sequence[os.PathLike[str] | str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    log_prefix: Path,
    grace_seconds: float = 5.0,
    poll_seconds: float = 0.1,
) -> DeadlineProcessResult:
    """Run an HLS subprocess without PIPE-backed deadline failure modes.

    Vitis descendants can fill or retain inherited pipes after the direct child
    exits.  Output is therefore written to ordinary files, while the calling
    worker itself polls the fixed monotonic deadline.  At timeout, both the
    isolated process group and a procfs descendant snapshot receive TERM/KILL.
    """

    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    if grace_seconds < 0:
        raise ValueError("grace_seconds must be non-negative")
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_prefix.with_suffix(".stdout.log")
    stderr_path = log_prefix.with_suffix(".stderr.log")
    evidence_path = log_prefix.with_suffix(".deadline.json")

    t0 = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr_handle:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            start_new_session=True,
            text=True,
        )
        deadline = t0 + timeout
        completed = _wait_until_exit(
            process,
            deadline=deadline,
            poll_seconds=poll_seconds,
        )
        timed_out = not completed
        if timed_out:
            _signal_process_tree(process, signal.SIGTERM)
            completed = _wait_until_exit(
                process,
                deadline=time.monotonic() + grace_seconds,
                poll_seconds=poll_seconds,
            )
            if not completed:
                _signal_process_tree(process, signal.SIGKILL)

        try:
            return_code = process.wait(timeout=max(1.0, grace_seconds + 1.0))
        except subprocess.TimeoutExpired:
            _signal_process_tree(process, signal.SIGKILL)
            return_code = process.wait(timeout=max(1.0, grace_seconds + 1.0))

    t1 = time.monotonic()
    stdout = _read_log(stdout_path)
    stderr = _read_log(stderr_path)
    payload = {
        "schema": "mementohls-file-backed-deadline-v1",
        "command": [str(item) for item in command],
        "cwd": str(cwd),
        "pid": process.pid,
        "process_group": process.pid,
        "timeout_seconds": timeout,
        "grace_seconds": grace_seconds,
        "timed_out": timed_out,
        "return_code": return_code,
        "execution_time_seconds": t1 - t0,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    temporary = evidence_path.with_name(
        f".{evidence_path.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, evidence_path)
    return DeadlineProcessResult(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        t0=t0,
        t1=t1,
        execution_time=t1 - t0,
        timed_out=timed_out,
    )
