from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import psutil

if not hasattr(os, "getloadavg"):
    os.getloadavg = psutil.getloadavg  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CapacitySample:
    timestamp_unix: float
    cpu_percent: float
    load1: float
    decision: str
    memory_available_gib: float = 0.0
    process_tree_rss_gib: float = 0.0


def _process_tree_rss_gib() -> float:
    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return float(total) / (1024**3)


def sample_capacity(
    *,
    starting: bool,
    start_cpu_max: float = 45.0,
    start_load_max: float = 96.0,
    pause_cpu_max: float = 70.0,
    pause_load_max: float = 128.0,
    start_memory_available_gib_min: float = 0.0,
    pause_memory_available_gib_min: float = 0.0,
    process_tree_rss_gib_max: float = float("inf"),
) -> CapacitySample:
    cpu = float(psutil.cpu_percent(interval=1.0))
    load1 = float(os.getloadavg()[0])
    memory_available_gib = float(psutil.virtual_memory().available) / (1024**3)
    process_tree_rss_gib = _process_tree_rss_gib()
    if starting:
        allowed = (
            cpu < start_cpu_max
            and load1 < start_load_max
            and memory_available_gib > start_memory_available_gib_min
            and process_tree_rss_gib < process_tree_rss_gib_max
        )
        decision = "start" if allowed else "wait_start_gate"
    else:
        overloaded = (
            cpu > pause_cpu_max
            or load1 > pause_load_max
            or memory_available_gib < pause_memory_available_gib_min
            or process_tree_rss_gib > process_tree_rss_gib_max
        )
        decision = "pause_new_block" if overloaded else "continue"
    return CapacitySample(
        time.time(),
        cpu,
        load1,
        decision,
        memory_available_gib,
        process_tree_rss_gib,
    )


def wait_for_capacity(
    log_path: Path,
    *,
    stable_seconds: int = 300,
    poll_seconds: int = 15,
    sleep: Callable[[float], None] = time.sleep,
    immediate_if_initially_allowed: bool = False,
    continuation_gate: bool = False,
    start_cpu_max: float = 45.0,
    start_load_max: float = 96.0,
    pause_cpu_max: float = 70.0,
    pause_load_max: float = 128.0,
    start_memory_available_gib_min: float = 0.0,
    pause_memory_available_gib_min: float = 0.0,
    process_tree_rss_gib_max: float = float("inf"),
) -> list[CapacitySample]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[CapacitySample] = []
    stable_since: float | None = None
    recovering = False
    while True:
        value = sample_capacity(
            starting=not continuation_gate or recovering,
            start_cpu_max=start_cpu_max,
            start_load_max=start_load_max,
            pause_cpu_max=pause_cpu_max,
            pause_load_max=pause_load_max,
            start_memory_available_gib_min=start_memory_available_gib_min,
            pause_memory_available_gib_min=pause_memory_available_gib_min,
            process_tree_rss_gib_max=process_tree_rss_gib_max,
        )
        samples.append(value)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(value), sort_keys=True) + "\n")
        if continuation_gate and not recovering:
            if value.decision == "continue":
                return samples
            recovering = True
            stable_since = None
            sleep(float(poll_seconds))
            continue
        if (
            immediate_if_initially_allowed
            and len(samples) == 1
            and value.decision == "start"
        ):
            return samples
        if value.decision == "start":
            stable_since = stable_since or value.timestamp_unix
            if value.timestamp_unix - stable_since >= stable_seconds:
                return samples
        else:
            stable_since = None
        sleep(float(poll_seconds))


def validate_capacity_window(
    log_path: Path,
    *,
    stable_seconds: int = 300,
    max_age_seconds: int = 600,
    start_cpu_max: float = 67.0,
    start_load_max: float = 124.0,
    start_memory_available_gib_min: float = 0.0,
    now: float | None = None,
) -> list[CapacitySample]:
    """Validate a recent trailing stable window before a GPU deployment is used."""

    if not log_path.is_file() or log_path.is_symlink():
        raise ValueError(f"capacity evidence is missing or is a symlink: {log_path}")
    rows: list[CapacitySample] = []
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            row = CapacitySample(
                timestamp_unix=float(value["timestamp_unix"]),
                cpu_percent=float(value["cpu_percent"]),
                load1=float(value["load1"]),
                decision=str(value["decision"]),
                memory_available_gib=float(
                    value.get("memory_available_gib", float("inf"))
                ),
                process_tree_rss_gib=float(
                    value.get("process_tree_rss_gib", 0.0)
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid capacity evidence line {line_number}") from exc
        rows.append(row)
    if not rows:
        raise ValueError("capacity evidence is empty")
    trailing: list[CapacitySample] = []
    for row in reversed(rows):
        if (
            row.decision != "start"
            or row.cpu_percent >= start_cpu_max
            or row.load1 >= start_load_max
            or row.memory_available_gib <= start_memory_available_gib_min
        ):
            break
        trailing.append(row)
    trailing.reverse()
    if not trailing:
        raise ValueError("capacity evidence has no trailing allowed sample")
    if any(
        trailing[index].timestamp_unix <= trailing[index - 1].timestamp_unix
        for index in range(1, len(trailing))
    ):
        raise ValueError("capacity evidence timestamps are not strictly increasing")
    span = trailing[-1].timestamp_unix - trailing[0].timestamp_unix
    if span < stable_seconds:
        raise ValueError(
            f"capacity evidence stable span {span:.3f}s is below {stable_seconds}s"
        )
    current = time.time() if now is None else float(now)
    age = current - trailing[-1].timestamp_unix
    if age < -5.0 or age > max_age_seconds:
        raise ValueError(
            f"capacity evidence age {age:.3f}s is outside [-5,{max_age_seconds}]"
        )
    return trailing
