#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Iterable


SOURCE_DIRS = ("src", "configs", "memorybank", "scripts", "tests")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".orig"}


def source_files(root: Path) -> Iterable[Path]:
    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            yield path


def tree_manifest(root: Path) -> dict[str, object]:
    records = []
    digest = hashlib.sha256()
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_digest,
            }
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return {
        "root": str(root.resolve()),
        "file_count": len(records),
        "tree_sha256": digest.hexdigest(),
        "files": records,
    }


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-tree-sha256")
    parser.add_argument(
        "--protected-source",
        type=Path,
        default=Path("/home/xjzhu/HLS/DAC2027"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    protected = args.protected_source.resolve()
    if output == protected or protected in output.parents:
        raise SystemExit("Refusing to write a manifest inside the protected source")
    payload = {
        "schema_version": "dac-test-workspace-manifest-1",
        "created_at_unix": time.time(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "source": tree_manifest(args.repo.resolve()),
        "protected_source": str(protected),
        "protected_baseline_tree_sha256": args.baseline_tree_sha256,
        "vitis_version": command_output(
            ["/tools/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls", "-version"]
        ),
        "git_head": command_output(
            ["git", "-C", str(args.repo.resolve()), "rev-parse", "HEAD"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "source": payload["source"]}))


if __name__ == "__main__":
    main()
