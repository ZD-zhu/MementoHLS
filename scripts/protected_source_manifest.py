#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable


DEFAULT_RELATIVE_ROOTS = (
    "MementoHLS/main.py",
    "MementoHLS/src",
    "MementoHLS/configs",
    "MementoHLS/memorybank",
    "MementoHLS/scripts",
    "MementoHLS/tests",
    "MementoHLS/docs",
    "result/final",
    "result/preregistration",
)
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}


def _files(root: Path, relative_roots: tuple[str, ...]) -> Iterable[Path]:
    seen: set[Path] = set()
    for relative in relative_roots:
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"manifest input escapes protected root: {relative}")
        candidates = [target] if target.is_file() else (
            sorted(path for path in target.rglob("*") if path.is_file())
            if target.is_dir()
            else []
        )
        for path in candidates:
            if path in seen:
                continue
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            seen.add(path)
            yield path


def build_manifest(
    root: Path,
    relative_roots: tuple[str, ...] = DEFAULT_RELATIVE_ROOTS,
) -> dict[str, object]:
    protected = root.resolve()
    records = []
    digest = hashlib.sha256()
    for path in sorted(_files(protected, relative_roots)):
        content = path.read_bytes()
        stat = path.stat()
        relative = path.relative_to(protected).as_posix()
        file_sha = hashlib.sha256(content).hexdigest()
        record = {
            "path": relative,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": file_sha,
        }
        records.append(record)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
    return {
        "schema_version": "dac-test-protected-source-manifest-1",
        "protected_root": str(protected),
        "relative_roots": list(relative_roots),
        "file_count": len(records),
        "manifest_sha256": digest.hexdigest(),
        "files": records,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if args.command == "build":
        output = args.output.resolve()
        if output == root or root in output.parents:
            raise SystemExit("manifest output must be outside the protected root")
        payload = build_manifest(root)
        payload["created_at_unix"] = time.time()
        _atomic_json(output, payload)
        print(
            json.dumps(
                {
                    "status": "built",
                    "file_count": payload["file_count"],
                    "manifest_sha256": payload["manifest_sha256"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return

    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    relative_roots = tuple(expected["relative_roots"])
    actual = build_manifest(root, relative_roots)
    differences = []
    if actual["manifest_sha256"] != expected.get("manifest_sha256"):
        differences.append("manifest_sha256")
    if actual["file_count"] != expected.get("file_count"):
        differences.append("file_count")
    result = {
        "status": "pass" if not differences else "fail",
        "differences": differences,
        "expected_manifest_sha256": expected.get("manifest_sha256"),
        "actual_manifest_sha256": actual["manifest_sha256"],
        "file_count": actual["file_count"],
    }
    print(json.dumps(result, sort_keys=True))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
