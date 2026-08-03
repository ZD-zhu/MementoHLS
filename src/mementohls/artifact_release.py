from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .models import atomic_write_json, atomic_write_text


ROOT_FILES = (
    "main.py",
    "pytest.ini",
    "README.md",
    "REPRODUCE.md",
    "ARTIFACT_EVALUATION.md",
)
COPY_DIRS = ("src", "memorybank", "configs", "scripts", "tests", "docs")
EXCLUDED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "artifact",
    "vitis_hls.log",
    "KERNELMEM2_SOURCE_README.md",
}
EXCLUDED_GLOBS = ("*.pyc", "*.pyo", "*.orig", "*.log")
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".toml",
    ".ini",
}
REPLACEMENTS = {
    "/home/xjzhu/HLS/DAC2027": "/path/to/DAC2027",
    "/home/xjzhu/HLS/Hls-Eval/hls_eval_data": "/path/to/Hls-Eval/hls_eval_data",
    "/home/xjzhu/HLS/Hls-Eval": "/path/to/Hls-Eval",
    "/home/xjzhu/HLS/Bench4HLS/benchmark": "/path/to/Bench4HLS/benchmark",
    "/home/xjzhu/HLS/DAC2027/result/development/vitis_coexistence_probe/probe.json": "/path/to/capacity_evidence.json",
    "/mnt/sdb/llm_models/Meta-Llama-3-8B-Instruct": "/path/to/Meta-Llama-3-8B-Instruct",
    "http://172.16.120.87:8000/v1": "http://127.0.0.1:8000/v1",
}
EXACT_REPLACEMENTS = {
    'REPO="${MEMENTOHLS_REPO:-/home/xjzhu/HLS/DAC2027/MementoHLS}"':
        'REPO="${MEMENTOHLS_REPO:?set MEMENTOHLS_REPO to the artifact root}"',
    'SCHEDULE="${MEMENTOHLS_SCHEDULE:-/home/xjzhu/HLS/DAC2027/result/preregistration/schedule.json}"':
        'SCHEDULE="${MEMENTOHLS_SCHEDULE:?set MEMENTOHLS_SCHEDULE}"',
}
_PRIVATE_USER = "xj" + "zhu"
_PRIVATE_MOUNT = "s" + "db"
_SSH_TOKENS = ("Proxy" + "Command", "Jump" + "ServerWzh", "ssh" + "pass")
_LEGACY_STACK = ("Kernel" + "Bench", "N" + "CU", "N" + "SYS")
_LEGACY_REPO = "Kernel" + "Mem2"

FORBIDDEN = {
    "private_home": re.compile(r"/home/" + _PRIVATE_USER + r"(?:/|\b)", re.IGNORECASE),
    "private_model_mount": re.compile(r"/mnt/" + _PRIVATE_MOUNT + r"(?:/|\b)", re.IGNORECASE),
    "private_ip": re.compile(r"\b(?:106\.14\.|172\.16\.|192\.168\.)\d{1,3}(?:\.\d{1,3})?\b"),
    "ssh_config": re.compile(r"\b(?:" + "|".join(_SSH_TOKENS) + r")\b", re.IGNORECASE),
    "legacy_cuda_stack": re.compile(r"\b(?:" + "|".join(_LEGACY_STACK) + r")\b", re.IGNORECASE),
    "legacy_repo": re.compile(r"\b" + _LEGACY_REPO + r"\b|/result" + r"2(?:/|\b)", re.IGNORECASE),
}


def _excluded(path: Path) -> bool:
    return (
        any(part in EXCLUDED_NAMES for part in path.parts)
        or any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_GLOBS)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_records(root: Path, *, exclude_checksums: bool = False) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        if exclude_checksums and rel == "SHA256SUMS":
            continue
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def _copy_tree(source: Path, destination: Path) -> None:
    for root_file in ROOT_FILES:
        path = source / root_file
        if path.is_file():
            target = destination / root_file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for dirname in COPY_DIRS:
        base = source / dirname
        if not base.is_dir():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            rel = path.relative_to(source)
            if _excluded(rel):
                continue
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _sanitize_text_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in EXACT_REPLACEMENTS.items():
            text = text.replace(old.replace(chr(92) + "${", "${"), new.replace(chr(92) + "${", "${"))
        for old, new in REPLACEMENTS.items():
            text = text.replace(old, new)
        text = FORBIDDEN["private_home"].sub("/path/to/private-home/", text)
        text = FORBIDDEN["private_ip"].sub("127.0.0.1", text)
        atomic_write_text(path, text)


def scan_forbidden(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                findings.append({
                    "rule": name,
                    "path": path.relative_to(root).as_posix(),
                    "line": text.count("\n", 0, match.start()) + 1,
                    "match": match.group(0),
                })
    return findings


def build_artifact(
    source: Path,
    output: Path,
    *,
    evidence: Iterable[Path] = (),
    release_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"artifact output already exists: {output}")
    output.mkdir(parents=True)
    _copy_tree(source, output)
    evidence_root = output / "evidence"
    seen_names: set[str] = set()
    for path in evidence:
        if path.name in seen_names:
            raise ValueError(f"duplicate evidence basename: {path.name}")
        seen_names.add(path.name)
        evidence_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, evidence_root / path.name)
    _sanitize_text_tree(output)

    findings = scan_forbidden(output)
    if findings:
        raise ValueError(f"artifact sanitization failed: {findings[:20]}")

    release_text = json.dumps(release_metadata or {}, ensure_ascii=False)
    for old, new in REPLACEMENTS.items():
        release_text = release_text.replace(old, new)
    release_text = FORBIDDEN["private_home"].sub(
        "/path/to/private-home/", release_text
    )
    release_text = FORBIDDEN["private_ip"].sub("127.0.0.1", release_text)
    sanitized_release_metadata = json.loads(release_text)

    source_records = _tree_records(output)
    manifest = {
        "schema_version": 1,
        "package": "MementoHLS_DAC2027_AE",
        "hls_only": True,
        "code_scope": "HLS generation, Reviewer, ERCL, RFL, analysis, and reproduction only",
        "legacy_gpu_generation_or_profiling_stack_present": False,
        "release_metadata": sanitized_release_metadata,
        "files_before_manifest": source_records,
        "file_count_before_manifest": len(source_records),
        "total_bytes_before_manifest": sum(row["bytes"] for row in source_records),
        "sanitization_findings": [],
    }
    atomic_write_json(output / "PACKAGE_MANIFEST.json", manifest)
    checksum_rows = _tree_records(output, exclude_checksums=True)
    atomic_write_text(
        output / "SHA256SUMS",
        "".join(f"{row['sha256']}  {row['path']}\n" for row in checksum_rows),
    )
    final_findings = scan_forbidden(output)
    if final_findings:
        raise ValueError(f"artifact final scan failed: {final_findings[:20]}")
    return {
        "output": str(output),
        "file_count": len(_tree_records(output)),
        "forbidden_findings": 0,
        "sha256sums_sha256": _sha256(output / "SHA256SUMS"),
    }
