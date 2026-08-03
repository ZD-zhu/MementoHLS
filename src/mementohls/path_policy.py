from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_PROTECTED_ROOT = Path("/home/xjzhu/HLS/DAC2027")


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def is_within(path: Path, root: Path) -> bool:
    value = _resolved(path)
    parent = _resolved(root)
    return value == parent or parent in value.parents


def validate_write_target(
    path: Path,
    *,
    workspace_root: Path,
    protected_roots: Iterable[Path] = (DEFAULT_PROTECTED_ROOT,),
    label: str = "write target",
) -> Path:
    value = _resolved(path)
    allowed = _resolved(workspace_root)
    if not is_within(value, allowed):
        raise ValueError(
            f"{label} must stay inside the isolated workspace {allowed}: {value}"
        )
    for protected in protected_roots:
        if is_within(value, protected):
            raise ValueError(
                f"{label} resolves inside protected source {protected}: {value}"
            )
    return value


def validate_experiment_paths(
    *,
    repo_root: Path,
    output_dir: Path,
    workspace_root: Path,
    protected_roots: Iterable[Path] = (DEFAULT_PROTECTED_ROOT,),
) -> dict[str, str]:
    workspace = _resolved(workspace_root)
    repo = validate_write_target(
        repo_root,
        workspace_root=workspace,
        protected_roots=protected_roots,
        label="repository root",
    )
    output = validate_write_target(
        output_dir,
        workspace_root=workspace,
        protected_roots=protected_roots,
        label="experiment output",
    )
    return {
        "workspace_root": str(workspace),
        "repo_root": str(repo),
        "output_dir": str(output),
        "protected_roots": ",".join(
            str(_resolved(path)) for path in protected_roots
        ),
    }
