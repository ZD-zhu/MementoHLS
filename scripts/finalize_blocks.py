#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mementohls.models import atomic_write_json, file_sha256
from mementohls.reporting import MODES, load_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    expected_blocks = {int(item["block_id"]) for item in schedule["blocks"]}
    expected_tasks = int(schedule["dataset_case_count"])
    dataset_kind = str(schedule["dataset_kind"])
    expected_trajectories = expected_tasks * 5
    for mode in MODES[1:]:
        root = args.iteration_root / mode
        metadata = sorted(root.glob("run_metadata_block__*.json"))
        errors = sorted(root.glob("run_errors_block__*.json"))
        ids = {int(path.stem.rsplit("__", 1)[1]) for path in metadata}
        if ids != expected_blocks:
            raise SystemExit(f"{mode}: block metadata mismatch {sorted(ids)}")
        if len(errors) != len(expected_blocks) or any(json.loads(path.read_text()) != [] for path in errors):
            raise SystemExit(f"{mode}: missing or non-empty block error ledger")
        trajectories = load_trajectories(root)
        if (
            len(trajectories) != expected_trajectories
            or len({item["benchmark_case_name"] for item in trajectories})
            != expected_tasks
            or any(item.get("dataset_kind") != dataset_kind for item in trajectories)
        ):
            raise SystemExit(
                f"{mode}: incomplete {expected_tasks}x5 {dataset_kind} trajectory set"
            )
        manifest = {
            "mode": mode,
            "dataset_kind": dataset_kind,
            "task_count": expected_tasks,
            "status": "all_stratified_blocks_complete",
            "trajectory_count": len(trajectories),
            "block_count": len(expected_blocks),
            "schedule_sha256": file_sha256(args.schedule),
            "block_metadata": [{"path": str(path), "sha256": file_sha256(path)} for path in metadata],
        }
        atomic_write_json(root / "run_metadata.json", manifest)
        atomic_write_json(root / "run_errors.json", [])
        print(f"finalized {mode}: {len(trajectories)} trajectories")


if __name__ == "__main__":
    main()
