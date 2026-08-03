#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mementohls.metrics import aggregate_case_metrics, per_case_metrics
from mementohls.models import atomic_create_json, file_sha256
from mementohls.orchestrator import _round0_source_manifest
from mementohls.provenance import canonical_sha256
from mementohls.reporting import ORDERED_METRICS, load_trajectories


def main() -> None:
    parser = argparse.ArgumentParser("Lock the +15pp engineering goal after fresh Zero-shot")
    parser.add_argument("--iteration-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.iteration_root.resolve()
    zero_root = root / "zero_shot"
    freeze_path = root / "protocol" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    dataset = freeze.get("dataset") or {}
    expected_tasks = int(dataset.get("case_count") or 0)
    dataset_kind = str(dataset.get("kind") or "")
    expected_trajectories = expected_tasks * 5
    trajectories = load_trajectories(zero_root)
    if (
        len(trajectories) != expected_trajectories
        or len({item["benchmark_case_name"] for item in trajectories})
        != expected_tasks
    ):
        raise SystemExit(
            "target lock requires exactly "
            f"{expected_tasks}x5 {dataset_kind} Zero-shot trajectories"
        )
    if any(item.get("dataset_kind") != dataset_kind for item in trajectories):
        raise SystemExit("target lock dataset identity mismatch")
    if any(item.get("mode") != "zero_shot" or len(item.get("rounds") or []) != 1 for item in trajectories):
        raise SystemExit("target lock source is not strict Round 0 Zero-shot")
    baseline = aggregate_case_metrics(per_case_metrics(trajectories))
    rows = []
    for metric in ORDERED_METRICS:
        target = baseline[metric] if metric.startswith("parse_") else min(1.0, baseline[metric] + 0.15)
        rows.append({"metric": metric, "fresh_zero_shot": baseline[metric], "target": target, "required_difference_pp": 100.0 * (target - baseline[metric])})
    stable = {
        "status": "locked-after-fresh-zero-shot-before-repair-modes",
        "iteration_id": root.name,
        "formula": "Parse may not fall; every other metric targets min(1, fresh_zero_shot + 0.15)",
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "round0_source_manifest": _round0_source_manifest(zero_root),
        "dataset_kind": dataset_kind,
        "task_count": expected_tasks,
        "trajectory_count": expected_trajectories,
        "rows": rows,
    }
    output = root / "protocol" / "locked_targets.json"
    payload = {**stable, "locked_at_unix": time.time(), "lock_content_sha256": canonical_sha256(stable)}
    if not atomic_create_json(output, payload):
        existing = json.loads(output.read_text(encoding="utf-8"))
        check = dict(existing)
        check.pop("locked_at_unix", None)
        digest = check.pop("lock_content_sha256", None)
        if digest != canonical_sha256(check) or check != stable:
            raise SystemExit("existing target lock differs from the fresh baseline")
        payload = existing
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
