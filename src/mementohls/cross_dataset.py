from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import atomic_write_json


REQUIRED_SUMMARY_KEYS = ("metrics", "cost", "case_count", "trajectory_count")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("modes"), list) or not isinstance(value.get("summaries"), dict):
        raise ValueError(f"invalid analysis schema: {path}")
    missing = [
        mode
        for mode in value["modes"]
        if mode not in value["summaries"]
        or any(key not in value["summaries"][mode] for key in REQUIRED_SUMMARY_KEYS)
    ]
    if missing:
        raise ValueError(f"incomplete summaries in {path}: {missing}")
    return value


def _direction(value: float, *, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def synthesize_datasets(
    inputs: Iterable[tuple[str, Path]],
) -> dict[str, Any]:
    loaded: list[tuple[str, Path, dict[str, Any]]] = []
    labels: set[str] = set()
    for label, path in inputs:
        if label in labels:
            raise ValueError(f"duplicate dataset label: {label}")
        labels.add(label)
        loaded.append((label, path, _load(path)))
    if len(loaded) < 2:
        raise ValueError("cross-dataset synthesis requires at least two analyses")

    reference_modes = loaded[0][2]["modes"]
    for label, _, analysis in loaded[1:]:
        if analysis["modes"] != reference_modes:
            raise ValueError(f"mode order mismatch for {label}")

    datasets: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    ordering_rows: list[dict[str, Any]] = []
    factorial_rows: list[dict[str, Any]] = []

    for label, path, analysis in loaded:
        audit = analysis.get("experiment_audit") or {}
        datasets.append({
            "dataset": label,
            "analysis_path": str(path),
            "dataset_kind": audit.get("dataset_kind"),
            "task_count": audit.get("task_count"),
            "trajectory_count_per_mode": audit.get("expected_trajectory_count_per_mode"),
            "audit_passed": bool(audit.get("passed")),
            "all_targets_met": bool(analysis.get("all_targets_met")),
            "all_preregistered_orders_met": bool(
                analysis.get("all_preregistered_orders_met")
            ),
            "all_statistical_checks_met": bool(
                analysis.get("all_statistical_checks_met")
            ),
        })
        for mode in reference_modes:
            summary = analysis["summaries"][mode]
            for metric, value in summary["metrics"].items():
                metric_rows.append({
                    "dataset": label,
                    "mode": mode,
                    "metric": metric,
                    "value": float(value),
                    "task_count": int(summary["case_count"]),
                    "trajectory_count": int(summary["trajectory_count"]),
                })
            for key, value in summary["cost"].items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    cost_rows.append({
                        "dataset": label,
                        "mode": mode,
                        "cost_metric": key,
                        "value": float(value),
                    })
        for row in analysis.get("paired_comparisons") or []:
            item = {"dataset": label, **row}
            comparison_rows.append(item)
        for row in analysis.get("targets") or []:
            target_rows.append({"dataset": label, **row})
        for row in analysis.get("ordering") or []:
            ordering_rows.append({"dataset": label, **row})
        for row in analysis.get("factorial_effects") or []:
            factorial_rows.append({"dataset": label, **row})

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[(str(row["comparison"]), str(row["metric"]))].append(row)
    consistency_rows: list[dict[str, Any]] = []
    for (comparison, metric), rows in sorted(grouped.items()):
        directions = [_direction(float(row["difference"])) for row in rows]
        ci_lows = [row.get("ci95_low") for row in rows]
        consistency_rows.append({
            "comparison": comparison,
            "metric": metric,
            "dataset_count": len(rows),
            "directions": "|".join(
                f"{row['dataset']}:{direction}"
                for row, direction in zip(rows, directions)
            ),
            "same_direction": len(set(directions)) == 1,
            "all_positive": all(direction == "positive" for direction in directions),
            "all_ci95_low_gt_zero": all(
                low is not None and float(low) > 0.0 for low in ci_lows
            ),
        })

    # Deliberately no pooled pass@k: task suites retain independent denominators.
    return {
        "schema_version": 1,
        "aggregation_policy": "separate_dataset_denominators_no_pooled_pass_at_k",
        "modes": reference_modes,
        "datasets": datasets,
        "metric_rows": metric_rows,
        "cost_rows": cost_rows,
        "comparison_rows": comparison_rows,
        "consistency_rows": consistency_rows,
        "target_rows": target_rows,
        "ordering_rows": ordering_rows,
        "factorial_rows": factorial_rows,
    }


def write_cross_dataset_artifacts(
    inputs: Iterable[tuple[str, Path]],
    output: Path,
) -> dict[str, Any]:
    payload = synthesize_datasets(inputs)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "cross_dataset_summary.json", payload)
    for key, filename in [
        ("metric_rows", "cross_dataset_metrics.csv"),
        ("cost_rows", "cross_dataset_costs.csv"),
        ("comparison_rows", "cross_dataset_comparisons.csv"),
        ("consistency_rows", "cross_dataset_consistency.csv"),
        ("target_rows", "cross_dataset_targets.csv"),
        ("ordering_rows", "cross_dataset_ordering.csv"),
        ("factorial_rows", "cross_dataset_factorial_effects.csv"),
    ]:
        _write_csv(output / filename, payload[key])
    return payload
