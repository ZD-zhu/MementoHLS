#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mementohls.models import atomic_write_json
from mementohls.reporting import (
    ORDERED_METRICS,
    _paired_rows,
    load_trajectories,
    summarize_mode,
)


DEFAULT_MODES = (
    "zero_shot",
    "feedback",
    "rfl",
    "ercl",
    "mementohls",
)
ORDERING = (
    ("feedback", "zero_shot"),
    ("rfl", "feedback"),
    ("ercl", "feedback"),
    ("mementohls", "rfl"),
    ("mementohls", "ercl"),
)


def _run_errors(root: Path, mode: str) -> list[Any]:
    path = root / mode / "run_errors.json"
    if not path.is_file():
        return [f"missing {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [f"invalid {path}"]


def _sample_indices(items: list[dict[str, Any]]) -> set[int]:
    return {int(item["sample_index"]) for item in items}


def _one_seed_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "compile": "pass_compile",
        "tb": "pass_tb",
        "synth": "pass_synth",
        "tb_and_synth": "pass_tb_and_synth",
    }
    metrics: dict[str, float | None] = {}
    for label, key in keys.items():
        metrics[f"{label}_p@1"] = sum(
            bool(item["final_review"][key]) for item in items
        ) / len(items)
        metrics[f"{label}_p@5"] = None
    return {
        "metrics": metrics,
        "effective_sample_count": 1,
        "p_at_5_status": "not_estimable_from_one-seed-smoke",
    }


def analyze(root: Path, modes: tuple[str, ...]) -> dict[str, Any]:
    trajectories = {
        mode: load_trajectories(root / mode, require_complete=True)
        for mode in modes
    }
    sample_sets = {
        mode: _sample_indices(items)
        for mode, items in trajectories.items()
    }
    summaries = {}
    for mode in modes:
        if sample_sets[mode] == {0, 1, 2, 3, 4}:
            summaries[mode] = summarize_mode(root / mode)
        elif sample_sets[mode] == {0}:
            summaries[mode] = _one_seed_summary(trajectories[mode])
        else:
            raise ValueError(
                f"{mode}: unsupported development sample indices "
                f"{sorted(sample_sets[mode])}"
            )
    identities = {
        mode: {
            (
                str(item["benchmark_case_name"]),
                int(item["sample_index"]),
            ): (
                item.get("seed_prompt_sha256"),
                item["rounds"][0].get("response_sha256"),
                item["rounds"][0].get("code_sha256"),
            )
            for item in items
        }
        for mode, items in trajectories.items()
    }
    common_keys = set.intersection(
        *(set(values) for values in identities.values())
    )
    identity_mismatches = [
        list(key)
        for key in sorted(common_keys)
        if len({identities[mode][key] for mode in modes}) != 1
    ]
    run_errors = {
        mode: _run_errors(root, mode)
        for mode in modes
    }
    ordering = []
    for candidate, baseline in ORDERING:
        if candidate not in summaries or baseline not in summaries:
            continue
        for metric in ORDERED_METRICS:
            if metric.startswith("parse_"):
                continue
            candidate_value = summaries[candidate]["metrics"].get(metric)
            baseline_value = summaries[baseline]["metrics"].get(metric)
            if candidate_value is None or baseline_value is None:
                continue
            delta = candidate_value - baseline_value
            ordering.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "difference": delta,
                    "strict_order_met": delta > 0,
                }
            )
    counts = {mode: len(items) for mode, items in trajectories.items()}
    expected_count = counts[modes[0]]
    integrity_errors = []
    if len(set(counts.values())) != 1:
        integrity_errors.append(f"trajectory counts differ: {counts}")
    if identity_mismatches:
        integrity_errors.append(
            f"Round 0 identity mismatches: {len(identity_mismatches)}"
        )
    for mode, errors in run_errors.items():
        if errors:
            integrity_errors.append(f"{mode}: {len(errors)} run errors")
    result = {
        "schema_version": "dac-test-development-analysis-1",
        "root": str(root.resolve()),
        "modes": list(modes),
        "trajectory_count_per_mode": counts,
        "expected_count_per_mode": expected_count,
        "summaries": summaries,
        "paired_comparisons": (
            _paired_rows(summaries)
            if all(indices == {0, 1, 2, 3, 4} for indices in sample_sets.values())
            else []
        ),
        "ordering": ordering,
        "all_strict_orders_met": bool(ordering)
        and all(row["strict_order_met"] for row in ordering),
        "round0_identity_mismatches": identity_mismatches,
        "run_errors": run_errors,
        "integrity": {
            "passed": not integrity_errors,
            "errors": integrity_errors,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, tuple(args.modes))
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "integrity": result["integrity"]["passed"],
                "all_strict_orders_met": result["all_strict_orders_met"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
