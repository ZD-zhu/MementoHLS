#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NON_PARSE_METRICS = [
    "compile_p@1",
    "compile_p@5",
    "tb_p@1",
    "tb_p@5",
    "synth_p@1",
    "synth_p@5",
    "tb_and_synth_p@1",
    "tb_and_synth_p@5",
]


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def benchmark_checks(label: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in analysis.get("ordering") or []:
        if row.get("metric") not in NON_PARSE_METRICS:
            continue
        checks.append(
            {
                "scope": label,
                "check": (
                    f"{row.get('candidate')}>{row.get('baseline')}:"
                    f"{row.get('metric')}"
                ),
                "passed": bool(row.get("strict_order_met")),
                "value": row.get("difference"),
                "minimum": 0.0,
                "strict": True,
            }
        )
    checks.append(
        {
            "scope": label,
            "check": "plus_15pp_and_parse_guard",
            "passed": bool(analysis.get("all_targets_met")),
        }
    )
    full_zero_rows = [
        row
        for row in analysis.get("paired_comparisons") or []
        if row.get("comparison") == "mementohls-zero_shot"
        and row.get("metric") in NON_PARSE_METRICS
    ]
    checks.append(
        {
            "scope": label,
            "check": "full_vs_zero_ci95_positive",
            "passed": len(full_zero_rows) == len(NON_PARSE_METRICS)
            and all(float(row.get("ci95_low", -1.0)) > 0.0 for row in full_zero_rows),
        }
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-analysis", type=Path, required=True)
    parser.add_argument("--hls-analysis", type=Path, required=True)
    parser.add_argument("--hls-v11-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bench = read(args.bench_analysis)
    hls = read(args.hls_analysis)
    old = read(args.hls_v11_analysis)
    checks = benchmark_checks("bench4hls", bench) + benchmark_checks(
        "hls_eval", hls
    )
    current_metrics = hls["summaries"]["mementohls"]["metrics"]
    old_metrics = old["summaries"]["mementohls"]["metrics"]
    for metric in NON_PARSE_METRICS:
        delta = float(current_metrics[metric]) - float(old_metrics[metric])
        checks.append(
            {
                "scope": "hls_eval",
                "check": f"v11_noninferiority:{metric}",
                "passed": delta >= -0.01 - 1e-12,
                "value": delta,
                "minimum": -0.01,
            }
        )
    for metric, minimum in (
        ("tb_and_synth_p@1", 0.6623),
        ("tb_and_synth_p@5", 0.7879),
    ):
        value = float(current_metrics[metric])
        checks.append(
            {
                "scope": "hls_eval",
                "check": f"absolute_floor:{metric}",
                "passed": value >= minimum - 1e-12,
                "value": value,
                "minimum": minimum,
            }
        )

    payload = {
        "schema_version": "dac-test-dual-acceptance-1",
        "passed": all(bool(item["passed"]) for item in checks),
        "checks": checks,
        "failed_checks": [
            item["check"] for item in checks if not bool(item["passed"])
        ],
        "seed_or_version_splicing_allowed": False,
        "failure_policy": (
            "create a new frozen version, pass both small gates, and rerun the "
            "complete affected formal matrix"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
