#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


NON_PARSE = {
    "compile_p@1",
    "compile_p@5",
    "tb_p@1",
    "tb_p@5",
    "synth_p@1",
    "synth_p@5",
    "tb_and_synth_p@1",
    "tb_and_synth_p@5",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-analysis", type=Path, required=True)
    parser.add_argument("--hls-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for label, path in (
        ("bench4hls", args.bench_analysis),
        ("hls_eval", args.hls_analysis),
    ):
        analysis = load(path)
        ordering = [
            row
            for row in analysis.get("ordering") or []
            if row.get("metric") in NON_PARSE
        ]
        rows.append(
            {
                "dataset": label,
                "analysis_path": str(path.resolve()),
                "analysis_sha256": sha256(path),
                "ordering_row_count": len(ordering),
                "passed": len(ordering) == 5 * len(NON_PARSE)
                and all(bool(row.get("strict_order_met")) for row in ordering),
                "failed_orderings": [
                    (
                        f"{row.get('candidate')}>{row.get('baseline')}:"
                        f"{row.get('metric')}"
                    )
                    for row in ordering
                    if not bool(row.get("strict_order_met"))
                ],
            }
        )
    payload = {
        "schema_version": "dac-test-development-gate-1",
        "status": "pass" if all(row["passed"] for row in rows) else "fail",
        "datasets": rows,
        "required_before_formal_freeze": True,
        "seed_selection_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
