#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mementohls.dual_report_writer import DatasetEvidence, write_combined_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the final non-pooled HLS-Eval + Bench4HLS reports."
    )
    parser.add_argument(
        "--dataset", action="append", nargs=5,
        metavar=("LABEL", "ANALYSIS_JSON", "RUN_ROOT", "AUDIT_JSON", "REPO_ROOT"),
        required=True,
    )
    parser.add_argument("--cross-dataset-json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    evidence = []
    for label, analysis_path, run_root, audit_path, repo_root in args.dataset:
        analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
        audit = None
        if audit_path != "-":
            audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        evidence.append(DatasetEvidence(
            label=label,
            analysis=analysis,
            run_root=Path(run_root),
            independent_audit=audit,
            repo_root=Path(repo_root),
        ))
    cross = None
    if args.cross_dataset_json:
        cross = json.loads(Path(args.cross_dataset_json).read_text(encoding="utf-8"))
    paths = write_combined_reports(
        evidence, Path(args.output_dir), cross_dataset=cross
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
