#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mementohls.paper_figures import generate_all
from mementohls.report_writer import write_all_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    write_all_reports(
        analysis,
        output_dir=args.output,
        run_root=args.run_root,
        repo_root=args.repo_root,
    )
    generate_all(args.output / "figures", args.analysis)


if __name__ == "__main__":
    main()

