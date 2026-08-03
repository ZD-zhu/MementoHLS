#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mementohls.paper_figures import generate_all


def _analysis(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator:
        path = Path(value).expanduser().resolve()
        return path.stem, path
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/analysis.json")
    return label.strip(), Path(raw_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analysis",
        action="append",
        type=_analysis,
        default=[],
        help="Repeat as LABEL=/path/to/analysis.json.",
    )
    args = parser.parse_args()
    generate_all(args.output, args.analysis)


if __name__ == "__main__":
    main()
