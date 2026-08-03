from __future__ import annotations

import argparse
import json
from pathlib import Path

from mementohls.artifact_release import build_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument(
        "--release-metadata",
        type=Path,
        help="Strict JSON containing candidate-freeze and reporting-release hashes.",
    )
    args = parser.parse_args()
    metadata = (
        json.loads(args.release_metadata.read_text(encoding="utf-8"))
        if args.release_metadata
        else {}
    )
    result = build_artifact(
        args.source.resolve(),
        args.output.resolve(),
        evidence=[path.resolve() for path in args.evidence],
        release_metadata=metadata,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
