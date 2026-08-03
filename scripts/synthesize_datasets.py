from __future__ import annotations

import argparse
from pathlib import Path

from mementohls.cross_dataset import write_cross_dataset_artifacts


def _dataset(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/analysis.json")
    return label.strip(), Path(raw_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize audited analyses without pooling dataset denominators."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_dataset,
        required=True,
        help="Repeat as LABEL=/path/to/analysis.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_cross_dataset_artifacts(args.dataset, args.output.resolve())
    print(
        f"wrote {len(payload['datasets'])} datasets, "
        f"{len(payload['metric_rows'])} metric rows"
    )


if __name__ == "__main__":
    main()
