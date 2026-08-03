#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mementohls.deployment_evidence import (
    DeploymentEvidenceError,
    import_stop_evidence,
    validate_stop_evidence,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Import or validate local vLLM stop evidence. The source must already "
            "be on V80; this command never stores credentials and performs no SSH, "
            "network, GPU, vLLM, or Vitis operation."
        )
    )
    commands = value.add_subparsers(dest="command", required=True)
    copy = commands.add_parser(
        "import", help="validate source, copy only the allowlist, validate destination"
    )
    copy.add_argument("--source-stop-dir", type=Path, required=True)
    copy.add_argument("--destination", type=Path, required=True)
    copy.add_argument("--start-manifest", type=Path, required=True)

    validate = commands.add_parser("validate", help="validate imported stop evidence")
    validate.add_argument("--stop-dir", type=Path, required=True)
    validate.add_argument("--start-manifest", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "import":
            attestation = import_stop_evidence(
                args.source_stop_dir,
                args.destination,
                args.start_manifest,
            )
            stop_dir = args.destination.resolve()
            action = "imported-and-validated"
        else:
            attestation = validate_stop_evidence(
                args.stop_dir,
                args.start_manifest,
            )
            stop_dir = args.stop_dir.resolve()
            action = "validated"
    except DeploymentEvidenceError as exc:
        raise SystemExit(f"stop evidence rejected: {exc}") from exc

    print(
        json.dumps(
            {
                "status": action,
                "stop_dir": str(stop_dir),
                **attestation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
