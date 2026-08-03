#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mementohls.resource_gate import validate_capacity_window


def main() -> None:
    parser = argparse.ArgumentParser(
        "Validate a recent stable V80 capacity window before formal inference"
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stable-seconds", type=int, default=300)
    parser.add_argument("--max-age-seconds", type=int, default=600)
    parser.add_argument("--start-cpu-max", type=float, default=67.0)
    parser.add_argument("--start-load-max", type=float, default=124.0)
    parser.add_argument(
        "--start-memory-available-gib-min", type=float, default=380.0
    )
    args = parser.parse_args()
    rows = validate_capacity_window(
        args.log,
        stable_seconds=args.stable_seconds,
        max_age_seconds=args.max_age_seconds,
        start_cpu_max=args.start_cpu_max,
        start_load_max=args.start_load_max,
        start_memory_available_gib_min=(
            args.start_memory_available_gib_min
        ),
    )
    payload = {
        "status": "validated",
        "log": str(args.log.resolve()),
        "sample_count": len(rows),
        "stable_span_seconds": (
            rows[-1].timestamp_unix - rows[0].timestamp_unix
        ),
        "first": asdict(rows[0]),
        "last": asdict(rows[-1]),
        "policy": {
            "stable_seconds": args.stable_seconds,
            "max_age_seconds": args.max_age_seconds,
            "start_cpu_max": args.start_cpu_max,
            "start_load_max": args.start_load_max,
            "start_memory_available_gib_min": (
                args.start_memory_available_gib_min
            ),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
