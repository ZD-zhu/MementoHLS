#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mementohls.resource_gate import wait_for_capacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stable-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--immediate-if-allowed", action="store_true")
    parser.add_argument("--continuation-gate", action="store_true")
    parser.add_argument("--start-cpu-max", type=float, default=45.0)
    parser.add_argument("--start-load-max", type=float, default=96.0)
    parser.add_argument("--pause-cpu-max", type=float, default=70.0)
    parser.add_argument("--pause-load-max", type=float, default=128.0)
    parser.add_argument(
        "--start-memory-available-gib-min", type=float, default=380.0
    )
    parser.add_argument(
        "--pause-memory-available-gib-min", type=float, default=350.0
    )
    parser.add_argument("--process-tree-rss-gib-max", type=float, default=300.0)
    args = parser.parse_args()
    samples = wait_for_capacity(
        args.log,
        stable_seconds=args.stable_seconds,
        poll_seconds=args.poll_seconds,
        immediate_if_initially_allowed=args.immediate_if_allowed,
        continuation_gate=args.continuation_gate,
        start_cpu_max=args.start_cpu_max,
        start_load_max=args.start_load_max,
        pause_cpu_max=args.pause_cpu_max,
        pause_load_max=args.pause_load_max,
        start_memory_available_gib_min=args.start_memory_available_gib_min,
        pause_memory_available_gib_min=args.pause_memory_available_gib_min,
        process_tree_rss_gib_max=args.process_tree_rss_gib_max,
    )
    print(f"capacity gate opened after {len(samples)} samples")


if __name__ == "__main__":
    main()
