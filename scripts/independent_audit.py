from __future__ import annotations

import argparse
from pathlib import Path

from mementohls.independent_audit import audit_run, write_audit
from mementohls.models import RunMode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, default=5)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--require-analysis-strict-json", action="store_true")
    parser.add_argument(
        "--runtime-deviation-manifest",
        type=Path,
        action="append",
        default=[],
    )
    args = parser.parse_args()
    result, metrics = audit_run(
        args.run_root.resolve(),
        modes=[mode.value for mode in RunMode],
        expected_tasks=args.expected_tasks,
        expected_samples=args.expected_samples,
        analysis_path=args.analysis.resolve() if args.analysis else None,
        require_analysis_strict_json=args.require_analysis_strict_json,
        runtime_deviation_manifests=[
            path.resolve() for path in args.runtime_deviation_manifest
        ],
    )
    write_audit(result, metrics, args.output.resolve())
    print(
        f"{result['status']}: {result['observed']['total_trajectories']} "
        f"trajectories, {result['data_quality']['error_count']} errors"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
