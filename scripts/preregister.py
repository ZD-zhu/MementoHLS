#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml


BENCH4HLS_REPAIR_MODES = [
    "feedback", "rfl", "ercl", "mementohls",
    "mementohls_core_rules", "mementohls_all_rules",
]
HLS_EVAL_REPAIR_MODES = ["feedback", "rfl", "ercl", "mementohls"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind",
        choices=("hls_eval", "bench4hls"),
        default="hls_eval",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    config_path = args.repo / "configs" / "experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(args.repo / "src"))
    from mementohls.compat import enable_python310_hls_eval
    from mementohls.datasets import load_bench4hls_cases

    by_suite: dict[str, list[str]] = defaultdict(list)
    if args.dataset_kind == "bench4hls":
        repair_modes = BENCH4HLS_REPAIR_MODES
        names = [case.name for case in load_bench4hls_cases(args.dataset_root)]
        by_suite["Bench4HLS"] = names
        expected_cases = 170
        block_count = 10
    else:
        repair_modes = HLS_EVAL_REPAIR_MODES
        hls_eval_root = args.dataset_root.parent
        enable_python310_hls_eval(hls_eval_root)
        from hls_eval.data import find_benchmark_case_dirs

        for case in find_benchmark_case_dirs(args.dataset_root):
            suite = case.relative_to(args.dataset_root).parts[0]
            by_suite[suite].append(case.name)
        expected_cases = 94
        block_count = 9
    for names in by_suite.values():
        names.sort()
    cases = [
        (suite, case)
        for suite, names in sorted(by_suite.items())
        for case in names
    ]
    if len(cases) != expected_cases:
        raise SystemExit(
            f"expected {expected_cases} {args.dataset_kind} cases, found {len(cases)}"
        )
    blocks: list[list[dict[str, str]]] = [[] for _ in range(block_count)]
    for suite, names in sorted(by_suite.items()):
        for index, case in enumerate(names):
            blocks[index % block_count].append({"suite": suite, "case": case})
    rng = random.Random(args.seed)
    base_order = list(repair_modes)
    rng.shuffle(base_order)
    schedule_blocks = []
    for index, cases_in_block in enumerate(blocks):
        order = base_order[index % len(base_order):] + base_order[:index % len(base_order)]
        schedule_blocks.append({"block_id": index, "case_count": len(cases_in_block), "cases": cases_in_block, "repair_mode_order": order})
    schedule = {
        "schema_version": "dac-test-latin-block-3",
        "dataset_kind": args.dataset_kind,
        "dataset_case_count": expected_cases,
        "seed": args.seed,
        "zero_shot_runs_first": True,
        "nonzero_mode_count": len(repair_modes),
        "block_count": len(schedule_blocks),
        "suite_case_counts": {key: len(value) for key, value in sorted(by_suite.items())},
        "blocks": schedule_blocks,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "schedule.json", schedule)
    ledger = {
        "status": "preregistered development hypothesis",
        "dataset_kind": args.dataset_kind,
        "dataset_case_count": expected_cases,
        "test_informed": True,
        "held_out_claim_allowed": False,
        "config_sha256": sha256(config_path),
        "core_ercl_sha256": sha256(args.repo / "memorybank" / "ercl_generic.yaml"),
        "primary_ercl_sha256": sha256(args.repo / "memorybank" / "ercl_primary.yaml"),
        "all_rules_ercl_sha256": sha256(args.repo / "memorybank" / "ercl_test_informed_all.yaml"),
        "schedule_sha256": sha256(args.output / "schedule.json"),
        "primary_metrics": config["primary_metrics"],
        "directional_hypotheses": [
            "zero_shot < feedback", "feedback < rfl", "feedback < ercl",
            "rfl < mementohls", "ercl < mementohls",
        ],
        "optimization_goal_pp": 15,
        "failed_order_policy": "record difference and CI; any result-driven change creates a new hashed development iteration and reruns affected modes",
        "qor_used_for_selection": False,
        "reference_kernel_read_or_prompted": False,
    }
    atomic_json(args.output / "PREREGISTRATION.json", ledger)
    text = """# MementoHLS DAC2027 preregistration\n\nThis is a **test-informed development benchmark**, not a held-out generalization test.\n\nPrimary endpoints are TB&Synth p@1 and p@5 from the same candidate. The directional hypotheses are Zero-shot < Feedback, Feedback < RFL, Feedback < ERCL, and MementoHLS above both RFL and ERCL. The engineering goal is at least +15 percentage points over fresh Zero-shot where the metric is not saturated.\n\nIf an ordering fails, the failure, paired difference and confidence interval remain in the iteration ledger. Result-driven optimization is permitted only in a new hashed development iteration; candidates are never spliced across versions.\n\nThe primary ERCL bank has 71 contracts: 60 task-generic contracts plus 11 reusable algorithm-family contracts. The 60-contract core is an ablation, and the 81-contract bank adds 10 exact task contracts only as a labeled test-informed sensitivity upper bound. QoR is observational and never controls repair or selection.\n"""
    (args.output / "PREREGISTRATION.md").write_text(text, encoding="utf-8")
    print(json.dumps(ledger, sort_keys=True))


if __name__ == "__main__":
    main()
