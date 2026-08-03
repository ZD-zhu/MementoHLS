#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path

from mementohls.datasets import (
    DATASET_BENCH4HLS,
    DATASET_HLS_EVAL,
    SUPPORTED_DATASETS,
)
from mementohls.models import HLS_EVAL_VERIFICATION_MODES, RunMode
from mementohls.orchestrator import ExperimentConfig, ExperimentRunner
from mementohls.path_policy import DEFAULT_PROTECTED_ROOT, validate_experiment_paths
from mementohls.reporting import analyze_experiment, summarize_mode


ROOT = Path(__file__).resolve().parent


@contextmanager
def exclusive_mode_lock(output_dir: Path, mode: str):
    """Refuse a second writer for the same iteration/mode."""
    lock_dir = output_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{mode}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"Another writer holds {lock_path}: {owner}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.getfqdn(),
                    "mode": mode,
                },
                sort_keys=True,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MementoHLS: executable, falsifiable memory for HLS repair")
    parser.add_argument("--mode", choices=[item.value for item in RunMode])
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--dataset-kind",
        choices=SUPPORTED_DATASETS,
        default=DATASET_HLS_EVAL,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/xjzhu/HLS/DAC_test/datasets/Hls-Eval/hls_eval_data"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=ROOT.parent,
        help="Every mutable experiment path must resolve below this directory.",
    )
    parser.add_argument(
        "--protected-source-root",
        type=Path,
        default=DEFAULT_PROTECTED_ROOT,
        help="Immutable source tree that must never receive experiment writes.",
    )
    parser.add_argument(
        "--hls-eval-root",
        type=Path,
        default=Path("/home/xjzhu/HLS/DAC_test/datasets/Hls-Eval"),
    )
    parser.add_argument("--rules", type=Path, help="Override the mode-selected ERCL bank (development only)")
    parser.add_argument("--vitis-dir", type=Path, default=Path("/tools/Xilinx/Vitis_HLS/2024.2"))
    parser.add_argument("--base-url", default="http://172.16.120.87:8000/v1")
    parser.add_argument("--model", default="meta-llama/Llama-3-8b-chat-hf")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-repair-rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=None,
        help=(
            "Defaults to 94 for HLS-Eval and 170 for Bench4HLS; set to 0 "
            "only for explicit --case/--limit development smoke tests"
        ),
    )
    parser.add_argument(
        "--bench4hls-frozen-epoch",
        default="@2027-01-01 00:00:00",
        help="Frozen libfaketime epoch used only by Bench4HLS TB executables",
    )
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--deployment-binding", type=Path)
    parser.add_argument("--vllm-run-metadata", type=Path)
    parser.add_argument("--inference-server", default="A100")
    parser.add_argument("--fpga-part", default="xczu9eg-ffvb1156-2-e")
    parser.add_argument("--clock-ns", type=float, default=5.0)
    parser.add_argument("--design-workers", type=int, default=16)
    parser.add_argument("--llm-workers", type=int, default=8)
    parser.add_argument("--csim-workers", type=int, default=12)
    parser.add_argument("--synth-workers", type=int, default=8)
    parser.add_argument("--llm-timeout", type=float, default=600.0)
    parser.add_argument("--csim-timeout", type=float, default=360.0)
    parser.add_argument("--synth-timeout", type=float, default=360.0)
    parser.add_argument("--review-infrastructure-retries", type=int, default=2)
    parser.add_argument("--runtime-start-cpu-max", type=float, default=45.0)
    parser.add_argument("--runtime-start-load-max", type=float, default=96.0)
    parser.add_argument("--runtime-pause-cpu-max", type=float, default=70.0)
    parser.add_argument("--runtime-pause-load-max", type=float, default=120.0)
    parser.add_argument(
        "--runtime-start-memory-available-gib-min",
        type=float,
        default=380.0,
    )
    parser.add_argument(
        "--runtime-pause-memory-available-gib-min",
        type=float,
        default=350.0,
    )
    parser.add_argument(
        "--runtime-process-tree-rss-gib-max",
        type=float,
        default=300.0,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--round0-source", type=Path)
    parser.add_argument("--causal-replay-source", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iteration-id", default="development")
    parser.add_argument("--formal", action="store_true", help="Enforce the frozen DAC2027 protocol")
    parser.add_argument("--block-id", type=int, help="Frozen schedule block for a formal stratified run")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Development-only: defer aggregation to analyze_development_matrix.py",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.analyze_only:
        result = analyze_experiment(args.output_dir)
        print(f"all_targets_met={result['all_targets_met']}")
        print(f"all_statistical_checks_met={result['all_statistical_checks_met']}")
        return
    if args.mode is None:
        raise SystemExit("--mode is required unless --analyze-only is used")
    validate_experiment_paths(
        repo_root=ROOT,
        output_dir=args.output_dir,
        workspace_root=args.workspace_root,
        protected_roots=(args.protected_source_root,),
    )
    if args.samples != len(args.seeds):
        raise SystemExit("--samples must equal the number of --seeds")
    if args.formal and args.skip_summary:
        raise SystemExit("--skip-summary is development-only")
    if not 0 <= args.max_repair_rounds <= 5:
        raise SystemExit("--max-repair-rounds must be in [0, 5]")

    mode = RunMode(args.mode)
    core_rules = ROOT / "memorybank" / "ercl_generic.yaml"
    primary_rules = ROOT / "memorybank" / "ercl_primary.yaml"
    all_rules = ROOT / "memorybank" / "ercl_test_informed_all.yaml"
    selected_rules = (
        all_rules
        if mode.uses_all_rules
        else core_rules
        if mode.uses_core_rules
        else primary_rules
    )
    rules_path = args.rules or selected_rules
    formal_run = args.formal
    if formal_run and not mode.formal_enabled:
        raise SystemExit(
            f"{mode.value} is a retired v11 ablation and cannot run in DAC_test formal experiments"
        )
    if (
        formal_run
        and args.dataset_kind == DATASET_HLS_EVAL
        and mode not in HLS_EVAL_VERIFICATION_MODES
    ):
        raise SystemExit(
            f"{mode.value} is not part of the five-mode HLS-Eval verification matrix"
        )
    expected_rules = selected_rules
    if formal_run and mode.uses_ercl and rules_path.resolve() != expected_rules.resolve():
        raise SystemExit(
            f"Formal {mode.value} requires frozen ERCL bank {expected_rules}; "
            "custom --rules is development-only"
        )

    expected_case_count = args.expected_case_count
    if expected_case_count is None:
        expected_case_count = (
            170 if args.dataset_kind == DATASET_BENCH4HLS else 94
        )

    config = ExperimentConfig(
        mode=mode,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dataset_kind=args.dataset_kind,
        hls_eval_root=args.hls_eval_root,
        repo_root=ROOT,
        rules_path=rules_path,
        vitis_dir=args.vitis_dir,
        base_url=args.base_url,
        model=args.model,
        samples=args.samples,
        temperature=args.temperature,
        seeds=tuple(args.seeds),
        max_repair_rounds=args.max_repair_rounds,
        max_tokens=args.max_tokens,
        context_limit=args.context_limit,
        expected_case_count=(
            None if expected_case_count == 0 else expected_case_count
        ),
        freeze_manifest=args.freeze_manifest,
        deployment_binding=args.deployment_binding,
        vllm_run_metadata=args.vllm_run_metadata,
        inference_server=args.inference_server,
        fpga_part=args.fpga_part,
        clock_ns=args.clock_ns,
        design_workers=args.design_workers,
        llm_workers=args.llm_workers,
        csim_workers=args.csim_workers,
        synth_workers=args.synth_workers,
        llm_timeout=args.llm_timeout,
        csim_timeout=args.csim_timeout,
        synth_timeout=args.synth_timeout,
        review_infrastructure_retries=args.review_infrastructure_retries,
        runtime_start_cpu_max=args.runtime_start_cpu_max,
        runtime_start_load_max=args.runtime_start_load_max,
        runtime_pause_cpu_max=args.runtime_pause_cpu_max,
        runtime_pause_load_max=args.runtime_pause_load_max,
        runtime_start_memory_available_gib_min=(
            args.runtime_start_memory_available_gib_min
        ),
        runtime_pause_memory_available_gib_min=(
            args.runtime_pause_memory_available_gib_min
        ),
        runtime_process_tree_rss_gib_max=(
            args.runtime_process_tree_rss_gib_max
        ),
        resume=args.resume,
        round0_source=args.round0_source,
        causal_replay_source=args.causal_replay_source,
        case_names=tuple(args.case),
        limit=args.limit,
        iteration_id=args.iteration_id,
        formal_protocol=args.formal,
        block_id=args.block_id,
        bench4hls_frozen_epoch=args.bench4hls_frozen_epoch,
    )
    with exclusive_mode_lock(config.output_dir, config.mode.value):
        runner = ExperimentRunner(config)
        try:
            runner.run()
            if not args.skip_summary:
                summary = summarize_mode(config.output_dir / config.mode.value)
                print(summary["metrics"])
        finally:
            runner.close()


if __name__ == "__main__":
    main()
