#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mementohls.compat import enable_python310_hls_eval
from mementohls.datasets import (
    DATASET_BENCH4HLS,
    DATASET_HLS_EVAL,
    load_bench4hls_cases,
)
from mementohls.models import (
    FORMAL_RUN_MODES,
    HLS_EVAL_VERIFICATION_MODES,
    atomic_write_json,
    file_sha256,
)
from mementohls.orchestrator import _command_version, _dataset_manifest, _hls_eval_source_manifest, _tree_sha256
from mementohls.provenance import python_environment_manifest


def junit(path: Path, required: str | None = None) -> dict[str, object]:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failures = [item for item in cases if item.find("failure") is not None or item.find("error") is not None]
    named = [item for item in cases if item.get("name") == required] if required else []
    required_pass = not required or bool(named) and all(item.find("failure") is None and item.find("error") is None and item.find("skipped") is None for item in named)
    if failures or not required_pass:
        raise SystemExit(f"JUnit evidence failed: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "tests": len(cases), "required": required, "required_passed": required_pass}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--hls-eval-root", type=Path, default=Path("/home/xjzhu/HLS/DAC_test/datasets/Hls-Eval"))
    parser.add_argument(
        "--dataset-kind",
        choices=(DATASET_HLS_EVAL, DATASET_BENCH4HLS),
        default=DATASET_HLS_EVAL,
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/xjzhu/HLS/DAC_test/datasets/Hls-Eval/hls_eval_data"))
    parser.add_argument("--expected-case-count", type=int)
    parser.add_argument(
        "--bench4hls-frozen-epoch",
        default="@2027-01-01 00:00:00",
    )
    parser.add_argument("--vitis-dir", type=Path, default=Path("/tools/Xilinx/Vitis_HLS/2024.2"))
    parser.add_argument("--unit-junit", type=Path, required=True)
    parser.add_argument("--vitis-junit", type=Path, required=True)
    parser.add_argument("--token-junit", type=Path)
    parser.add_argument("--development-gate", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enable_python310_hls_eval(args.hls_eval_root)
    from hls_eval.data import BenchmarkCase, find_benchmark_case_dirs
    from hls_eval.tools import build_vitis_hls_env

    expected_cases = args.expected_case_count or (
        170 if args.dataset_kind == DATASET_BENCH4HLS else 94
    )
    if args.dataset_kind == DATASET_BENCH4HLS:
        cases = list(load_bench4hls_cases(args.dataset_root))
    else:
        cases = [
            BenchmarkCase(path, name=path.name)
            for path in find_benchmark_case_dirs(args.dataset_root)
        ]
        cases.sort(key=lambda item: item.name)
    if len(cases) != expected_cases:
        raise SystemExit(
            f"expected {expected_cases} {args.dataset_kind} cases, found {len(cases)}"
        )
    dataset, dataset_sha = _dataset_manifest(cases)
    hls_eval, hls_eval_sha = _hls_eval_source_manifest(args.hls_eval_root)
    vitis_executable = args.vitis_dir / "bin" / "vitis_hls"
    version_output = _command_version(
        [str(vitis_executable), "-version"],
        env=build_vitis_hls_env(args.vitis_dir),
    )
    unit = junit(args.unit_junit)
    vitis_required = (
        "test_bench4hls_prob001_passes_csim_and_synthesis"
        if args.dataset_kind == DATASET_BENCH4HLS
        else "test_synthetic_kernel_passes_csim_and_synthesis"
    )
    token_required = (
        "test_bench4hls_round0_and_memory_prompts_fit_native_context"
        if args.dataset_kind == DATASET_BENCH4HLS
        else "test_all_round0_and_worst_case_memory_prompts_fit_native_context"
    )
    vitis = junit(args.vitis_junit, vitis_required)
    token = junit(args.token_junit, token_required) if args.token_junit else None
    development_gate = json.loads(
        args.development_gate.read_text(encoding="utf-8")
    )
    if (
        development_gate.get("status") != "pass"
        or not development_gate.get("required_before_formal_freeze")
    ):
        raise SystemExit("both fixed-seed development gates must pass before freeze")
    payload = {
        "status": "pass",
        "iteration_id": args.iteration_id,
        "benchmark_status": "test-informed development benchmark",
        "source_tree_sha256": _tree_sha256(args.repo.resolve()),
        "ercl": {
            "core_sha256": file_sha256(args.repo / "memorybank" / "ercl_generic.yaml"),
            "primary_sha256": file_sha256(args.repo / "memorybank" / "ercl_primary.yaml"),
            "all_rules_sha256": file_sha256(args.repo / "memorybank" / "ercl_test_informed_all.yaml"),
            "core_count": 60,
            "primary_count": 71,
            "all_rules_count": 81,
        },
        "dataset": {
            "kind": args.dataset_kind,
            "manifest_sha256": dataset_sha,
            "case_count": expected_cases,
            "reference_kernel_excluded": True,
            "testbench_hidden_from_prompt": True,
            "bench4hls_reference_design_never_read": (
                args.dataset_kind == DATASET_BENCH4HLS
            ),
        },
        "hls_eval": {**hls_eval, "source_tree_sha256": hls_eval_sha},
        "python_environment": python_environment_manifest(),
        "vitis": {"directory": str(args.vitis_dir.resolve()), "version_output": version_output, "executable_sha256": file_sha256(vitis_executable)},
        "preregistration": {"schedule_path": str(args.schedule.resolve()), "schedule_sha256": file_sha256(args.schedule)},
        "tests": {
            "unit_status": "pass",
            "vitis_smoke_status": "pass",
            "live_token_status": "pass" if token else "not_run",
            "unit": unit,
            "vitis_smoke": vitis,
            "live_token": token,
            "development_gate_status": "pass",
            "development_gate": {
                "path": str(args.development_gate.resolve()),
                "sha256": file_sha256(args.development_gate),
            },
            "reference_kernel_read_or_prompted": False,
        },
        "fixed_protocol": {
            "served_model": "meta-llama/Llama-3-8b-chat-hf",
            "context_limit": 8192,
            "samples": 5,
            "seeds": [0, 1, 2, 3, 4],
            "temperature": 0.7,
            "max_repair_rounds": 5,
            "max_tokens": 4096,
            "bench4hls_frozen_epoch": (
                args.bench4hls_frozen_epoch
                if args.dataset_kind == DATASET_BENCH4HLS
                else None
            ),
            "modes": [
                item.value
                for item in (
                    FORMAL_RUN_MODES
                    if args.dataset_kind == DATASET_BENCH4HLS
                    else HLS_EVAL_VERIFICATION_MODES
                )
            ],
            "fpga_part": "xczu9eg-ffvb1156-2-e",
            "clock_ns": 5.0,
            "workers": {"design": 16, "llm": 8, "csim": 12, "synth": 8},
            "resource_profile": {
                "name": "v80-short-run-coexistence",
                "nice": 10,
                "ionice_class": 2,
                "ionice_level": 7,
                "start_cpu_max": 45.0,
                "start_load_max": 96.0,
                "start_memory_available_gib_min": 380.0,
                "pause_cpu_max": 70.0,
                "pause_load_max": 120.0,
                "pause_memory_available_gib_min": 350.0,
                "mementohls_logical_cpu_cap": 80,
                "mementohls_rss_gib_cap": 300,
                "vitis_threads_per_task": 2,
                "prevalidated_window_max_age_seconds": 600,
                "current_ceiling_recheck": True,
            },
            "timeouts_seconds": {"llm": 600.0, "csim": 360.0, "synth": 360.0},
            "same_candidate_success": True,
            "qor_selection": False,
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
