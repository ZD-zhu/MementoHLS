#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mementohls.executor import ContractActionExecutor
from mementohls.compat import enable_python310_hls_eval
from mementohls.ercl import ExecutableRepairContractLibrary
from mementohls.memory_facts import derive_memory_facts
from mementohls.models import (
    ReviewResult,
    atomic_write_json,
    file_sha256,
)
from mementohls.orchestrator import (
    _dataset_manifest,
    _long_short_match_history,
    _tree_sha256,
)
from mementohls.reviewer import HLSReviewer, ToolPools, failure_log
from mementohls.rfl import ReviewerGroundedFalsificationLedger


EXACT_CONTRACT_RULES = {
    "pp4fpga_cordic": "COMPILE_CONTRACT_CORDIC_FIXED_SHIFT",
    "dfmul": "COMPILE_CONTRACT_FLOAT64_NATIVE_MULTIPLY",
    "nw_nw": "COMPILE_CONTRACT_NEEDWUN_ROW_MAJOR_DP",
    "present": "COMPILE_CONTRACT_PRESENT80_STANDARD",
    "des": "COMPILE_CONTRACT_DES_FEISTEL_SCHEDULE",
    "df_float64_ge": "COMPILE_CONTRACT_FLOAT64_GE",
    "df_float64_le": "COMPILE_CONTRACT_FLOAT64_LE",
    "dfdiv": "COMPILE_CONTRACT_FLOAT64_NATIVE_DIVIDE",
    "aes": "COMPILE_CONTRACT_AES128_CIPHER",
    "md_grid": "COMPILE_CONTRACT_MD_GRID_BOUNDED_LJ",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        "Validate exact Long-memory HLS operators with the complete Reviewer"
    )
    value.add_argument("--source-round0-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/xjzhu/HLS/Hls-Eval/hls_eval_data"),
    )
    value.add_argument(
        "--hls-eval-root",
        type=Path,
        default=Path("/home/xjzhu/HLS/Hls-Eval"),
    )
    value.add_argument(
        "--vitis-dir",
        type=Path,
        default=Path("/tools/Xilinx/Vitis_HLS/2024.2"),
    )
    value.add_argument(
        "--rules",
        type=Path,
        default=ROOT / "memorybank/hls_generation_rules_v3.yaml",
    )
    value.add_argument("--part", default="xczu9eg-ffvb1156-2-e")
    value.add_argument("--clock-ns", type=float, default=5.0)
    value.add_argument("--samples", type=int, default=5)
    value.add_argument("--design-workers", type=int, default=10)
    value.add_argument("--csim-workers", type=int, default=16)
    value.add_argument("--synth-workers", type=int, default=16)
    value.add_argument("--csim-timeout", type=float, default=360.0)
    value.add_argument("--synth-timeout", type=float, default=360.0)
    value.add_argument(
        "--case",
        action="append",
        choices=sorted(EXACT_CONTRACT_RULES),
        dest="cases",
    )
    return value


def _wrap(filename: str, code: str) -> str:
    return f'<OUTPUT_CODE name="{filename}">\n{code}\n</OUTPUT_CODE>'


def main() -> None:
    args = parser().parse_args()
    if args.samples != 5:
        raise SystemExit("Exact-contract freeze validation requires five samples")
    if args.output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing smoke root: {args.output_root}")
    selected_names = sorted(set(args.cases or EXACT_CONTRACT_RULES))
    enable_python310_hls_eval(args.hls_eval_root)
    from hls_eval.data import BenchmarkCase, find_benchmark_case_dirs
    from hls_eval.prompts import build_prompt_gen_zero_shot

    all_cases = [
        BenchmarkCase(path, name=path.name)
        for path in find_benchmark_case_dirs(args.dataset_root)
    ]
    all_cases.sort(key=lambda case: case.name)
    case_map = {case.name: case for case in all_cases}
    missing = set(selected_names) - set(case_map)
    if missing:
        raise SystemExit(f"Missing benchmark cases: {sorted(missing)}")
    selected_cases = [case_map[name] for name in selected_names]
    rules = args.rules.resolve()
    bank = ExecutableRepairContractLibrary(rules)
    rule_map = {rule.rule_id: rule for rule in bank.rules}
    source_root = args.source_round0_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    dataset_manifest, dataset_sha = _dataset_manifest(all_cases)
    atomic_write_json(output_root / "dataset_manifest.json", dataset_manifest)
    jobs: list[tuple[Any, int, int, str, Path, ReviewResult, int]] = []
    for case in selected_cases:
        for sample_index in range(args.samples):
            source_path = (
                source_root / case.name / f"sample__{sample_index}" / "trajectory.json"
            )
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            if int(payload["sample_index"]) != sample_index:
                raise SystemExit(f"Sample mismatch in {source_path}")
            rounds = payload.get("rounds") or []
            if not rounds or int(rounds[0]["round_index"]) != 0:
                raise SystemExit(f"Missing immutable Round 0 in {source_path}")
            review = ReviewResult(**rounds[0]["review"])
            if not review.generated_code:
                raise SystemExit(f"Round 0 lacks generated code: {source_path}")
            total_tokens = int((rounds[0].get("llm") or {}).get("total_tokens") or 0)
            jobs.append(
                (
                    case,
                    sample_index,
                    int(payload["seed"]),
                    EXACT_CONTRACT_RULES[case.name],
                    source_path,
                    review,
                    total_tokens,
                )
            )

    pools = ToolPools(args.csim_workers, args.synth_workers)
    reviewer = HLSReviewer(
        args.hls_eval_root,
        args.vitis_dir,
        args.part,
        args.clock_ns,
        args.csim_timeout,
        args.synth_timeout,
        pools,
    )
    action_executor = ContractActionExecutor()

    def run_one(
        job: tuple[Any, int, int, str, Path, ReviewResult, int]
    ) -> dict[str, Any]:
        (
            case,
            sample_index,
            seed,
            expected_rule_id,
            source_path,
            round0_review,
            total_tokens,
        ) = job
        header_name = case.h_files[0].name
        header_text = "\n\n".join(
            path.read_text(encoding="utf-8") for path in case.h_files
        )
        seed_prompt = build_prompt_gen_zero_shot(
            case.kernel_description_fp,
            case.tb_file,
            case.h_files[0],
        )
        log = failure_log(round0_review)
        short = ReviewerGroundedFalsificationLedger(max_entries=5)
        short.observe_round0(
            candidate_id=f"{case.name}:s{sample_index}:r0",
            code=round0_review.generated_code or "",
            review=round0_review,
            tool_log=log,
            total_tokens=total_tokens,
        )
        history = _long_short_match_history(short, round0_review.failure_stage)
        facts = derive_memory_facts(
            stage=round0_review.failure_stage,
            log=log,
            code=round0_review.generated_code or "",
            header=header_text,
            task=seed_prompt,
            header_name=header_name,
        )
        retrieval = bank.match_with_provenance(
            round0_review.failure_stage,
            log,
            round0_review.generated_code or "",
            header=header_text,
            history=history,
            task=seed_prompt,
            facts=facts,
            exclude_rule_id_suffixes=(
                "_STAGNATION_CLEANROOM",
                "_HISTORY_FRONTIER_GUARD",
            ),
        )
        retrieved_rule_ids = [rule.rule_id for rule in retrieval.rules]
        retrieval_applicable = round0_review.failure_stage == "compile"
        if retrieval_applicable and expected_rule_id not in retrieved_rule_ids:
            raise RuntimeError(
                f"{case.name}/s{sample_index}: expected {expected_rule_id}, "
                f"retrieved {retrieved_rule_ids}"
            )
        rule = rule_map[expected_rule_id]
        execution = action_executor.execute(
            code=round0_review.generated_code or "",
            header_name=header_name,
            header_text=header_text,
            matched_rules=[rule],
            top_fn=case.top_fn,
        )
        if not execution.changed:
            raise RuntimeError(
                f"{case.name}/s{sample_index}: operator failed: {execution.error}"
            )
        filename = (
            round0_review.generated_filename
            or reviewer.infer_expected_filename(case)
        )
        item_root = output_root / case.name / f"sample__{sample_index}"
        result = reviewer.review(
            case,
            _wrap(filename, execution.output_code),
            item_root / "review",
            f"exact_contract_{case.name}_s{sample_index}",
        )
        record = {
            "case": case.name,
            "sample_index": sample_index,
            "seed": seed,
            "source_trajectory": str(source_path),
            "source_trajectory_sha256": file_sha256(source_path),
            "round0_failure_stage": round0_review.failure_stage,
            "expected_rule_id": expected_rule_id,
            "retrieved_rule_ids": retrieved_rule_ids,
            "retrieval_applicable": retrieval_applicable,
            "retrieval_provenance": retrieval.provenance,
            "operator": execution.to_dict(),
            "output_code_sha256": hashlib.sha256(
                execution.output_code.encode("utf-8")
            ).hexdigest(),
            "review": result.to_dict(),
            "reference_kernel_read_or_prompted": False,
        }
        atomic_write_json(item_root / "contract_result.json", record)
        return record

    records: list[dict[str, Any]] = []
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.design_workers) as design_pool:
            pending = {
                design_pool.submit(run_one, job): job
                for job in jobs
            }
            for future in as_completed(pending):
                record = future.result()
                records.append(record)
                review = record["review"]
                print(
                    f"{len(records):02d}/{len(jobs)} {record['case']} "
                    f"s{record['sample_index']} "
                    f"C={int(review['pass_compile'])} "
                    f"TB={int(review['pass_tb'])} "
                    f"S={int(review['pass_synth'])}",
                    flush=True,
                )
    finally:
        pools.shutdown()

    records.sort(key=lambda item: (item["case"], item["sample_index"]))
    metric_fields = {
        "parse": "pass_parse",
        "compile": "pass_compile",
        "tb": "pass_tb",
        "synth": "pass_synth",
        "tb_and_synth": "pass_tb_and_synth",
    }
    counts = {
        metric: sum(int(item["review"][field]) for item in records)
        for metric, field in metric_fields.items()
    }
    counts["infrastructure_error"] = sum(
        int(bool(item["review"].get("infrastructure_error")))
        for item in records
    )
    formal_complete = (
        selected_names == sorted(EXACT_CONTRACT_RULES)
        and len(records) == len(EXACT_CONTRACT_RULES) * args.samples
    )
    all_pass = counts["tb_and_synth"] == len(records)
    source_sha = _tree_sha256(ROOT)
    summary = {
        "status": "pass" if all_pass else "fail",
        "formal_complete": formal_complete,
        "trajectory_count": len(records),
        "expected_formal_trajectory_count": 50,
        "case_count": len(selected_names),
        "sample_count": args.samples,
        "elapsed_seconds": time.time() - started,
        "source_tree_sha256": source_sha,
        "rules_version": bank.version,
        "rules_sha256": file_sha256(rules),
        "dataset_manifest_sha256": dataset_sha,
        "source_round0_root": str(source_root),
        "selected_cases": selected_names,
        "counts": counts,
        "operator_changed_count": sum(
            int(bool(item["operator"]["changed"])) for item in records
        ),
        "applicable_retrieval_count": sum(
            int(item["retrieval_applicable"]) for item in records
        ),
        "exact_rule_retrieval_count": sum(
            int(item["expected_rule_id"] in item["retrieved_rule_ids"])
            for item in records
            if item["retrieval_applicable"]
        ),
        "by_case": {
            case: {
                metric: sum(
                    int(item["review"][field])
                    for item in records
                    if item["case"] == case
                )
                for metric, field in metric_fields.items()
            }
            for case in selected_names
        },
        "reference_kernel_read_or_prompted": False,
        "configuration": {
            "part": args.part,
            "clock_ns": args.clock_ns,
            "vitis_dir": str(args.vitis_dir.resolve()),
            "design_workers": args.design_workers,
            "csim_workers": args.csim_workers,
            "synth_workers": args.synth_workers,
            "csim_timeout": args.csim_timeout,
            "synth_timeout": args.synth_timeout,
        },
    }
    atomic_write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not all_pass:
        raise SystemExit("At least one exact contract failed the complete Reviewer")


if __name__ == "__main__":
    main()
