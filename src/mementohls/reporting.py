from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .candidate_selection import last_parse_valid_round_index
from .metrics import (
    METRICS,
    aggregate_case_metrics,
    nearest_rank_percentile,
    paired_bootstrap,
    per_case_metrics,
    round_cumulative_metrics,
)
from .extended_analysis import (
    budget_curve_rows,
    repeated_error_cleanroom_analysis,
    round_gain_rows,
    suite_stratified_rows,
)
from .models import (
    FORMAL_RUN_MODES,
    HLS_EVAL_VERIFICATION_MODES,
    RunMode,
    atomic_write_json,
    atomic_write_text,
)


MODES = [item.value for item in FORMAL_RUN_MODES]
HLS_EVAL_MODES = [item.value for item in HLS_EVAL_VERIFICATION_MODES]
PRIMARY_METRICS = ("tb_and_synth_p@1", "tb_and_synth_p@5")
ORDERED_METRICS = tuple(
    f"{metric}_p@{k}" for metric in METRICS for k in (1, 5)
)
BOOTSTRAP_ITERATIONS = 10_000
STATISTICAL_SEED = 2027


def expected_feature_dict(mode: str) -> dict[str, bool]:
    return RunMode(mode).feature_flags


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def load_trajectories(mode_root: Path, *, require_complete: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(mode_root.glob("*/sample__*/trajectory.json")):
        item = _json(path)
        item["_trajectory_path"] = str(path)
        if require_complete and not (
            item.get("terminal") is True and item.get("status") == "complete"
        ):
            raise ValueError(f"non-terminal trajectory: {path}")
        rows.append(item)
    if require_complete and not rows:
        raise ValueError(f"no complete trajectories in {mode_root}")
    return rows


def _tool_seconds(review: dict[str, Any]) -> float:
    total = 0.0
    for key in ("compile", "testbench", "synthesis"):
        value = review.get(key)
        if isinstance(value, dict):
            total += float(value.get("execution_time") or 0.0)
    return total


def _call_cost(call: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(call, dict):
        return {"logical": 0, "physical": 0, "http_attempts": 0, "prompt": 0, "completion": 0, "total": 0, "latency": 0}
    physical = int(bool(call.get("physical_call", True)))
    return {
        "logical": 1,
        "physical": physical,
        "http_attempts": int(call.get("attempts") or 0) if physical else 0,
        "prompt": int(call.get("prompt_tokens") or 0),
        "completion": int(call.get("completion_tokens") or 0),
        "total": int(call.get("total_tokens") or 0),
        "latency": float(call.get("elapsed_seconds") or 0.0),
    }


def _trajectory_cost(item: dict[str, Any]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for round_payload in item.get("rounds", []):
        for field in ("diagnoser_llm", "llm"):
            totals.update(_call_cost(round_payload.get(field)))
        totals["tool_seconds"] += _tool_seconds(round_payload.get("review") or {})
    repairs = max(0, len(item.get("rounds", [])) - 1)
    return {
        "benchmark_case_name": item["benchmark_case_name"],
        "sample_index": int(item["sample_index"]),
        "seed": int(item["seed"]),
        "mode": item["mode"],
        "success": bool(item.get("success")),
        "repair_rounds": repairs,
        "logical_llm_calls": int(totals["logical"]),
        "physical_llm_calls": int(totals["physical"]),
        "http_attempts": int(totals["http_attempts"]),
        "prompt_tokens": int(totals["prompt"]),
        "completion_tokens": int(totals["completion"]),
        "total_tokens": int(totals["total"]),
        "llm_seconds": float(totals["latency"]),
        "tool_seconds": float(totals["tool_seconds"]),
        "critical_path_proxy_seconds": float(totals["latency"] + totals["tool_seconds"]),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def summarize_costs(trajectories: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    per_trajectory = [_trajectory_cost(item) for item in trajectories]
    repair_rounds = [row["repair_rounds"] for row in per_trajectory]
    success_rounds = [row["repair_rounds"] for row in per_trajectory if row["success"]]
    failure_rounds = [row["repair_rounds"] for row in per_trajectory if not row["success"]]
    summary: dict[str, Any] = {
        "trajectory_count": len(per_trajectory),
        "mean_repair_rounds": _mean(repair_rounds),
        "median_repair_rounds": float(statistics.median(repair_rounds)) if repair_rounds else 0.0,
        "p95_repair_rounds": nearest_rank_percentile(repair_rounds, 0.95),
        "mean_success_repair_rounds": _mean(success_rounds),
        "mean_failure_repair_rounds": _mean(failure_rounds),
    }
    for key in (
        "logical_llm_calls", "physical_llm_calls", "http_attempts", "prompt_tokens",
        "completion_tokens", "total_tokens", "llm_seconds", "tool_seconds",
        "critical_path_proxy_seconds",
    ):
        values = [float(row[key]) for row in per_trajectory]
        summary[f"mean_{key}"] = _mean(values)
        summary[f"total_{key}"] = float(sum(values))
        summary[f"p95_{key}"] = nearest_rank_percentile(values, 0.95)

    by_round: dict[int, Counter[str]] = defaultdict(Counter)
    reached: Counter[int] = Counter()
    for item in trajectories:
        for payload in item.get("rounds", []):
            index = int(payload["round_index"])
            reached[index] += 1
            for field in ("diagnoser_llm", "llm"):
                by_round[index].update(_call_cost(payload.get(field)))
            by_round[index]["tool_seconds"] += _tool_seconds(payload.get("review") or {})
    round_rows: list[dict[str, Any]] = []
    for index in range(6):
        count = reached[index]
        values = by_round[index]
        row: dict[str, Any] = {"round": index, "trajectories_reached": count}
        for key in ("logical", "physical", "http_attempts", "prompt", "completion", "total", "latency", "tool_seconds"):
            row[f"total_{key}"] = float(values[key])
            row[f"mean_{key}"] = float(values[key] / count) if count else 0.0
        round_rows.append(row)
    return summary, per_trajectory, round_rows


def _stage_name(review: dict[str, Any]) -> str:
    return "success" if review.get("pass_tb_and_synth") else str(review.get("failure_stage") or "unknown")


def _stage_score(vector: Any) -> int:
    if isinstance(vector, dict):
        return sum(int(bool(vector.get(key))) for key in ("parse", "compile", "tb", "synth", "tb_and_synth"))
    return sum(int(bool(value)) for value in (vector or []))


def mechanism_summary(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    rule_hits: Counter[str] = Counter()
    for item in trajectories:
        rounds = item.get("rounds", [])
        counts["archive_activated"] += int(bool(item.get("archive_activated")))
        for before, after in zip(rounds, rounds[1:]):
            transitions[f"{_stage_name(before['review'])}->{_stage_name(after['review'])}"] += 1
        for payload in rounds[1:]:
            retrieval = payload.get("ercl_retrieval") or {}
            selected = list(payload.get("retrieved_rule_ids") or [])
            counts["ercl_queries"] += int(bool(retrieval) or "ercl_retrieval" in payload)
            counts["ercl_abstain"] += int(not selected and bool(retrieval))
            for rule_id in selected:
                rule_hits[str(rule_id)] += 1
            operator = payload.get("operator") or {}
            counts["operator_attempted"] += int(bool(operator.get("attempted")))
            counts["operator_changed"] += int(bool(operator.get("changed")))
            if operator.get("changed"):
                before_score = _stage_score(payload.get("before_stage_vector"))
                after_score = _stage_score(payload.get("stage_vector"))
                counts["operator_progress"] += int(after_score > before_score)
                counts["operator_success"] += int(bool((payload.get("review") or {}).get("pass_tb_and_synth")))
            gate = payload.get("diagnosis_novelty_gate") or {}
            counts["veto_applied"] += int(bool(gate.get("applied")))
            counts["veto_active_hits"] += len(gate.get("active_hit_ids") or [])
            clean = payload.get("cleanroom") or {}
            counts["cleanroom_requested"] += int(bool(clean.get("requested")))
            counts["cleanroom_triggered"] += int(bool(clean.get("triggered")))
            if clean.get("triggered"):
                before_score = _stage_score(payload.get("before_stage_vector"))
                after_score = _stage_score(payload.get("stage_vector"))
                counts["cleanroom_immediate_progress"] += int(after_score > before_score)
                counts["cleanroom_immediate_success"] += int(bool((payload.get("review") or {}).get("pass_tb_and_synth")))
            state = payload.get("rfl_state") or {}
            entries = state.get("entries") or []
            if entries:
                status = str(entries[-1].get("hypothesis_status") or "unresolved")
                counts[f"credit_{status}"] += 1
                counts["stage_regression"] += int(bool(entries[-1].get("stage_regression")))
    return {
        "counts": dict(sorted(counts.items())),
        "rule_hits": dict(rule_hits.most_common()),
        "failure_stage_transitions": dict(sorted(transitions.items())),
    }


def _archive_counterfactual(trajectories: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    counterfactual: list[dict[str, Any]] = []
    for item in trajectories:
        index = last_parse_valid_round_index(item["rounds"])
        last_review = item["rounds"][index]["review"]
        copy = dict(item)
        copy["final_review"] = last_review
        counterfactual.append(copy)
        rows.append({
            "benchmark_case_name": item["benchmark_case_name"],
            "sample_index": int(item["sample_index"]),
            "selected_round": int(item["selected_round_index"]),
            "last_parse_valid_round": index,
            "archive_activated": int(int(item["selected_round_index"]) != index),
            "selected_tb_and_synth": int(bool(item["final_review"]["pass_tb_and_synth"])),
            "last_tb_and_synth": int(bool(last_review["pass_tb_and_synth"])),
        })
    selected = aggregate_case_metrics(per_case_metrics(trajectories))
    last = aggregate_case_metrics(per_case_metrics(counterfactual))
    delta = {key: selected[key] - last[key] for key in ORDERED_METRICS}
    return delta, rows


def summarize_mode(mode_root: Path, *, write_files: bool = True) -> dict[str, Any]:
    trajectories = load_trajectories(mode_root)
    per_case = per_case_metrics(trajectories)
    metrics = aggregate_case_metrics(per_case)
    rounds = round_cumulative_metrics(trajectories, max_round=5)
    cost, per_trajectory_cost, cost_by_round = summarize_costs(trajectories)
    mechanism = mechanism_summary(trajectories)
    archive_delta, archive_rows = _archive_counterfactual(trajectories)
    summary = {
        "mode": mode_root.name,
        "trajectory_count": len(trajectories),
        "case_count": len(per_case),
        "metrics": metrics,
        "per_case": {
            case: {f"{metric}_p@{k}": value for (metric, k), value in row.items()}
            for case, row in per_case.items()
        },
        "round_cumulative_metrics": {str(key): value for key, value in rounds.items()},
        "cost": cost,
        "cost_by_round": cost_by_round,
        "mechanism": mechanism,
        "archive_counterfactual_delta": archive_delta,
    }
    if write_files:
        atomic_write_json(mode_root / "summary.json", summary)
        _write_csv(mode_root / "per_trajectory_cost.csv", per_trajectory_cost)
        _write_csv(mode_root / "cost_by_round.csv", cost_by_round)
        _write_csv(mode_root / "archive_counterfactual.csv", archive_rows)
    return summary


def _case_map(summary: dict[str, Any], metric_key: str) -> dict[str, float]:
    return {case: float(row[metric_key]) for case, row in summary["per_case"].items()}


def _bootstrap_array(values: np.ndarray, *, seed: int, iterations: int = BOOTSTRAP_ITERATIONS) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def paired_permutation_pvalue(values: np.ndarray, *, seed: int = STATISTICAL_SEED, iterations: int = BOOTSTRAP_ITERATIONS) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(iterations, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (iterations + 1))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (m - rank) * pvalues[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _paired_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("feedback", "zero_shot"),
        ("mementohls", "zero_shot"),
        ("rfl", "feedback"),
        ("ercl", "feedback"),
        ("mementohls", "rfl"),
        ("mementohls", "ercl"),
        ("mementohls", "mementohls_core_rules"),
        ("mementohls_all_rules", "mementohls"),
    ]
    rows: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for candidate, baseline in comparisons:
        if candidate not in summaries or baseline not in summaries:
            continue
        for metric_key in ORDERED_METRICS:
            names = sorted(summaries[baseline]["per_case"])
            values = np.asarray([
                summaries[candidate]["per_case"][name][metric_key]
                - summaries[baseline]["per_case"][name][metric_key]
                for name in names
            ])
            estimate, low, high = _bootstrap_array(values, seed=STATISTICAL_SEED)
            identifier = f"{candidate}-{baseline}:{metric_key}"
            pvalue = (
                paired_permutation_pvalue(values)
                if metric_key in PRIMARY_METRICS
                else None
            )
            if pvalue is not None:
                raw_p[identifier] = pvalue
            rows.append({
                "comparison": f"{candidate}-{baseline}",
                "metric": metric_key,
                "difference": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "permutation_p": pvalue,
                "holm_p": None,
            })
    adjusted = holm_adjust(raw_p)
    for row in rows:
        identifier = f"{row['comparison']}:{row['metric']}"
        if identifier in adjusted:
            row["holm_p"] = adjusted[identifier]
    return rows


def _factorial_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = sorted(summaries["feedback"]["per_case"])
    formulas = {
        "RFL_given_ERCL": lambda m, e, r, f: m - e,
        "ERCL_given_RFL": lambda m, e, r, f: m - r,
        "interaction": lambda m, e, r, f: m - e - r + f,
    }
    for metric_key in ORDERED_METRICS:
        values_by_term = {
            name: np.asarray([
                summaries[name]["per_case"][case][metric_key] for case in names
            ])
            for name in ("mementohls", "ercl", "rfl", "feedback")
        }
        for effect, formula in formulas.items():
            values = formula(
                values_by_term["mementohls"], values_by_term["ercl"],
                values_by_term["rfl"], values_by_term["feedback"],
            )
            estimate, low, high = _bootstrap_array(values, seed=STATISTICAL_SEED + 17)
            rows.append({"effect": effect, "metric": metric_key, "estimate": estimate, "ci95_low": low, "ci95_high": high})
    return rows


def _target_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = summaries["zero_shot"]["metrics"]
    full = summaries["mementohls"]["metrics"]
    rows = []
    for key in ORDERED_METRICS:
        target = baseline[key] if key.startswith("parse_") else min(1.0, baseline[key] + 0.15)
        rows.append({
            "metric": key,
            "zero_shot": baseline[key],
            "mementohls": full[key],
            "difference": full[key] - baseline[key],
            "target": target,
            "target_met": bool(full[key] >= target - 1e-12),
        })
    return rows


def _ordering_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("feedback", "zero_shot"), ("rfl", "feedback"), ("ercl", "feedback"),
        ("mementohls", "rfl"), ("mementohls", "ercl"),
    ]
    rows = []
    for candidate, baseline in comparisons:
        for key in (item for item in ORDERED_METRICS if not item.startswith("parse_")):
            delta = summaries[candidate]["metrics"][key] - summaries[baseline]["metrics"][key]
            rows.append({"candidate": candidate, "baseline": baseline, "metric": key, "difference": delta, "strict_order_met": bool(delta > 0)})
    return rows


def experiment_audit(
    root: Path,
    trajectories: dict[str, list[dict[str, Any]]],
    modes: list[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    freeze_path = root / "protocol" / "freeze_manifest.json"
    freeze = _json(freeze_path) if freeze_path.is_file() else {}
    dataset = freeze.get("dataset") or {}
    expected_cases = int(dataset.get("case_count") or 0)
    dataset_kind = str(dataset.get("kind") or "")
    expected = expected_cases * 5
    if not expected_cases or not dataset_kind:
        errors.append("freeze manifest lacks dataset kind/count")
    identities: dict[tuple[str, int], dict[str, tuple[Any, ...]]] = defaultdict(dict)
    source_hashes: set[str] = set()
    dataset_hashes: set[str] = set()
    dataset_kinds: set[str] = set()
    for mode in modes or MODES:
        items = trajectories.get(mode, [])
        if len(items) != expected:
            errors.append(f"{mode}: expected {expected} trajectories, found {len(items)}")
        seen: set[tuple[str, int, int, str]] = set()
        run_errors = root / mode / "run_errors.json"
        if not run_errors.is_file() or _json(run_errors) != []:
            errors.append(f"{mode}: run_errors.json is missing or non-empty")
        for item in items:
            key = (str(item.get("benchmark_case_name")), int(item.get("sample_index", -1)), int(item.get("seed", -1)), str(item.get("mode")))
            if key in seen:
                errors.append(f"{mode}: duplicate {key}")
            seen.add(key)
            source_hashes.add(str(item.get("source_sha256")))
            dataset_hashes.add(str(item.get("dataset_manifest_sha256")))
            dataset_kinds.add(str(item.get("dataset_kind")))
            rounds = item.get("rounds") or []
            if not rounds:
                errors.append(f"{mode}:{key}: no rounds")
                continue
            r0 = rounds[0]
            identities[(key[0], key[1])][mode] = (
                item.get("seed_prompt_sha256"), r0.get("response_sha256"), r0.get("code_sha256")
            )
            selected = int(item.get("selected_round_index", -1))
            if selected < 0 or selected >= len(rounds) or item.get("final_review") != rounds[selected].get("review"):
                errors.append(f"{mode}:{key}: selected candidate mismatch")
            review = item.get("final_review") or {}
            if bool(review.get("pass_tb_and_synth")) != bool(review.get("pass_tb") and review.get("pass_synth")):
                errors.append(f"{mode}:{key}: TB&Synth is not same-candidate conjunction")
            for round_payload in rounds:
                if (round_payload.get("prompt_provenance") or {}).get("reference_kernel_read_or_prompted") is not False:
                    errors.append(f"{mode}:{key}: reference-kernel provenance is not false")
                state = round_payload.get("rfl_state")
                if state and len(state.get("entries") or []) > int(round_payload["round_index"]) + 1:
                    errors.append(f"{mode}:{key}: RFL future-state leakage")
    for key, by_mode in identities.items():
        if len(set(by_mode.values())) > 1:
            errors.append(f"round0 identity mismatch for {key}")
    if len(source_hashes) != 1:
        errors.append(f"source hash count is {len(source_hashes)}")
    if len(dataset_hashes) != 1:
        errors.append(f"dataset hash count is {len(dataset_hashes)}")
    if dataset_kinds != {dataset_kind}:
        errors.append(
            f"dataset kind mismatch: expected {dataset_kind}, found {sorted(dataset_kinds)}"
        )
    return {
        "passed": not errors,
        "dataset_kind": dataset_kind,
        "task_count": expected_cases,
        "expected_trajectory_count_per_mode": expected,
        "error_count": len(errors),
        "errors": errors[:1000],
        "round0_identity_pairs": len(identities),
        "source_hashes": sorted(source_hashes),
        "dataset_hashes": sorted(dataset_hashes),
    }


def analyze_experiment(root: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    destination = output_dir or root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    freeze_path = root / "protocol" / "freeze_manifest.json"
    freeze = _json(freeze_path) if freeze_path.is_file() else {}
    dataset_kind = str((freeze.get("dataset") or {}).get("kind") or "")
    modes = HLS_EVAL_MODES if dataset_kind == "hls_eval" else MODES
    trajectories = {mode: load_trajectories(root / mode) for mode in modes}
    summaries = {mode: summarize_mode(root / mode) for mode in modes}
    paired = _paired_rows(summaries)
    factorial = _factorial_rows(summaries)
    targets = _target_rows(summaries)
    ordering = _ordering_rows(summaries)
    audit = experiment_audit(root, trajectories, modes)
    suite_rows = suite_stratified_rows(trajectories)
    budget_rows = budget_curve_rows(trajectories)
    gain_rows = round_gain_rows(summaries)
    if "mementohls_no_cleanroom" in trajectories:
        cleanroom_summary, cleanroom_rows = repeated_error_cleanroom_analysis(
            trajectories
        )
    else:
        cleanroom_summary = {
            "status": "not_run",
            "reason": "retired_no_cleanroom_ablation",
        }
        cleanroom_rows = []
    main_rows = [{"mode": mode, **summaries[mode]["metrics"], **summaries[mode]["cost"]} for mode in modes]
    result = {
        "modes": modes,
        "summaries": summaries,
        "paired_comparisons": paired,
        "factorial_effects": factorial,
        "targets": targets,
        "ordering": ordering,
        "experiment_audit": audit,
        "suite_stratified": suite_rows,
        "budget_curves": budget_rows,
        "round_gains": gain_rows,
        "repeated_error_cleanroom": cleanroom_summary,
        "all_targets_met": all(row["target_met"] for row in targets),
        "all_preregistered_orders_met": all(row["strict_order_met"] for row in ordering),
        "all_statistical_checks_met": all(
            row["ci95_low"] > 0 for row in paired
            if row["comparison"] == "mementohls-zero_shot" and not row["metric"].startswith("parse_")
        ),
    }
    atomic_write_json(destination / "analysis.json", result)
    atomic_write_json(destination / "experiment_audit.json", audit)
    _write_csv(destination / "main_metrics_and_cost.csv", main_rows)
    _write_csv(destination / "paired_bootstrap_and_permutation.csv", paired)
    _write_csv(destination / "factorial_rfl_ercl_effects.csv", factorial)
    _write_csv(destination / "fifteen_point_targets.csv", targets)
    _write_csv(destination / "preregistered_ordering.csv", ordering)
    _write_csv(destination / "suite_stratified_metrics.csv", suite_rows)
    _write_csv(destination / "budget_fairness_and_pareto.csv", budget_rows)
    _write_csv(destination / "round_cumulative_and_marginal.csv", gain_rows)
    atomic_write_json(
        destination / "repeated_error_cleanroom_summary.json",
        cleanroom_summary,
    )
    _write_csv(
        destination / "repeated_error_cleanroom_trajectories.csv",
        cleanroom_rows,
    )
    return result
