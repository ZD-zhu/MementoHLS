from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .metrics import aggregate_case_metrics, per_case_metrics


HLS_EVAL_SUITES = (
    "c2hlsc",
    "chstone",
    "flowgnn",
    "gnnbuilder",
    "machsuite",
    "polybench",
    "pp4fpga",
    "rosetta",
)
BENCH4HLS_SUITE = "Bench4HLS"
SUITES = HLS_EVAL_SUITES
_ALL_SUITES = (*HLS_EVAL_SUITES, BENCH4HLS_SUITE)
BUDGET_RESOURCES = (
    "total_tokens",
    "logical_llm_calls",
    "physical_llm_calls",
    "tool_seconds",
)
BUDGET_METRICS = ("tb_and_synth_p@1", "tb_and_synth_p@5")


def _suite(item: dict[str, Any]) -> str:
    for value in item.get("benchmark_case_tags") or []:
        if str(value) in _ALL_SUITES:
            return str(value)
    for key in ("benchmark_suite", "source_suite", "suite"):
        if str(item.get(key) or "") in _ALL_SUITES:
            return str(item[key])
    raise ValueError(
        f"missing suite provenance for {item.get('benchmark_case_name')}"
    )


def suite_stratified_rows(
    trajectories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, items in trajectories.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[_suite(item)].append(item)
        dataset_kinds = {str(item.get("dataset_kind") or "") for item in items}
        expected_suites = (
            (BENCH4HLS_SUITE,)
            if dataset_kinds == {"bench4hls"}
            else HLS_EVAL_SUITES
        )
        missing = sorted(set(expected_suites) - set(grouped))
        extra = sorted(set(grouped) - set(expected_suites))
        if missing or extra:
            raise ValueError(
                f"{mode}: suite mismatch missing={missing}, extra={extra}"
            )
        for suite in expected_suites:
            per_case = per_case_metrics(grouped[suite])
            rows.append(
                {
                    "mode": mode,
                    "suite": suite,
                    "case_count": len(per_case),
                    "trajectory_count": len(grouped[suite]),
                    **aggregate_case_metrics(per_case),
                }
            )
    return rows


def _call_cost(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {
            "total_tokens": 0.0,
            "logical_llm_calls": 0.0,
            "physical_llm_calls": 0.0,
        }
    physical = float(bool(value.get("physical_call", True)))
    return {
        "total_tokens": float(value.get("total_tokens") or 0.0),
        "logical_llm_calls": 1.0,
        "physical_llm_calls": physical,
    }


def _review_tool_seconds(review: Any) -> float:
    if not isinstance(review, dict):
        return 0.0
    result = 0.0
    for key in ("compile", "testbench", "synthesis"):
        payload = review.get(key)
        if isinstance(payload, dict):
            result += float(payload.get("execution_time") or 0.0)
    return result


def _round_resource(payload: dict[str, Any]) -> dict[str, float]:
    result: Counter[str] = Counter()
    for key in ("diagnoser_llm", "llm"):
        result.update(_call_cost(payload.get(key)))
    result["tool_seconds"] += _review_tool_seconds(payload.get("review"))
    return {key: float(result[key]) for key in BUDGET_RESOURCES}


def _review_rank(review: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(bool(review.get(key)))
        for key in (
            "pass_tb_and_synth",
            "pass_tb",
            "pass_synth",
            "pass_compile",
            "pass_parse",
        )
    )


def _false_review() -> dict[str, bool]:
    return {
        "pass_parse": False,
        "pass_compile": False,
        "pass_tb": False,
        "pass_synth": False,
        "pass_tb_and_synth": False,
    }


def _cumulative_resource_values(
    items: list[dict[str, Any]], resource: str
) -> list[float]:
    values: list[float] = []
    for item in items:
        total = 0.0
        for payload in item.get("rounds") or []:
            total += _round_resource(payload)[resource]
            values.append(total)
    return values


def _budget_grid(
    trajectories: dict[str, list[dict[str, Any]]], resource: str
) -> list[float]:
    pooled = np.asarray(
        [
            value
            for items in trajectories.values()
            for value in _cumulative_resource_values(items, resource)
        ],
        dtype=float,
    )
    if pooled.size == 0:
        return [0.0]
    quantiles = np.quantile(
        pooled,
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    values = {0.0, *[float(value) for value in quantiles]}
    if resource != "tool_seconds":
        values = {float(int(round(value))) for value in values}
    return sorted(values)


def _review_under_budget(
    item: dict[str, Any], resource: str, budget: float
) -> tuple[dict[str, Any], int]:
    cumulative = 0.0
    eligible: list[tuple[tuple[int, ...], int, dict[str, Any]]] = []
    for payload in item.get("rounds") or []:
        cumulative += _round_resource(payload)[resource]
        if cumulative <= budget + 1e-12:
            review = payload.get("review") or _false_review()
            round_index = int(payload.get("round_index") or 0)
            eligible.append((_review_rank(review), -round_index, review))
    if not eligible:
        return _false_review(), -1
    _, negative_round, review = max(eligible, key=lambda row: (row[0], row[1]))
    return review, -negative_round


def budget_curve_rows(
    trajectories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resource in BUDGET_RESOURCES:
        for budget in _budget_grid(trajectories, resource):
            for mode, items in trajectories.items():
                projected: list[dict[str, Any]] = []
                generated = 0
                selected_rounds: list[int] = []
                for item in items:
                    review, selected_round = _review_under_budget(
                        item, resource, budget
                    )
                    generated += int(selected_round >= 0)
                    if selected_round >= 0:
                        selected_rounds.append(selected_round)
                    copy = dict(item)
                    copy["final_review"] = review
                    projected.append(copy)
                metrics = aggregate_case_metrics(per_case_metrics(projected))
                rows.append(
                    {
                        "resource": resource,
                        "budget": budget,
                        "mode": mode,
                        "generated_trajectory_fraction": generated / len(items),
                        "mean_selected_round": (
                            float(np.mean(selected_rounds))
                            if selected_rounds
                            else -1.0
                        ),
                        **{key: metrics[key] for key in BUDGET_METRICS},
                    }
                )
    for resource in BUDGET_RESOURCES:
        for metric in BUDGET_METRICS:
            subset = [
                row
                for row in rows
                if row["resource"] == resource
            ]
            for row in subset:
                dominated = any(
                    other["budget"] <= row["budget"]
                    and other[metric] >= row[metric]
                    and (
                        other["budget"] < row["budget"]
                        or other[metric] > row[metric]
                    )
                    for other in subset
                )
                row[f"pareto_{metric}"] = not dominated
    return rows


def round_gain_rows(
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, summary in summaries.items():
        prior: dict[str, float] | None = None
        for round_index in range(6):
            metrics = summary["round_cumulative_metrics"][str(round_index)]
            row: dict[str, Any] = {"mode": mode, "round": round_index}
            for key, value in metrics.items():
                row[f"cumulative_{key}"] = value
                row[f"marginal_{key}"] = (
                    value - prior[key] if prior is not None else value
                )
            rows.append(row)
            prior = metrics
    return rows


def _cleanroom_event(item: dict[str, Any]) -> dict[str, Any] | None:
    for payload in item.get("rounds") or []:
        clean = payload.get("cleanroom") or {}
        if clean.get("triggered"):
            state = payload.get("rfl_state") or {}
            signature = clean.get("trigger_failure_signature")
            signature_count = int(
                (state.get("signature_counts") or {}).get(signature, 0)
            )
            reason = str(clean.get("reason") or "unknown")
            before = payload.get("before_stage_vector") or {}
            after = payload.get("stage_vector") or {}
            before_score = sum(
                int(bool(before.get(key)))
                for key in ("parse", "compile", "tb", "synth", "tb_and_synth")
            )
            after_score = sum(
                int(bool(after.get(key)))
                for key in ("parse", "compile", "tb", "synth", "tb_and_synth")
            )
            return {
                "round": int(payload.get("round_index") or 0),
                "reason": reason,
                "signature": signature,
                "signature_count": signature_count,
                "repeated_signature": bool(
                    signature_count >= 2 or "signature" in reason
                ),
                "immediate_progress": bool(after_score > before_score),
                "immediate_success": bool(
                    (payload.get("review") or {}).get("pass_tb_and_synth")
                ),
            }
    return None


def _task_balanced_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seed: int = 2027,
    iterations: int = 10_000,
) -> tuple[float, float, float]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[row["benchmark_case_name"]].append(float(row["paired_delta"]))
    values = np.asarray(
        [float(np.mean(value)) for value in by_case.values()],
        dtype=float,
    )
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def repeated_error_cleanroom_analysis(
    trajectories: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    treated = trajectories["mementohls"]
    control = {
        (item["benchmark_case_name"], int(item["sample_index"])): item
        for item in trajectories["mementohls_no_cleanroom"]
    }
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for item in treated:
        event = _cleanroom_event(item)
        if event is None:
            continue
        key = (item["benchmark_case_name"], int(item["sample_index"]))
        comparison = control[key]
        treated_success = int(
            bool(item["final_review"]["pass_tb_and_synth"])
        )
        control_success = int(
            bool(comparison["final_review"]["pass_tb_and_synth"])
        )
        reasons[event["reason"]] += 1
        rows.append(
            {
                "benchmark_case_name": key[0],
                "sample_index": key[1],
                "seed": int(item["seed"]),
                **event,
                "mementohls_success": treated_success,
                "no_cleanroom_success": control_success,
                "paired_delta": treated_success - control_success,
            }
        )
    summary: dict[str, Any] = {
        "interpretation": (
            "Mechanism-selected paired observational analysis; the subgroup "
            "is selected from MementoHLS outcomes and is not an unbiased "
            "causal estimate."
        ),
        "trigger_reason_counts": dict(reasons.most_common()),
    }
    for name, selected in (
        ("all_cleanroom", rows),
        (
            "repeated_signature",
            [row for row in rows if row["repeated_signature"]],
        ),
    ):
        estimate, low, high = _task_balanced_bootstrap(selected)
        summary[name] = {
            "trajectory_count": len(selected),
            "task_count": len(
                {row["benchmark_case_name"] for row in selected}
            ),
            "mementohls_success_rate": (
                float(np.mean([row["mementohls_success"] for row in selected]))
                if selected
                else 0.0
            ),
            "no_cleanroom_success_rate": (
                float(np.mean([row["no_cleanroom_success"] for row in selected]))
                if selected
                else 0.0
            ),
            "trajectory_weighted_difference": (
                float(np.mean([row["paired_delta"] for row in selected]))
                if selected
                else 0.0
            ),
            "task_balanced_difference": estimate,
            "task_bootstrap_ci95_low": low,
            "task_bootstrap_ci95_high": high,
            "immediate_progress_rate": (
                float(np.mean([row["immediate_progress"] for row in selected]))
                if selected
                else 0.0
            ),
            "immediate_success_rate": (
                float(np.mean([row["immediate_success"] for row in selected]))
                if selected
                else 0.0
            ),
        }
    return summary, rows

