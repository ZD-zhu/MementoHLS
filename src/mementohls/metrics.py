from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import numpy as np


METRICS = {
    "parse": "pass_parse",
    "compile": "pass_compile",
    "tb": "pass_tb",
    "synth": "pass_synth",
    "tb_and_synth": "pass_tb_and_synth",
}


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got n={n}, c={c}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n], got n={n}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _group_trajectories(
    trajectories: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[str(trajectory["benchmark_case_name"])].append(trajectory)
    for case_name, items in grouped.items():
        items.sort(key=lambda item: int(item["sample_index"]))
        sample_indices = [int(item["sample_index"]) for item in items]
        if sample_indices != [0, 1, 2, 3, 4]:
            raise ValueError(
                f"Expected samples 0..4 for {case_name}, got {sample_indices}"
            )
    return dict(grouped)


def per_case_metrics(
    trajectories: list[dict[str, Any]],
    review_selector: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, dict[tuple[str, int], float]]:
    selector = review_selector or (lambda item: item["final_review"])
    output: dict[str, dict[tuple[str, int], float]] = {}
    for case_name, items in _group_trajectories(trajectories).items():
        row: dict[tuple[str, int], float] = {}
        reviews = [selector(item) for item in items]
        for metric_name, key in METRICS.items():
            c = sum(bool(review[key]) for review in reviews)
            row[(metric_name, 1)] = pass_at_k(5, c, 1)
            row[(metric_name, 5)] = pass_at_k(5, c, 5)
        output[case_name] = row
    return output


def aggregate_case_metrics(
    per_case: dict[str, dict[tuple[str, int], float]],
) -> dict[str, float]:
    if not per_case:
        raise ValueError("No per-case metrics")
    output: dict[str, float] = {}
    for metric_name in METRICS:
        for k in (1, 5):
            output[f"{metric_name}_p@{k}"] = float(
                np.mean([row[(metric_name, k)] for row in per_case.values()])
            )
    return output


def round_cumulative_metrics(
    trajectories: list[dict[str, Any]],
    max_round: int = 5,
) -> dict[int, dict[str, float]]:
    grouped = _group_trajectories(trajectories)
    output: dict[int, dict[str, float]] = {}
    for cutoff in range(max_round + 1):
        per_case: dict[str, dict[tuple[str, int], float]] = {}
        for case_name, items in grouped.items():
            row: dict[tuple[str, int], float] = {}
            for metric_name, key in METRICS.items():
                successes = 0
                for item in items:
                    eligible = [
                        round_payload["review"]
                        for round_payload in item["rounds"]
                        if int(round_payload["round_index"]) <= cutoff
                    ]
                    successes += int(any(bool(review[key]) for review in eligible))
                row[(metric_name, 1)] = pass_at_k(5, successes, 1)
                row[(metric_name, 5)] = pass_at_k(5, successes, 5)
            per_case[case_name] = row
        output[cutoff] = aggregate_case_metrics(per_case)
    return output


def paired_bootstrap(
    baseline: dict[str, dict[tuple[str, int], float]],
    candidate: dict[str, dict[tuple[str, int], float]],
    metric: tuple[str, int],
    *,
    iterations: int = 10000,
    seed: int = 20260712,
) -> tuple[float, float, float]:
    names = sorted(baseline)
    if names != sorted(candidate):
        raise ValueError("Baseline and candidate case sets differ")
    differences = np.asarray(
        [candidate[name][metric] - baseline[name][metric] for name in names],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(iterations, len(differences)),
    )
    samples = differences[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def nearest_rank_percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]
