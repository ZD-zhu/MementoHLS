from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "input": "#EDE7F6",
    "reviewer": "#E3F2FD",
    "ercl": "#FFF3CD",
    "rfl": "#FCE4EC",
    "action": "#E8F5E9",
    "decision": "#FFE0B2",
    "ink": "#263238",
}


def _box(ax, x: float, y: float, w: float, h: float, title: str, lines: list[str], color: str, *, lw: float = 1.5) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.02", facecolor=color, edgecolor=COLORS["ink"], linewidth=lw)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=10.8, fontweight="bold", color=COLORS["ink"])
    ax.text(x + 0.018, y + h - 0.105, "\n".join(lines), ha="left", va="top", fontsize=8.0, linespacing=1.32, color=COLORS["ink"])


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "#455A64", style: str = "-|>", lw: float = 1.6, connection: str = "arc3") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=lw, color=color, connectionstyle=connection))


def _save(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, "MementoHLS: Executable Contracts, Reviewer-Grounded Falsification", ha="center", va="top", fontsize=16, fontweight="bold", color=COLORS["ink"])

    _box(ax, 0.02, 0.57, 0.16, 0.28, "Immutable HLS Task", ["Description + header", "Top function + dimensions", "Testbench + data stay private", "Reference kernel is excluded"], COLORS["input"])
    _box(ax, 0.22, 0.57, 0.16, 0.28, "Round 0 Generator", ["Original HLS-Eval prompt", "Meta-Llama-3-8B-Instruct", "One complete .cpp candidate", "Same seed across all modes"], COLORS["input"])
    _box(ax, 0.42, 0.57, 0.18, 0.28, "Stage-aware Reviewer", ["1  Parse output contract", "2  Compile in C simulation", "3  Run testbench", "4  Vitis synthesis", "Success = TB AND Synth", "from the same candidate"], COLORS["reviewer"], lw=2.1)
    _box(ax, 0.64, 0.57, 0.16, 0.28, "Structured Diagnoser", ["Earliest failing stage", "Tool evidence + root cause", "Allowed / forbidden actions", "Immutable interface contract"], COLORS["reviewer"])
    _box(ax, 0.84, 0.57, 0.14, 0.28, "Single-action Arbiter", ["1 ERCL operator", "2 authorized clean-room", "3 local LLM repair", "Exactly one candidate"], COLORS["decision"], lw=2.1)

    _box(ax, 0.08, 0.14, 0.34, 0.30, "ERCL - Executable Repair Contract Library", ["71 frozen primary contracts: 60 generic + 11 algorithm-family", "Positive, negative, header, facts and history eligibility", "Allowed/forbidden actions and immutable invariants", "Optional operator with pre/postconditions", "Top-1 retrieval; abstain when evidence is insufficient"], COLORS["ercl"], lw=2.1)
    _box(ax, 0.55, 0.14, 0.38, 0.30, "RFL - Reviewer-Grounded Falsification Ledger", ["Isolated to one task / sample / seed trajectory", "Failure signature + code SHA + action fingerprint", "Reviewer credit: supported / contradicted / unresolved", "Trajectory-local contract/root/action veto", "Stagnation, one clean-room request, best-candidate archive", "Token, call and tool cost per round"], COLORS["rfl"], lw=2.1)
    _box(ax, 0.405, 0.035, 0.18, 0.075, "Candidate source", ["latest / archive / immutable task"], COLORS["action"])

    _arrow(ax, (0.18, 0.71), (0.22, 0.71))
    _arrow(ax, (0.38, 0.71), (0.42, 0.71))
    _arrow(ax, (0.60, 0.71), (0.64, 0.71))
    _arrow(ax, (0.80, 0.71), (0.84, 0.71))
    _arrow(ax, (0.72, 0.57), (0.72, 0.44))
    _arrow(ax, (0.84, 0.36), (0.84, 0.57))
    _arrow(ax, (0.42, 0.29), (0.64, 0.57), color="#C47F00", connection="arc3,rad=-0.15")
    _arrow(ax, (0.74, 0.44), (0.74, 0.57), color="#AD1457")
    _arrow(ax, (0.58, 0.26), (0.42, 0.26), color="#AD1457", style="-|>")
    ax.text(0.50, 0.275, "trajectory-local veto\n(falsifies ERCL hypothesis)", ha="center", va="bottom", fontsize=8.5, color="#AD1457", fontweight="bold")
    _arrow(ax, (0.84, 0.57), (0.58, 0.11), color="#2E7D32", connection="arc3,rad=0.13")
    _arrow(ax, (0.405, 0.072), (0.42, 0.57), color="#2E7D32", connection="arc3,rad=-0.12")
    _arrow(ax, (0.93, 0.70), (0.60, 0.79), color="#1565C0", connection="arc3,rad=0.27")
    ax.text(0.77, 0.89, "full Reviewer validates every action;\nresults update RFL credit and archive", ha="center", fontsize=8.7, color="#1565C0", fontweight="bold")
    ax.text(0.5, 0.005, "One control loop, one candidate per round, at most five repairs; no QoR-based selection", ha="center", va="bottom", fontsize=10, fontstyle="italic", color=COLORS["ink"])
    _save(fig, output, "mementohls_architecture")


def rfl_trajectory_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "RFL trajectory: plan -> action -> Reviewer outcome -> falsification credit", ha="center", va="top", fontsize=15, fontweight="bold", color=COLORS["ink"])
    positions = [0.04, 0.20, 0.36, 0.52, 0.68, 0.84]
    for index, x in enumerate(positions):
        color = COLORS["input"] if index == 0 else COLORS["rfl"]
        _box(ax, x, 0.56, 0.125, 0.22, f"Round {index}", ["candidate SHA", "stage vector", "failure signature", "token + tool cost"], color)
        if index < 5:
            _arrow(ax, (x + 0.125, 0.67), (positions[index + 1], 0.67))
    _box(ax, 0.06, 0.15, 0.22, 0.25, "Credit assignment", ["supported: Reviewer frontier improves", "contradicted: regression / same basin", "unresolved: evidence insufficient", "log wording change alone is not progress"], COLORS["reviewer"])
    _box(ax, 0.39, 0.15, 0.22, 0.25, "Stagnation controller", ["repeated signature or action", "same code or A-B-A loop", "two rounds without progress", "one clean-room request in rounds 3-5"], COLORS["ercl"])
    _box(ax, 0.72, 0.15, 0.22, 0.25, "Best-candidate archive", ["TB&Synth > TB > Synth", "> Compile > Parse", "then compile frontier, earlier round", "and lower cost; same trajectory only"], COLORS["action"])
    for x in (0.17, 0.50, 0.83):
        _arrow(ax, (x, 0.56), (x, 0.40), color="#6A1B9A")
    _arrow(ax, (0.28, 0.275), (0.39, 0.275), color="#AD1457")
    _arrow(ax, (0.61, 0.275), (0.72, 0.275), color="#2E7D32")
    ax.text(0.5, 0.055, "RFL never shares state across tasks, samples, seeds, or modes; the archive does not add candidates", ha="center", fontsize=10, fontstyle="italic", color=COLORS["ink"])
    _save(fig, output, "rfl_trajectory")


def _bootstrap_mean_interval(
    summary: dict[str, Any],
    metric: str,
    *,
    seed: int = 2027,
    iterations: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(
        [
            float(metrics[metric])
            for metrics in (summary.get("per_case") or {}).values()
        ],
        dtype=float,
    )
    if values.size == 0:
        value = float(summary["metrics"][metric])
        return value, value
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draws[index] = values[rng.integers(0, values.size, values.size)].mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _mode_labels(modes: list[str]) -> list[str]:
    label_by_mode = {
        "zero_shot": "Zero",
        "feedback": "Feedback",
        "rfl": "RFL",
        "ercl": "ERCL",
        "mementohls": "Memento",
        "mementohls_no_executor": "No Exec",
        "mementohls_no_veto": "No Veto",
        "mementohls_no_cleanroom": "No Clean",
        "mementohls_core_rules": "Core 60",
        "mementohls_all_rules": "All 81",
    }
    return [label_by_mode.get(mode, mode) for mode in modes]


def result_cost_figure(
    analyses: Path | list[tuple[str, Path]],
    output: Path,
) -> None:
    legacy_single = isinstance(analyses, Path)
    specifications = (
        [(analyses.stem, analyses)]
        if legacy_single
        else list(analyses)
    )
    if not specifications:
        return
    loaded = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in specifications
    ]
    modes = loaded[0][1]["modes"]
    if any(analysis["modes"] != modes for _, analysis in loaded[1:]):
        raise ValueError("all result-cost panels require the same mode order")
    labels = _mode_labels(modes)
    rows = len(loaded)
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(22.0, 4.9 * rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.65, 1.0, 1.15, 1.0]},
    )
    x = np.arange(len(modes))
    width = 0.36
    blue = "#2563A6"
    blue_open = "#A9C7E8"
    gold = "#D89B2B"
    gold_open = "#F2D69B"
    ink = COLORS["ink"]

    for row_index, (dataset_label, analysis) in enumerate(loaded):
        summaries = analysis["summaries"]
        p1 = np.asarray([
            100.0 * summaries[mode]["metrics"]["tb_and_synth_p@1"]
            for mode in modes
        ])
        p5 = np.asarray([
            100.0 * summaries[mode]["metrics"]["tb_and_synth_p@5"]
            for mode in modes
        ])
        p1_ci = [
            _bootstrap_mean_interval(
                summaries[mode], "tb_and_synth_p@1", seed=2027 + index
            )
            for index, mode in enumerate(modes)
        ]
        p5_ci = [
            _bootstrap_mean_interval(
                summaries[mode], "tb_and_synth_p@5", seed=3027 + index
            )
            for index, mode in enumerate(modes)
        ]
        p1_yerr = np.maximum(
            0.0,
            np.asarray([
                [p1[index] - 100.0 * low for index, (low, _) in enumerate(p1_ci)],
                [100.0 * high - p1[index] for index, (_, high) in enumerate(p1_ci)],
            ], dtype=float),
        )
        p5_yerr = np.maximum(
            0.0,
            np.asarray([
                [p5[index] - 100.0 * low for index, (low, _) in enumerate(p5_ci)],
                [100.0 * high - p5[index] for index, (_, high) in enumerate(p5_ci)],
            ], dtype=float),
        )
        tokens = np.asarray([
            summaries[mode]["cost"]["mean_total_tokens"] / 1000.0 for mode in modes
        ])
        calls = np.asarray([
            summaries[mode]["cost"].get("mean_physical_llm_calls", 0.0) for mode in modes
        ])
        rounds = np.asarray([
            summaries[mode]["cost"]["mean_repair_rounds"] for mode in modes
        ])
        tool_seconds = np.asarray([
            summaries[mode]["cost"].get("mean_tool_seconds", 0.0) for mode in modes
        ])

        accuracy, token_ax, call_ax, tool_ax = axes[row_index]
        accuracy.bar(
            x - width / 2,
            p1,
            width,
            yerr=p1_yerr,
            capsize=2,
            label="TB&Synth p@1",
            color=blue,
            edgecolor=ink,
            linewidth=0.55,
        )
        accuracy.bar(
            x + width / 2,
            p5,
            width,
            yerr=p5_yerr,
            capsize=2,
            label="TB&Synth p@5",
            color=blue_open,
            edgecolor=ink,
            linewidth=0.55,
        )
        accuracy.set_ylim(0, 100)
        accuracy.set_ylabel(
            dataset_label + chr(10) + "Success rate (%)", fontweight="bold"
        )
        accuracy.set_title("Functional and synthesis success")
        accuracy.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")

        token_ax.bar(x, tokens, color=gold, edgecolor=ink, linewidth=0.55)
        token_ax.set_ylabel("Mean tokens (thousands)")
        token_ax.set_title("LLM token cost")

        call_ax.bar(
            x - width / 2,
            calls,
            width,
            color=blue,
            edgecolor=ink,
            linewidth=0.55,
            label="Physical LLM calls",
        )
        call_ax.bar(
            x + width / 2,
            rounds,
            width,
            color=gold_open,
            edgecolor=ink,
            linewidth=0.55,
            label="Repair rounds",
        )
        call_ax.set_ylabel("Count per trajectory")
        call_ax.set_title("Calls and repair depth")
        call_ax.legend(frameon=False, fontsize=8, loc="upper left")

        tool_ax.bar(x, tool_seconds, color=gold, edgecolor=ink, linewidth=0.55)
        tool_ax.set_ylabel("Mean tool seconds")
        tool_ax.set_title("Vitis Reviewer cost")

        for axis in axes[row_index]:
            axis.set_xticks(x, labels, rotation=28, ha="right")
            axis.grid(axis="y", color="#D9DEE3", linewidth=0.65, alpha=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(labelsize=8)

    fig.suptitle(
        "TB&Synth accuracy and per-trajectory cost under the frozen protocol",
        fontsize=16,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.5,
        0.012,
        "Task-bootstrap 95% intervals are shown on success bars; datasets retain separate denominators.",
        ha="center",
        fontsize=9,
        color="#455A64",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    _save(fig, output, "result_cost" if legacy_single else "result_cost_dual_dataset")


def generate_all(
    output: Path,
    analyses: list[tuple[str, Path]] | None = None,
) -> None:
    architecture_figure(output)
    rfl_trajectory_figure(output)
    existing = [
        (label, path) for label, path in (analyses or []) if path.is_file()
    ]
    if existing:
        result_cost_figure(existing, output)
