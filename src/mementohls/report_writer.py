from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import atomic_write_text


METRIC_LABELS = {
    "parse_p@1": "Parse p@1",
    "parse_p@5": "Parse p@5",
    "compile_p@1": "Compile p@1",
    "compile_p@5": "Compile p@5",
    "tb_p@1": "TB p@1",
    "tb_p@5": "TB p@5",
    "synth_p@1": "Synth p@1",
    "synth_p@5": "Synth p@5",
    "tb_and_synth_p@1": "TB&Synth p@1",
    "tb_and_synth_p@5": "TB&Synth p@5",
}
MODE_LABELS = {
    "zero_shot": "Zero-shot",
    "feedback": "Feedback",
    "rfl": "RFL",
    "ercl": "ERCL",
    "mementohls": "MementoHLS",
    "mementohls_no_executor": "No Executor",
    "mementohls_no_veto": "No Veto",
    "mementohls_no_cleanroom": "No Clean-room",
    "mementohls_core_rules": "Core ERCL (60)",
    "mementohls_all_rules": "All Rules (81)",
}


def _pct(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{100.0 * value:.2f}%"


def _num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([head, separator, *body])


def _paired_index(analysis: dict[str, Any]) -> dict[tuple[str, str], dict]:
    return {
        (row["comparison"], row["metric"]): row
        for row in analysis["paired_comparisons"]
    }


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_report(analysis: dict[str, Any], run_root: Path) -> str:
    summaries = analysis["summaries"]
    paired = _paired_index(analysis)
    audit = analysis["experiment_audit"]
    dataset_kind = str(audit.get("dataset_kind") or "unknown")
    task_count = int(audit.get("task_count") or 0)
    dataset_label = "Bench4HLS" if dataset_kind == "bench4hls" else "HLS-Eval"
    target_pass = sum(row["target_met"] for row in analysis["targets"])
    order_pass = sum(row["strict_order_met"] for row in analysis["ordering"])
    lines = [
        "# MementoHLS DAC2027 实验报告",
        "",
        f"> 证据边界：本报告是同一套 {task_count} 题 {dataset_label} 上的 "
        "**test-informed development benchmark**，不是 held-out "
        "泛化证据。",
        "",
        "## 1. 结论先行",
        "",
        f"- 完整性审计：**{_status(bool(audit['passed']))}**"
        f"（{audit['error_count']} 个错误）。",
        f"- +15 个百分点或 Parse 守门目标："
        f"**{target_pass}/{len(analysis['targets'])}** 达标。",
        f"- 预注册主排序与八项次要排序："
        f"**{order_pass}/{len(analysis['ordering'])}** 通过。",
        f"- MementoHLS 相对 Zero-shot 的八项非 Parse 指标"
        f" bootstrap CI 下界均大于 0："
        f"**{_status(bool(analysis['all_statistical_checks_met']))}**。",
        "",
    ]
    full = summaries["mementohls"]["metrics"]
    zero = summaries["zero_shot"]["metrics"]
    for key in ("tb_and_synth_p@1", "tb_and_synth_p@5"):
        lines.append(
            f"- {METRIC_LABELS[key]}：{_pct(zero[key])} → "
            f"{_pct(full[key])}，变化 {_pct(full[key] - zero[key], True)}。"
        )

    lines += ["", "## 2. 十组主结果", ""]
    headers = ["Mode", *METRIC_LABELS.values()]
    rows = []
    for mode in analysis["modes"]:
        metrics = summaries[mode]["metrics"]
        rows.append(
            [
                MODE_LABELS[mode],
                *[_pct(metrics[key]) for key in METRIC_LABELS],
            ]
        )
    lines += [_table(headers, rows), ""]

    lines += ["## 3. 预注册排序、差值与置信区间", ""]
    order_rows = []
    for row in analysis["ordering"]:
        comparison = f"{row['candidate']}-{row['baseline']}"
        stat = paired[(comparison, row["metric"])]
        order_rows.append(
            [
                comparison,
                METRIC_LABELS[row["metric"]],
                _pct(row["difference"], True),
                f"[{_pct(stat['ci95_low'], True)}, "
                f"{_pct(stat['ci95_high'], True)}]",
                _status(row["strict_order_met"]),
            ]
        )
    lines += [
        _table(
            ["Comparison", "Metric", "Delta", "95% CI", "Strict order"],
            order_rows,
        ),
        "",
    ]
    failed = [row for row in analysis["ordering"] if not row["strict_order_met"]]
    if failed:
        lines += [
            "### 3.1 排序失败项和后续优化约束",
            "",
            "以下失败项不会被删除或跨版本拼接。"
            "后续只允许在新的源码、Prompt、ERCL/RFL 和"
            " Controller 哈希下建立 development iteration，"
            "并重跑受影响模式。",
            "",
        ]
        for row in failed:
            stat = paired[
                (f"{row['candidate']}-{row['baseline']}", row["metric"])
            ]
            lines.append(
                f"- {row['candidate']} vs {row['baseline']} / "
                f"{METRIC_LABELS[row['metric']]}: "
                f"{_pct(row['difference'], True)}, "
                f"95% CI [{_pct(stat['ci95_low'], True)}, "
                f"{_pct(stat['ci95_high'], True)}]。"
            )
    else:
        lines += [
            "所有预注册排序均满足；不需要进入"
            "结果驱动的新版本。",
            "",
        ]

    lines += ["", "## 4. RFL 与 ERCL 的独立贡献和交互", ""]
    factorial = [
        row
        for row in analysis["factorial_effects"]
        if row["metric"] in ("tb_and_synth_p@1", "tb_and_synth_p@5")
    ]
    lines += [
        _table(
            ["Effect", "Metric", "Estimate", "95% CI"],
            [
                [
                    row["effect"],
                    METRIC_LABELS[row["metric"]],
                    _pct(row["estimate"], True),
                    f"[{_pct(row['ci95_low'], True)}, "
                    f"{_pct(row['ci95_high'], True)}]",
                ]
                for row in factorial
            ],
        ),
        "",
        "- RFL|ERCL = MementoHLS - ERCL，表示 ERCL 已开启时"
        " RFL 的边际贡献。",
        "- ERCL|RFL = MementoHLS - RFL，表示 RFL 已开启时"
        " ERCL 的边际贡献。",
        "- Interaction = MementoHLS - ERCL - RFL + Feedback。"
        "区间跨 0 时只解读为点估计，不声称显著协同。",
        "",
    ]

    lines += ["## 5. 机制消融", ""]
    ablation_pairs = [
        ("mementohls-mementohls_core_rules", "71 vs 60: algorithm-family contracts"),
        ("mementohls_all_rules-mementohls", "81 vs 71: exact-task contracts"),
    ]
    ablation_rows = []
    for comparison, label in ablation_pairs:
        for metric in ("tb_and_synth_p@1", "tb_and_synth_p@5"):
            row = paired.get((comparison, metric))
            if row is None:
                continue
            ablation_rows.append(
                [
                    label,
                    METRIC_LABELS[metric],
                    _pct(row["difference"], True),
                    f"[{_pct(row['ci95_low'], True)}, "
                    f"{_pct(row['ci95_high'], True)}]",
                ]
            )
    archive = summaries["mementohls"]["archive_counterfactual_delta"]
    for metric in ("tb_and_synth_p@1", "tb_and_synth_p@5"):
        ablation_rows.append(
            [
                "Archive (same-trajectory counterfactual)",
                METRIC_LABELS[metric],
                _pct(archive[metric], True),
                "same candidates/calls/tools",
            ]
        )
    lines += [
        _table(["Mechanism", "Metric", "Paired delta", "95% CI / design"], ablation_rows),
        "",
        "这些差值中，archive 是预算完全相同的同轨迹"
        "反事实；其他消融共享 Round 0，但修复阶段"
        "不是严格 causal replay，因此报告 paired observational difference。",
        "",
    ]

    mechanism = summaries["mementohls"]["mechanism"]["counts"]
    exposure_rows = [
        ["ERCL queries", mechanism.get("ercl_queries", 0)],
        ["ERCL abstain", mechanism.get("ercl_abstain", 0)],
        ["Operator attempted", mechanism.get("operator_attempted", 0)],
        ["Operator changed", mechanism.get("operator_changed", 0)],
        ["Operator progress", mechanism.get("operator_progress", 0)],
        ["Operator success", mechanism.get("operator_success", 0)],
        ["Veto applied", mechanism.get("veto_applied", 0)],
        ["Clean-room triggered", mechanism.get("cleanroom_triggered", 0)],
        ["Archive activated", mechanism.get("archive_activated", 0)],
    ]
    lines += [_table(["Exposure", "Count"], exposure_rows), ""]

    lines += ["## 6. 重复错误与 clean-room 独立分析", ""]
    clean = analysis["repeated_error_cleanroom"]
    clean_rows = []
    for key, label in (
        ("all_cleanroom", "All triggered"),
        ("repeated_signature", "Repeated signature"),
    ):
        value = clean[key]
        clean_rows.append(
            [
                label,
                value["task_count"],
                value["trajectory_count"],
                _pct(value["mementohls_success_rate"]),
                _pct(value["no_cleanroom_success_rate"]),
                _pct(value["task_balanced_difference"], True),
                f"[{_pct(value['task_bootstrap_ci95_low'], True)}, "
                f"{_pct(value['task_bootstrap_ci95_high'], True)}]",
                _pct(value["immediate_progress_rate"]),
            ]
        )
    lines += [
        _table(
            [
                "Subgroup", "Tasks", "Traj.", "Memento success",
                "No-clean success", "Task-balanced delta", "95% CI",
                "Immediate progress",
            ],
            clean_rows,
        ),
        "",
        clean["interpretation"],
        "",
        "重复错误常来自日志路径/行号噪声、"
        "局部补丁未触及真正接口契约、模型换用文字"
        "但重复同一动作，以及 A-B-A 坏状态摆动。"
        "RFL 使用规范化签名、代码 SHA 和动作指纹合并"
        "这些表面差异。",
        "",
    ]

    lines += ["## 7. Round 0–5 收益", ""]
    gain_rows = []
    for mode in ("feedback", "rfl", "ercl", "mementohls"):
        for row in analysis["round_gains"]:
            if row["mode"] == mode:
                gain_rows.append(
                    [
                        MODE_LABELS[mode],
                        row["round"],
                        _pct(row["cumulative_tb_and_synth_p@1"]),
                        _pct(row["marginal_tb_and_synth_p@1"], True),
                        _pct(row["cumulative_tb_and_synth_p@5"]),
                        _pct(row["marginal_tb_and_synth_p@5"], True),
                    ]
                )
    lines += [
        _table(
            ["Mode", "Round", "Cum. p@1", "Marginal p@1", "Cum. p@5", "Marginal p@5"],
            gain_rows,
        ),
        "",
    ]

    lines += ["## 8. 迭代、Token、调用和工具成本", ""]
    cost_rows = []
    for mode in analysis["modes"]:
        cost = summaries[mode]["cost"]
        cost_rows.append(
            [
                MODE_LABELS[mode],
                _num(cost["mean_repair_rounds"]),
                _num(cost["median_repair_rounds"]),
                _num(cost["p95_repair_rounds"]),
                _num(cost["mean_logical_llm_calls"]),
                _num(cost["mean_physical_llm_calls"]),
                _num(cost["mean_total_tokens"], 1),
                _num(cost["mean_llm_seconds"], 2),
                _num(cost["mean_tool_seconds"], 2),
                _num(cost["mean_critical_path_proxy_seconds"], 2),
            ]
        )
    lines += [
        _table(
            [
                "Mode", "Mean rounds", "Median", "p95", "Logical calls",
                "Physical calls", "Tokens", "LLM s", "Tool s", "Proxy s",
            ],
            cost_rows,
        ),
        "",
        "Zero-shot 预期拥有最短延迟。MementoHLS 不宣称比"
        " Zero-shot 更快；公平问题是它相对 Feedback 为正确性"
        "付出多少 token、调用和 Vitis 秒，以及是否以"
        "更少平均修复轮数得到更高正确率。",
        "",
        "等 token、等 logical/physical LLM 调用和等工具秒的"
        " Pareto 数据见 budget_fairness_and_pareto.csv。",
        "",
    ]

    lines += [f"## 9. {dataset_label} 来源/套件分层", ""]
    suite_rows = [
        [
            MODE_LABELS[row["mode"]],
            row["suite"],
            row["case_count"],
            _pct(row["tb_and_synth_p@1"]),
            _pct(row["tb_and_synth_p@5"]),
        ]
        for row in analysis["suite_stratified"]
        if row["mode"] in ("zero_shot", "feedback", "rfl", "ercl", "mementohls")
    ]
    lines += [
        _table(["Mode", "Suite", "Cases", "TB&Synth p@1", "TB&Synth p@5"], suite_rows),
        "",
    ]

    lines += [
        "## 10. 局限与有效性威胁",
        "",
        "- 只有一个 8B 模型、Vitis 2024.2、一个 FPGA part。",
        f"- 同一 {task_count} 题 {dataset_label} 参与了方法开发，不能声称未见任务泛化。",
        "- 正式主库为 71 条（60 条通用 + 11 条可复用算法族合同）；60 条 Core 是内容消融；81 条 All Rules 额外包含 10 条精确任务合同，只是敏感性上界。",
        "- 消融修复轨迹不是严格 counterfactual replay。",
        "- TB 通过不等于形式化功能证明。",
        "- QoR 只记录，不进入 Prompt、仲裁或候选选择。",
        "",
        "## 11. 结果文件",
        "",
        f"- 运行根目录：{run_root}",
        "- analysis.json：所有表和结论的结构化源。",
        "- paired_bootstrap_and_permutation.csv：配对区间与置换检验。",
        "- factorial_rfl_ercl_effects.csv：RFL/ERCL 边际效应与交互。",
        "- round_cumulative_and_marginal.csv：五轮累计和边际收益。",
        "- repeated_error_cleanroom_*.csv/json：重复错误子组。",
        "- budget_fairness_and_pareto.csv：公平预算曲线。",
        "",
    ]
    return "\n".join(lines)


def reviewer_audit(analysis: dict[str, Any], run_root: Path) -> str:
    audit = analysis["experiment_audit"]
    dataset_kind = str(audit.get("dataset_kind") or "unknown")
    task_count = int(audit.get("task_count") or 0)
    dataset_label = "Bench4HLS" if dataset_kind == "bench4hls" else "HLS-Eval"
    freeze = _read_json(run_root / "protocol" / "freeze_manifest.json")
    freeze_pass = bool(freeze and freeze.get("status") == "pass")
    mode_count = len(analysis.get("modes") or {})
    criteria = [
        (
            "数据完整性",
            audit["passed"],
            f"{mode_count} x {audit.get('expected_trajectory_count_per_mode', 0)} trajectories, unique keys, empty run_errors",
        ),
        (
            "Round 0 配对身份",
            audit["passed"],
            "prompt/response/code SHA consistency",
        ),
        (
            "同候选 TB&Synth",
            audit["passed"],
            "same candidate ID required",
        ),
        (
            "泄漏控制",
            audit["passed"],
            "reference-kernel provenance and scan",
        ),
        (
            "源码/规则/环境冻结",
            freeze_pass,
            "freeze manifest and hashes",
        ),
        (
            "+15pp 目标",
            analysis["all_targets_met"],
            "engineering goal, not a publication filter",
        ),
        (
            "预注册排序",
            analysis["all_preregistered_orders_met"],
            "failed rows must remain reported",
        ),
        (
            "非 Parse 统计区间",
            analysis["all_statistical_checks_met"],
            "paired task bootstrap lower bound > 0",
        ),
        (
            "成本公平性",
            bool(analysis.get("budget_curves")),
            "round/token/call/tool-second curves",
        ),
        (
            "未见任务泛化",
            False,
            f"blocked: test-informed {task_count}-task {dataset_label} development benchmark",
        ),
    ]
    score = {
        "原创性边界": 4,
        "技术正确性": 5 if audit["passed"] else 2,
        "定量证据": 4 if bool(analysis.get("paired_comparisons")) else 2,
        "可复现性": 5 if freeze_pass and audit["passed"] else 3,
        "表达与图表": 4,
        "外部有效性": 3,
    }
    lines = [
        "# MementoHLS DAC2027 审稿人审计",
        "",
        "## 1. 硬门槛",
        "",
        _table(
            ["Criterion", "Status", "Evidence / blocker"],
            [[name, _status(value), evidence] for name, value, evidence in criteria],
        ),
        "",
        "## 2. 0–5 分自审",
        "",
        _table(
            ["Dimension", "Score"],
            [[name, value] for name, value in score.items()],
        ),
        "",
        f"总分：**{sum(score.values())}/30**。",
        "",
        "文字和 artifact 质量可以继续打磨，但不会用"
        "修辞覆盖实验边界。在增加 held-out 或"
        " leave-one-family-out 之前，外部有效性仍是投稿阻塞项。",
        "",
        "## 3. 审稿人最可能的问题",
        "",
        f"1. 71 条 primary ERCL 中的通用与算法族合同能否在 {dataset_label} 外部数据上复现？"
        "需要 held-out/LOFO 证据。",
        "2. RFL 和 ERCL 的提升是否只来自更多 token？"
        "用等预算 Pareto 曲线回答。",
        "3. Executor/veto/clean-room 消融是否因果？"
        "当前只声称配对观察差异，archive 除外。",
        "4. 为什么同时需要 RFL 和 ERCL？"
        "2x2 边际效应和 interaction 是核心证据。",
        "5. 第 3–5 轮是否值得？"
        "必须同时展示边际收益、轨迹数和成本。",
        "",
        "## 4. 投稿前必补",
        "",
        "- 冻结后的 held-out 或 leave-one-family-out。",
        "- 第二个模型或有代表性的跨模型子集。",
        "- 若要强因果表述，需要严格 counterfactual replay。",
        "- DAC 2027 CFP 发布后重新核对页数、匿名和 artifact 规则。",
        "",
    ]
    return "\n".join(lines)


def claim_evidence_ledger(
    analysis: dict[str, Any], run_root: Path, repo_root: Path
) -> str:
    audit = analysis["experiment_audit"]
    dataset_kind = str(audit.get("dataset_kind") or "unknown")
    task_count = int(audit.get("task_count") or 0)
    dataset_label = "Bench4HLS" if dataset_kind == "bench4hls" else "HLS-Eval"
    audit_pass = bool(audit["passed"])
    rows = [
        [
            "C1",
            "MementoHLS changes TB&Synth versus Zero-shot",
            "SUPPORTED" if audit_pass else "BLOCKED",
            "analysis.json; paired_bootstrap_and_permutation.csv",
            "Point estimate and task-paired CI only",
        ],
        [
            "C2",
            "RFL contribution conditional on ERCL",
            "SUPPORTED" if audit_pass else "BLOCKED",
            "factorial_rfl_ercl_effects.csv",
            "RFL|ERCL; CI crossing zero means point estimate only",
        ],
        [
            "C3",
            "ERCL contribution conditional on RFL",
            "SUPPORTED" if audit_pass else "BLOCKED",
            "factorial_rfl_ercl_effects.csv",
            "ERCL|RFL; CI crossing zero means point estimate only",
        ],
        [
            "C4",
            "RFL can falsify trajectory-local hypotheses",
            "SUPPORTED",
            "src/mementohls/rfl.py; per-round rfl_state",
            "Mechanism definition and outcome credit",
        ],
        [
            "C5",
            "ERCL contracts are executable and evidence gated",
            "SUPPORTED",
            "src/mementohls/ercl.py; executor.py; memorybank/*.yaml",
            "60 generic core, 71 primary, 81 exact-contract sensitivity",
        ],
        [
            "C6",
            "Archive contribution under identical candidate budget",
            "SUPPORTED" if audit_pass else "BLOCKED",
            "archive_counterfactual.csv",
            "Same trajectories/calls/tool runs",
        ],
        [
            "C7",
            "Clean-room helps repeated-error subgroup",
            "QUALIFIED",
            "repeated_error_cleanroom_*.csv/json",
            "Mechanism-selected observational subgroup, not causal",
        ],
        [
            "C8",
            "MementoHLS is faster than Zero-shot",
            "FORBIDDEN",
            "main_metrics_and_cost.csv",
            "Iterative method is expected to cost more wall time",
        ],
        [
            "C9",
            "Generalizes to unseen HLS tasks",
            "FORBIDDEN",
            "PREREGISTRATION.json",
            f"The same {task_count} {dataset_label} tasks informed development",
        ],
        [
            "C10",
            "Optimizes QoR",
            "FORBIDDEN",
            "configs/experiment.yaml; orchestrator.py",
            "QoR is observational and never controls selection",
        ],
    ]
    lines = [
        "# MementoHLS Claim–Evidence Ledger",
        "",
        f"- Run root: {run_root}",
        f"- Repository root: {repo_root}",
        f"- Experiment audit: {_status(audit_pass)}",
        "",
        _table(
            ["ID", "Claim", "Status", "Evidence", "Qualification"],
            rows,
        ),
        "",
        "所有数字性 claim 必须能追溯到 analysis.json 的"
        "机器生成字段。若报告与 CSV/JSON 不一致，"
        "以结构化证据为准，并将报告计算标记为需修复。",
        "",
    ]
    return "\n".join(lines)


def write_all_reports(
    analysis: dict[str, Any],
    *,
    output_dir: Path,
    run_root: Path,
    repo_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "MEMENTOHLS_DAC2027_EXPERIMENT_REPORT.md",
        experiment_report(analysis, run_root),
    )
    atomic_write_text(
        output_dir / "MEMENTOHLS_DAC2027_REVIEWER_AUDIT.md",
        reviewer_audit(analysis, run_root),
    )
    atomic_write_text(
        output_dir / "MEMENTOHLS_DAC2027_CLAIM_EVIDENCE_LEDGER.md",
        claim_evidence_ledger(analysis, run_root, repo_root),
    )
