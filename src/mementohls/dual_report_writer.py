from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .report_writer import claim_evidence_ledger, experiment_report, reviewer_audit


METRICS = [
    "parse_p@1", "parse_p@5", "compile_p@1", "compile_p@5",
    "tb_p@1", "tb_p@5", "synth_p@1", "synth_p@5",
    "tb_and_synth_p@1", "tb_and_synth_p@5",
]
CORE_METRICS = METRICS[2:]
MODES = [
    "zero_shot", "feedback", "rfl", "ercl", "mementohls",
    "mementohls_core_rules", "mementohls_all_rules",
]
LABELS = {
    "zero_shot": "Zero-shot", "feedback": "Feedback", "rfl": "RFL",
    "ercl": "ERCL", "mementohls": "MementoHLS",
    "mementohls_no_executor": "No Executor",
    "mementohls_no_veto": "No Veto",
    "mementohls_no_cleanroom": "No Clean-room",
    "mementohls_core_rules": "Core Rules",
    "mementohls_all_rules": "All Rules",
}


@dataclass(frozen=True)
class DatasetEvidence:
    label: str
    analysis: dict[str, Any]
    run_root: Path
    independent_audit: dict[str, Any] | None = None
    repo_root: Path | None = None


def _pct(value: Any, signed: bool = False) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NA"
    n = 100.0 * float(value)
    return f"{n:+.2f}" if signed else f"{n:.2f}"


def _num(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rows = [[str(v) for v in row] for row in rows]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def _summary(ev: DatasetEvidence, mode: str) -> dict[str, Any]:
    return ev.analysis.get("summaries", {}).get(mode, {})


def _find(rows: list[dict[str, Any]], key: str, value: str,
          metric: str) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value and row.get("metric") == metric:
            return row
    return {}


def _passed(audit: dict[str, Any] | None) -> str:
    if not audit:
        return "未提供"
    value = audit.get("passed")
    if value is None:
        value = audit.get("status") == "PASS"
    return "通过" if value else "未通过"


def _recovery_summary(ev: DatasetEvidence) -> dict[str, Any]:
    audit = ev.independent_audit or {}
    value = audit.get("recovery_deployment_audit")
    return value if isinstance(value, dict) else {}


def _recovery_overview(ev: DatasetEvidence) -> str:
    recovery = _recovery_summary(ev)
    if not recovery:
        return ""
    old_epoch = recovery.get("old_deployment_epoch", "NA")
    new_epoch = recovery.get("new_deployment_epoch", "NA")
    old_count = recovery.get("observed_old_epoch_trajectories", "NA")
    new_count = recovery.get("observed_new_epoch_trajectories", "NA")
    return (
        f"\uFF1B\u53D7\u63A7\u6062\u590D {old_epoch}\u2192{new_epoch}"
        f"\uFF08\u8F68\u8FF9 {old_count}/{new_count}\uFF09\uFF0C"
        "\u72EC\u7ACB\u5BA1\u8BA1\u786E\u8BA4\u6A21\u578B"
        "\u5236\u54C1\u3001\u542F\u52A8\u914D\u7F6E\u4E0E"
        "\u51BB\u7ED3\u6E90\u7801\u8BED\u4E49\u4E00\u81F4"
    )


def _overview(ev: DatasetEvidence) -> str:
    z = _summary(ev, "zero_shot")
    m = _summary(ev, "mementohls")
    zm, mm = z.get("metrics", {}), m.get("metrics", {})
    p1 = float(mm.get("tb_and_synth_p@1", 0)) - float(zm.get("tb_and_synth_p@1", 0))
    p5 = float(mm.get("tb_and_synth_p@5", 0)) - float(zm.get("tb_and_synth_p@5", 0))
    targets = ev.analysis.get("targets", [])
    orders = ev.analysis.get("ordering", [])
    nt = sum(bool(r.get("target_met", r.get("met", False))) for r in targets)
    no = sum(bool(r.get("strict_order_met", r.get("met", False))) for r in orders)
    audit = ev.analysis.get("experiment_audit", {})
    return (
        f"- {ev.label}：{z.get('case_count', audit.get('task_count', 'NA'))} 个任务；"
        f"MementoHLS TB&Synth p@1/p@5={_pct(mm.get('tb_and_synth_p@1'))}%/"
        f"{_pct(mm.get('tb_and_synth_p@5'))}%，相对 Zero-shot "
        f"{_pct(p1, True)}/{_pct(p5, True)} 个百分点；"
        f"+15pp 目标 {nt}/{len(targets)}，预注册排序 {no}/{len(orders)}；"
        f"内置审计{_passed(audit)}，独立审计{_passed(ev.independent_audit)}"
        f"{_recovery_overview(ev)}\u3002"
    )


def _metrics(ev: DatasetEvidence) -> str:
    rows = []
    for mode in MODES:
        s = _summary(ev, mode)
        if not s:
            continue
        m = s.get("metrics", {})
        rows.append([LABELS[mode], *[_pct(m.get(k)) for k in METRICS]])
    return _table(
        ["模式", "Parse p@1", "Parse p@5", "Compile p@1", "Compile p@5",
         "TB p@1", "TB p@5", "Synth p@1", "Synth p@5",
         "TB&Synth p@1", "TB&Synth p@5"], rows
    )


def _factorial(ev: DatasetEvidence) -> str:
    rows = []
    factors = ev.analysis.get("factorial_effects", [])
    for metric in CORE_METRICS:
        r = _find(factors, "effect", "RFL_given_ERCL", metric)
        e = _find(factors, "effect", "ERCL_given_RFL", metric)
        i = _find(factors, "effect", "interaction", metric)
        rows.append([
            metric,
            _pct(r.get("estimate"), True),
            f"[{_pct(r.get('ci95_low'), True)}, {_pct(r.get('ci95_high'), True)}]",
            _pct(e.get("estimate"), True),
            f"[{_pct(e.get('ci95_low'), True)}, {_pct(e.get('ci95_high'), True)}]",
            _pct(i.get("estimate"), True),
            f"[{_pct(i.get('ci95_low'), True)}, {_pct(i.get('ci95_high'), True)}]",
        ])
    return _table(
        ["指标", "RFL|ERCL pp", "95% CI", "ERCL|RFL pp", "95% CI",
         "交互项 pp", "95% CI"], rows
    )


def _paired(ev: DatasetEvidence) -> str:
    pairs = ev.analysis.get("paired_comparisons", [])
    names = [
        "feedback-zero_shot", "rfl-feedback", "ercl-feedback",
        "mementohls-rfl", "mementohls-ercl", "mementohls-zero_shot",
    ]
    rows = []
    for name in names:
        for metric in ("tb_and_synth_p@1", "tb_and_synth_p@5"):
            r = _find(pairs, "comparison", name, metric)
            if r:
                rows.append([
                    name, metric, _pct(r.get("difference"), True),
                    f"[{_pct(r.get('ci95_low'), True)}, "
                    f"{_pct(r.get('ci95_high'), True)}]",
                    _num(r.get("permutation_p"), 4),
                    _num(r.get("holm_p"), 4),
                ])
    return _table(
        ["配对比较", "指标", "差值 pp", "任务级 bootstrap 95% CI",
         "置换 p", "Holm p"], rows
    )


def _rounds(ev: DatasetEvidence) -> str:
    curve = _summary(ev, "mementohls").get("round_cumulative_metrics", {})
    rows, previous1, previous5 = [], None, None
    for r in range(6):
        m = curve.get(str(r), {})
        p1, p5 = m.get("tb_and_synth_p@1"), m.get("tb_and_synth_p@5")
        rows.append([
            r, _pct(p1), _pct(p5),
            "—" if previous1 is None else _pct(float(p1) - previous1, True),
            "—" if previous5 is None else _pct(float(p5) - previous5, True),
        ])
        if isinstance(p1, (int, float)):
            previous1 = float(p1)
        if isinstance(p5, (int, float)):
            previous5 = float(p5)
    return _table(
        ["轮次", "累计 p@1", "累计 p@5", "p@1 边际 pp", "p@5 边际 pp"], rows
    )


def _cost(ev: DatasetEvidence) -> str:
    rows = []
    for mode in ("zero_shot", "feedback", "rfl", "ercl", "mementohls"):
        c = _summary(ev, mode).get("cost", {})
        rows.append([
            LABELS[mode], _num(c.get("mean_repair_rounds"), 3),
            _num(c.get("median_repair_rounds"), 1),
            _num(c.get("p95_repair_rounds"), 1),
            _num(c.get("mean_logical_llm_calls"), 3),
            _num(c.get("mean_physical_llm_calls"), 3),
            _num(c.get("mean_total_tokens"), 1),
            _num(c.get("mean_llm_seconds"), 2),
            _num(c.get("mean_tool_seconds"), 2),
            _num(c.get("mean_critical_path_proxy_seconds"), 2),
        ])
    return _table(
        ["模式", "平均修复轮", "中位", "p95", "逻辑调用", "物理调用",
         "平均 token", "LLM 秒", "工具秒", "关键路径代理秒"], rows
    )


def _mechanisms(ev: DatasetEvidence) -> str:
    s = _summary(ev, "mementohls")
    counts = s.get("mechanism", {}).get("counts", {})
    keys = [
        "ercl_queries", "ercl_abstain", "operator_attempted", "operator_changed",
        "operator_progress", "operator_success", "credit_supported",
        "credit_contradicted", "credit_unresolved", "veto_active_hits",
        "veto_applied", "archive_activated", "cleanroom_requested",
        "cleanroom_triggered", "cleanroom_immediate_progress",
        "cleanroom_immediate_success", "stage_regression",
    ]
    archive = s.get("archive_counterfactual_delta", {})
    clean = ev.analysis.get("repeated_error_cleanroom", {})
    allc, rep = clean.get("all_cleanroom", {}), clean.get("repeated_signature", {})
    return "\n".join([
        _table(["机制", "次数"], [[k, counts.get(k, 0)] for k in keys]),
        "",
        "Archive 轨迹内反事实差值：Compile p@1/p@5="
        f"{_pct(archive.get('compile_p@1'), True)}/"
        f"{_pct(archive.get('compile_p@5'), True)} pp；TB&Synth p@1/p@5="
        f"{_pct(archive.get('tb_and_synth_p@1'), True)}/"
        f"{_pct(archive.get('tb_and_synth_p@5'), True)} pp。",
        "",
        "### 重复错误与 clean-room 的独立机制分析",
        "",
        _table(
            ["子组", "轨迹", "任务", "Memento 成功", "No-clean 成功",
             "轨迹差 pp", "任务平衡差 pp", "任务 bootstrap 95% CI",
             "立即进步", "立即成功"],
            [
                ["全部触发", allc.get("trajectory_count"), allc.get("task_count"),
                 _pct(allc.get("mementohls_success_rate")),
                 _pct(allc.get("no_cleanroom_success_rate")),
                 _pct(allc.get("trajectory_weighted_difference"), True),
                 _pct(allc.get("task_balanced_difference"), True),
                 f"[{_pct(allc.get('task_bootstrap_ci95_low'), True)}, "
                 f"{_pct(allc.get('task_bootstrap_ci95_high'), True)}]",
                 _pct(allc.get("immediate_progress_rate")),
                 _pct(allc.get("immediate_success_rate"))],
                ["重复失败签名", rep.get("trajectory_count"), rep.get("task_count"),
                 _pct(rep.get("mementohls_success_rate")),
                 _pct(rep.get("no_cleanroom_success_rate")),
                 _pct(rep.get("trajectory_weighted_difference"), True),
                 _pct(rep.get("task_balanced_difference"), True),
                 f"[{_pct(rep.get('task_bootstrap_ci95_low'), True)}, "
                 f"{_pct(rep.get('task_bootstrap_ci95_high'), True)}]",
                 _pct(rep.get("immediate_progress_rate")),
                 _pct(rep.get("immediate_success_rate"))],
            ],
        ),
        "",
        "重复通常源于同一失败签名下重复动作、失败代码 SHA 不变、A-B-A 循环、"
        "相互冲突的诊断，或连续两轮阶段向量与 Compile 子前沿都没有改善。"
        "clean-room 返回 immutable task，但保留禁止重做的压缩负面历史。"
        "该子组由机制结果选择，因此是配对观察性解释，不是无偏因果估计。",
    ])


def combined_experiment_report(
    datasets: list[DatasetEvidence],
    *,
    cross_dataset: dict[str, Any] | None = None,
    include_dataset_appendices: bool = True,
) -> str:
    lines = [
        "# MementoHLS DAC2027 双数据集正式实验报告",
        "",
        "## 技术摘要",
        "",
        *[_overview(ev) for ev in datasets],
        "",
        "两套数据分别计算 pass@k 与置信区间，禁止合并不同任务分母。"
        "TB&Synth 必须来自同一候选；QoR 不进入提示、动作仲裁或候选选择。",
        "",
        "## 范围、协议与方法",
        "",
        "实验固定 Meta-Llama-3-8B-Instruct、temperature 0.7、五条 seed trajectory、"
        "Round 0 加最多五轮修复、Vitis HLS 2024.2、"
        "xczu9eg-ffvb1156-2-e 与 5ns。HLS-Eval 使用 94 题，Bench4HLS 仅使用"
        "冻结 test split。两者都是 test-informed development benchmark，"
        "不宣称未见任务泛化。",
        "",
        "Feedback 是无状态工具反馈。RFL（Reviewer-Grounded Falsification Ledger）"
        "是 task/sample/seed 内隔离的轨迹账本，记录失败签名、动作指纹、"
        "supported/contradicted/unresolved 信用、局部 veto、停滞和 archive。"
        "ERCL（Executable Repair Contract Library）是冻结的长期合同库，"
        "以正反证据、历史条件和不变量决定 eligibility，并可授权一个确定性 operator。"
        "MementoHLS 每轮执行唯一动作：高置信 operator；否则获授权 clean-room；"
        "否则 LLM 局部修复。完整 Reviewer 结果再回写 RFL，因而 RFL 与 ERCL"
        "同时工作但不会同时修改代码。",
        "",
    ]
    for ev in datasets:
        lines.extend([
            f"## {ev.label}：十组主结果", "", _metrics(ev), "",
            f"### {ev.label}：RFL 与 ERCL 的 2×2 条件贡献", "",
            _factorial(ev), "",
            f"### {ev.label}：配对统计证据", "", _paired(ev), "",
            f"### {ev.label}：五轮累计与边际收益", "", _rounds(ev), "",
            f"### {ev.label}：轮数、token、调用和工具成本", "", _cost(ev), "",
            f"### {ev.label}：机制、消融与重复错误", "", _mechanisms(ev), "",
        ])
    lines.extend([
        "## 跨数据集解释",
        "",
        "只比较效应方向、量级和成本，不报告 pooled pass@k。方向不一致时报告"
        "异质性，不选择性强调有利数据。",
        "",
    ])
    if cross_dataset is not None:
        lines.extend([
            f"跨数据集结构化综合记录了 "
            f"{len(cross_dataset.get('consistency_rows', cross_dataset.get('direction_consistency', [])))} 条方向比较。",
            "",
        ])
    lines.extend([
        "## 数据质量与有效性威胁",
        "",
        "- 每套数据要求十种模式各有 task×5 个唯一终态，run_errors=0，"
        "Round 0 prompt/response/code SHA 跨模式相同。",
        "- 独立审计重算 pass@k，并检查同候选 TB&Synth、RFL 无未来状态、"
        "严格 JSON、reference kernel 泄漏以及源码和数据哈希。",
        "- Bench4HLS 披露 180 秒 csim runtime guard；这是基础设施超时执行偏差，"
        "不改变已冻结的候选语义协议，但不能从审计中隐藏。",
        "- 多轮方法必然比 Zero-shot 多消耗时间。论文只能讨论准确率—token—"
        "LLM 调用—工具秒交换，不声称 MementoHLS 比 Zero-shot 更快。",
        "- 除轨迹内 archive 反事实外，普通消融和 clean-room 子组均使用"
        "配对观察性措辞；后续轨迹不是严格 counterfactual replay。",
        "- 单模型、单 Vitis、单 part 和两个已知基准限制外部效度；"
        "TB 不是形式验证，本文不优化 QoR。",
        "",
        "## 复现与后续工作",
        "",
        "复现必须绑定模型、Prompt、源码、ERCL、数据划分、调度和环境哈希，"
        "先通过 synthetic 与真实 Vitis smoke test，再运行 Zero-shot 和"
        "Latin-square 修复块，最后执行内置分析、独立审计、跨数据综合与报告生成。"
        "任何改变候选路径的修复都建立新 run，不能覆盖正式证据。",
        "",
        "投稿前优先补充真正 held-out 或 leave-one-family-out、第二模型、"
        "等 token/调用/工具预算曲线，以及 core/primary/all ERCL 的跨家族稳定性。",
    ])
    if include_dataset_appendices:
        for ev in datasets:
            lines.extend([
                "", f"# 附录：{ev.label} 单数据集完整机器生成报告", "",
                experiment_report(ev.analysis, ev.run_root),
            ])
    return "\n".join(lines).rstrip() + "\n"


def combined_reviewer_audit(datasets: list[DatasetEvidence]) -> str:
    rows = []
    for ev in datasets:
        a = ev.analysis.get("experiment_audit", {})
        ia = ev.independent_audit or {}
        epochs = (ia.get("observed") or {}).get("deployment_epochs") or []
        recovery = _recovery_summary(ev)
        rows.append([
            ev.label, a.get("task_count"),
            a.get("expected_trajectory_count_per_mode"),
            _passed(a), a.get("error_count", len(a.get("errors", []))),
            _passed(ia), ia.get("error_count", len(ia.get("errors", []))),
            len(epochs), recovery.get("status", "\u65E0\u6062\u590D"),
        ])
    lines = [
        "# MementoHLS DAC2027 \u5BA1\u7A3F\u7EA7\u8BC1\u636E\u5BA1\u8BA1", "",
        "## \u603B\u7ED3", "",
        _table([
            "\u6570\u636E\u96C6", "\u4EFB\u52A1",
            "\u6BCF\u6A21\u5F0F\u8F68\u8FF9",
            "\u5185\u7F6E\u5BA1\u8BA1", "\u9519\u8BEF",
            "\u72EC\u7ACB\u5BA1\u8BA1", "\u9519\u8BEF",
            "\u90E8\u7F72 epoch", "\u6062\u590D\u5BA1\u8BA1",
        ], rows), "",
        "\u6700\u7EC8 claim \u53EA\u6709\u5728\u4E24\u5957\u6570\u636E"
        "\u5341\u7EC4\u5B8C\u6574\u3001\u4E25\u683C JSON\u3001"
        "\u6CC4\u6F0F\u626B\u63CF\u3001\u540C\u5019\u9009\u7EA6"
        "\u675F\u548C\u72EC\u7ACB\u91CD\u7B97\u5168\u90E8\u901A"
        "\u8FC7\u540E\u624D\u53EF\u5F15\u7528\u3002Bench4HLS runtime "
        "guard \u662F\u5FC5\u987B\u62AB\u9732\u7684\u57FA\u7840"
        "\u8BBE\u65BD\u504F\u5DEE\uFF0C\u4E0D\u7B49\u540C\u4E8E"
        "\u5019\u9009\u8BED\u4E49\u53D8\u66F4\u3002",
    ]
    for ev in datasets:
        recovery = _recovery_summary(ev)
        lines.extend(["", f"## {ev.label} \u8BE6\u7EC6\u5BA1\u8BA1", ""])
        if recovery:
            lines.extend([
                "### \u53D7\u63A7\u90E8\u7F72\u6062\u590D\u62AB\u9732", "",
                f"- \u65E7\u90E8\u7F72 epoch\uFF1A"
                f"{recovery.get('old_deployment_epoch', 'NA')}\uFF1B"
                f"\u65B0\u90E8\u7F72 epoch\uFF1A"
                f"{recovery.get('new_deployment_epoch', 'NA')}\u3002",
                f"- \u65E7/\u65B0 epoch \u7EC8\u6001\u8F68\u8FF9\uFF1A"
                f"{recovery.get('observed_old_epoch_trajectories', 'NA')}/"
                f"{recovery.get('observed_new_epoch_trajectories', 'NA')}\u3002",
                "- \u6062\u590D\u5BA1\u8BA1\u72B6\u6001\uFF1A"
                f"{recovery.get('status', 'NA')}\uFF1B\u4EC5\u5728\u6A21\u578B"
                "\u5236\u54C1\u3001\u542F\u52A8\u53C2\u6570\u3001"
                "\u51BB\u7ED3\u6E90\u7801\u4E0E\u6570\u636E\u5951"
                "\u7EA6\u5168\u90E8\u4E00\u81F4\u65F6\u5141\u8BB8"
                "\u5408\u5E76\u5206\u6790\u3002",
                "- \u8BE5\u6062\u590D\u5C5E\u4E8E\u5FC5\u987B\u62AB"
                "\u9732\u7684\u57FA\u7840\u8BBE\u65BD\u8FDE\u7EED"
                "\u6027\u504F\u5DEE\uFF1B\u4E0D\u80FD\u8868\u8FF0"
                "\u4E3A\u5355\u4E00\u4E0D\u95F4\u65AD vLLM \u8FDB"
                "\u7A0B\u3002", "",
            ])
        lines.append(reviewer_audit(ev.analysis, ev.run_root))
    return "\n".join(lines).rstrip() + "\n"


def combined_claim_ledger(datasets: list[DatasetEvidence]) -> str:
    rows = []
    for ev in datasets:
        z = _summary(ev, "zero_shot").get("metrics", {})
        m = _summary(ev, "mementohls").get("metrics", {})
        for metric in ("tb_and_synth_p@1", "tb_and_synth_p@5"):
            rows.append([
                ev.label, f"MementoHLS 相对 Zero-shot 的 {metric}",
                f"{_pct(m.get(metric))}% vs {_pct(z.get(metric))}%",
                "summaries + paired bootstrap + independent audit",
            ])
        rows.extend([
            [ev.label, "RFL 与 ERCL 各自贡献",
             "2×2 条件主效应与 interaction", "factorial_effects"],
            [ev.label, "五轮收益与成本",
             "轮次、token、调用、工具秒", "round metrics + cost_by_round"],
            [ev.label, "重复错误 clean-room",
             "机制选择的配对观察性子组", "repeated_error_cleanroom"],
        ])
    lines = [
        "# MementoHLS DAC2027 Claim–Evidence Ledger", "",
        _table(["范围", "主张", "定量表达", "结构化证据"], rows), "",
        "## 禁止越界", "",
        "- 不宣称未见任务泛化、全部指标均超过 15pp、全部排序均成立或全部 CI 显著。",
        "- 不把 All Rules 的 test-informed 上界当成主方法。",
        "- 不把普通消融、机制选择子组或 runtime guard 写成随机因果试验。",
        "- 不宣称 QoR 优化或形式正确性。",
    ]
    for ev in datasets:
        lines.extend([
            "", f"## {ev.label} 单数据集 claim ledger", "",
            claim_evidence_ledger(
                ev.analysis, ev.run_root, ev.repo_root or ev.run_root
            ),
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_combined_reports(
    datasets: list[DatasetEvidence],
    output_dir: Path,
    *,
    cross_dataset: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_dir / "MEMENTOHLS_DAC2027_EXPERIMENT_REPORT.md",
        "audit": output_dir / "MEMENTOHLS_DAC2027_REVIEWER_AUDIT.md",
        "ledger": output_dir / "MEMENTOHLS_DAC2027_CLAIM_EVIDENCE_LEDGER.md",
    }
    paths["report"].write_text(
        combined_experiment_report(datasets, cross_dataset=cross_dataset),
        encoding="utf-8",
    )
    paths["audit"].write_text(combined_reviewer_audit(datasets), encoding="utf-8")
    paths["ledger"].write_text(combined_claim_ledger(datasets), encoding="utf-8")
    for ev in datasets:
        slug = ev.label.lower().replace("-", "_").replace(" ", "_")
        (output_dir / f"{slug}_experiment_report.md").write_text(
            experiment_report(ev.analysis, ev.run_root), encoding="utf-8"
        )
        (output_dir / f"{slug}_reviewer_audit.md").write_text(
            reviewer_audit(ev.analysis, ev.run_root), encoding="utf-8"
        )
    return paths
