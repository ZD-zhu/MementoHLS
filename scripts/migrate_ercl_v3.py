#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PRESERVED_PRIMARY: set[str] = set()

EXACT_RULES = {
    "COMPILE_CONTRACT_CORDIC_FIXED_SHIFT",
    "COMPILE_CONTRACT_FLOAT64_NATIVE_MULTIPLY",
    "COMPILE_CONTRACT_NEEDWUN_ROW_MAJOR_DP",
    "COMPILE_CONTRACT_PRESENT80_STANDARD",
    "COMPILE_CONTRACT_DES_FEISTEL_SCHEDULE",
    "COMPILE_CONTRACT_FLOAT64_GE",
    "COMPILE_CONTRACT_FLOAT64_LE",
    "COMPILE_CONTRACT_FLOAT64_NATIVE_DIVIDE",
    "COMPILE_CONTRACT_AES128_CIPHER",
    "COMPILE_CONTRACT_MD_GRID_BOUNDED_LJ",
}

REVIEW = [
    "Run the complete Parse, C-sim compile/TB, and synthesis Reviewer on the single produced candidate"
]
GLOBAL_INVARIANTS = [
    "Preserve the exact public TopModule or header-defined signature",
    "Do not read or infer from a reference implementation or testbench body",
    "Use only finite statically bounded synthesizable control and storage",
]
TB_FAILURE_EVIDENCE = [
    "mismatch",
    "expected",
    "actual",
    "Test Failed",
    r"\bpassed\s*[=:]\s*false\b",
    r"\btimeout\s*[=:]\s*true\b",
    r"\breturn_code\s*[=:]\s*-\d+\b",
]


def rule(
    *,
    rule_id: str,
    stage: str,
    priority: int,
    match: dict[str, list[str]],
    diagnosis: str,
    allowed: list[str],
    forbidden: list[str],
    confidence: str = "high",
    operator_id: str | None = None,
    minimum_sources: int = 3,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "rule_id": rule_id,
        "stage": stage,
        "priority": priority,
        "match": match,
        "diagnosis": diagnosis,
        "allowed_actions": allowed,
        "forbidden_actions": forbidden,
        "must_hold_invariants": GLOBAL_INVARIANTS,
        "confidence": confidence,
        "confidence_score": 0.96 if confidence == "high" else 0.72,
        "minimum_evidence_sources": minimum_sources,
        "source": (
            "Public task, generated candidate, and Vitis feedback; "
            "AMD UG1399 and Vitis HLS introductory examples"
        ),
        "expected_stage_transition": (
            "The current earliest failed stage advances without regressing a "
            "previously passed Reviewer component"
        ),
        "validation_required": REVIEW,
    }
    if operator_id:
        value.update(
            {
                "operator_id": operator_id,
                "operator_preconditions": [
                    "The public task, generated code, and tool log jointly prove the edit"
                ],
                "operator_postconditions": [
                    "Only the authorized sites change and the public interface remains exact"
                ],
            }
        )
    return value


NEW_PRIMARY = [
    rule(
        rule_id="COMPILE_AP_INT_CALLABLE_BIT_ACCESS",
        stage="compile",
        priority=195,
        match={
            "log_any": [
                "called object type",
                "does not provide a call operator",
                "no match for call to",
                "expression cannot be used as a function",
            ],
            "code_all": [r"\bap_(?:u)?int\s*<\s*\d+\s*>"],
            "facts_all": ["compile_ap_int_callable_bit_access"],
        },
        diagnosis=(
            "An ap_int/ap_uint value is invoked with function-call syntax even "
            "though the public task and compiler require single-bit indexing."
        ),
        allowed=[
            "Replace only compiler-proven value(index) sites with value[index]"
        ],
        forbidden=[
            "Do not rewrite actual function calls or ap_int constructors",
            "Do not change widths, signedness, or the public signature",
        ],
        operator_id="ap_int_callable_bit_access",
    ),
    rule(
        rule_id="TB_STATIC_RESET_ENABLE_PRIORITY",
        stage="tb",
        priority=192,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_all": [
                r"\b(?:sequential|clock(?:ed)?|flip[- ]?flop|register|counter|shift register|lfsr|finite state machine|fsm|state machine|on each (?:clock|cycle)|rising edge)\b"
            ],
            "facts_all": [
                "task_sequential_state",
                "code_nonstatic_sequential_state_candidate",
            ],
        },
        diagnosis=(
            "State that must survive calls is recreated as an automatic local, "
            "or reset and enable priority is not implemented explicitly."
        ),
        allowed=[
            "Promote only state-like automatic locals to static storage",
            "Apply the public task's reset-versus-enable priority",
        ],
        forbidden=[
            "Do not make combinational temporaries static",
            "Do not change synchronous versus asynchronous behavior",
        ],
        operator_id="sequential_state_storage",
        minimum_sources=2,
    ),
    rule(
        rule_id="TB_OLD_NEXT_STATE_COMMIT",
        stage="tb",
        priority=191,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_all": [
                r"\b(?:sequential|clock(?:ed)?|register|counter|shift register|lfsr|finite state machine|fsm|state machine|on each (?:clock|cycle)|rising edge)\b"
            ],
            "code_any": [r"\b(?:state|next_state|counter|shift|lfsr)\b"],
            "facts_any": ["task_sequential_state"],
            "facts_none": ["code_nonstatic_sequential_state_candidate"],
        },
        diagnosis=(
            "The candidate observes already-updated state where the public "
            "contract requires outputs and next-state to be derived from old state."
        ),
        allowed=[
            "Snapshot old state, derive outputs and next state, then commit once",
            "Keep reset behavior and output timing exactly as stated",
        ],
        forbidden=[
            "Do not introduce an extra cycle of latency",
            "Do not compute Moore outputs from already-committed next state",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="TB_COUNTER_MODULO_CASCADE",
        stage="tb",
        priority=188,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_all": [
                r"\b(?:counter|count from|modulo|decade|saturat|cascade)\b"
            ],
            "code_any": [r"\+\+|--|\+=|-=|%\s*\d+|counter|count"],
            "facts_any": ["task_counter_contract"],
        },
        diagnosis=(
            "Counter recurrence, terminal comparison, rollover/carry, saturation, "
            "or reset/enable priority is inconsistent with the public task."
        ),
        allowed=[
            "Derive the exact terminal value and update order before editing",
            "Generate cascade carry from the pre-update terminal condition",
        ],
        forbidden=[
            "Do not replace saturation with wraparound or vice versa",
            "Do not infer a terminal count from observed test values",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="TB_SHIFT_REGISTER_LFSR_TAPS",
        stage="tb",
        priority=187,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_all": [
                r"\b(?:shift register|lfsr|linear feedback|tap|feedback bit)\b"
            ],
            "code_any": [r"<<|>>|\^|shift|lfsr|feedback"],
            "facts_any": ["task_shift_lfsr"],
        },
        diagnosis=(
            "Shift direction, serial insertion bit, LFSR taps, seed handling, "
            "or old-state feedback timing does not match the public task."
        ),
        allowed=[
            "Write the old-state recurrence bit by bit before compacting it",
            "Use exactly the task-stated taps and shift direction",
        ],
        forbidden=[
            "Do not reverse bit numbering or invent a maximal-length polynomial",
            "Do not update the register before computing old-state feedback",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="TB_FSM_TRANSITION_OUTPUT_TIMING",
        stage="tb",
        priority=186,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_all": [
                r"\b(?:finite state machine|fsm|moore|mealy|state transition)\b"
            ],
            "code_any": [r"switch\s*\(|state|case\s+"],
            "facts_any": ["task_fsm_contract"],
        },
        diagnosis=(
            "FSM state encoding, transition condition, Moore/Mealy output timing, "
            "or old-state/next-state commit order is incorrect."
        ),
        allowed=[
            "Build a complete transition/output table from the public task",
            "Compute Moore outputs from current state and Mealy outputs from current state plus input",
            "Commit next state exactly once after outputs are determined",
        ],
        forbidden=[
            "Do not merge states without a task-proven equivalence",
            "Do not compute Moore output from already-updated state",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="TB_REDUCTION_POPCOUNT_BIT_REVERSE",
        stage="tb",
        priority=185,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_any": [
                r"\b(?:reduction|population count|popcount|bit reverse|reverse bits|leading zeros|count leading)\b"
            ],
            "code_any": [r"\[|\]|\^|&|\||<<|>>|switch\s*\(|if\s*\("],
            "facts_any": [
                "task_reduction_contract",
            ],
        },
        diagnosis=(
            "The reduction identity, accumulation width, bit traversal, popcount, "
            "leading-zero, or bit-reversal mapping violates the public contract."
        ),
        allowed=[
            "Map every visited input bit to its reduction or reversed output position",
            "Initialize reductions with the correct identity and visit each input bit once",
        ],
        forbidden=[
            "Do not use callable syntax for ap_int/ap_uint bit access",
            "Do not hard-code observed vectors",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="TB_PACKED_MUX_PRIORITY_BIT_ORDER",
        stage="tb",
        priority=184,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_any": [
                r"\b(?:multiplexer|mux|priority encoder|encoder|decoder|pack|unpack|concatenat|packed)\b"
            ],
            "code_any": [r"\[|\]|<<|>>|switch\s*\(|if\s*\("],
            "facts_any": ["task_bit_access_pack", "task_mux_encoder"],
        },
        diagnosis=(
            "Packed slice order, mux selection, or encoder priority is reversed, "
            "ambiguous, or incomplete; alternatively, a modular ap_int/ap_uint "
            "loop index cannot represent the exclusive bound and wraps forever."
        ),
        allowed=[
            "Map every packed output slice and define selection priority from the public task",
            "Use explicit ap_uint indexing or range extraction",
            "Use a finite induction type that can represent the exclusive loop bound, or write an explicit bounded priority chain",
            "Assign the no-match default before the scan so a later default cannot overwrite a matched position",
        ],
        forbidden=[
            "Do not silently reverse MSB/LSB numbering",
            "Do not invent a priority direction when the task evidence is insufficient",
            "Do not compare a modular loop counter against a value outside its representable range",
            "Do not overwrite a matched encoder result after break or return",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="TB_TRUTH_TABLE_WAVEFORM_BOOLEAN",
        stage="tb",
        priority=183,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_any": [
                r"\b(?:truth table|waveform|boolean function|logic equation|IEEE[- ]?754|NaN|floating-point comparison)\b"
            ],
            "code_any": [r"\^|&|\||~|<<|>>|switch\s*\(|if\s*\("],
            "facts_any": ["task_truth_waveform", "task_ieee754_bit_exact"],
        },
        diagnosis=(
            "Boolean polarity, truth-table coverage, waveform timing, or a "
            "bit-level predicate is inconsistent with the public contract."
        ),
        allowed=[
            "Derive the complete Boolean table or predicate before editing",
            "Assign every output on every reachable branch",
        ],
        forbidden=[
            "Do not infer expected values from private test output",
            "Do not add state to a combinational contract",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="TB_WIDTH_SIGN_SHIFT_CARRY",
        stage="tb",
        priority=182,
        match={
            "log_any": TB_FAILURE_EVIDENCE,
            "task_any": [
                r"\b(?:bit width|width|signed|unsigned|arithmetic shift|logical shift|carry|overflow|underflow|saturat)\b"
            ],
            "code_any": [r"ap_(?:u)?int\s*<|<<|>>|\+|-|\*"],
            "facts_any": ["task_width_carry"],
        },
        diagnosis=(
            "An intermediate width, signed conversion, shift type, carry, or "
            "saturation boundary truncates information required by the public task."
        ),
        allowed=[
            "Size intermediates for the full mathematical range before narrowing",
            "Make signedness and logical-versus-arithmetic shift intent explicit",
        ],
        forbidden=[
            "Do not widen or change the public interface",
            "Do not mask away a carry required by the output contract",
        ],
        confidence="medium",
    ),
    rule(
        rule_id="COMPILE_SIGNATURE_ARRAY_POINTER_INPLACE",
        stage="compile",
        priority=181,
        match={
            "log_any": [
                "conflicting types",
                "no matching function",
                "cannot convert",
                "invalid conversion",
                "subscripted value",
                "incompatible pointer",
            ],
            "task_any": [
                r"\b(?:function signature|prototype|array|pointer|in-place|in place|TopModule)\b"
            ],
            "code_any": [r"\[[^\]]*\]|\*|&"],
            "facts_any": ["task_signature_array_inplace"],
        },
        diagnosis=(
            "The implementation does not preserve the public function signature, "
            "array extent, pointer level, or in-place ownership semantics."
        ),
        allowed=[
            "Restore the exact public prototype and use the stated array/pointer level",
            "Preserve in-place reads before overwriting aliased outputs",
        ],
        forbidden=[
            "Do not add wrappers, overloads, or alternate top functions",
            "Do not change fixed array dimensions or pointer ownership",
        ],
        confidence="medium",
    ),
]


EMPIRICAL = {
    "COMPILE_UNRESOLVED_HELPER_CLOSURE": {
        "selected": 277,
        "progress": 9,
        "success": 0,
        "regressions": 7,
    },
    "TB_FORMULA_OR_REDUCTION": {
        "selected": 22,
        "progress": 1,
        "success": 0,
        "regressions": 2,
    },
    "TB_SIGNED_WIDTH_SHIFT_MASK": {
        "selected": 825,
        "progress": 21,
        "success": 18,
        "regressions": 16,
    },
    "TB_BRANCH_OUTPUT_COVERAGE": {
        "selected": 168,
        "progress": 12,
        "success": 2,
        "regressions": 27,
    },
}


def sanitize_source_text(value: object) -> str:
    text = str(value)
    replacements = {
        "Bench4HLS": "public instruction-to-HLS benchmark",
        "HLS-Eval": "public HLS benchmark",
        "HLS Eval": "public HLS benchmark",
    }
    for needle, replacement in replacements.items():
        text = text.replace(needle, replacement)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    existing = {item["rule_id"]: item for item in payload["rules"]}
    generic = [
        dict(item)
        for item in payload["rules"]
        if item.get("evidence_tier") == "generic"
    ]
    if len(generic) != 60:
        raise SystemExit(f"expected 60 generic rules, found {len(generic)}")
    for item in generic:
        item["source"] = sanitize_source_text(item.get("source", "public evidence"))
        if item["rule_id"] in EMPIRICAL:
            item["empirical"] = EMPIRICAL[item["rule_id"]]
            item["minimum_evidence_sources"] = max(
                2, int(item.get("minimum_evidence_sources", 1))
            )
    for rule_id in (
        "TB_REDUCTION_IDENTITY_ACCUMULATOR",
        "TB_INPLACE_SEMANTICS",
        "TB_SIGNED_WIDTH_SHIFT_MASK",
        "TB_BRANCH_OUTPUT_COVERAGE",
    ):
        if rule_id in existing:
            item = next(value for value in generic if value["rule_id"] == rule_id)
            item["minimum_evidence_sources"] = 3

    preserved = [dict(existing[rule_id]) for rule_id in sorted(PRESERVED_PRIMARY)]
    exact = [dict(existing[rule_id]) for rule_id in sorted(EXACT_RULES)]
    for item in preserved + exact:
        item["source"] = sanitize_source_text(item.get("source", "public evidence"))
    rules = generic + preserved + NEW_PRIMARY + exact
    if len(rules) != 81 or len({item["rule_id"] for item in rules}) != 81:
        raise SystemExit("v3 bank must contain 81 unique rules")
    policy = dict(payload.get("retrieval_policy") or {})
    policy.update(
        {
            "top_k": 1,
            "fallback_policy": "abstain_to_feedback",
            "minimum_authorization_score": 0.58,
            "minimum_runner_up_margin": 0.04,
            "independent_evidence_policy": (
                "Primary reusable contracts require evidence from the public "
                "task, generated candidate, derived facts, or Vitis log; "
                "dataset labels and problem IDs are forbidden."
            ),
            "empirical_quarantine_policy": (
                "Rules selected at least 20 times with zero success or below "
                "5% progress receive deterministic reliability penalties."
            ),
        }
    )
    payload["retrieval_policy"] = policy
    payload["rules"] = rules
    payload["sources"] = list(
        dict.fromkeys(
            [
                sanitize_source_text(value)
                for value in list(payload.get("sources") or [])
            ]
            + [
                "Public instruction-to-HLS prompts and v11 generated-code/tool-log audit; no testbench or reference implementation content",
                "AMD UG1399 Vitis HLS arbitrary-precision, state, and interface guidance",
                "Xilinx Vitis-HLS-Introductory-Examples synthesizable modeling patterns",
            ]
        )
    )
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        {
            "generic": len(generic),
            "primary_extra": len(preserved) + len(NEW_PRIMARY),
            "all_extra": len(exact),
            "all": len(rules),
        }
    )


if __name__ == "__main__":
    main()
