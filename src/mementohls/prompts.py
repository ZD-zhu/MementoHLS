from __future__ import annotations

import json
from pathlib import Path

from .ercl import MemoryRule
from .model_input_contract import MODEL_INPUT_SCHEMA_VERSION
from .models import Diagnosis


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...<clipped>...\n" + text[-half:]


def _render_rfl_payload(text: str, limit: int) -> str:
    """Bypass the legacy compactor for a validated RFL model-input contract."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _clip_json(text, limit)
    if isinstance(payload, dict) and payload.get("v") == MODEL_INPUT_SCHEMA_VERSION:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if rendered != text:
            raise ValueError("RFL model-input contract is not canonical JSON")
        if len(rendered) > limit:
            raise ValueError("RFL model-input contract exceeds its profile budget")
        return rendered
    return _clip_json(text, limit)


def _clip_json(text: str, limit: int) -> str:
    """Return valid trajectory JSON whose character length never exceeds limit."""
    if limit < 2:
        raise ValueError("JSON clip limit must be at least two characters")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {"memory_error": "invalid JSON omitted"}
    if not isinstance(payload, dict):
        payload = {"memory_error": "non-object JSON omitted"}

    def render(value: dict, *, pretty: bool = False) -> str:
        if pretty:
            return json.dumps(value, ensure_ascii=False, indent=2)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def first_present(value: dict, *keys: str):
        """Read the current schema first, then explicitly supported aliases."""
        for key in keys:
            if key in value:
                return value[key]
        return None

    def bounded_list(value, count: int) -> list:
        if isinstance(value, list):
            return value[:count]
        if value in (None, ""):
            return []
        return [value]

    rendered = render(payload, pretty=True)
    if len(rendered) <= limit:
        return rendered

    attempts = payload.get("attempts")
    while isinstance(attempts, list) and len(attempts) > 1:
        attempts.pop(0)
        payload["older_attempts_omitted"] = True
        rendered = render(payload)
        if len(rendered) <= limit:
            return rendered
    payload.pop("root_cause_counts", None)
    payload.pop("failed_strategies", None)
    rendered = render(payload)
    if len(rendered) <= limit:
        return rendered

    current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
    best = payload.get("best") if isinstance(payload.get("best"), dict) else {}
    stagnation = (
        payload.get("stagnation")
        if isinstance(payload.get("stagnation"), dict)
        else {}
    )
    constraints = (
        payload.get("next_action_constraints")
        if isinstance(payload.get("next_action_constraints"), dict)
        else {}
    )
    # ReviewerGroundedFalsificationLedger.to_prompt() uses round/candidate/failure_signature. The
    # *_index/*_id/*_after names are retained solely for older stored payloads.
    current_round = first_present(current, "round", "round_index")
    current_candidate = first_present(current, "candidate", "candidate_id")
    current_stage = first_present(current, "failure_stage", "stage")
    current_signature = first_present(
        current,
        "failure_signature",
        "failure_signature_after",
        "signature",
    )
    best_candidate = first_present(best, "candidate", "candidate_id")
    if best_candidate is None:
        best_candidate = first_present(
            current, "best_candidate", "best_candidate_id"
        )
    best_outcome = first_present(best, "outcome", "best_outcome")
    if best_outcome is None:
        best_outcome = first_present(current, "best_outcome")
    compact_current = {
        "round": current_round,
        "candidate": current_candidate,
        "stage": current_stage,
        "signature": current_signature,
        "rules": bounded_list(
            first_present(current, "rule_ids", "rules"), 3
        ),
        "action": first_present(current, "action_type", "action"),
        "cause": str(
            first_present(current, "root_cause", "cause") or ""
        )[:120],
        "repeat": first_present(
            current, "repeated_signature_count", "repeat"
        ),
        "no_progress": first_present(
            current, "no_progress_streak", "no_progress"
        ),
    }
    compact = {
        "scope": payload.get("scope") or "current trajectory only",
        "current": {key: value for key, value in compact_current.items() if value not in (None, "", [])},
        "best": {
            "candidate": best_candidate,
            "outcome": best_outcome,
        },
        "stagnation": {
            key: stagnation.get(key)
            for key in (
                "repeated_signature_count",
                "no_progress_streak",
                "frontier_no_improvement_streak",
                "current_duplicate_code",
                "current_two_cycle",
            )
            if stagnation.get(key) is not None
        },
        "next_action_constraints": {
            key: constraints.get(key)
            for key in (
                "requires_new_root_cause_or_action_family",
                "do_not_repeat_mechanism_ids",
                "do_not_repeat_action_fingerprints",
                "contradicted_root_causes",
                "must_not_reintroduce_invariant_violations",
                "must_improve_or_preserve_frontier",
                "if_cleanroom",
            )
            if constraints.get(key) is not None
        },
        "memory_compacted": True,
    }
    rendered = render(compact)
    if len(rendered) <= limit:
        return rendered

    minimal = {
        "scope": payload.get("scope") or "current trajectory only",
        "round": current_round,
        "candidate": current_candidate,
        "stage": current_stage,
        "signature": str(current_signature or "")[:48],
        "best": {
            key: value
            for key, value in {
                "candidate": best_candidate,
                "outcome": best_outcome,
            }.items()
            if value not in (None, "", [])
        },
        "next_action_constraints": {
            key: value
            for key, value in {
                "requires_new_root_cause_or_action_family": constraints.get(
                    "requires_new_root_cause_or_action_family"
                ),
                "do_not_repeat_mechanism_ids": bounded_list(
                    constraints.get("do_not_repeat_mechanism_ids"), 2
                ),
                "must_improve_or_preserve_frontier": constraints.get(
                    "must_improve_or_preserve_frontier"
                ),
            }.items()
            if value not in (None, "", [])
        },
        "repeat": (
            stagnation.get("repeated_signature_count")
            or first_present(current, "repeated_signature_count", "repeat")
        ),
        "no_progress": (
            stagnation.get("no_progress_streak")
            or first_present(current, "no_progress_streak", "no_progress")
        ),
        "frontier_no_improvement": stagnation.get(
            "frontier_no_improvement_streak"
        ),
        "memory_compacted": True,
    }
    minimal = {
        key: value
        for key, value in minimal.items()
        if value not in (None, "", [], {})
    }
    rendered = render(minimal)
    if len(rendered) <= limit:
        return rendered

    best_minimal = minimal.get("best")
    if isinstance(best_minimal, dict):
        best_minimal.pop("outcome", None)
        if not best_minimal:
            minimal.pop("best", None)
    constraint_minimal = minimal.get("next_action_constraints")
    if isinstance(constraint_minimal, dict):
        constraint_minimal.pop("must_improve_or_preserve_frontier", None)
        if not constraint_minimal:
            minimal.pop("next_action_constraints", None)
    rendered = render(minimal)
    if len(rendered) <= limit:
        return rendered

    for key in (
        "frontier_no_improvement",
        "no_progress",
        "repeat",
        "scope",
        "next_action_constraints",
        "best",
        "signature",
        "candidate",
        "round",
        "stage",
    ):
        minimal.pop(key, None)
        rendered = render(minimal)
        if len(rendered) <= limit:
            return rendered
    fallback = '{"memory_compacted":true}'
    return fallback if len(fallback) <= limit else "{}"


_DIAGNOSIS_PROFILES = (
    {"short": 3600, "log": 3200, "code": 4800, "rules": 0},
    {"short": 2400, "log": 2800, "code": 4800, "rules": 1},
    {"short": 1400, "log": 2200, "code": 4500, "rules": 2},
    {"short": 800, "log": 1600, "code": 4000, "rules": 3},
    {"short": 400, "log": 1200, "code": 3500, "rules": 3},
)
_REPAIR_PROFILES = (
    {"original": 8500, "code": 7000, "short": 5000, "diagnosis": 4000, "rules": 0},
    {"original": 7000, "code": 8000, "short": 3200, "diagnosis": 3200, "rules": 1},
    {"original": 5200, "code": 9000, "short": 2000, "diagnosis": 2600, "rules": 2},
    {"original": 3800, "code": 10000, "short": 1200, "diagnosis": 2000, "rules": 3},
    {"original": 2600, "code": 11000, "short": 700, "diagnosis": 1600, "rules": 3},
    {"original": 1600, "code": 9500, "short": 400, "diagnosis": 1200, "rules": 3},
    {"original": 1200, "code": 6500, "short": 250, "diagnosis": 800, "rules": 3},
    {"original": 900, "code": 4500, "short": 180, "diagnosis": 600, "rules": 3},
)
_CLEAN_PROFILES = (
    {"original": 10000, "short": 4200, "diagnosis": 4000, "rules": 0},
    {"original": 8000, "short": 2800, "diagnosis": 3200, "rules": 1},
    {"original": 6000, "short": 1600, "diagnosis": 2600, "rules": 2},
    {"original": 4500, "short": 900, "diagnosis": 2000, "rules": 3},
    {"original": 3000, "short": 500, "diagnosis": 1400, "rules": 3},
    {"original": 1800, "short": 250, "diagnosis": 900, "rules": 3},
    {"original": 1000, "short": 180, "diagnosis": 600, "rules": 3},
)

DIAGNOSIS_PROFILE_COUNT = len(_DIAGNOSIS_PROFILES)
REPAIR_PROFILE_COUNT = len(_REPAIR_PROFILES)
CLEAN_PROFILE_COUNT = len(_CLEAN_PROFILES)
DIAGNOSIS_RFL_BUDGETS = tuple(item["short"] for item in _DIAGNOSIS_PROFILES)
REPAIR_RFL_BUDGETS = tuple(item["short"] for item in _REPAIR_PROFILES)
CLEAN_RFL_BUDGETS = tuple(item["short"] for item in _CLEAN_PROFILES)


def _profile(profiles: tuple[dict[str, int], ...], level: int) -> dict[str, int]:
    if not 0 <= level < len(profiles):
        raise ValueError(f"Unsupported prompt compact_level={level}")
    return profiles[level]


def _rule_prompt_dict(rule: MemoryRule, compact_level: int) -> dict:
    value = rule.to_prompt_dict()
    if compact_level == 0:
        return value
    keep = {
        "rule_id",
        "stage",
        "diagnosis",
        "allowed_actions",
        "forbidden_actions",
        "must_hold_invariants",
        "operator_id",
        "expected_stage_transition",
    }
    if compact_level >= 3:
        keep.discard("diagnosis")
        keep.discard("expected_stage_transition")
    return {key: item for key, item in value.items() if key in keep}


DIAGNOSER_SYSTEM = """You are the Diagnoser in a Vitis HLS code-generation workflow.
Identify one root cause for the current earliest failing stage. Use concrete tool
evidence and only the supplied deterministic ERCL rules when enabled.
Do not write C++ code. Return one JSON object and no Markdown."""


def build_diagnosis_prompt(
    stage: str,
    tool_log: str,
    code: str,
    rules: list[MemoryRule],
    ercl_enabled: bool,
    rfl: str | None = None,
    retrieval_policy: dict | None = None,
    compact_level: int = 0,
) -> str:
    profile = _profile(_DIAGNOSIS_PROFILES, compact_level)
    rule_payload = (
        [_rule_prompt_dict(rule, profile["rules"]) for rule in rules]
        if ercl_enabled
        else []
    )
    return f"""Diagnose the earliest failing HLS stage.

Expected failure_stage: {stage}

Deterministic ERCL retrieval policy and matched rules:
{json.dumps({"policy": retrieval_policy or {}, "rules": rule_payload}, indent=2) if rule_payload else "ERCL disabled or no rule matched."}

Trajectory-local RFL for diagnosis credit assignment:
<TRAJECTORY_MEMORY>
{_render_rfl_payload(rfl, profile["short"]) if rfl else "RFL disabled."}
</TRAJECTORY_MEMORY>

Tool evidence:
<TOOL_LOG>
{_clip(tool_log, profile["log"])}
</TOOL_LOG>

Current generated C++ (empty when parsing failed):
<CURRENT_CODE>
{_clip(code, profile["code"])}
</CURRENT_CODE>

Use exactly one canonical_root_intent_id from:
missing_header_or_declaration, top_signature_mismatch, array_or_pointer_semantics,
initialization_or_state, loop_or_index, inplace_or_dataflow,
numeric_precision_or_width, missing_output_commit, unsynthesizable_construct,
stage_unknown.
Use exactly one canonical_action_intent_id from:
header_include, signature_restore, array_dereference, array_copy_elementwise,
remove_illegal_main, initialize_state, adjust_loop_bound, fix_index_map,
preserve_inplace_order, precision_or_width, commit_output,
remove_unsynthesizable_construct, stage_generic_other.
The target must name an identifier present in CURRENT_CODE/project header, the exact
project header filename, or null. Do not invent a target.

Return exactly this JSON shape:
{{
  "failure_stage": "{stage}",
  "evidence": ["concrete evidence"],
  "root_cause": "one root cause",
  "canonical_root_intent_id": "one frozen root taxonomy ID",
  "canonical_action_intent_id": "one frozen action taxonomy ID",
  "canonical_action_target_id": "a Controller-verifiable target such as param:state, array:out, include:types.h, function:main, or null",
  "predicted_observation": "one concrete error disappearance or stage transition expected after the repair",
  "allowed_actions": ["minimal actions"],
  "forbidden_actions": ["changes that must not be made"],
  "must_hold_invariants": ["interface and functional invariants"],
  "rule_ids": ["only IDs from the matched rules, or an empty list"]
}}"""


REPAIRER_SYSTEM = """You are the Repairer in a Vitis HLS generation workflow.
Repair only the diagnosed earliest failure. The header, testbench, top function,
fixed dimensions, data files, and semantics are immutable. Including the exact
project header is required when its declarations are used and is not a header
modification. Never request, use, infer from, or reproduce a reference kernel.
Return exactly one OUTPUT_CODE XML element containing one complete .cpp file."""


def _rules_payload(
    rules: list[MemoryRule],
    retrieval_policy: dict | None = None,
    compact_level: int = 0,
) -> str:
    if not rules:
        return "ERCL disabled or no rule matched."
    return json.dumps(
        {
            "policy": retrieval_policy or {},
            "rules": [_rule_prompt_dict(rule, compact_level) for rule in rules],
        },
        indent=2,
    )


def build_repair_prompt(
    original_generation_prompt: str,
    filename: str,
    current_code: str,
    diagnosis: Diagnosis,
    rules: list[MemoryRule],
    rfl: str | None,
    retrieval_policy: dict | None = None,
    compact_level: int = 0,
) -> str:
    profile = _profile(_REPAIR_PROFILES, compact_level)
    diagnosis_text = _clip(
        json.dumps(diagnosis.to_dict(), indent=2), profile["diagnosis"]
    )
    return f"""Perform one local repair of the current HLS candidate.

Repair contract:
- Eliminate the validated earliest failure with one concrete edit strategy.
- Do not return byte-identical code or repeat an attempt in RFL.
- Preserve the exact public signature, typedefs, fixed dimensions, and semantics.
- Include the supplied project header exactly; never duplicate its declarations.
- Keep every required helper and use only statically bounded hardware constructs.
- For TB mismatch, explicitly check initialization, bounds, index mapping,
  in-place state, precision/overflow, reduction formula, and output commit.
- For synthesis failure, eliminate unsupported dynamic/recursive/host constructs.
- Do not optimize QoR and do not modify or emit header/testbench files.
- Never use, request, infer from, or reproduce a reference kernel.

Immutable original task:
<ORIGINAL_TASK>
{_clip(original_generation_prompt, profile["original"])}
</ORIGINAL_TASK>

Validated diagnosis:
{diagnosis_text}

Matched ERCL knowledge:
{_rules_payload(rules, retrieval_policy, profile["rules"])}

Compressed RFL state for this trajectory only:
{_render_rfl_payload(rfl, profile["short"]) if rfl else "RFL disabled."}

Current candidate:
<CURRENT_CODE>
{_clip(current_code, profile["code"])}
</CURRENT_CODE>

Output only:
<OUTPUT_CODE name="{Path(filename).name}">
// one complete repaired C++ implementation
</OUTPUT_CODE>"""


def build_cleanroom_prompt(
    original_generation_prompt: str,
    filename: str,
    diagnosis: Diagnosis,
    rules: list[MemoryRule],
    rfl: str,
    trigger_reason: str,
    retrieval_policy: dict | None = None,
    compact_level: int = 0,
) -> str:
    profile = _profile(_CLEAN_PROFILES, compact_level)
    diagnosis_text = _clip(
        json.dumps(diagnosis.to_dict(), indent=2), profile["diagnosis"]
    )
    return f"""Regenerate one clean HLS implementation because local repair stagnated.

Clean-room trigger: {trigger_reason}

Mandatory clean-room contract:
- Derive the algorithm anew only from IMMUTABLE_TASK and its supplied header and
  testbench contract; the prior failed source is intentionally absent.
- Treat the compressed RFL only as constraints describing approaches and
  failure signatures not to repeat, never as source code to patch.
- Include the exact project header and implement the exact top signature.
- Use fixed storage, finite statically bounded loops, and synthesizable C++.
- Preserve dimensions, numerical semantics, in-place behavior, and all outputs.
- Never read, request, or use a reference kernel.
- Produce one candidate; this consumes the current repair round.

<IMMUTABLE_TASK>
{_clip(original_generation_prompt, profile["original"])}
</IMMUTABLE_TASK>

Failure constraint:
{diagnosis_text}

Matched ERCL authorization/constraints:
{_rules_payload(rules, retrieval_policy, profile["rules"])}

Compressed trajectory-only RFL:
{_render_rfl_payload(rfl, profile["short"])}

Output only:
<OUTPUT_CODE name="{Path(filename).name}">
// one complete independently re-derived C++ implementation
</OUTPUT_CODE>"""
