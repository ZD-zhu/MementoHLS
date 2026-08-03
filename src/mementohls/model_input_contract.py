from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Diagnosis
    from .rfl import ReviewerGroundedFalsificationLedger

MODEL_INPUT_SCHEMA_VERSION = "dac2027-rfl-input-1.1.0"
_ROLE_CODES = {"diagnoser": "d", "repairer": "r", "cleanroom": "c"}
_STAGE_CODES = {"parse": "p", "compile": "c", "tb": "t", "synth": "s", "success": "ok"}
_IMMUTABLE_IDS = ["HDR", "TOP", "DIM", "TB", "NOREF", "NOQOR"]
_STAGE_GUIDANCE = {
    "parse": {
        "focus": "output_format_and_parse_contract",
        "avoid": "semantic_or_qor_change_before_parse",
        "preserve": "immutable_task_and_public_interface",
    },
    "compile": {
        "focus": "declaration_type_signature_and_static_interface",
        "avoid": "functional_guess_without_compile_evidence",
        "preserve": "immutable_public_signature_and_dimensions",
    },
    "tb": {
        "focus": "functional_semantics_state_bounds_index_width_and_output_commit",
        "avoid": "synthesis_only_action_for_tb_failure",
        "preserve": "parse_compile_and_any_passing_later_frontier",
    },
    "synth": {
        "focus": "bounded_synthesizable_construct_and_interface",
        "avoid": "functional_rewrite_without_synthesis_evidence",
        "preserve": "parse_compile_and_any_passing_tb_frontier",
    },
    "success": {
        "focus": "no_repair_required",
        "avoid": "unnecessary_change",
        "preserve": "all_verified_frontiers",
    },
}
_FORBIDDEN_MODEL_KEYS = {
    "trajectory_key",
    "trajectory_key_sha256",
    "benchmark_case_name",
    "task",
    "sample",
    "sample_index",
    "seed",
    "mode",
    "candidate_id",
    "source_candidate_id",
    "code_sha256",
    "changed_span_sha256",
    "tool_seconds",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "path",
    "host",
    "user",
    "ip",
}


@dataclass(frozen=True)
class ModelInputContract:
    role: str
    compact_level: int
    budget_chars: int
    feasible: bool
    json_text: str
    contract_sha256: str
    retained_capsule_ids: list[str] = field(default_factory=list)
    dropped_capsule_ids: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def audit_dict(self) -> dict[str, Any]:
        return asdict(self)


def _opaque(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def _validate_payload(payload: dict[str, Any], text: str, budget: int, role: str) -> list[str]:
    errors: list[str] = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid_json:{exc}"]
    if parsed != payload:
        errors.append("roundtrip_mismatch")
    if len(text) > budget:
        errors.append("budget_exceeded")
    if payload.get("v") != MODEL_INPUT_SCHEMA_VERSION:
        errors.append("schema_mismatch")
    if payload.get("r") != _ROLE_CODES[role]:
        errors.append("role_mismatch")
    leaked = _walk_keys(payload) & _FORBIDDEN_MODEL_KEYS
    if leaked:
        errors.append("forbidden_keys:" + ",".join(sorted(leaked)))
    serialized = text.lower()
    for sentinel in ("unresolved_root", "root_cause", "diagnostic_atoms", "machine_history"):
        if sentinel in serialized:
            errors.append(f"unresolved_or_machine_leak:{sentinel}")
    return errors


def _stage_guidance_capsule(
    *,
    role: str,
    stage: str,
    has_negative_evidence: bool,
) -> dict[str, Any]:
    guidance = dict(_STAGE_GUIDANCE.get(stage, _STAGE_GUIDANCE["compile"]))
    guidance.update(
        {
            "id": "stage_guide",
            "k": "policy",
            "role": role,
            "next": (
                "choose_distinct_evidence_grounded_action_family"
                if has_negative_evidence
                else "choose_one_evidence_grounded_action_family"
            ),
        }
    )
    return guidance


def build_model_input(
    *,
    memory: "ReviewerGroundedFalsificationLedger",
    role: str,
    compact_level: int,
    budget_chars: int,
    chosen_source_stage: str,
    effective_diagnosis: "Diagnosis | None" = None,
) -> ModelInputContract:
    """Build one schema-aware model view; audit identities never enter the payload."""
    if role not in _ROLE_CODES:
        raise ValueError(f"Unsupported model-input role {role!r}")
    if budget_chars < 2:
        raise ValueError("Model-input budget must be at least two characters")

    active = memory.active_ledgers_for()
    veto = [
        _opaque("v", item.family_id)
        for item in active
        if item.status == "vetoed"
    ]
    exhausted = [
        _opaque("e", item.family_id)
        for item in active
        if item.status == "exhausted"
    ]
    mandatory = {
        "v": MODEL_INPUT_SCHEMA_VERSION,
        "r": _ROLE_CODES[role],
        "s": _STAGE_CODES.get(chosen_source_stage, "u"),
        "i": _IMMUTABLE_IDS,
        "x": list(dict.fromkeys(veto))[:5],
        "e": list(dict.fromkeys(exhausted))[:5],
        "c": int(memory.cleanroom_used),
        "r5": 1,
        "ev": "tool",
    }
    text = _render(mandatory)
    if len(text) > budget_chars:
        errors = _validate_payload(mandatory, text, budget_chars, role)
        errors.append("mandatory_core_infeasible")
        return ModelInputContract(
            role=role,
            compact_level=compact_level,
            budget_chars=budget_chars,
            feasible=False,
            json_text="",
            contract_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            validation_errors=list(dict.fromkeys(errors)),
        )

    supported = [
        entry
        for entry in reversed(memory.entries)
        if entry.round_index > 0 and entry.action_status == "supported"
    ][:2]
    negative = [
        entry
        for entry in reversed(memory.entries)
        if entry.round_index > 0 and entry.action_status in {"contradicted", "exhausted"}
    ][:3]
    unresolved_count = sum(
        1
        for entry in memory.entries
        if entry.round_index > 0 and entry.action_status == "unresolved"
    )

    capsules: list[tuple[str, dict[str, Any]]] = [
        (
            "stage_guide",
            _stage_guidance_capsule(
                role=role,
                stage=chosen_source_stage,
                has_negative_evidence=bool(negative or veto or exhausted),
            ),
        )
    ]
    for entry in negative:
        capsule_id = _opaque("n", entry.action_family_id or entry.edit_instance_fingerprint)
        capsules.append(
            (
                capsule_id,
                {
                    "id": capsule_id,
                    "k": "neg",
                    "a": entry.canonical_action_intent_id,
                    "t": entry.canonical_action_target_id,
                    "o": entry.action_status,
                    "b": entry.action_credit_basis[:2],
                    "q": entry.rule_ids[:2],
                    "d": "do_not_repeat_action_family",
                },
            )
        )
    if role != "cleanroom":
        for entry in supported:
            capsule_id = _opaque("p", entry.action_family_id or entry.edit_instance_fingerprint)
            capsules.append(
                (
                    capsule_id,
                    {
                        "id": capsule_id,
                        "k": "pos",
                        "a": entry.canonical_action_intent_id,
                        "t": entry.canonical_action_target_id,
                        "b": entry.action_credit_basis[:2],
                        "d": "reuse_only_when_current_evidence_matches",
                    },
                )
            )
    if unresolved_count:
        capsules.append(("u_count", {"id": "u_count", "k": "cnt", "n": unresolved_count}))

    payload = dict(mandatory)
    retained: list[str] = []
    dropped: list[str] = []
    accepted: list[dict[str, Any]] = []
    for capsule_id, capsule in capsules:
        trial = dict(payload)
        trial["m"] = accepted + [capsule]
        trial_text = _render(trial)
        if len(trial_text) <= budget_chars:
            accepted.append(capsule)
            retained.append(capsule_id)
            payload = trial
            text = trial_text
        else:
            dropped.append(capsule_id)
    if accepted:
        payload["m"] = accepted
        text = _render(payload)

    errors = _validate_payload(payload, text, budget_chars, role)
    contract_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ModelInputContract(
        role=role,
        compact_level=compact_level,
        budget_chars=budget_chars,
        feasible=not errors,
        json_text=text if not errors else "",
        contract_sha256=contract_sha,
        retained_capsule_ids=retained,
        dropped_capsule_ids=dropped,
        validation_errors=errors,
    )
