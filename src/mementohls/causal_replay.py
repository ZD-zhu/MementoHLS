from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from .models import LLMCallResult, RunMode, file_sha256
from .provenance import canonical_sha256
from .rfl import normalize_failure_text


CAUSAL_REPLAY_VERSION = "2-canonical-global-diagnosis-guards"
CONTROL_MECHANISMS = {
    RunMode.MEMENTOHLS_NO_CLEANROOM: "cleanroom",
    RunMode.MEMENTOHLS_NO_EXECUTOR: "postprocess",
}
# ERCL policy appends these guards after replay identity is checked, while
# completed Full trajectories persist the post-policy diagnosis. Treat them as
# implicit policy in both views; every other diagnosis value remains exact.
MANDATORY_GLOBAL_FORBIDDEN_ACTIONS = (
    "Never modify headers, testbenches, data files, top function, or fixed dimensions",
    "Never retrieve or prompt the reference kernel",
    "Never use QoR to choose a candidate",
)


# `full_no_format_recovery` is intentionally deferred until the controller
# persists the raw pre-retry response and retry activation as separate events;
# without those artifacts an exact activation branch cannot be audited.
_REVIEW_IDENTITY_FIELDS = (
    "pass_parse",
    "pass_compile",
    "pass_tb",
    "pass_synth",
    "pass_tb_and_synth",
    "failure_stage",
    "generated_filename",
)


class CausalReplayMismatch(RuntimeError):
    """The control no longer has the exact Full prefix required for attribution."""


def canonical_prompt_log(raw_log: str) -> str:
    """Remove path/time/tool-run volatility before a prompt SHA is computed.

    Rule matching and full raw logs remain unchanged. Only the Diagnoser prompt
    receives this deterministic evidence view, so Full and its causal controls
    can share an exact prompt-response cache.
    """
    normalized = normalize_failure_text(raw_log, limit=6000)
    return "CANONICAL REVIEW EVIDENCE:\n" + (normalized or "no diagnostic text")


def canonical_diagnosis_identity(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Return the diagnosis identity after applying mandatory global guards.

    Full persists a diagnosis after ERCL appends the global veto, whereas
    a causal control verifies the replayed diagnosis immediately before that
    append. The veto is deterministic policy rather than an LLM-side decision,
    so both representations canonically contain it exactly once and in one
    fixed order. Non-global actions, their order, and all other fields are left
    untouched so substantive replay divergence is still rejected.
    """

    canonical = dict(diagnosis)
    actions = canonical.get("forbidden_actions")
    if actions is None:
        actions = []
    if not isinstance(actions, list) or not all(
        isinstance(action, str) for action in actions
    ):
        # Preserve malformed values verbatim. The schema validator normally
        # prevents this, and identity comparison must not conceal corruption.
        return canonical
    mandatory = frozenset(MANDATORY_GLOBAL_FORBIDDEN_ACTIONS)
    canonical["forbidden_actions"] = [
        action for action in actions if action not in mandatory
    ] + list(MANDATORY_GLOBAL_FORBIDDEN_ACTIONS)
    return canonical


def _review_identity(review: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(review.get(key) for key in _REVIEW_IDENTITY_FIELDS)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key(case_name: str, sample_index: int, seed: int) -> str:
    return f"{case_name}\0{sample_index}\0{seed}"


def _activation_round(payload: dict[str, Any], mechanism: str) -> int | None:
    for round_payload in (payload.get("rounds") or [])[1:]:
        if mechanism == "cleanroom" and (
            round_payload.get("cleanroom") or {}
        ).get("triggered"):
            return int(round_payload["round_index"])
        if mechanism == "postprocess" and (
            round_payload.get("operator") or {}
        ).get("changed"):
            return int(round_payload["round_index"])
    return None


def build_full_source_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors_path = root / "run_errors.json"
    if not errors_path.is_file() or json.loads(errors_path.read_text()) != []:
        raise ValueError("Causal Full source requires an empty run_errors.json")
    entries: dict[str, dict[str, Any]] = {}
    bindings: set[tuple[Any, ...]] = set()
    for path in sorted(root.glob("*/sample__*/trajectory.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "complete"
            or payload.get("terminal") is not True
            or payload.get("mode") != RunMode.MEMENTOHLS.value
        ):
            raise ValueError(f"Invalid causal Full source trajectory: {path}")
        case_name = str(payload["benchmark_case_name"])
        sample_index = int(payload["sample_index"])
        seed = int(payload["seed"])
        identity = _key(case_name, sample_index, seed)
        if identity in entries:
            raise ValueError(f"Duplicate causal Full source identity: {identity}")
        bindings.add(
            (
                payload.get("iteration_id"),
                payload.get("source_sha256"),
                payload.get("dataset_manifest_sha256"),
                payload.get("hls_eval_source_sha256"),
                payload.get("deployment_epoch"),
            )
        )
        entries[identity] = {
            "case": case_name,
            "sample_index": sample_index,
            "seed": seed,
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "round_count": len(payload.get("rounds") or []),
            "cleanroom_activation_round": _activation_round(payload, "cleanroom"),
            "postprocess_activation_round": _activation_round(payload, "postprocess"),
        }
    if len(bindings) != 1:
        raise ValueError("Causal Full source trajectories do not share one binding")
    manifest: dict[str, Any] = {
        "version": CAUSAL_REPLAY_VERSION,
        "source_mode": RunMode.MEMENTOHLS.value,
        "root": str(root),
        "trajectory_count": len(entries),
        "binding": list(next(iter(bindings))) if bindings else [],
        "entries": entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


@dataclass(frozen=True)
class ReplayCall:
    call: LLMCallResult
    reused_from: str
    effective_prompt: str


class CausalReplayPlan:
    def __init__(
        self,
        *,
        source_root: Path,
        source_path: Path,
        source_sha256: str,
        payload: dict[str, Any],
        control_mode: RunMode,
        source_manifest_sha256: str,
    ) -> None:
        self.source_root = source_root
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self.payload = payload
        self.control_mode = control_mode
        self.mechanism = CONTROL_MECHANISMS[control_mode]
        self.activation_round = _activation_round(payload, self.mechanism)
        self.source_manifest_sha256 = source_manifest_sha256
        self._llm_fields = {field.name for field in fields(LLMCallResult)}

    def source_round(self, round_index: int) -> dict[str, Any]:
        rounds = self.payload.get("rounds") or []
        if round_index < 0 or round_index >= len(rounds):
            raise CausalReplayMismatch(
                f"Full source has no round {round_index}: {self.source_path}"
            )
        return rounds[round_index]

    def should_replay(self, role: str, round_index: int) -> bool:
        if role not in {"diagnosis", "repair"}:
            raise ValueError(f"Unknown causal replay role: {role}")
        if self.activation_round is None:
            return True
        if round_index < self.activation_round:
            return True
        return role == "diagnosis" and round_index == self.activation_round

    def assert_parent(
        self,
        round_index: int,
        *,
        parent_round_index: int,
        response_sha256: str,
        code_sha256: str | None,
        review: dict[str, Any],
    ) -> None:
        source_round = self.source_round(round_index)
        source_parent_id = source_round.get("parent_candidate_id")
        source_parent = next(
            (
                value
                for value in self.payload.get("rounds") or []
                if value.get("candidate_id") == source_parent_id
            ),
            None,
        )
        if source_parent is None:
            raise CausalReplayMismatch(
                f"Full source parent is missing at round {round_index}"
            )
        mismatches = {}
        expected = {
            "parent_round_index": int(source_parent["round_index"]),
            "response_sha256": source_parent.get("response_sha256"),
            "code_sha256": source_parent.get("code_sha256"),
            "review_identity": _review_identity(source_parent.get("review") or {}),
        }
        actual = {
            "parent_round_index": parent_round_index,
            "response_sha256": response_sha256,
            "code_sha256": code_sha256,
            "review_identity": _review_identity(review),
        }
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                mismatches[key] = {"full": expected_value, "control": actual[key]}
        if mismatches:
            raise CausalReplayMismatch(
                f"Control parent diverged before {self.mechanism} activation: "
                + json.dumps(mismatches, sort_keys=True)
            )

    def replay_call(
        self,
        *,
        role: str,
        round_index: int,
        prompt_variants: Iterable[str],
    ) -> ReplayCall:
        if not self.should_replay(role, round_index):
            raise CausalReplayMismatch(
                f"Attempted post-activation replay: {role} round {round_index}"
            )
        source_round = self.source_round(round_index)
        if role == "diagnosis":
            call_payload = source_round.get("diagnoser_llm") or {}
            expected_user_sha = source_round.get("diagnosis_prompt_sha256")
        else:
            call_payload = source_round.get("llm") or {}
            expected_user_sha = source_round.get("repair_prompt_sha256")
        variants = list(prompt_variants)
        match = next(
            (value for value in variants if _sha256_text(value) == expected_user_sha),
            None,
        )
        if not call_payload or match is None:
            raise CausalReplayMismatch(
                f"Exact {role} prompt SHA miss before activation at round "
                f"{round_index}; refusing a live fallback"
            )
        source_text = call_payload.get("text")
        if not isinstance(source_text, str) or not source_text:
            raise CausalReplayMismatch("Full source cache entry has no response text")
        values = {
            key: value for key, value in call_payload.items() if key in self._llm_fields
        }
        values.update(
            {
                "text": source_text,
                "attempts": 0,
                "effective_user_prompt": match,
                "prompt_variant_index": variants.index(match),
            }
        )
        call = LLMCallResult(**values)
        return ReplayCall(
            call=call,
            reused_from=(
                f"{self.source_path}:round={round_index}:role={role}:"
                f"prompt_sha256={expected_user_sha}"
            ),
            effective_prompt=match,
        )

    def assert_upstream(
        self,
        round_index: int,
        *,
        diagnosis_prompt_sha256: str,
        diagnosis: dict[str, Any],
        retrieved_rule_ids: list[str],
        selected_rule_ids: list[str],
    ) -> None:
        if self.activation_round is not None and round_index > self.activation_round:
            return
        source = self.source_round(round_index)
        expected = {
            "diagnosis_prompt_sha256": source.get("diagnosis_prompt_sha256"),
            "diagnosis": canonical_diagnosis_identity(
                source.get("diagnosis") or {}
            ),
            "retrieved_rule_ids": source.get("retrieved_rule_ids") or [],
            "selected_rule_ids": source.get("selected_rule_ids") or [],
        }
        actual = {
            "diagnosis_prompt_sha256": diagnosis_prompt_sha256,
            "diagnosis": canonical_diagnosis_identity(diagnosis),
            "retrieved_rule_ids": retrieved_rule_ids,
            "selected_rule_ids": selected_rule_ids,
        }
        if actual != expected:
            raise CausalReplayMismatch(
                "Diagnosis-side identity differs before causal branch: "
                + json.dumps({"full": expected, "control": actual}, sort_keys=True)
            )

    def assert_completed_prefix_round(self, round_payload: dict[str, Any]) -> None:
        round_index = int(round_payload["round_index"])
        if self.activation_round is not None and round_index >= self.activation_round:
            return
        source = self.source_round(round_index)
        expected = {
            "response_sha256": source.get("response_sha256"),
            "code_sha256": source.get("code_sha256"),
            "action_origin": source.get("action_origin"),
            "review_identity": _review_identity(source.get("review") or {}),
        }
        actual = {
            "response_sha256": round_payload.get("response_sha256"),
            "code_sha256": round_payload.get("code_sha256"),
            "action_origin": round_payload.get("action_origin"),
            "review_identity": _review_identity(round_payload.get("review") or {}),
        }
        if actual != expected:
            raise CausalReplayMismatch(
                "Reviewed control prefix differs from Full source: "
                + json.dumps({"full": expected, "control": actual}, sort_keys=True)
            )

    def assert_activation_branch(
        self,
        round_payload: dict[str, Any],
        *,
        repair_reused: bool,
    ) -> None:
        round_index = int(round_payload["round_index"])
        if self.activation_round != round_index:
            return
        source = self.source_round(round_index)
        if repair_reused:
            raise CausalReplayMismatch("Activation outcome was copied from Full")
        if self.mechanism == "cleanroom" and round_payload.get("action_origin") == "cleanroom_llm":
            raise CausalReplayMismatch("No-clean control executed clean-room")
        if self.mechanism == "postprocess" and round_payload.get("action_origin") == "operator":
            raise CausalReplayMismatch("No-postprocess control executed operator")
        if self.mechanism == "cleanroom" and not (
            source.get("cleanroom") or {}
        ).get("triggered"):
            raise CausalReplayMismatch("Full source clean activation is missing")
        if self.mechanism == "postprocess" and not (
            source.get("operator") or {}
        ).get("changed"):
            raise CausalReplayMismatch("Full source operator activation is missing")

    def assert_terminal(self, control_payload: dict[str, Any]) -> None:
        if self.activation_round is not None:
            return
        source_rounds = self.payload.get("rounds") or []
        control_rounds = control_payload.get("rounds") or []
        expected = {
            "round_count": len(source_rounds),
            "final_review_identity": _review_identity(
                self.payload.get("final_review") or {}
            ),
            "selected_round_index": self.payload.get("selected_round_index"),
        }
        actual = {
            "round_count": len(control_rounds),
            "final_review_identity": _review_identity(
                control_payload.get("final_review") or {}
            ),
            "selected_round_index": control_payload.get("selected_round_index"),
        }
        if actual != expected:
            raise CausalReplayMismatch(
                "No-activation control did not completely replay Full: "
                + json.dumps({"full": expected, "control": actual}, sort_keys=True)
            )

    def round_provenance(
        self,
        round_index: int,
        *,
        diagnosis_reused_from: str | None,
        repair_reused_from: str | None,
    ) -> dict[str, Any]:
        if self.activation_round is None or round_index < self.activation_round:
            state = "replayed_prefix"
        elif round_index == self.activation_round:
            state = "activation_branch"
        else:
            state = "post_activation_live"
        source = self.source_round(round_index) if round_index < len(self.payload["rounds"]) else {}
        return {
            "version": CAUSAL_REPLAY_VERSION,
            "state": state,
            "mechanism": self.mechanism,
            "activation_round": self.activation_round,
            "source_round_index": round_index if source else None,
            "source_candidate_id": source.get("candidate_id"),
            "source_response_sha256": source.get("response_sha256"),
            "diagnosis_reused_from": diagnosis_reused_from,
            "repair_reused_from": repair_reused_from,
            "reviewer_reexecuted": True,
        }

    def provenance(self) -> dict[str, Any]:
        source_round = (
            self.source_round(self.activation_round)
            if self.activation_round is not None
            else None
        )
        return {
            "version": CAUSAL_REPLAY_VERSION,
            "source_mode": RunMode.MEMENTOHLS.value,
            "source_root": str(self.source_root),
            "source_trajectory": str(self.source_path),
            "source_trajectory_sha256": self.source_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "mechanism": self.mechanism,
            "activation_round": self.activation_round,
            "activation_event_sha256": (
                canonical_sha256(
                    {
                        "round_index": self.activation_round,
                        "cleanroom": source_round.get("cleanroom"),
                        "operator": source_round.get("operator"),
                        "diagnosis_prompt_sha256": source_round.get(
                            "diagnosis_prompt_sha256"
                        ),
                    }
                )
                if source_round is not None
                else None
            ),
            "policy": (
                "exact prompt-SHA response replay before activation; diagnosis-only "
                "replay at activation; live action and outcome after branch"
            ),
        }


class CausalReplaySource:
    def __init__(self, root: Path, *, expected_trajectories: int | None) -> None:
        self.root = root.resolve()
        self.manifest = build_full_source_manifest(self.root)
        if (
            expected_trajectories is not None
            and self.manifest["trajectory_count"] != expected_trajectories
        ):
            raise ValueError(
                "Causal Full source count mismatch: "
                f"{self.manifest['trajectory_count']} != {expected_trajectories}"
            )

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    def validate_binding(
        self,
        *,
        iteration_id: str,
        source_sha256: str,
        dataset_manifest_sha256: str,
        hls_eval_source_sha256: str,
        deployment_epoch: str,
    ) -> None:
        expected = [
            iteration_id,
            source_sha256,
            dataset_manifest_sha256,
            hls_eval_source_sha256,
            deployment_epoch,
        ]
        if self.manifest.get("binding") != expected:
            raise ValueError(
                "Causal Full source binding mismatch: "
                + json.dumps(
                    {"full": self.manifest.get("binding"), "control": expected},
                    sort_keys=True,
                )
            )

    def plan_for(
        self,
        *,
        case_name: str,
        sample_index: int,
        seed: int,
        control_mode: RunMode,
    ) -> CausalReplayPlan:
        if control_mode not in CONTROL_MECHANISMS:
            raise ValueError(f"Mode is not a causal Full control: {control_mode}")
        identity = _key(case_name, sample_index, seed)
        entry = (self.manifest.get("entries") or {}).get(identity)
        if entry is None:
            raise ValueError(f"Missing causal Full source trajectory: {identity}")
        source_path = (self.root / entry["relative_path"]).resolve()
        if self.root not in source_path.parents:
            raise ValueError(f"Unsafe causal Full source path: {source_path}")
        if file_sha256(source_path) != entry["sha256"]:
            raise ValueError(f"Causal Full source changed after binding: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        return CausalReplayPlan(
            source_root=self.root,
            source_path=source_path,
            source_sha256=entry["sha256"],
            payload=payload,
            control_mode=control_mode,
            source_manifest_sha256=self.manifest_sha256,
        )
