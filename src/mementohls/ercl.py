from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import Diagnosis, FailureStage


_SOURCES = ("log", "code", "header", "history", "task", "facts")
_MATCH_KEYS = {
    *(f"{source}_{quantifier}" for source in _SOURCES for quantifier in ("all", "any", "none")),
    "fallback",
}
_STAGE_RULE_PREFIX = {
    FailureStage.PARSE.value: "PARSE_",
    FailureStage.COMPILE.value: "COMPILE_",
    FailureStage.TB.value: "TB_",
    FailureStage.SYNTH.value: "SYNTH_",
}
_ROUTING_CHANNELS = {
    "diagnoser",
    "cleanroom_authorization",
    "trajectory_frontier_guard",
}
_ROUTING_STATUSES = {"active", "reserved_inactive"}
_INTERFACE_CONSTRAINT_RE = re.compile(
    r"(?:\bheader\b|\.h\b|\bpublic\s+signature\b|"
    r"\bfunction\s+signature\b|\btop(?:[- ]function)?\b|"
    r"\binterface\b|\bprototype\b|\btypedef\b|\bABI\b|"
    r"\barray\s+(?:extent|dimension)|\bfixed\s+dimension)",
    flags=re.IGNORECASE,
)
_PROTECTIVE_CONSTRAINT_RE = re.compile(
    r"(?:\bimmutable\b|\bexact(?:ly)?\b|\bauthoritative\b|"
    r"\bpreserv\w*\b|\bremain\w*\b|\bunchanged\b|"
    r"\bdo\s+not\b|\bnever\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryRule:
    rule_id: str
    stage: str
    priority: int
    diagnosis: str
    allowed_actions: list[str]
    forbidden_actions: list[str]
    must_hold_invariants: list[str]
    match: dict[str, list[str]]
    operator_id: str | None = None
    confidence: str = "medium"
    source: str | None = None
    operator_preconditions: list[str] | None = None
    operator_postconditions: list[str] | None = None
    expected_stage_transition: str | None = None
    validation_required: list[str] | None = None
    eligibility: dict[str, list[str]] | None = None
    routing_channel: str = "diagnoser"
    routing_status: str = "active"
    routing_reason: str | None = None
    evidence_tier: str = "generic"
    confidence_score: float | None = None
    minimum_evidence_sources: int = 1
    conflicts_with: list[str] | None = None
    empirical: dict[str, float | int] | None = None

    @staticmethod
    def _pattern_matches(pattern: str, text: str) -> bool:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    def matches(
        self,
        *,
        log: str,
        code: str,
        header: str = "",
        history: str = "",
        task: str = "",
        facts: str = "",
    ) -> bool:
        haystacks = {
            "log": log,
            "code": code,
            "header": header,
            "history": history,
            "task": task,
            "facts": facts,
        }
        constrained = False
        for source, haystack in haystacks.items():
            all_patterns = self.match.get(f"{source}_all", [])
            any_patterns = self.match.get(f"{source}_any", [])
            none_patterns = self.match.get(f"{source}_none", [])
            if all_patterns:
                constrained = True
                if not all(self._pattern_matches(pattern, haystack) for pattern in all_patterns):
                    return False
            if any_patterns:
                constrained = True
                if not any(self._pattern_matches(pattern, haystack) for pattern in any_patterns):
                    return False
            if none_patterns:
                constrained = True
                if any(self._pattern_matches(pattern, haystack) for pattern in none_patterns):
                    return False
        return constrained or bool(self.match.get("fallback"))

    def conflict_class(self) -> str:
        if self.match.get("fallback"):
            return "general_hls_compatibility"
        if self._has_immutable_interface_contract():
            return "immutable_interface"
        if any(
            self.match.get(key)
            for key in ("code_none", "header_none")
        ):
            return "exact_negative_evidence"
        if self.match.get("log_all"):
            return "exact_tool_error"
        if any(
            self.match.get(key)
            for key in (
                "code_all",
                "header_all",
                "code_any",
                "header_any",
                "history_all",
                "task_all",
                "task_any",
                "facts_all",
                "facts_any",
            )
        ):
            return "exact_code_or_header_feature"
        if self.stage == FailureStage.TB.value:
            return "tb_semantic_hypothesis"
        return "general_hls_compatibility"

    def _has_immutable_interface_contract(self) -> bool:
        """Recognize only evidence-backed interface-preservation contracts.

        Merely mentioning an invariant is intentionally insufficient. The rule
        must combine an explicit protective interface constraint with exact
        immutable evidence: a complete header contract, a derived fact clause
        explicitly tied to an immutable header, or a three-source
        log/header/code declaration conflict.
        """
        constraint_items = [
            *self.forbidden_actions,
            *self.must_hold_invariants,
            *(self.operator_preconditions or []),
            *(self.operator_postconditions or []),
        ]
        protected_items = [
            item
            for item in constraint_items
            if _INTERFACE_CONSTRAINT_RE.search(item)
            and _PROTECTIVE_CONSTRAINT_RE.search(item)
        ]
        if not protected_items:
            return False

        exact_header_contract = bool(self.match.get("header_all"))
        interface_action = any(
            re.search(
                r"(?:header|typedef|signature|interface|parameter|"
                r"pointer[- ]to[- ]array|array\s+(?:extent|dimension))",
                item,
                re.IGNORECASE,
            )
            for item in self.allowed_actions
        )
        immutable_fact_contract = (
            bool(self.match.get("facts_all"))
            and interface_action
            and any(
                re.search(r"(?:immutable|header[- ]proven)", item, re.IGNORECASE)
                for item in constraint_items + [self.diagnosis]
            )
        )
        cross_source_declaration_conflict = all(
            self.match.get(key) for key in ("log_any", "header_any", "code_any")
        )
        return any(
            (
                exact_header_contract,
                immutable_fact_contract,
                cross_source_declaration_conflict,
            )
        )

    def specificity(self) -> int:
        """Score independent clauses; alternatives in an `any` clause stay broad."""
        score = 0
        for key, patterns in self.match.items():
            if key == "fallback" or not patterns:
                continue
            quantifier = key.rsplit("_", 1)[-1]
            if quantifier == "all":
                score += 4 * len(patterns)
            elif quantifier == "none":
                score += 3 * len(patterns)
            elif quantifier == "any":
                score += 2
        return score

    def evidence_sources(
        self,
        *,
        log: str,
        code: str,
        header: str = "",
        history: str = "",
        task: str = "",
        facts: str = "",
    ) -> list[str]:
        haystacks = {
            "log": log,
            "code": code,
            "header": header,
            "history": history,
            "task": task,
            "facts": facts,
        }
        sources: list[str] = []
        for source, haystack in haystacks.items():
            positive = [
                *self.match.get(f"{source}_all", []),
                *self.match.get(f"{source}_any", []),
            ]
            negative = self.match.get(f"{source}_none", [])
            if positive and any(
                self._pattern_matches(pattern, haystack)
                for pattern in positive
            ):
                sources.append(source)
            elif negative and all(
                not self._pattern_matches(pattern, haystack)
                for pattern in negative
            ):
                sources.append(source)
        return sources

    def authorization_score(self, evidence_source_count: int) -> float:
        base = (
            float(self.confidence_score)
            if self.confidence_score is not None
            else {"high": 0.90, "medium": 0.70, "low": 0.45}.get(
                self.confidence.lower(), 0.45
            )
        )
        score = base
        score += min(0.08, self.specificity() / 200.0)
        score += min(0.06, max(0, evidence_source_count - 1) * 0.03)
        empirical = self.empirical or {}
        selected = int(empirical.get("selected", 0))
        progress = int(empirical.get("progress", 0))
        success = int(empirical.get("success", 0))
        regressions = int(empirical.get("regressions", 0))
        if selected >= 20:
            progress_rate = progress / selected
            success_rate = success / selected
            regression_rate = regressions / selected
            if success == 0:
                score -= 0.15
            if progress_rate < 0.05:
                score -= 0.20
            if regression_rate > progress_rate:
                score -= min(0.15, regression_rate - progress_rate)
            score += min(0.08, success_rate * 0.30)
        elif selected and success:
            score += min(0.06, (success / selected) * 0.08)
        return max(0.0, min(1.0, score))

    def eligibility_decision(self, *, facts: str) -> tuple[bool, list[str]]:
        """Evaluate deterministic pre-ranking gates against derived facts.

        Eligibility is separate from matching: a rule can match broad compiler
        text yet still be rejected before top-k ranking when immutable evidence
        does not authorize its action.
        """
        conditions = self.eligibility or {}
        # Direct legacy callers did not provide derived facts. Preserve their
        # match behavior; the production orchestrator always supplies schema-v2
        # facts before retrieval, so action authorization remains fail-closed.
        if conditions and not facts:
            return True, []
        failures: list[str] = []
        for quantifier in ("all", "any", "none"):
            patterns = conditions.get(f"facts_{quantifier}", [])
            if not patterns:
                continue
            matched = [
                self._pattern_matches(pattern, facts) for pattern in patterns
            ]
            if quantifier == "all" and not all(matched):
                failures.append(
                    "facts_all_missing:"
                    + ",".join(
                        pattern
                        for pattern, present in zip(patterns, matched)
                        if not present
                    )
                )
            elif quantifier == "any" and not any(matched):
                failures.append("facts_any_missing:" + ",".join(patterns))
            elif quantifier == "none" and any(matched):
                failures.append(
                    "facts_none_present:"
                    + ",".join(
                        pattern
                        for pattern, present in zip(patterns, matched)
                        if present
                    )
                )
        return not failures, failures

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "stage": self.stage,
            "diagnosis": self.diagnosis,
            "allowed_actions": self.allowed_actions,
            "forbidden_actions": self.forbidden_actions,
            "must_hold_invariants": self.must_hold_invariants,
            "operator_id": self.operator_id,
            "evidence_tier": self.evidence_tier,
            "confidence": self.confidence,
            "source": self.source,
            "operator_preconditions": self.operator_preconditions or [],
            "operator_postconditions": self.operator_postconditions or [],
            "expected_stage_transition": self.expected_stage_transition,
            "validation_required": self.validation_required or [],
            "eligibility": self.eligibility or {},
            "conflict_class": self.conflict_class(),
            "specificity": self.specificity(),
            "routing_channel": self.routing_channel,
            "routing_status": self.routing_status,
            "confidence_score": self.confidence_score,
            "minimum_evidence_sources": self.minimum_evidence_sources,
            "conflicts_with": self.conflicts_with or [],
            "empirical": self.empirical or {},
        }


@dataclass(frozen=True)
class RetrievalResult:
    """Selected rules plus bounded, serializable retrieval provenance."""

    rules: list[MemoryRule]
    provenance: dict[str, Any]


class ExecutableRepairContractLibrary:
    def __init__(self, rules_path: Path) -> None:
        self.rules_path = rules_path
        payload = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        self.version = str(payload.get("version", "unknown"))
        self.sources = list(payload.get("sources", []))
        self.retrieval_policy = dict(payload.get("retrieval_policy") or {})
        self.evidence_tier_counts = dict(payload.get("evidence_tier_counts") or {})
        self.rules_content_sha256 = payload.get("rules_sha256")
        self.retrieval_policy.setdefault("top_k", 1)
        self.retrieval_policy.setdefault("fallback_policy", "abstain_to_feedback")
        self.retrieval_policy.setdefault("minimum_authorization_score", 0.58)
        self.retrieval_policy.setdefault("minimum_runner_up_margin", 0.04)
        self.retrieval_policy.setdefault(
            "conflict_precedence",
            [
                "immutable_interface",
                "exact_negative_evidence",
                "exact_tool_error",
                "exact_code_or_header_feature",
                "general_hls_compatibility",
                "tb_semantic_hypothesis",
            ],
        )
        top_k = int(self.retrieval_policy["top_k"])
        if not 1 <= top_k <= 4:
            raise ValueError("ERCL retrieval_policy.top_k must be in [1, 4]")
        self.rules = [self._parse_rule(rule) for rule in payload.get("rules", [])]
        if not self.rules:
            raise ValueError(f"No HLS memory rules found in {rules_path}")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError(f"Duplicate rule IDs found in {rules_path}")
        for rule in self.rules:
            expected_prefix = _STAGE_RULE_PREFIX[rule.stage]
            if not rule.rule_id.startswith(expected_prefix):
                raise ValueError(
                    f"Rule {rule.rule_id} stage {rule.stage!r} must use "
                    f"the {expected_prefix!r} prefix"
                )
            if rule.routing_channel not in _ROUTING_CHANNELS:
                raise ValueError(
                    f"Rule {rule.rule_id} uses unsupported routing_channel: "
                    f"{rule.routing_channel!r}"
                )
            if rule.routing_status not in _ROUTING_STATUSES:
                raise ValueError(
                    f"Rule {rule.rule_id} uses unsupported routing_status: "
                    f"{rule.routing_status!r}"
                )
            if rule.rule_id.endswith("_STAGNATION_CLEANROOM"):
                if (
                    rule.routing_channel != "cleanroom_authorization"
                    or rule.routing_status != "active"
                ):
                    raise ValueError(
                        f"Rule {rule.rule_id} must be an active "
                        "cleanroom_authorization rule"
                    )
            elif rule.rule_id.endswith("_HISTORY_FRONTIER_GUARD"):
                if (
                    rule.routing_channel != "trajectory_frontier_guard"
                    or rule.routing_status != "reserved_inactive"
                ):
                    raise ValueError(
                        f"Rule {rule.rule_id} must be a reserved_inactive "
                        "trajectory_frontier_guard"
                    )
            elif (
                rule.routing_channel != "diagnoser"
                or rule.routing_status != "active"
            ):
                raise ValueError(
                    f"Ordinary rule {rule.rule_id} must use the active "
                    "diagnoser route"
                )
            unknown = set(rule.match) - _MATCH_KEYS
            if unknown:
                raise ValueError(
                    f"Rule {rule.rule_id} uses unsupported match keys: {sorted(unknown)}"
                )
            unknown_eligibility = set(rule.eligibility or {}) - {
                "facts_all",
                "facts_any",
                "facts_none",
            }
            if unknown_eligibility:
                raise ValueError(
                    f"Rule {rule.rule_id} uses unsupported eligibility keys: "
                    f"{sorted(unknown_eligibility)}"
                )
            if not 1 <= rule.minimum_evidence_sources <= len(_SOURCES):
                raise ValueError(
                    f"Rule {rule.rule_id} minimum_evidence_sources must be in "
                    f"[1, {len(_SOURCES)}]"
                )
            if (
                rule.confidence_score is not None
                and not 0.0 <= rule.confidence_score <= 1.0
            ):
                raise ValueError(
                    f"Rule {rule.rule_id} confidence_score must be in [0, 1]"
                )
            for patterns in [
                *rule.match.values(),
                *(rule.eligibility or {}).values(),
            ]:
                for pattern in patterns:
                    re.compile(pattern, flags=re.IGNORECASE)

    @staticmethod
    def _as_patterns(value: Any) -> list[str]:
        if value is None or value is False:
            return []
        if value is True:
            return ["true"]
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            raise ValueError(f"Rule match value must be a list or string, got {value!r}")
        return [str(item) for item in value]

    @classmethod
    def _parse_rule(cls, payload: dict[str, Any]) -> MemoryRule:
        required = {
            "rule_id",
            "stage",
            "priority",
            "diagnosis",
            "allowed_actions",
            "forbidden_actions",
            "must_hold_invariants",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Rule is missing fields {sorted(missing)}: {payload}")
        stage = str(payload["stage"])
        valid_stages = {
            item.value for item in FailureStage if item is not FailureStage.SUCCESS
        }
        if stage not in valid_stages:
            raise ValueError(f"Unsupported rule stage: {stage}")
        match = {
            str(key): cls._as_patterns(value)
            for key, value in (payload.get("match") or {}).items()
        }
        return MemoryRule(
            rule_id=str(payload["rule_id"]),
            stage=stage,
            priority=int(payload["priority"]),
            diagnosis=str(payload["diagnosis"]),
            allowed_actions=[str(item) for item in payload["allowed_actions"]],
            forbidden_actions=[str(item) for item in payload["forbidden_actions"]],
            must_hold_invariants=[
                str(item) for item in payload["must_hold_invariants"]
            ],
            match=match,
            operator_id=(
                str(payload["operator_id"]) if payload.get("operator_id") else None
            ),
            confidence=str(payload.get("confidence", "medium")),
            source=(str(payload["source"]) if payload.get("source") else None),
            operator_preconditions=[
                str(item) for item in payload.get("operator_preconditions", [])
            ],
            operator_postconditions=[
                str(item) for item in payload.get("operator_postconditions", [])
            ],
            expected_stage_transition=(
                str(payload["expected_stage_transition"])
                if payload.get("expected_stage_transition")
                else None
            ),
            validation_required=[
                str(item) for item in payload.get("validation_required", [])
            ],
            eligibility={
                str(key): cls._as_patterns(value)
                for key, value in (payload.get("eligibility") or {}).items()
            },
            routing_channel=str(payload.get("routing_channel", "diagnoser")),
            routing_status=str(payload.get("routing_status", "active")),
            routing_reason=(
                str(payload["routing_reason"])
                if payload.get("routing_reason")
                else None
            ),
            evidence_tier=str(payload.get("evidence_tier", "generic")),
            confidence_score=(
                float(payload["confidence_score"])
                if payload.get("confidence_score") is not None
                else None
            ),
            minimum_evidence_sources=int(
                payload.get("minimum_evidence_sources", 1)
            ),
            conflicts_with=[
                str(item) for item in payload.get("conflicts_with", [])
            ],
            empirical={
                str(key): value
                for key, value in (payload.get("empirical") or {}).items()
            },
        )

    @staticmethod
    def _rank_rules(
        rules: list[MemoryRule],
        precedence_order: list[str],
        authorization_scores: dict[str, float] | None = None,
    ) -> list[MemoryRule]:
        precedence = {
            name: index for index, name in enumerate(precedence_order)
        }
        confidence = {"high": 2, "medium": 1, "low": 0}
        rules.sort(
            key=lambda rule: (
                precedence.get(rule.conflict_class(), len(precedence)),
                -(authorization_scores or {}).get(rule.rule_id, 0.0),
                -confidence.get(rule.confidence.lower(), 0),
                -rule.specificity(),
                -rule.priority,
                rule.rule_id,
            )
        )
        return rules

    def match_route(
        self,
        routing_channel: str,
        stage: str,
        log: str,
        code: str,
        *,
        header: str = "",
        history: str = "",
        task: str = "",
        facts: str = "",
        include_inactive: bool = False,
        limit: int = 1,
    ) -> list[MemoryRule]:
        """Match one independent control route without entering Diagnoser top-k."""
        if routing_channel == "diagnoser":
            raise ValueError("Use match_with_provenance for the diagnoser route")
        if routing_channel not in _ROUTING_CHANNELS:
            raise ValueError(f"Unsupported routing channel: {routing_channel}")
        if limit < 0:
            raise ValueError("ERCL route limit must be non-negative")
        matched: list[MemoryRule] = []
        authorization_scores: dict[str, float] = {}
        for rule in self.rules:
            if rule.stage != stage or rule.routing_channel != routing_channel:
                continue
            if rule.routing_status != "active" and not include_inactive:
                continue
            if not rule.matches(
                log=log,
                code=code,
                header=header,
                history=history,
                task=task,
                facts=facts,
            ):
                continue
            evidence_source_count = len(
                rule.evidence_sources(
                    log=log,
                    code=code,
                    header=header,
                    history=history,
                    task=task,
                    facts=facts,
                )
            )
            if evidence_source_count < rule.minimum_evidence_sources:
                continue
            is_eligible, _ = rule.eligibility_decision(facts=facts)
            if is_eligible:
                matched.append(rule)
                authorization_scores[rule.rule_id] = (
                    rule.authorization_score(evidence_source_count)
                )
        return self._rank_rules(
            matched,
            self.retrieval_policy["conflict_precedence"],
            authorization_scores,
        )[:limit]

    @staticmethod
    def _bounded_fact_projection(facts: str) -> dict[str, Any]:
        if not facts:
            return {}
        try:
            payload = json.loads(facts)
        except json.JSONDecodeError:
            return {"invalid_json": True, "character_count": len(facts)}
        if not isinstance(payload, dict):
            return {"invalid_payload_type": type(payload).__name__}
        projection: dict[str, Any] = {
            "schema_version": payload.get("schema_version"),
            "stage": payload.get("stage"),
            "markers": list(payload.get("markers") or [])[:32],
        }
        if payload.get("eligibility"):
            projection["eligibility"] = payload["eligibility"]
        error_set = payload.get("compile_error_set")
        if isinstance(error_set, dict):
            projection["compile_error_set"] = {
                key: error_set.get(key)
                for key in (
                    "unresolved_symbol_count",
                    "unresolved_symbols",
                    "helper_closure",
                    "omitted_symbol_count",
                )
                if key in error_set
            }
        return projection

    def match_with_provenance(
        self,
        stage: str,
        log: str,
        code: str,
        *,
        header: str = "",
        history: str = "",
        task: str = "",
        facts: str = "",
        limit: int | None = None,
        exclude_rule_id_suffixes: tuple[str, ...] = (),
        exclude_rule_ids: tuple[str, ...] = (),
    ) -> RetrievalResult:
        """Prefilter action eligibility before ranking and top-k truncation."""
        excluded_rule_ids = frozenset(exclude_rule_ids)
        stage_rules = [
            rule
            for rule in self.rules
            if rule.stage == stage
            and rule.rule_id not in excluded_rule_ids
            and not any(
                rule.rule_id.endswith(suffix)
                for suffix in exclude_rule_id_suffixes
            )
        ]
        matched_substantive: list[MemoryRule] = []
        eligibility_rejections: list[dict[str, Any]] = []
        pattern_matched_rule_ids: list[str] = []
        routed_rule_ids: dict[str, list[str]] = {
            channel: [] for channel in sorted(_ROUTING_CHANNELS)
        }
        routed_eligible_rule_ids: dict[str, list[str]] = {
            channel: [] for channel in sorted(_ROUTING_CHANNELS)
        }
        routing_decisions: list[dict[str, Any]] = []
        evidence_source_counts: dict[str, int] = {}
        authorization_scores: dict[str, float] = {}
        for rule in stage_rules:
            if rule.match.get("fallback"):
                continue
            if not rule.matches(
                log=log,
                code=code,
                header=header,
                history=history,
                task=task,
                facts=facts,
            ):
                continue
            pattern_matched_rule_ids.append(rule.rule_id)
            routed_rule_ids[rule.routing_channel].append(rule.rule_id)
            evidence_source_count = len(
                rule.evidence_sources(
                    log=log,
                    code=code,
                    header=header,
                    history=history,
                    task=task,
                    facts=facts,
                )
            )
            evidence_source_counts[rule.rule_id] = evidence_source_count
            authorization_scores[rule.rule_id] = rule.authorization_score(
                evidence_source_count
            )
            if evidence_source_count < rule.minimum_evidence_sources:
                eligibility_rejections.append(
                    {
                        "rule_id": rule.rule_id,
                        "phase": "independent_evidence_gate",
                        "evidence_source_count": evidence_source_count,
                        "minimum_evidence_sources": (
                            rule.minimum_evidence_sources
                        ),
                    }
                )
                continue
            is_eligible, failed_conditions = rule.eligibility_decision(facts=facts)
            if is_eligible:
                routed_eligible_rule_ids[rule.routing_channel].append(rule.rule_id)
            if is_eligible and (
                rule.routing_channel != "diagnoser"
                or rule.routing_status != "active"
            ):
                routing_decisions.append(
                    {
                        "rule_id": rule.rule_id,
                        "routing_channel": rule.routing_channel,
                        "routing_status": rule.routing_status,
                        "disposition": (
                            "reserved_inactive"
                            if rule.routing_status == "reserved_inactive"
                            else "independent_control_route"
                        ),
                        "reason": rule.routing_reason,
                    }
                )
            elif is_eligible:
                matched_substantive.append(rule)
            else:
                eligibility_rejections.append(
                    {
                        "rule_id": rule.rule_id,
                        "phase": "eligibility_prefilter",
                        "failed_conditions": failed_conditions,
                    }
                )

        fallback_used = False
        if matched_substantive:
            candidates = matched_substantive
        elif self.retrieval_policy.get("fallback_policy") == "abstain_to_feedback":
            candidates = []
        else:
            fallback_used = True
            candidates = []
            for rule in stage_rules:
                if not rule.match.get("fallback") or not rule.matches(
                    log=log,
                    code=code,
                    header=header,
                    history=history,
                    task=task,
                    facts=facts,
                ):
                    continue
                is_eligible, failed_conditions = rule.eligibility_decision(
                    facts=facts
                )
                if is_eligible:
                    candidates.append(rule)
                else:
                    eligibility_rejections.append(
                        {
                            "rule_id": rule.rule_id,
                            "phase": "eligibility_prefilter",
                            "failed_conditions": failed_conditions,
                        }
                    )

        self._rank_rules(
            candidates,
            self.retrieval_policy["conflict_precedence"],
            authorization_scores,
        )
        selected_limit = (
            int(self.retrieval_policy["top_k"]) if limit is None else int(limit)
        )
        if selected_limit < 0:
            raise ValueError("ERCL retrieval limit must be non-negative")
        selected = candidates[:selected_limit]
        abstain_reason: str | None = None
        if selected:
            top = selected[0]
            top_score = authorization_scores.get(top.rule_id, 0.0)
            minimum_score = float(
                self.retrieval_policy["minimum_authorization_score"]
            )
            if top_score < minimum_score:
                abstain_reason = "authorization_score_below_minimum"
                selected = []
            elif len(candidates) > 1:
                runner_up = candidates[1]
                direct_conflict = (
                    runner_up.rule_id in (top.conflicts_with or [])
                    or top.rule_id in (runner_up.conflicts_with or [])
                )
                comparable = (
                    runner_up.conflict_class() == top.conflict_class()
                    or direct_conflict
                )
                margin = top_score - authorization_scores.get(
                    runner_up.rule_id, 0.0
                )
                if (
                    comparable
                    and margin
                    < float(
                        self.retrieval_policy[
                            "minimum_runner_up_margin"
                        ]
                    )
                ):
                    abstain_reason = "runner_up_margin_below_minimum"
                    selected = []
        provenance = {
            "schema_version": "2",
            "stage": stage,
            "prefilter_before_top_k": True,
            "top_k": selected_limit,
            "fallback_used": fallback_used,
            "excluded_rule_id_suffixes": list(exclude_rule_id_suffixes),
            "trajectory_vetoed_rule_ids": sorted(excluded_rule_ids),
            "stage_rule_count": len(stage_rules),
            "pattern_matched_rule_ids": pattern_matched_rule_ids,
            "eligibility_rejections": eligibility_rejections,
            "ranked_rule_ids": [rule.rule_id for rule in candidates],
            "selected_rule_ids": [rule.rule_id for rule in selected],
            "abstain_reason": abstain_reason,
            "authorization_scores": {
                rule.rule_id: authorization_scores.get(rule.rule_id, 0.0)
                for rule in candidates
            },
            "evidence_source_counts": evidence_source_counts,
            "route_rule_counts": dict(
                Counter(rule.routing_channel for rule in stage_rules)
            ),
            "route_pattern_match_counts": {
                channel: len(rule_ids)
                for channel, rule_ids in routed_rule_ids.items()
            },
            "route_eligible_match_counts": {
                channel: len(rule_ids)
                for channel, rule_ids in routed_eligible_rule_ids.items()
            },
            "routed_pattern_matched_rule_ids": routed_rule_ids,
            "routed_eligible_rule_ids": routed_eligible_rule_ids,
            "routing_decisions": routing_decisions,
            "selected_conflict_classes": [
                rule.conflict_class() for rule in selected
            ],
            "fact_projection": self._bounded_fact_projection(facts),
        }
        return RetrievalResult(rules=selected, provenance=provenance)

    def match(
        self,
        stage: str,
        log: str,
        code: str,
        *,
        header: str = "",
        history: str = "",
        task: str = "",
        facts: str = "",
        limit: int | None = None,
        exclude_rule_id_suffixes: tuple[str, ...] = (),
        exclude_rule_ids: tuple[str, ...] = (),
    ) -> list[MemoryRule]:
        """Backward-compatible selected-rule interface."""
        return self.match_with_provenance(
            stage,
            log,
            code,
            header=header,
            history=history,
            task=task,
            facts=facts,
            limit=limit,
            exclude_rule_id_suffixes=exclude_rule_id_suffixes,
            exclude_rule_ids=exclude_rule_ids,
        ).rules

    @staticmethod
    def rules_to_prompt(rules: list[MemoryRule]) -> str:
        if not rules:
            return (
                "No long-term rule matched. Use only the raw evidence and HLS "
                "invariants."
            )
        return json.dumps([rule.to_prompt_dict() for rule in rules], indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found in diagnosis response")


def validate_diagnosis(
    text: str,
    expected_stage: str,
    matched_rules: list[MemoryRule],
    *,
    current_code: str = "",
    header: str = "",
    header_name: str = "",
) -> Diagnosis:
    rule_ids = [rule.rule_id for rule in matched_rules]
    try:
        value = extract_json_object(text)
        stage = str(value.get("failure_stage", expected_stage))
        if stage != expected_stage:
            raise ValueError(
                f"Diagnosis stage {stage!r} does not match {expected_stage!r}"
            )

        def string_list(key: str) -> list[str]:
            raw = value.get(key, [])
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, list):
                raise ValueError(f"{key} must be a list")
            return [str(item) for item in raw if str(item).strip()]

        root_cause = str(value.get("root_cause", "")).strip()
        predicted_observation = str(
            value.get("predicted_observation", "")
        ).strip()
        evidence = string_list("evidence")
        if not root_cause or not evidence:
            raise ValueError("Diagnosis requires root_cause and evidence")
        returned_rule_ids = string_list("rule_ids")
        if any(item not in rule_ids for item in returned_rule_ids):
            raise ValueError("Diagnosis selected a rule outside the deterministic gate")
        # Exact, high-confidence deterministic operators are part of ERCL,
        # not optional model advice. Promote the top gated operator when the JSON
        # diagnosis omits every rule ID, while retaining one-candidate-per-round.
        if (
            not returned_rule_ids
            and matched_rules
            and matched_rules[0].operator_id
            and matched_rules[0].confidence.lower() == "high"
        ):
            returned_rule_ids = [matched_rules[0].rule_id]
        diagnosed_allowed = string_list("allowed_actions")
        diagnosed_forbidden = string_list("forbidden_actions")
        diagnosed_invariants = string_list("must_hold_invariants")
        from .rfl import ACTION_INTENT_TAXONOMY, ROOT_INTENT_TAXONOMY

        validation_notes: list[str] = []
        canonical_root = str(
            value.get("canonical_root_intent_id", "stage_unknown")
        ).strip()
        if canonical_root not in ROOT_INTENT_TAXONOMY:
            validation_notes.append("invalid_root_intent_fell_back_to_stage_unknown")
            canonical_root = "stage_unknown"
        canonical_action = str(
            value.get("canonical_action_intent_id", "stage_generic_other")
        ).strip()
        if canonical_action not in ACTION_INTENT_TAXONOMY:
            validation_notes.append("invalid_action_intent_fell_back_to_stage_generic_other")
            canonical_action = "stage_generic_other"
        raw_target = value.get("canonical_action_target_id")
        canonical_target = str(raw_target).strip() if raw_target is not None else None
        selected_operator = next(
            (
                rule.operator_id
                for rule in matched_rules
                if rule.rule_id in returned_rule_ids and rule.operator_id
            ),
            None,
        )
        operator_intents = {
            "exact_header_include": ("missing_header_or_declaration", "header_include"),
            "array_typedef_deref": ("array_or_pointer_semantics", "array_dereference"),
            "remove_illegal_main": ("unsynthesizable_construct", "remove_illegal_main"),
            "remove_host_io": ("unsynthesizable_construct", "remove_unsynthesizable_construct"),
            "remove_invalid_vitis_include": ("missing_header_or_declaration", "header_include"),
        }
        if selected_operator and canonical_action == "stage_generic_other":
            canonical_root, canonical_action = operator_intents.get(
                selected_operator,
                (canonical_root, "stage_generic_other"),
            )
        if selected_operator and not canonical_target:
            canonical_target = f"operator:{selected_operator}"
        target_valid = False
        if canonical_target:
            prefix, separator, name = canonical_target.partition(":")
            if separator and prefix == "operator" and selected_operator == name:
                target_valid = True
            elif separator and prefix == "include" and name == header_name:
                target_valid = True
            elif separator and prefix in {"param", "array", "function", "output"}:
                target_valid = bool(
                    re.fullmatch(r"[A-Za-z_]\w*", name)
                    and re.search(rf"\b{re.escape(name)}\b", current_code + "\n" + header)
                )
        if canonical_target and not target_valid:
            validation_notes.append("unverified_action_target_was_removed")
            canonical_target = None
        action_veto_eligible = bool(
            canonical_action != "stage_generic_other" and canonical_target and target_valid
        )
        root_veto_eligible = bool(
            canonical_root != "stage_unknown" and canonical_target and target_valid
        )
        selected_rules = [
            rule for rule in matched_rules if rule.rule_id in returned_rule_ids
        ]
        if selected_rules:
            allowed = list(
                dict.fromkeys(
                    action for rule in selected_rules for action in rule.allowed_actions
                )
            )
            forbidden = list(
                dict.fromkeys(
                    action for rule in selected_rules for action in rule.forbidden_actions
                )
            )
            invariants = list(
                dict.fromkeys(
                    item
                    for rule in selected_rules
                    for item in rule.must_hold_invariants
                )
            )
        else:
            allowed = diagnosed_allowed
            forbidden = diagnosed_forbidden
            invariants = diagnosed_invariants
        return Diagnosis(
            failure_stage=stage,
            evidence=evidence,
            root_cause=root_cause,
            allowed_actions=allowed,
            forbidden_actions=forbidden,
            must_hold_invariants=invariants,
            rule_ids=returned_rule_ids,
            fallback=False,
            predicted_observation=predicted_observation,
            canonical_root_intent_id=canonical_root,
            canonical_action_intent_id=canonical_action,
            canonical_action_target_id=canonical_target,
            action_veto_eligible=action_veto_eligible,
            root_veto_eligible=root_veto_eligible,
            validation_notes=validation_notes,
        )
    except Exception as exc:
        fallback_rules = matched_rules[:1]
        allowed = [
            action for rule in fallback_rules for action in rule.allowed_actions
        ]
        forbidden = [
            action for rule in fallback_rules for action in rule.forbidden_actions
        ]
        invariants = [
            item for rule in fallback_rules for item in rule.must_hold_invariants
        ]
        return Diagnosis(
            failure_stage=expected_stage,
            evidence=[f"Diagnosis JSON validation failed: {exc}"],
            root_cause=(
                f"Use stage-specific repair for {expected_stage} based on the "
                "raw tool evidence."
            ),
            allowed_actions=allowed
            or [f"Repair the {expected_stage} failure without changing the interface"],
            forbidden_actions=forbidden
            or ["Do not modify the testbench, header, or top-function signature"],
            must_hold_invariants=invariants
            or ["Preserve functional behavior and all fixed interface dimensions"],
            rule_ids=[rule.rule_id for rule in fallback_rules],
            fallback=True,
            predicted_observation=(
                f"The current {expected_stage} diagnostic disappears or the "
                "candidate advances to the next Reviewer stage."
            ),
            canonical_root_intent_id="stage_unknown",
            canonical_action_intent_id="stage_generic_other",
            canonical_action_target_id=None,
            action_veto_eligible=False,
            root_veto_eligible=False,
            validation_notes=["fallback_diagnosis_is_not_veto_eligible"],
        )
