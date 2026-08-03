from __future__ import annotations

import difflib
import hashlib
import json
import re
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from .models import Diagnosis, ReviewResult


RFL_SCHEMA_VERSION = "dac2027-rfl-1.1.0"

ROOT_INTENT_TAXONOMY = frozenset(
    {
        "missing_header_or_declaration",
        "top_signature_mismatch",
        "array_or_pointer_semantics",
        "initialization_or_state",
        "loop_or_index",
        "inplace_or_dataflow",
        "numeric_precision_or_width",
        "missing_output_commit",
        "unsynthesizable_construct",
        "stage_unknown",
    }
)
ACTION_INTENT_TAXONOMY = frozenset(
    {
        "header_include",
        "signature_restore",
        "array_dereference",
        "array_copy_elementwise",
        "remove_illegal_main",
        "initialize_state",
        "adjust_loop_bound",
        "fix_index_map",
        "preserve_inplace_order",
        "precision_or_width",
        "commit_output",
        "remove_unsynthesizable_construct",
        "stage_generic_other",
    }
)


def _canonical_sha256(namespace: str, *values: Any) -> str:
    payload = [namespace, *values]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TrajectoryKey:
    benchmark_case_name: str
    sample_index: int
    seed: int
    mode: str

    def __post_init__(self) -> None:
        if not self.benchmark_case_name.strip():
            raise ValueError("benchmark_case_name must be non-empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if not self.mode.strip():
            raise ValueError("mode must be non-empty")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass
class NegativeLedgerItem:
    trajectory_key_sha256: str
    scope_kind: str
    scope_id: str
    family_kind: str
    family_id: str
    status: str
    first_round: int
    latest_round: int
    evidence_rounds: list[int] = field(default_factory=list)
    evidence_basis: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    operator_ids: list[str] = field(default_factory=list)


def failure_basin_id(
    stage: str, signature: str | None, *, source: bool
) -> str | None:
    if stage == "success" or signature is None:
        return None
    namespace = "source-basin-v1" if source else "result-basin-v1"
    return _canonical_sha256(namespace, stage, signature)[:24]


def root_family_id(
    stage: str,
    canonical_root_intent_id: str,
    canonical_action_target_id: str | None,
) -> str:
    return _canonical_sha256(
        "root-family-v1",
        stage,
        canonical_root_intent_id,
        canonical_action_target_id,
    )[:24]


def action_family_id(
    *,
    stage: str,
    action_type: str,
    operator_id: str | None,
    canonical_action_intent_id: str,
    canonical_action_target_id: str | None,
    rule_ids: list[str],
) -> str:
    return _canonical_sha256(
        "action-family-v1",
        stage,
        action_type,
        operator_id,
        canonical_action_intent_id,
        canonical_action_target_id,
        sorted(set(rule_ids)),
    )[:24]


def stage_vector_from_mapping(outcome: dict[str, bool]) -> list[int]:
    return [
        int(bool(outcome.get(key)))
        for key in ("parse", "compile", "tb", "synth", "tb_and_synth")
    ]


def frontier_episode_id(
    stage: str,
    outcome: dict[str, bool],
    compile_frontier: dict[str, Any],
) -> str:
    return _canonical_sha256(
        "frontier-episode-v1",
        stage,
        stage_vector_from_mapping(outcome),
        list(compile_subfrontier_score(compile_frontier)),
    )[:24]

_STAGE_KEYS = ("tb_and_synth", "tb", "synth", "compile", "parse")
_INTERESTING_LOG = re.compile(
    r"error|fatal|fail|mismatch|expected|actual|undefined|undeclared|timeout|"
    r"segmentation|recursive|synthesi|unsupported|bound",
    re.IGNORECASE,
)
_ABSOLUTE_PATH = re.compile(r"(?:/[A-Za-z0-9_.+-]+){2,}")
_SOURCE_LOCATION = re.compile(
    r"(?P<file>\b[A-Za-z0-9_.+-]+\.(?:c|cc|cpp|cxx|h|hh|hpp)):\d+(?::\d+)?",
    re.IGNORECASE,
)
_RUN_NAME = re.compile(r"\b[A-Za-z0-9_.+-]+__s\d+__r\d+\b")


_COMPILE_ERROR_LINE = re.compile(
    r"(?:^|\s)(?:fatal\s+)?error\s*:|"
    r"\berror\s+[A-Z0-9-]+\s*:|"
    r"\bundefined reference\b",
    re.IGNORECASE,
)
_COMPILE_ERROR_SUMMARY = re.compile(
    r"\b(\d+)\s+errors?\s+generated\b", re.IGNORECASE
)
_COMPILE_TIMEOUT = re.compile(
    r"\b(?:timed?\s*out|timeout)\b|"
    r"\b(?:exit|return)\s*(?:code|status)?\s*[:=]?\s*124\b|"
    r"\breturncode\s*[:=]\s*124\b",
    re.IGNORECASE,
)
_UNRESOLVED_SYMBOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:use of\s+)?undeclared identifier\s*[\x60'\"\u2018]([^\x60'\"\u2019\s;:,()]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[\x60'\"\u2018]([^\x60'\"\u2019]+)[\x60'\"\u2019]\s+(?:was|is)\s+not\s+declared",
        re.IGNORECASE,
    ),
    re.compile(
        r"unknown type name\s*[\x60'\"\u2018]?([A-Za-z_]\w*(?:::\w+)*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"undefined reference to\s*[\x60'\"\u2018]([^\x60'\"\u2019]+)[\x60'\"\u2019]",
        re.IGNORECASE,
    ),
    re.compile(
        r"no (?:member|matching function) named\s*[\x60'\"\u2018]([^\x60'\"\u2019]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"implicit declaration of function\s*[\x60'\"\u2018]([^\x60'\"\u2019]+)",
        re.IGNORECASE,
    ),
)


_MECHANISM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("format_contract", re.compile(r"output_code|xml|markdown|filename", re.IGNORECASE)),
    ("header_include", re.compile(r"header|include|unknown type", re.IGNORECASE)),
    ("helper_body", re.compile(r"undeclared|helper|forward declaration|not declared", re.IGNORECASE)),
    ("signature_or_type", re.compile(r"signature|argument|conversion|type", re.IGNORECASE)),
    ("initialization", re.compile(r"initiali[sz]|uninitialized|identity", re.IGNORECASE)),
    ("loop_bounds", re.compile(r"loop bound|off.by.one|out of bounds|tripcount", re.IGNORECASE)),
    ("index_mapping", re.compile(r"index|stride|row|column|flatten", re.IGNORECASE)),
    ("inplace_update", re.compile(r"in.place|read.before.write|overwrite|copy back", re.IGNORECASE)),
    ("bit_width_shift", re.compile(r"shift|mask|bit width|pack|unpack", re.IGNORECASE)),
    ("division_rounding", re.compile(r"division|divide|round|truncate|quotient", re.IGNORECASE)),
    ("reduction_identity", re.compile(r"reduction|accumulat|identity|dot product|checksum", re.IGNORECASE)),
    ("output_validity", re.compile(r"valid count|output length|sentinel|valid flag", re.IGNORECASE)),
    ("branch_coverage", re.compile(r"branch|piecewise|threshold|fallthrough|boundary predicate", re.IGNORECASE)),
    ("precision", re.compile(r"precision|overflow|signed|cast|nan|fraction|exponent", re.IGNORECASE)),
    ("formula_or_reduction", re.compile(r"formula|reduction|accumulat|operator|axis", re.IGNORECASE)),
    ("output_commit", re.compile(r"output|result|commit|assign", re.IGNORECASE)),
    ("dynamic_storage", re.compile(r"dynamic|malloc|new|vector|stl", re.IGNORECASE)),
    ("recursion", re.compile(r"recurs", re.IGNORECASE)),
    ("unbounded_loop", re.compile(r"unbounded|finite maximum|loop trip", re.IGNORECASE)),
    ("interface_or_pointer", re.compile(r"interface|pointer|port", re.IGNORECASE)),
    ("pragma", re.compile(r"pragma|directive", re.IGNORECASE)),
)


def diagnostic_atoms(text: str, limit: int = 6) -> list[str]:
    atoms: list[str] = []
    for line in normalize_failure_text(text).split(" | "):
        if line and line not in atoms:
            atoms.append(line[:220])
        if len(atoms) >= limit:
            break
    return atoms


def infer_mechanism_id(*, stage: str, diagnosis: Diagnosis, rule_ids: list[str], operator_id: str | None, action_type: str) -> str:
    if action_type == "cleanroom_llm":
        return "cleanroom_rederive"
    if operator_id:
        return f"operator:{operator_id}"
    evidence = " ".join([diagnosis.root_cause, *diagnosis.allowed_actions, *rule_ids])
    for name, pattern in _MECHANISM_PATTERNS:
        if pattern.search(evidence):
            return name
    return f"{stage}_other"


def stage_vector(review: ReviewResult) -> dict[str, bool]:
    return {
        "parse": bool(review.pass_parse),
        "compile": bool(review.pass_compile),
        "tb": bool(review.pass_tb),
        "synth": bool(review.pass_synth),
        "tb_and_synth": bool(review.pass_tb_and_synth),
    }


def score_vector(outcome: dict[str, bool]) -> tuple[int, ...]:
    return tuple(int(bool(outcome[key])) for key in _STAGE_KEYS)


def normalize_failure_text(text: str, limit: int = 900) -> str:
    """Normalize volatile evidence without erasing error types or symbol names."""

    def compact_path(match: re.Match[str]) -> str:
        return f"<path>/{match.group(0).rsplit('/', 1)[-1]}"

    text = _ABSOLUTE_PATH.sub(compact_path, text)
    text = _SOURCE_LOCATION.sub(
        lambda match: f"{match.group('file')}:<loc>", text
    )
    text = _RUN_NAME.sub("<design_run>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<time>", text)
    text = re.sub(
        r"\b(line|column)\s+#?\d+\b", r"\1 <n>", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(round|iteration|attempt)\s*#?\s*\d+\b",
        r"\1 <n>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b\d+\s+(?:warnings?|errors?)\s+generated\b",
        "<n> diagnostics generated",
        text,
        flags=re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    selected = [line for line in lines if _INTERESTING_LOG.search(line)]
    normalized = " | ".join((selected or lines)[:8]).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:limit]


def failure_signature(stage: str, tool_log: str) -> tuple[str | None, str]:
    if stage == "success":
        return None, ""
    normalized = normalize_failure_text(tool_log)
    digest = hashlib.sha256(f"{stage}\n{normalized}".encode("utf-8")).hexdigest()
    return f"{stage}:{digest[:16]}", normalized


def compile_diagnostic_frontier(stage: str, tool_log: str) -> dict[str, Any]:
    """Extract a bounded, trajectory-local Compile diagnostic frontier.

    The frontier intentionally uses only compiler evidence already available to
    the current trajectory. It never inspects another task, reference kernel,
    or a later round.
    """

    if stage != "compile":
        return {
            "applicable": False,
            "normalized_error_count": 0,
            "unresolved_symbols": [],
            "evidence": [],
            "timed_out": False,
        }

    log = tool_log or ""
    unique_errors: list[str] = []
    for raw_line in log.splitlines():
        if not _COMPILE_ERROR_LINE.search(raw_line):
            continue
        normalized = normalize_failure_text(raw_line, limit=260)
        if normalized and normalized not in unique_errors:
            unique_errors.append(normalized)
        if len(unique_errors) >= 64:
            break

    explicit_counts = [
        int(match.group(1)) for match in _COMPILE_ERROR_SUMMARY.finditer(log)
    ]
    error_count = max(len(unique_errors), max(explicit_counts, default=0))

    symbols: set[str] = set()
    for pattern in _UNRESOLVED_SYMBOL_PATTERNS:
        for match in pattern.finditer(log):
            symbol = re.sub(r"\s+", " ", match.group(1)).strip(
                " \t\r\n\x60'\"\u2018\u2019"
            )
            if symbol and len(symbol) <= 120 and not symbol.isdigit():
                symbols.add(symbol)
            if len(symbols) >= 32:
                break
        if len(symbols) >= 32:
            break

    return {
        "applicable": True,
        "normalized_error_count": min(int(error_count), 999),
        "unresolved_symbols": sorted(symbols, key=lambda value: value.lower()),
        "evidence": diagnostic_atoms(log, limit=6),
        "timed_out": bool(_COMPILE_TIMEOUT.search(log)),
    }


def compile_subfrontier_score(frontier: dict[str, Any]) -> tuple[int, ...]:
    """Higher is better; an opaque log cannot outrank an evidenced failure."""

    if not frontier.get("applicable"):
        return (0, 0, 0, 0, 0)
    evidence_available = bool(frontier.get("evidence"))
    return (
        1,
        int(not bool(frontier.get("timed_out"))),
        int(evidence_available),
        -min(int(frontier.get("normalized_error_count") or 0), 999),
        -min(len(frontier.get("unresolved_symbols") or []), 99),
    )


def compile_frontier_delta(before, after):
    before_symbols = set(before.get("unresolved_symbols") or [])
    after_symbols = set(after.get("unresolved_symbols") or [])
    return {
        "error_count_before": int(before.get("normalized_error_count") or 0),
        "error_count_after": int(after.get("normalized_error_count") or 0),
        "error_count_reduction": (
            int(before.get("normalized_error_count") or 0)
            - int(after.get("normalized_error_count") or 0)
        ),
        "resolved_symbols": sorted(before_symbols - after_symbols),
        "introduced_symbols": sorted(after_symbols - before_symbols),
        "timeout_cleared": bool(
            before.get("timed_out") and not after.get("timed_out")
        ),
        "evidence_changed": before.get("evidence") != after.get("evidence"),
    }


def _bounded_prompt_strings(
    values: list[Any] | set[str],
    *,
    count: int,
    width: int,
) -> list[str]:
    return [str(value)[:width] for value in sorted(values)[:count]]


def summarize_code_change(before: str, after: str) -> str:
    matcher = difflib.SequenceMatcher(
        a=before.splitlines(), b=after.splitlines(), autojunk=False
    )
    added = removed = replaced = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            replaced += 1
            removed += i2 - i1
            added += j2 - j1
    return f"replace_blocks={replaced}; added_lines={added}; removed_lines={removed}"


def changed_span_sha256(before: str, after: str) -> str:
    matcher = difflib.SequenceMatcher(
        a=before.splitlines(), b=after.splitlines(), autojunk=False
    )
    changed: list[dict[str, Any]] = []
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed.append(
            {
                "tag": tag,
                "before": [line.strip() for line in before_lines[i1:i2]],
                "after": [line.strip() for line in after_lines[j1:j2]],
            }
        )
    return hashlib.sha256(
        json.dumps(changed, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def action_fingerprint(
    *,
    action_type: str,
    operator_id: str | None,
    rule_ids: list[str],
    root_cause: str,
    code_change_summary: str,
    changed_span_hash: str,
) -> str:
    """Stable trajectory-local action identity for repetition/credit assignment."""
    payload = {
        "action_type": action_type,
        "operator_id": operator_id,
        "rule_ids": sorted(set(rule_ids)),
        "root_cause": normalize_failure_text(root_cause, limit=240),
        "code_change_summary": code_change_summary,
        "changed_span_sha256": changed_span_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def review_tool_seconds(review: ReviewResult) -> dict[str, float]:
    def elapsed(value: dict[str, Any] | None) -> float:
        return float((value or {}).get("execution_time") or 0.0)

    return {
        "compile": elapsed(review.compile),
        "tb": elapsed(review.testbench),
        "synth": elapsed(review.synthesis),
    }


@dataclass
class CandidateRecord:
    candidate_id: str
    round_index: int
    code_sha256: str
    outcome: dict[str, bool]
    score: list[int]
    compile_frontier: dict[str, Any]
    compile_subfrontier_score: list[int]
    total_tokens: int
    tool_seconds: dict[str, float]
    trajectory_key_sha256: str
    result_failure_stage: str
    result_failure_signature: str | None
    result_failure_basin_id: str | None


@dataclass
class RFLEntry:
    round_index: int
    candidate_id: str
    parent_candidate_id: str | None
    base_origin: str
    source_code_sha256: str | None
    candidate_code_sha256: str
    failure_signature_before: str | None
    failure_signature_after: str | None
    normalized_failure_after: str
    failure_stage: str
    rule_ids: list[str]
    root_cause: str
    diagnosis_conflict: bool
    diagnosis_conflict_count: int
    duplicate_code_count: int
    two_cycle_detected: bool
    action_type: str
    operator_id: str | None
    action_fingerprint: str
    mechanism_id: str
    diagnostic_atoms: list[str]
    predicted_observation: str
    observed_delta: str
    compile_frontier_before: dict[str, Any]
    compile_frontier_after: dict[str, Any]
    compile_frontier_delta: dict[str, Any]
    compile_subfrontier_progress: bool
    unresolved_symbol_union: list[str]
    new_unresolved_symbols: list[str]
    resolved_unresolved_symbols: list[str]
    hypothesis_credit_basis: list[str]
    format_recovery_kind: str | None
    changed_span_sha256: str
    same_action_count: int
    hypothesis_status: str
    invariant_violations: list[str]
    before_outcome: dict[str, bool] | None
    after_outcome: dict[str, bool]
    stage_regression: bool
    component_regression: bool
    repeated_signature_count: int
    no_progress_streak: int
    frontier_no_improvement_streak: int
    cleanroom_used: bool
    operator_used: bool
    best_candidate_id: str
    best_outcome: dict[str, bool]
    code_change_summary: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_seconds: dict[str, float]
    trajectory_key_sha256: str = ""
    source_failure_stage: str | None = None
    source_failure_signature: str | None = None
    source_failure_basin_id: str | None = None
    result_failure_stage: str = "parse"
    result_failure_signature: str | None = None
    result_failure_basin_id: str | None = None
    frontier_episode_id_before: str = ""
    frontier_episode_id_after: str = ""
    canonical_root_intent_id: str = "stage_unknown"
    canonical_action_intent_id: str = "stage_generic_other"
    canonical_action_target_id: str | None = None
    action_veto_eligible: bool = False
    root_veto_eligible: bool = False
    root_family_id: str = ""
    root_status: str = "unresolved"
    root_credit_basis: list[str] = field(default_factory=list)
    action_family_id: str = ""
    action_status: str = "unresolved"
    action_credit_basis: list[str] = field(default_factory=list)
    edit_instance_fingerprint: str = ""
    visibility_decision: dict[str, str] = field(default_factory=dict)


def _opaque_family(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


class ReviewerGroundedFalsificationLedger:
    """Trajectory-local state, stagnation detector, and candidate archive."""

    def __init__(
        self,
        max_entries: int = 5,
        *,
        trajectory_key: TrajectoryKey | None = None,
        schema_version: str = RFL_SCHEMA_VERSION,
    ) -> None:
        if not 1 <= max_entries <= 5:
            raise ValueError("RFL retains between one and five entries")
        if schema_version != RFL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported RFL schema {schema_version!r}; "
                f"expected {RFL_SCHEMA_VERSION!r}"
            )
        self._explicit_trajectory_key = trajectory_key is not None
        self.trajectory_key = trajectory_key or TrajectoryKey(
            benchmark_case_name="__legacy_test_fixture__",
            sample_index=0,
            seed=0,
            mode="test_compat",
        )
        self.trajectory_key_sha256 = self.trajectory_key.sha256
        self.max_entries = max_entries
        self.entries: list[RFLEntry] = []
        self.candidates: list[CandidateRecord] = []
        self.signature_counts: Counter[str] = Counter()
        self.signature_history: list[str | None] = []
        self.code_sha_counts: Counter[str] = Counter()
        self.code_sha_history: list[str] = []
        self.current_duplicate_code = False
        self.current_two_cycle = False
        self.root_cause_counts: Counter[str] = Counter()
        self.signature_root_causes: dict[str, Counter[str]] = defaultdict(Counter)
        self.failed_strategy_counts: Counter[str] = Counter()
        self.action_fingerprint_counts: Counter[str] = Counter()
        self.mechanism_counts: Counter[str] = Counter()
        self.failed_mechanism_counts: Counter[str] = Counter()
        self.current_mechanism_id: str | None = None
        self.format_recovery_counts: Counter[str] = Counter()
        self.current_action_fingerprint: str | None = None
        self.current_action_failed = False
        self.hypothesis_status_counts: Counter[str] = Counter()
        self.diagnosis_conflict_count = 0
        self.current_diagnosis_conflict = False
        self.recovery_count = 0
        self.no_progress_streak = 0
        self.frontier_no_improvement_streak = 0
        self.current_frontier_improved = False
        self.current_compile_subfrontier_progress = False
        self.compile_unresolved_symbol_union: set[str] = set()
        self.cleanroom_used = False
        self.cleanroom_trigger_reason: str | None = None
        self.cleanroom_trigger_round: int | None = None
        self.cleanroom_trigger_candidate_id: str | None = None
        self.cleanroom_trigger_signature: str | None = None
        self.cleanroom_trigger_stage: str | None = None
        self.cleanroom_trigger_outcome: dict[str, bool] | None = None
        self.current_signature: str | None = None
        self.best_candidate_id: str | None = None
        self.best_outcome: dict[str, bool] = {
            "parse": False,
            "compile": False,
            "tb": False,
            "synth": False,
            "tb_and_synth": False,
        }
        self._best_score = score_vector(self.best_outcome)
        self._best_subfrontier_score = (0, 0, 0, 0, 0)
        self._best_rank = (self._best_score, self._best_subfrontier_score)
        self._best_round = 10**9
        self._best_cost = 10**18
        self.current_frontier_episode_id = ""
        self.negative_ledger: list[NegativeLedgerItem] = []
        self.active_source_candidate_id: str | None = None
        self.active_source_failure_basin_id: str | None = None
        self.action_family_outcomes: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.root_family_contradicted_actions: dict[tuple[str, str], set[str]] = defaultdict(set)

    def _upsert_ledger(
        self,
        *,
        scope_kind: str,
        scope_id: str | None,
        family_kind: str,
        family_id: str,
        status: str,
        round_index: int,
        evidence_basis: list[str],
        rule_ids: list[str],
        operator_id: str | None,
    ) -> None:
        if not scope_id:
            return
        for item in self.negative_ledger:
            if (
                item.trajectory_key_sha256 == self.trajectory_key_sha256
                and item.scope_kind == scope_kind
                and item.scope_id == scope_id
                and item.family_kind == family_kind
                and item.family_id == family_id
            ):
                item.status = status
                item.latest_round = round_index
                if round_index not in item.evidence_rounds:
                    item.evidence_rounds.append(round_index)
                item.evidence_basis = list(
                    dict.fromkeys(item.evidence_basis + evidence_basis)
                )
                item.rule_ids = list(dict.fromkeys(item.rule_ids + rule_ids))
                if operator_id and operator_id not in item.operator_ids:
                    item.operator_ids.append(operator_id)
                return
        self.negative_ledger.append(
            NegativeLedgerItem(
                trajectory_key_sha256=self.trajectory_key_sha256,
                scope_kind=scope_kind,
                scope_id=scope_id,
                family_kind=family_kind,
                family_id=family_id,
                status=status,
                first_round=round_index,
                latest_round=round_index,
                evidence_rounds=[round_index],
                evidence_basis=list(dict.fromkeys(evidence_basis)),
                rule_ids=list(dict.fromkeys(rule_ids)),
                operator_ids=[operator_id] if operator_id else [],
            )
        )

    def _frontier_episode_from_best(self) -> str:
        if self.best_candidate_id is None:
            return _canonical_sha256("frontier-episode-v1", "empty")[:24]
        record = next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.best_candidate_id
        )
        return frontier_episode_id(
            record.result_failure_stage,
            record.outcome,
            record.compile_frontier,
        )

    def _record_candidate(
        self,
        candidate_id: str,
        round_index: int,
        code: str,
        review: ReviewResult,
        tool_log: str,
        total_tokens: int,
    ) -> CandidateRecord:
        outcome = stage_vector(review)
        score = score_vector(outcome)
        compile_frontier = compile_diagnostic_frontier(
            review.failure_stage, tool_log
        )
        subfrontier_score = compile_subfrontier_score(compile_frontier)
        rank = (score, subfrontier_score)
        result_signature, _ = failure_signature(review.failure_stage, tool_log)
        record = CandidateRecord(
            candidate_id=candidate_id,
            round_index=round_index,
            code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
            outcome=outcome,
            score=list(score),
            compile_frontier=compile_frontier,
            compile_subfrontier_score=list(subfrontier_score),
            total_tokens=total_tokens,
            tool_seconds=review_tool_seconds(review),
            trajectory_key_sha256=self.trajectory_key_sha256,
            result_failure_stage=review.failure_stage,
            result_failure_signature=result_signature,
            result_failure_basin_id=failure_basin_id(
                review.failure_stage, result_signature, source=False
            ),
        )
        self.candidates.append(record)
        self.code_sha_counts[record.code_sha256] += 1
        self.code_sha_history.append(record.code_sha256)
        self.current_duplicate_code = self.code_sha_counts[record.code_sha256] >= 2
        code_two_cycle = (
            len(self.code_sha_history) >= 3
            and self.code_sha_history[-1] == self.code_sha_history[-3]
            and self.code_sha_history[-1] != self.code_sha_history[-2]
        )
        self.current_two_cycle = code_two_cycle
        is_better = (
            rank > self._best_rank
            or (
                rank == self._best_rank
                and (
                    round_index < self._best_round
                    or (
                        round_index == self._best_round
                        and total_tokens < self._best_cost
                    )
                )
            )
        )
        if self.best_candidate_id is None or is_better:
            self.best_candidate_id = candidate_id
            self.best_outcome = outcome
            self._best_score = score
            self._best_subfrontier_score = subfrontier_score
            self._best_rank = rank
            self._best_round = round_index
            self._best_cost = total_tokens
        self.current_frontier_episode_id = self._frontier_episode_from_best()
        return record

    def observe_round0(
        self,
        candidate_id: str,
        code: str,
        review: ReviewResult,
        tool_log: str,
        total_tokens: int,
    ) -> None:
        signature, normalized = failure_signature(review.failure_stage, tool_log)
        if signature:
            self.signature_counts[signature] += 1
        self.current_signature = signature
        self.signature_history.append(signature)
        compile_frontier = compile_diagnostic_frontier(
            review.failure_stage, tool_log
        )
        self.compile_unresolved_symbol_union.update(
            compile_frontier.get("unresolved_symbols") or []
        )
        self._record_candidate(
            candidate_id, 0, code, review, tool_log, total_tokens
        )
        outcome = stage_vector(review)
        self.entries.append(
            RFLEntry(
                round_index=0,
                candidate_id=candidate_id,
                parent_candidate_id=None,
                base_origin="immutable_original",
                source_code_sha256=None,
                candidate_code_sha256=hashlib.sha256(
                    code.encode("utf-8")
                ).hexdigest(),
                failure_signature_before=None,
                failure_signature_after=signature,
                normalized_failure_after=normalized,
                failure_stage=review.failure_stage,
                rule_ids=[],
                root_cause="",
                diagnosis_conflict=False,
                diagnosis_conflict_count=0,
                duplicate_code_count=1,
                two_cycle_detected=False,
                action_type="round0",
                operator_id=None,
                action_fingerprint="round0",
                mechanism_id="round0_generation",
                diagnostic_atoms=diagnostic_atoms(tool_log),
                predicted_observation="",
                observed_delta="initial_candidate",
                compile_frontier_before={
                    "applicable": False,
                    "normalized_error_count": 0,
                    "unresolved_symbols": [],
                    "evidence": [],
                    "timed_out": False,
                },
                compile_frontier_after=compile_frontier,
                compile_frontier_delta={},
                compile_subfrontier_progress=False,
                unresolved_symbol_union=sorted(
                    self.compile_unresolved_symbol_union
                ),
                new_unresolved_symbols=list(
                    compile_frontier.get("unresolved_symbols") or []
                ),
                resolved_unresolved_symbols=[],
                hypothesis_credit_basis=[],
                format_recovery_kind=None,
                changed_span_sha256=changed_span_sha256("", code),
                same_action_count=1,
                hypothesis_status="new",
                invariant_violations=[],
                before_outcome=None,
                after_outcome=outcome,
                stage_regression=False,
                component_regression=False,
                repeated_signature_count=self.signature_counts.get(signature or "", 0),
                no_progress_streak=0,
                frontier_no_improvement_streak=0,
                cleanroom_used=False,
                operator_used=False,
                best_candidate_id=self.best_candidate_id or candidate_id,
                best_outcome=dict(self.best_outcome),
                code_change_summary="initial candidate",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=total_tokens,
                tool_seconds=review_tool_seconds(review),
                trajectory_key_sha256=self.trajectory_key_sha256,
                result_failure_stage=review.failure_stage,
                result_failure_signature=signature,
                result_failure_basin_id=failure_basin_id(
                    review.failure_stage, signature, source=False
                ),
                frontier_episode_id_before=self.current_frontier_episode_id,
                frontier_episode_id_after=self.current_frontier_episode_id,
                visibility_decision={
                    "diagnoser": "hidden",
                    "repairer": "hidden",
                    "cleanroom": "hidden",
                    "reason": "round0_has_no_repair_outcome",
                },
            )
        )

    def record_repair(
        self,
        *,
        round_index: int,
        candidate_id: str,
        source_code: str | None,
        candidate_code: str,
        source_review: ReviewResult,
        candidate_review: ReviewResult,
        source_log: str,
        candidate_log: str,
        diagnosis: Diagnosis,
        rule_ids: list[str],
        action_type: str,
        operator_id: str | None,
        cleanroom_used: bool,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        parent_candidate_id: str | None = None,
        base_origin: str = "latest",
        invariant_violations: list[str] | None = None,
        format_recovery: dict[str, Any] | None = None,
    ) -> None:
        before = stage_vector(source_review)
        after = stage_vector(candidate_review)
        before_score = score_vector(before)
        after_score = score_vector(after)
        before_compile_frontier = compile_diagnostic_frontier(
            source_review.failure_stage, source_log
        )
        after_compile_frontier = compile_diagnostic_frontier(
            candidate_review.failure_stage, candidate_log
        )
        before_subfrontier = compile_subfrontier_score(before_compile_frontier)
        after_subfrontier = compile_subfrontier_score(after_compile_frontier)
        before_rank = (before_score, before_subfrontier)
        after_rank = (after_score, after_subfrontier)
        best_rank_before = self._best_rank
        frontier_episode_before = self.current_frontier_episode_id
        compile_subfrontier_progress = bool(
            after_score == before_score
            and after_compile_frontier.get("applicable")
            and after_subfrontier > before_subfrontier
        )
        self.current_compile_subfrontier_progress = (
            compile_subfrontier_progress
        )
        self.current_frontier_improved = after_rank > before_rank
        if after_rank > best_rank_before:
            self.frontier_no_improvement_streak = 0
        else:
            self.frontier_no_improvement_streak += 1
        if after_rank > before_rank:
            self.no_progress_streak = 0
        elif candidate_review.failure_stage == source_review.failure_stage:
            self.no_progress_streak += 1
        else:
            # A stage transition starts a new repair basin; stagnation must not
            # accumulate across unrelated Parse/Compile/TB/Synth failures.
            self.no_progress_streak = 0

        signature_before, _ = failure_signature(source_review.failure_stage, source_log)
        signature_after, normalized_after = failure_signature(
            candidate_review.failure_stage, candidate_log
        )
        source_record = next(
            (
                item
                for item in self.candidates
                if item.candidate_id == parent_candidate_id
            ),
            None,
        )
        source_basin = (
            source_record.result_failure_basin_id
            if source_record is not None
            else failure_basin_id(
                source_review.failure_stage, signature_before, source=False
            )
        )
        result_basin = failure_basin_id(
            candidate_review.failure_stage, signature_after, source=False
        )
        root_key = normalize_failure_text(diagnosis.root_cause, limit=240)
        source_text = source_code or ""
        change_summary = (
            "cleanroom_rederived_from_immutable_task"
            if source_code is None
            else summarize_code_change(source_text, candidate_code)
        )
        span_hash = changed_span_sha256(source_text, candidate_code)
        fingerprint = action_fingerprint(
            action_type=action_type,
            operator_id=operator_id,
            rule_ids=rule_ids,
            root_cause=diagnosis.root_cause,
            code_change_summary=change_summary,
            changed_span_hash=span_hash,
        )
        self.action_fingerprint_counts[fingerprint] += 1
        self.current_action_fingerprint = fingerprint
        mechanism_id = infer_mechanism_id(
            stage=source_review.failure_stage,
            diagnosis=diagnosis,
            rule_ids=rule_ids,
            operator_id=operator_id,
            action_type=action_type,
        )
        self.mechanism_counts[mechanism_id] += 1
        self.current_mechanism_id = mechanism_id
        same_action_count = self.action_fingerprint_counts[fingerprint]
        format_payload = format_recovery or {}
        format_kind = (
            str(format_payload.get("kind"))
            if format_payload.get("checked")
            else None
        )
        if format_kind:
            self.format_recovery_counts[format_kind] += 1
        source_sha = (
            hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            if source_code is not None
            else None
        )
        candidate_sha = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
        signature_changed = signature_before != signature_after
        union_before = set(self.compile_unresolved_symbol_union)
        before_symbols = set(
            before_compile_frontier.get("unresolved_symbols") or []
        )
        after_symbols = set(
            after_compile_frontier.get("unresolved_symbols") or []
        )
        new_unresolved_symbols = sorted(after_symbols - union_before)
        resolved_unresolved_symbols = sorted(before_symbols - after_symbols)
        self.compile_unresolved_symbol_union.update(after_symbols)
        frontier_delta = compile_frontier_delta(
            before_compile_frontier, after_compile_frontier
        )
        component_regression = any(
            before[key] and not after[key] for key in before
        )
        invariant_failure = bool(invariant_violations)
        hypothesis_credit_basis: list[str] = []
        if after["tb_and_synth"]:
            hypothesis_credit_basis.append("joint_success")
        if after_score > before_score:
            hypothesis_credit_basis.append("stage_frontier_improved")
        if compile_subfrontier_progress:
            if frontier_delta["timeout_cleared"]:
                hypothesis_credit_basis.append("compile_timeout_cleared")
            if frontier_delta["error_count_reduction"] > 0:
                hypothesis_credit_basis.append(
                    "compile_error_count_reduced"
                )
            if resolved_unresolved_symbols:
                hypothesis_credit_basis.append(
                    "compile_unresolved_symbols_resolved"
                )
            if not hypothesis_credit_basis:
                hypothesis_credit_basis.append(
                    "compile_diagnostic_subfrontier_improved"
                )
        if (
            hypothesis_credit_basis
            and not component_regression
            and not invariant_failure
        ):
            hypothesis_status = "supported"
        elif (
            after_rank < before_rank
            or signature_before == signature_after
            or source_sha == candidate_sha
            or component_regression
            or invariant_failure
        ):
            hypothesis_status = "contradicted"
        else:
            # A changed error string is evidence, not repair credit. Without a
            # stage or Compile diagnostic subfrontier gain the hypothesis stays
            # unresolved and cannot displace an earlier/lower-cost candidate.
            hypothesis_status = "unresolved"
        canonical_root = (
            diagnosis.canonical_root_intent_id
            if diagnosis.canonical_root_intent_id in ROOT_INTENT_TAXONOMY
            else "stage_unknown"
        )
        canonical_action = (
            diagnosis.canonical_action_intent_id
            if diagnosis.canonical_action_intent_id in ACTION_INTENT_TAXONOMY
            else "stage_generic_other"
        )
        validated_target = diagnosis.canonical_action_target_id
        action_eligible = bool(
            diagnosis.action_veto_eligible
            and canonical_action != "stage_generic_other"
            and validated_target
        )
        root_eligible = bool(
            diagnosis.root_veto_eligible
            and canonical_root != "stage_unknown"
            and validated_target
        )
        root_family = root_family_id(
            source_review.failure_stage, canonical_root, validated_target
        )
        action_family = action_family_id(
            stage=source_review.failure_stage,
            action_type=action_type,
            operator_id=operator_id,
            canonical_action_intent_id=canonical_action,
            canonical_action_target_id=validated_target,
            rule_ids=rule_ids,
        )
        instance_fingerprint = _canonical_sha256(
            "edit-instance-v1", action_family, change_summary, span_hash
        )[:24]
        action_status = hypothesis_status
        action_credit_basis = list(hypothesis_credit_basis)
        family_episode_key = (frontier_episode_before, action_family)
        self.action_family_outcomes[family_episode_key].append(action_status)
        unresolved_count = self.action_family_outcomes[family_episode_key].count(
            "unresolved"
        )
        if action_eligible and action_status == "contradicted":
            self._upsert_ledger(
                scope_kind="source_failure_basin",
                scope_id=source_basin,
                family_kind="action_family",
                family_id=action_family,
                status="vetoed",
                round_index=round_index,
                evidence_basis=["objective_contradiction"],
                rule_ids=rule_ids,
                operator_id=operator_id,
            )
        elif action_eligible and action_status == "unresolved" and unresolved_count >= 2:
            action_status = "exhausted"
            action_credit_basis.append("two_unresolved_no_gain_in_frontier_episode")
            self._upsert_ledger(
                scope_kind="frontier_episode",
                scope_id=frontier_episode_before,
                family_kind="action_family",
                family_id=action_family,
                status="exhausted",
                round_index=round_index,
                evidence_basis=["two_unresolved_no_gain_in_frontier_episode"],
                rule_ids=rule_ids,
                operator_id=operator_id,
            )
        root_status = "unresolved"
        root_credit_basis: list[str] = []
        if action_status == "supported" and (
            signature_changed
            or after_rank > before_rank
            or compile_subfrontier_progress
        ):
            root_status = "supported"
            root_credit_basis.append(
                "supported_action_and_objective_or_signature_gain"
            )
        elif (
            action_status == "contradicted"
            and action_eligible
            and root_eligible
            and source_basin
        ):
            root_key_scoped = (source_basin, root_family)
            self.root_family_contradicted_actions[root_key_scoped].add(
                action_family
            )
            if len(self.root_family_contradicted_actions[root_key_scoped]) >= 2:
                root_status = "contradicted"
                root_credit_basis.append(
                    "two_distinct_contradicted_action_families"
                )
                self._upsert_ledger(
                    scope_kind="source_failure_basin",
                    scope_id=source_basin,
                    family_kind="root_family",
                    family_id=root_family,
                    status="vetoed",
                    round_index=round_index,
                    evidence_basis=root_credit_basis,
                    rule_ids=rule_ids,
                    operator_id=operator_id,
                )
        self.current_action_failed = action_status == "contradicted"
        if self.current_action_failed:
            self.failed_mechanism_counts[mechanism_id] += 1
        self.hypothesis_status_counts[action_status] += 1
        diagnosis_conflict = False
        if root_key and signature_before:
            prior_roots = self.signature_root_causes[signature_before]
            diagnosis_conflict = bool(prior_roots and root_key not in prior_roots)
            if diagnosis_conflict:
                self.diagnosis_conflict_count += 1
            prior_roots[root_key] += 1
        if root_key:
            self.root_cause_counts[root_key] += 1
        if after_rank <= before_rank:
            strategy = (
                f"{signature_before or source_review.failure_stage}|{fingerprint}"
            )
            self.failed_strategy_counts[strategy] += 1
        if signature_after:
            self.signature_counts[signature_after] += 1
        self.current_signature = signature_after
        self.signature_history.append(signature_after)
        signature_two_cycle = (
            len(self.signature_history) >= 3
            and self.signature_history[-1] is not None
            and self.signature_history[-1] == self.signature_history[-3]
            and self.signature_history[-1] != self.signature_history[-2]
        )
        self.current_diagnosis_conflict = bool(
            diagnosis_conflict
            and signature_before is not None
            and signature_before == signature_after
        )
        self._record_candidate(
            candidate_id,
            round_index,
            candidate_code,
            candidate_review,
            candidate_log,
            total_tokens,
        )
        frontier_episode_after = self.current_frontier_episode_id
        self.current_two_cycle = self.current_two_cycle or signature_two_cycle
        entry = RFLEntry(
            round_index=round_index,
            candidate_id=candidate_id,
            parent_candidate_id=parent_candidate_id,
            base_origin=base_origin,
            source_code_sha256=source_sha,
            candidate_code_sha256=hashlib.sha256(
                candidate_code.encode("utf-8")
            ).hexdigest(),
            failure_signature_before=signature_before,
            failure_signature_after=signature_after,
            normalized_failure_after=normalized_after,
            failure_stage=candidate_review.failure_stage,
            rule_ids=list(rule_ids),
            root_cause=diagnosis.root_cause,
            diagnosis_conflict=diagnosis_conflict,
            diagnosis_conflict_count=self.diagnosis_conflict_count,
            duplicate_code_count=self.code_sha_counts[candidate_sha],
            two_cycle_detected=self.current_two_cycle,
            action_type=action_type,
            operator_id=operator_id,
            action_fingerprint=instance_fingerprint,
            mechanism_id=mechanism_id,
            diagnostic_atoms=diagnostic_atoms(candidate_log),
            predicted_observation=(
                diagnosis.predicted_observation
                or f"Eliminate the {source_review.failure_stage} diagnostic with mechanism {mechanism_id}"
            ),
            observed_delta=(
                "stage_improved" if after_score > before_score else
                "stage_regressed" if after_score < before_score else
                "compile_subfrontier_improved"
                if compile_subfrontier_progress else
                "failure_signature_changed_without_frontier_credit"
                if signature_changed else
                "no_observed_progress"
            ),
            compile_frontier_before=before_compile_frontier,
            compile_frontier_after=after_compile_frontier,
            compile_frontier_delta=frontier_delta,
            compile_subfrontier_progress=compile_subfrontier_progress,
            unresolved_symbol_union=sorted(
                self.compile_unresolved_symbol_union
            ),
            new_unresolved_symbols=new_unresolved_symbols,
            resolved_unresolved_symbols=resolved_unresolved_symbols,
            hypothesis_credit_basis=hypothesis_credit_basis,
            format_recovery_kind=format_kind,
            changed_span_sha256=span_hash,
            same_action_count=len(self.action_family_outcomes[family_episode_key]),
            hypothesis_status=action_status,
            invariant_violations=list(invariant_violations or []),
            before_outcome=before,
            after_outcome=after,
            stage_regression=after_score < before_score,
            component_regression=component_regression,
            repeated_signature_count=self.signature_counts.get(
                signature_after or "", 0
            ),
            no_progress_streak=self.no_progress_streak,
            frontier_no_improvement_streak=self.frontier_no_improvement_streak,
            cleanroom_used=cleanroom_used,
            operator_used=operator_id is not None and action_type == "operator",
            best_candidate_id=self.best_candidate_id or candidate_id,
            best_outcome=dict(self.best_outcome),
            code_change_summary=change_summary,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
            tool_seconds=review_tool_seconds(candidate_review),
            trajectory_key_sha256=self.trajectory_key_sha256,
            source_failure_stage=source_review.failure_stage,
            source_failure_signature=signature_before,
            source_failure_basin_id=source_basin,
            result_failure_stage=candidate_review.failure_stage,
            result_failure_signature=signature_after,
            result_failure_basin_id=result_basin,
            frontier_episode_id_before=frontier_episode_before,
            frontier_episode_id_after=frontier_episode_after,
            canonical_root_intent_id=canonical_root,
            canonical_action_intent_id=canonical_action,
            canonical_action_target_id=validated_target,
            action_veto_eligible=action_eligible,
            root_veto_eligible=root_eligible,
            root_family_id=root_family,
            root_status=root_status,
            root_credit_basis=root_credit_basis,
            action_family_id=action_family,
            action_status=action_status,
            action_credit_basis=action_credit_basis,
            edit_instance_fingerprint=instance_fingerprint,
            visibility_decision={
                "diagnoser": (
                    "visible"
                    if action_status in {"supported", "contradicted", "exhausted"}
                    else "count_only"
                ),
                "repairer": (
                    "visible"
                    if action_status in {"supported", "contradicted", "exhausted"}
                    else "count_only"
                ),
                "cleanroom": (
                    "visible"
                    if action_status in {"contradicted", "exhausted"}
                    else "count_only"
                ),
                "reason": "reviewer_grounded_credit",
            },
        )
        self.entries.append(entry)
        self.entries = self.entries[-self.max_entries :]

    def active_ledgers_for(
        self,
        candidate_id: str | None = None,
    ) -> list[NegativeLedgerItem]:
        chosen_id = (
            candidate_id
            or self.active_source_candidate_id
            or (self.candidates[-1].candidate_id if self.candidates else None)
        )
        record = next(
            (
                item
                for item in self.candidates
                if item.candidate_id == chosen_id
            ),
            None,
        )
        chosen_basin = record.result_failure_basin_id if record else None
        active = [
            item
            for item in self.negative_ledger
            if item.trajectory_key_sha256 == self.trajectory_key_sha256
            and item.status in {"vetoed", "exhausted"}
            and (
                (
                    item.scope_kind == "source_failure_basin"
                    and item.scope_id == chosen_basin
                )
                or (
                    item.scope_kind == "frontier_episode"
                    and item.scope_id == self.current_frontier_episode_id
                )
            )
        ]
        return list(active)

    def activate_source(self, candidate_id: str) -> list[NegativeLedgerItem]:
        record = next(
            (item for item in self.candidates if item.candidate_id == candidate_id),
            None,
        )
        if record is None:
            raise ValueError(f"Unknown trajectory candidate {candidate_id!r}")
        if record.trajectory_key_sha256 != self.trajectory_key_sha256:
            raise RuntimeError("Cross-trajectory candidate activation rejected")
        self.active_source_candidate_id = candidate_id
        self.active_source_failure_basin_id = record.result_failure_basin_id
        return self.active_ledgers_for(candidate_id)

    def should_cleanroom(self, next_round: int) -> tuple[bool, str | None]:
        if self.cleanroom_used or next_round not in {3, 4}:
            return False, None
        latest_candidate = self.candidates[-1].candidate_id if self.candidates else None
        active = self.active_ledgers_for(latest_candidate)
        repeated = bool(
            self.current_signature
            and self.signature_counts[self.current_signature] >= 2
        )
        active_exhausted = any(item.status == "exhausted" for item in active)
        active_veto = any(item.status == "vetoed" for item in active)
        latest_entry = self.entries[-1] if self.entries else None
        source_basin = latest_entry.result_failure_basin_id if latest_entry else None
        contradicted_actions = max(
            (
                len(action_ids)
                for (basin, _), action_ids in self.root_family_contradicted_actions.items()
                if basin == source_basin
            ),
            default=0,
        )
        strong_conditions = [
            (self.current_duplicate_code and self.current_action_failed, "duplicate_failed_code_sha256"),
            (self.current_two_cycle, "two_cycle_code_or_failure_oscillation"),
            (
                repeated and self.current_diagnosis_conflict,
                "repeated_signature_with_conflicting_diagnoses",
            ),
            (active_exhausted, "action_family_exhausted_in_frontier_episode"),
            (
                contradicted_actions >= 2,
                "two_contradicted_action_families_in_source_basin",
            ),
        ]
        for condition, reason in strong_conditions:
            if condition:
                return True, reason
        if next_round == 4:
            if (
                repeated
                and self.current_action_failed
                and self.current_mechanism_id
                and self.failed_mechanism_counts[self.current_mechanism_id] >= 2
            ):
                return True, "same_signature_repeated_failed_mechanism"
            if self.frontier_no_improvement_streak >= 2:
                return True, "two_rounds_without_objective_frontier_gain"
            if repeated and (active_veto or active_exhausted):
                return True, "repeated_signature_with_active_negative_ledger"
        return False, None

    def mark_cleanroom(
        self,
        reason: str,
        round_index: int,
        *,
        trigger_candidate_id: str | None = None,
        trigger_signature: str | None = None,
        trigger_stage: str | None = None,
        trigger_outcome: dict[str, bool] | None = None,
    ) -> None:
        if self.cleanroom_used:
            raise RuntimeError("Clean-room regeneration is limited to once per trajectory")
        self.cleanroom_used = True
        self.cleanroom_trigger_reason = reason
        self.cleanroom_trigger_round = round_index
        self.cleanroom_trigger_candidate_id = trigger_candidate_id
        self.cleanroom_trigger_signature = trigger_signature
        self.cleanroom_trigger_stage = trigger_stage
        self.cleanroom_trigger_outcome = (
            dict(trigger_outcome) if trigger_outcome is not None else None
        )

    @property
    def best_round_index(self) -> int:
        if self.best_candidate_id is None:
            raise RuntimeError("No candidate has been archived")
        return next(
            item.round_index
            for item in self.candidates
            if item.candidate_id == self.best_candidate_id
        )

    def recovery_source(
        self,
        current_candidate_id: str,
        current_review: ReviewResult,
    ) -> tuple[str | None, str | None]:
        current_record = next(
            (
                candidate
                for candidate in reversed(self.candidates)
                if candidate.candidate_id == current_candidate_id
            ),
            None,
        )
        best_record = next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == self.best_candidate_id
            ),
            None,
        )
        if current_record is None or best_record is None:
            return None, None
        if best_record.candidate_id == current_candidate_id:
            return None, None
        if best_record.code_sha256 == current_record.code_sha256:
            return None, None
        current_score = score_vector(stage_vector(current_review))
        current_subfrontier = tuple(current_record.compile_subfrontier_score)
        current_rank = (current_score, current_subfrontier)
        latest_entry = next(
            (
                entry
                for entry in reversed(self.entries)
                if entry.candidate_id == current_candidate_id
            ),
            None,
        )
        reasons: list[str] = []
        if self._best_rank > current_rank:
            reasons.append("historical_best_has_higher_stage_frontier")
        if latest_entry is not None:
            if latest_entry.component_regression:
                reasons.append("component_regression")
            if latest_entry.invariant_violations:
                reasons.append("immutable_invariant_violation")
            if latest_entry.action_status == "contradicted":
                reasons.append("contradicted_action_outcome")
            if (
                latest_entry.action_status in {"unresolved", "exhausted"}
                and latest_entry.action_veto_eligible
                and self.frontier_no_improvement_streak >= 2
                and any(
                    item.family_kind == "action_family"
                    and item.family_id == latest_entry.action_family_id
                    and item.status == "exhausted"
                    for item in self.negative_ledger
                    if item.scope_kind == "frontier_episode"
                    and item.scope_id == latest_entry.frontier_episode_id_before
                )
            ):
                reasons.append("exhausted_unresolved_action_without_frontier_gain")
            if (
                latest_entry.action_type == "cleanroom_llm"
                and not latest_entry.after_outcome.get("tb_and_synth", False)
                and self._best_rank >= current_rank
            ):
                reasons.append("failed_cleanroom_did_not_beat_archive_best")
        if reasons:
            return best_record.candidate_id, reasons[0]
        return None, None

    def mark_recovery(self) -> None:
        self.recovery_count += 1

    def ercl_veto_rule_ids(self, stage: str) -> list[str]:
        """Project active, trajectory-scoped action/root evidence to ERCL contract vetoes."""
        prefix = f"{stage.upper()}_"
        return sorted(
            {
                rule_id
                for item in self.active_ledgers_for()
                for rule_id in item.rule_ids
                if rule_id.startswith(prefix)
            }
        )

    def ercl_veto_operator_ids(self) -> list[str]:
        return sorted(
            {
                operator_id
                for item in self.active_ledgers_for()
                for operator_id in item.operator_ids
            }
        )

    def to_match_history(self, stage: str) -> str:
        relevant = [entry for entry in self.entries if entry.failure_stage == stage]
        consecutive = []
        for entry in reversed(self.entries):
            if entry.failure_stage != stage:
                break
            consecutive.append(entry)
        current = consecutive[0] if consecutive else None
        payload = {
            "schema_version": RFL_SCHEMA_VERSION,
            "stage": stage,
            # These fields prove authorization by a real trajectory-local
            # observation rather than by a mode name. In Round 1, current is
            # the immutable Round 0 candidate.
            "current_round_index": current.round_index if current else None,
            "current_action_type": current.action_type if current else None,
            "stage_attempt_count": len(relevant),
            "current_failure_signature": (
                current.failure_signature_after if current else None
            ),
            "repeated_signature_count": (
                current.repeated_signature_count if current else 0
            ),
            "no_progress_streak": sum(
                int(entry.hypothesis_status != "supported") for entry in consecutive
            ),
            "same_action_count": current.same_action_count if current else 0,
            "current_mechanism_id": current.mechanism_id if current else None,
            "frontier_no_improvement_streak": self.frontier_no_improvement_streak,
            "current_frontier_improved": self.current_frontier_improved,
            "current_compile_subfrontier_progress": (
                self.current_compile_subfrontier_progress
            ),
            "compile_unresolved_symbol_union": sorted(
                self.compile_unresolved_symbol_union
            )[:24],
            "current_action_failed": (
                current.hypothesis_status == "contradicted" if current else False
            ),
            "current_duplicate_code": self.current_duplicate_code,
            "current_two_cycle": self.current_two_cycle,
            "recovery_count": self.recovery_count,
            "cleanroom_already_used": self.cleanroom_used,
            "ercl_veto_rule_ids": self.ercl_veto_rule_ids(stage),
            "rfl_controller": {
                "active_source_failure_basin_id": self.active_source_failure_basin_id,
                "current_frontier_episode_id": self.current_frontier_episode_id,
                "active_action_family_ids": sorted(
                    item.family_id
                    for item in self.active_ledgers_for()
                    if item.family_kind == "action_family"
                ),
                "active_root_family_ids": sorted(
                    item.family_id
                    for item in self.active_ledgers_for()
                    if item.family_kind == "root_family"
                ),
                "active_rule_ids": self.ercl_veto_rule_ids(stage),
                "active_operator_ids": self.ercl_veto_operator_ids(),
            },
        }
        return json.dumps(payload, sort_keys=True)

    def apply_novelty_gate(
        self,
        diagnosis: Diagnosis,
        *,
        stage: str,
        action_type: str,
        operator_id: str | None,
        rule_ids: list[str],
    ) -> tuple[Diagnosis, dict[str, Any]]:
        action_family = action_family_id(
            stage=stage,
            action_type=action_type,
            operator_id=operator_id,
            canonical_action_intent_id=diagnosis.canonical_action_intent_id,
            canonical_action_target_id=diagnosis.canonical_action_target_id,
            rule_ids=rule_ids,
        )
        root_family = root_family_id(
            stage,
            diagnosis.canonical_root_intent_id,
            diagnosis.canonical_action_target_id,
        )
        active = self.active_ledgers_for()
        hits = [
            item
            for item in active
            if (
                item.family_kind == "action_family"
                and item.family_id == action_family
            )
            or (
                item.family_kind == "root_family"
                and item.family_id == root_family
            )
        ]
        telemetry = {
            "applied": bool(hits),
            "action_family_id": action_family,
            "root_family_id": root_family,
            "active_hit_ids": [item.family_id for item in hits],
            "llm_retry_added": False,
        }
        if not hits:
            return diagnosis, telemetry
        effective = Diagnosis(
            failure_stage=diagnosis.failure_stage,
            evidence=list(diagnosis.evidence),
            root_cause=(
                "The prior structured hypothesis is inactive in this failure "
                "scope; derive one new stage-specific action from current tool evidence."
            ),
            allowed_actions=[
                f"Use one new {stage}-specific action family not listed in the active veto set"
            ],
            forbidden_actions=list(
                dict.fromkeys(
                    diagnosis.forbidden_actions
                    + [f"Do not repeat trajectory-local veto family {_opaque_family(item.family_id)}" for item in hits]
                )
            ),
            must_hold_invariants=list(diagnosis.must_hold_invariants),
            rule_ids=[],
            fallback=True,
            predicted_observation=diagnosis.predicted_observation,
            canonical_root_intent_id="stage_unknown",
            canonical_action_intent_id="stage_generic_other",
            canonical_action_target_id=None,
            action_veto_eligible=False,
            root_veto_eligible=False,
            validation_notes=list(diagnosis.validation_notes)
            + ["novelty_gate_replaced_active_vetoed_hypothesis"],
        )
        return effective, telemetry

    def build_model_input_contract(
        self,
        *,
        role: str,
        compact_level: int,
        budget_chars: int,
        chosen_source_stage: str,
        effective_diagnosis: Diagnosis | None = None,
    ):
        from .model_input_contract import build_model_input

        return build_model_input(
            memory=self,
            role=role,
            compact_level=compact_level,
            budget_chars=budget_chars,
            chosen_source_stage=chosen_source_stage,
            effective_diagnosis=effective_diagnosis,
        )

    def _prompt_candidate_label(self, candidate_id: str | None) -> str | None:
        if candidate_id is None:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return f"r{candidate.round_index}"
        return "unknown"

    def to_prompt(self) -> str:
        """Compressed model-visible state with mode/server identities removed."""
        # Round 1 must remain a paired Feedback control: Round 0 contains no
        # repair plan/edit/result evidence yet. Inject RFL only after
        # one repair has produced an outcome that supports credit assignment.
        if not any(entry.round_index > 0 for entry in self.entries):
            return ""

        attempts = []
        for entry in self.entries:
            attempts.append(
                {
                    "round": entry.round_index,
                    "candidate": f"r{entry.round_index}",
                    "parent": self._prompt_candidate_label(entry.parent_candidate_id),
                    "base_origin": entry.base_origin,
                    "candidate_code_sha256": entry.candidate_code_sha256,
                    "failure_signature": entry.failure_signature_after,
                    "failure_summary": entry.normalized_failure_after[:600],
                    "failure_stage": entry.failure_stage,
                    "root_cause": entry.root_cause[:500],
                    "hypothesis_status": entry.hypothesis_status,
                    "rule_ids": entry.rule_ids,
                    "action_type": entry.action_type,
                    "operator_id": entry.operator_id,
                    "action_fingerprint": entry.action_fingerprint,
                    "mechanism_id": entry.mechanism_id,
                    "diagnostic_atoms": entry.diagnostic_atoms,
                    "predicted_observation": entry.predicted_observation,
                    "observed_delta": entry.observed_delta,
                    "compile_frontier": {
                        "normalized_error_count": entry.compile_frontier_after.get(
                            "normalized_error_count", 0
                        ),
                        "unresolved_symbols": (
                            _bounded_prompt_strings(
                                entry.compile_frontier_after.get(
                                    "unresolved_symbols", []
                                ),
                                count=8,
                                width=80,
                            )
                        ),
                        "evidence": _bounded_prompt_strings(
                            entry.compile_frontier_after.get(
                                "evidence", []
                            ),
                            count=2,
                            width=160,
                        ),
                        "timed_out": entry.compile_frontier_after.get(
                            "timed_out", False
                        ),
                    },
                    "compile_subfrontier_progress": (
                        entry.compile_subfrontier_progress
                    ),
                    "new_unresolved_symbols": (
                        _bounded_prompt_strings(
                            entry.new_unresolved_symbols, count=8, width=80
                        )
                    ),
                    "resolved_unresolved_symbols": (
                        _bounded_prompt_strings(
                            entry.resolved_unresolved_symbols,
                            count=8,
                            width=80,
                        )
                    ),
                    "hypothesis_credit_basis": (
                        entry.hypothesis_credit_basis[:6]
                    ),
                    "format_recovery_kind": entry.format_recovery_kind,
                    "changed_span_sha256": entry.changed_span_sha256,
                    "same_action_count": entry.same_action_count,
                    "action_failed": entry.hypothesis_status == "contradicted",
                    "duplicate_code_count": entry.duplicate_code_count,
                    "cleanroom_used": entry.cleanroom_used,
                    "invariant_violations": entry.invariant_violations[:8],
                    "before": entry.before_outcome,
                    "after": entry.after_outcome,
                    "stage_regression": entry.stage_regression,
                    "component_regression": entry.component_regression,
                    "repeated_signature_count": entry.repeated_signature_count,
                    "no_progress_streak": entry.no_progress_streak,
                    "frontier_no_improvement_streak": entry.frontier_no_improvement_streak,
                    "code_change_summary": entry.code_change_summary,
                    "round_total_tokens": entry.total_tokens,
                    "tool_seconds": entry.tool_seconds,
                }
            )
        current = attempts[-1] if attempts else None
        contradicted_root_causes = list(
            dict.fromkeys(
                entry.root_cause[:500]
                for entry in reversed(self.entries)
                if entry.hypothesis_status == "contradicted"
            )
        )[:3]
        failed_action_fingerprints = list(
            dict.fromkeys(
                entry.action_fingerprint
                for entry in reversed(self.entries)
                if entry.action_type != "round0"
                and entry.hypothesis_status == "contradicted"
                and entry.action_fingerprint
            )
        )[:5]
        failed_mechanism_ids = [
            mechanism for mechanism, _ in self.failed_mechanism_counts.most_common(5)
        ]
        prior_invariant_violations = list(
            dict.fromkeys(
                violation
                for entry in reversed(self.entries)
                for violation in entry.invariant_violations
            )
        )[:8]
        requires_novel_mechanism = bool(
            self.current_duplicate_code
            or self.current_two_cycle
            or self.current_diagnosis_conflict
            or self.no_progress_streak >= 2
            or (
                self.current_signature
                and self.signature_counts[self.current_signature] >= 2
            )
            or self.frontier_no_improvement_streak >= 2
        )
        payload = {
            "schema_version": RFL_SCHEMA_VERSION,
            "scope": "current trajectory only; never shared across tasks or samples",
            "decision_policy": {
                "avoid_identical_code_or_action": True,
                "reject_contradicted_hypotheses": True,
                "preserve_best_stage_frontier": True,
                "cleanroom_threshold": (
                    "same signature plus the same failed action/mechanism "
                    "without frontier gain, duplicate/A-B-A code, or "
                    "repeated-signature diagnosis conflict"
                ),
                "cleanroom_budget": "once, only for next repair round 3-5",
            },
            "instruction": (
                "Use plan-edit-result history for credit assignment. Do not repeat a "
                "contradicted hypothesis or identical failed action. Preserve the best "
                "stage frontier and every immutable interface invariant."
            ),
            "current": current,
            "next_action_constraints": {
                "requires_new_root_cause_or_action_family": requires_novel_mechanism,
                "do_not_repeat_action_fingerprints": failed_action_fingerprints,
                "do_not_repeat_mechanism_ids": failed_mechanism_ids,
                "contradicted_root_causes": contradicted_root_causes,
                "must_not_reintroduce_invariant_violations": prior_invariant_violations,
                "must_improve_or_preserve_frontier": self.best_outcome,
                "if_cleanroom": (
                    "rederive from immutable task; do not patch or imitate failed code"
                ),
            },
            "best": {
                "candidate": self._prompt_candidate_label(self.best_candidate_id),
                "outcome": self.best_outcome,
                "compile_subfrontier_score": list(
                    self._best_subfrontier_score
                ),
            },
            "stagnation": {
                "no_progress_streak": self.no_progress_streak,
                "frontier_no_improvement_streak": self.frontier_no_improvement_streak,
                "current_duplicate_code": self.current_duplicate_code,
                "current_two_cycle": self.current_two_cycle,
                "current_diagnosis_conflict": self.current_diagnosis_conflict,
                "current_action_repeat_count": self.action_fingerprint_counts.get(
                    self.current_action_fingerprint or "", 0
                ),
                "current_action_failed": self.current_action_failed,
                "current_frontier_improved": self.current_frontier_improved,
                "current_compile_subfrontier_progress": (
                    self.current_compile_subfrontier_progress
                ),
                "compile_unresolved_symbol_union": sorted(
                    self.compile_unresolved_symbol_union
                )[:24],
                "recovery_count": self.recovery_count,
                "cleanroom_already_used": self.cleanroom_used,
                "cleanroom_trigger_reason": self.cleanroom_trigger_reason,
                "cleanroom_trigger_round": self.cleanroom_trigger_round,
                "cleanroom_trigger_candidate": self._prompt_candidate_label(
                    self.cleanroom_trigger_candidate_id
                ),
                "cleanroom_trigger_signature": self.cleanroom_trigger_signature,
                "cleanroom_trigger_stage": self.cleanroom_trigger_stage,
            },
            "failed_strategies": self.failed_strategy_counts.most_common(5),
            "hypothesis_status_counts": dict(self.hypothesis_status_counts),
            "diagnosis_conflict_count": self.diagnosis_conflict_count,
            "root_cause_counts": self.root_cause_counts.most_common(5),
            "must_preserve_frontier": self.best_outcome,
            "attempts": attempts,
        }
        return json.dumps(payload, indent=2)

    def audit_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": RFL_SCHEMA_VERSION,
            "trajectory_key": asdict(self.trajectory_key),
            "trajectory_key_sha256": self.trajectory_key_sha256,
            "trajectory_key_explicit": self._explicit_trajectory_key,
            "max_entries": self.max_entries,
            "entry_count": len(self.entries),
            "entries": [asdict(entry) for entry in self.entries],
            "candidate_archive": [asdict(item) for item in self.candidates],
            "signature_counts": dict(self.signature_counts),
            "no_progress_streak": self.no_progress_streak,
            "cleanroom_used": self.cleanroom_used,
            "cleanroom_trigger_reason": self.cleanroom_trigger_reason,
            "cleanroom_trigger_round": self.cleanroom_trigger_round,
            "cleanroom_trigger_candidate_id": self.cleanroom_trigger_candidate_id,
            "cleanroom_trigger_signature": self.cleanroom_trigger_signature,
            "cleanroom_trigger_stage": self.cleanroom_trigger_stage,
            "cleanroom_trigger_outcome": self.cleanroom_trigger_outcome,
            "best_candidate_id": self.best_candidate_id,
            "best_round_index": self.best_round_index,
            "best_outcome": self.best_outcome,
            "recovery_count": self.recovery_count,
            "diagnosis_conflict_count": self.diagnosis_conflict_count,
            "current_diagnosis_conflict": self.current_diagnosis_conflict,
            "current_duplicate_code": self.current_duplicate_code,
            "current_two_cycle": self.current_two_cycle,
            "code_sha_counts": dict(self.code_sha_counts),
            "signature_history": self.signature_history,
            "code_sha_history": self.code_sha_history,
            "signature_root_causes": {
                signature: dict(roots)
                for signature, roots in self.signature_root_causes.items()
            },
            "failed_strategy_counts": dict(self.failed_strategy_counts),
            "action_fingerprint_counts": dict(self.action_fingerprint_counts),
            "mechanism_counts": dict(self.mechanism_counts),
            "failed_mechanism_counts": dict(self.failed_mechanism_counts),
            "format_recovery_counts": dict(self.format_recovery_counts),
            "current_mechanism_id": self.current_mechanism_id,
            "frontier_no_improvement_streak": self.frontier_no_improvement_streak,
            "current_frontier_improved": self.current_frontier_improved,
            "current_compile_subfrontier_progress": (
                self.current_compile_subfrontier_progress
            ),
            "compile_unresolved_symbol_union": sorted(
                self.compile_unresolved_symbol_union
            ),
            "best_compile_subfrontier_score": list(
                self._best_subfrontier_score
            ),
            "current_action_fingerprint": self.current_action_fingerprint,
            "current_action_failed": self.current_action_failed,
            "hypothesis_status_counts": dict(self.hypothesis_status_counts),
            "root_status_counts": dict(
                Counter(
                    entry.root_status
                    for entry in self.entries
                    if entry.round_index > 0
                )
            ),
            "action_status_counts": dict(
                Counter(
                    entry.action_status
                    for entry in self.entries
                    if entry.round_index > 0
                )
            ),
            "negative_ledger_status_counts": dict(
                Counter(item.status for item in self.negative_ledger)
            ),
            "negative_ledger_family_kind_counts": dict(
                Counter(item.family_kind for item in self.negative_ledger)
            ),
            "root_cause_counts": dict(self.root_cause_counts),
            "current_frontier_episode_id": self.current_frontier_episode_id,
            "active_source_candidate_id": self.active_source_candidate_id,
            "active_source_failure_basin_id": self.active_source_failure_basin_id,
            "negative_ledger": [asdict(item) for item in self.negative_ledger],
            "active_negative_ledger": [
                asdict(item) for item in self.active_ledgers_for()
            ],
        }
        # Round payloads live until terminal serialization. Detach all nested
        # state so later repairs cannot leak future information into old rounds.
        return deepcopy(payload)
