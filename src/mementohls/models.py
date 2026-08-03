from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEEDBACK = "feedback"
    RFL = "rfl"
    ERCL = "ercl"
    MEMENTOHLS = "mementohls"
    MEMENTOHLS_NO_EXECUTOR = "mementohls_no_executor"
    MEMENTOHLS_NO_VETO = "mementohls_no_veto"
    MEMENTOHLS_NO_CLEANROOM = "mementohls_no_cleanroom"
    MEMENTOHLS_CORE_RULES = "mementohls_core_rules"
    MEMENTOHLS_ALL_RULES = "mementohls_all_rules"

    @property
    def uses_rfl(self) -> bool:
        return self in {
            RunMode.RFL,
            RunMode.MEMENTOHLS,
            RunMode.MEMENTOHLS_NO_EXECUTOR,
            RunMode.MEMENTOHLS_NO_VETO,
            RunMode.MEMENTOHLS_NO_CLEANROOM,
            RunMode.MEMENTOHLS_CORE_RULES,
            RunMode.MEMENTOHLS_ALL_RULES,
        }

    @property
    def uses_ercl(self) -> bool:
        return self in {
            RunMode.ERCL,
            RunMode.MEMENTOHLS,
            RunMode.MEMENTOHLS_NO_EXECUTOR,
            RunMode.MEMENTOHLS_NO_VETO,
            RunMode.MEMENTOHLS_NO_CLEANROOM,
            RunMode.MEMENTOHLS_CORE_RULES,
            RunMode.MEMENTOHLS_ALL_RULES,
        }

    @property
    def uses_cleanroom(self) -> bool:
        return self in {
            RunMode.RFL,
            RunMode.MEMENTOHLS,
            RunMode.MEMENTOHLS_NO_EXECUTOR,
            RunMode.MEMENTOHLS_NO_VETO,
            RunMode.MEMENTOHLS_CORE_RULES,
            RunMode.MEMENTOHLS_ALL_RULES,
        }

    @property
    def uses_candidate_archive(self) -> bool:
        return self.uses_rfl

    @property
    def uses_veto(self) -> bool:
        return self.uses_rfl and self != RunMode.MEMENTOHLS_NO_VETO

    @property
    def uses_format_recovery(self) -> bool:
        return self.uses_action_executor

    @property
    def uses_action_executor(self) -> bool:
        return self in {
            RunMode.ERCL,
            RunMode.MEMENTOHLS,
            RunMode.MEMENTOHLS_NO_VETO,
            RunMode.MEMENTOHLS_NO_CLEANROOM,
            RunMode.MEMENTOHLS_CORE_RULES,
            RunMode.MEMENTOHLS_ALL_RULES,
        }

    @property
    def uses_core_rules(self) -> bool:
        return self == RunMode.MEMENTOHLS_CORE_RULES

    @property
    def uses_all_rules(self) -> bool:
        return self == RunMode.MEMENTOHLS_ALL_RULES

    @property
    def allows_repair(self) -> bool:
        return self != RunMode.ZERO_SHOT

    @property
    def feature_flags(self) -> dict[str, bool]:
        return {
            "feedback": self.allows_repair,
            "rfl": self.uses_rfl,
            "veto": self.uses_veto,
            "cleanroom": self.uses_cleanroom,
            "candidate_archive": self.uses_candidate_archive,
            "format_recovery": self.uses_format_recovery,
            "ercl": self.uses_ercl,
            "action_executor": self.uses_action_executor,
            "core_rules": self.uses_core_rules,
            "all_rules": self.uses_all_rules,
        }

    @property
    def formal_enabled(self) -> bool:
        return self in FORMAL_RUN_MODES


FORMAL_RUN_MODES = (
    RunMode.ZERO_SHOT,
    RunMode.FEEDBACK,
    RunMode.RFL,
    RunMode.ERCL,
    RunMode.MEMENTOHLS,
    RunMode.MEMENTOHLS_CORE_RULES,
    RunMode.MEMENTOHLS_ALL_RULES,
)

HLS_EVAL_VERIFICATION_MODES = (
    RunMode.ZERO_SHOT,
    RunMode.FEEDBACK,
    RunMode.RFL,
    RunMode.ERCL,
    RunMode.MEMENTOHLS,
)

RETIRED_ABLATION_MODES = (
    RunMode.MEMENTOHLS_NO_EXECUTOR,
    RunMode.MEMENTOHLS_NO_VETO,
    RunMode.MEMENTOHLS_NO_CLEANROOM,
)


@dataclass(frozen=True)
class RFLPolicyConfig:
    outcome_gated_model_contract: bool
    family_veto: bool
    validated_action_taxonomy: bool
    conservative_archive_recovery: bool
    clean_deadline_round: int
    round5_post_clean_recovery: bool
    max_repair_rounds: int = 5
    one_candidate_per_round: bool = True

    @classmethod
    def for_mode(cls, mode: RunMode) -> "RFLPolicyConfig":
        enabled = mode.uses_rfl
        return cls(
            outcome_gated_model_contract=enabled,
            family_veto=mode.uses_veto,
            validated_action_taxonomy=enabled,
            conservative_archive_recovery=mode.uses_candidate_archive,
            clean_deadline_round=4,
            round5_post_clean_recovery=enabled and mode.uses_cleanroom,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FailureStage(str, Enum):
    PARSE = "parse"
    COMPILE = "compile"
    TB = "tb"
    SYNTH = "synth"
    SUCCESS = "success"


@dataclass
class LLMCallResult:
    text: str
    raw_response: dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    elapsed_seconds: float
    attempts: int
    prompt_variant_index: int = 0
    effective_user_prompt: str | None = None
    request_seed: int | None = None
    request_temperature: float | None = None
    requested_max_tokens: int | None = None
    effective_max_tokens: int | None = None
    finish_reason: str | None = None
    message_roles: list[str] = field(default_factory=list)
    prompt_sha256: str | None = None
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    deployment_epoch: str | None = None
    physical_call: bool = True


@dataclass
class ReviewResult:
    pass_parse: bool = False
    pass_compile: bool = False
    pass_tb: bool = False
    pass_synth: bool = False
    pass_tb_and_synth: bool = False
    failure_stage: str = FailureStage.PARSE.value
    generated_filename: str | None = None
    generated_code: str | None = None
    parse_error: str | None = None
    compile: dict[str, Any] | None = None
    testbench: dict[str, Any] | None = None
    synthesis: dict[str, Any] | None = None
    qor: dict[str, Any] | None = None
    infrastructure_error: str | None = None
    infrastructure_attempts: int = 1
    infrastructure_retry_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnosis:
    failure_stage: str
    evidence: list[str]
    root_cause: str
    allowed_actions: list[str]
    forbidden_actions: list[str]
    must_hold_invariants: list[str]
    rule_ids: list[str]
    fallback: bool = False
    predicted_observation: str = ""
    canonical_root_intent_id: str = "stage_unknown"
    canonical_action_intent_id: str = "stage_generic_other"
    canonical_action_target_id: str | None = None
    action_veto_eligible: bool = False
    root_veto_eligible: bool = False
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _strict_json_value(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _strict_json_dumps(value: Any) -> str:
    return json.dumps(
        _strict_json_value(value),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, _strict_json_dumps(value))


def atomic_create_json(path: Path, value: Any) -> bool:
    """Create immutable JSON exactly once; return False if another writer won."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _strict_json_dumps(value)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError:
            return False
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
