from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


_FILENAME_BOOL_FIELDS = (
    "parent_filename_available",
    "changed_from_parent",
    "candidate_matches_inferred",
    "inherited_safe_alias",
    "credit_blocking",
)


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def filename_telemetry_errors(
    round_payload: dict[str, Any],
    parent_round: dict[str, Any] | None,
) -> list[str]:
    """Validate the parent-to-child filename credit boundary."""

    errors: list[str] = []
    review = round_payload.get("review")
    telemetry = round_payload.get("filename_telemetry")
    if not isinstance(review, dict):
        return ["review_missing"]
    if not isinstance(telemetry, dict):
        return ["filename_telemetry_missing"]

    required = {
        "inferred_expected_filename",
        "parent_generated_filename",
        "candidate_generated_filename",
        *_FILENAME_BOOL_FIELDS,
    }
    missing = sorted(required - set(telemetry))
    if missing:
        errors.append("filename_telemetry_missing_fields:" + ",".join(missing))

    candidate_parse = review.get("pass_parse")
    if type(candidate_parse) is not bool:
        errors.append("review_pass_parse_not_bool")
    candidate_filename = (
        review.get("generated_filename") if candidate_parse is True else None
    )
    if candidate_filename is not None and (
        not isinstance(candidate_filename, str) or not candidate_filename
    ):
        errors.append("candidate_generated_filename_invalid")

    parent_review = (
        parent_round.get("review") if isinstance(parent_round, dict) else None
    )
    parent_parse = (
        parent_review.get("pass_parse")
        if isinstance(parent_review, dict)
        else False
    )
    if parent_round is not None and type(parent_parse) is not bool:
        errors.append("parent_review_pass_parse_not_bool")
    parent_filename = (
        parent_review.get("generated_filename")
        if isinstance(parent_review, dict) and parent_parse is True
        else None
    )
    if parent_filename is not None and (
        not isinstance(parent_filename, str) or not parent_filename
    ):
        errors.append("parent_generated_filename_invalid")

    expected_filename = telemetry.get("inferred_expected_filename")
    if not isinstance(expected_filename, str) or not expected_filename:
        errors.append("inferred_expected_filename_invalid")

    changed = bool(
        parent_filename
        and candidate_filename
        and candidate_filename != parent_filename
    )
    derived = {
        "parent_generated_filename": parent_filename,
        "candidate_generated_filename": candidate_filename,
        "parent_filename_available": parent_filename is not None,
        "changed_from_parent": changed,
        "candidate_matches_inferred": bool(
            candidate_filename
            and candidate_filename == expected_filename
        ),
        "inherited_safe_alias": bool(
            parent_filename
            and candidate_filename == parent_filename
            and candidate_filename != expected_filename
        ),
        "credit_blocking": changed,
    }
    for field in _FILENAME_BOOL_FIELDS:
        if type(telemetry.get(field)) is not bool:
            errors.append(f"filename_telemetry_{field}_not_bool")
    for field, expected in derived.items():
        if telemetry.get(field) != expected:
            errors.append(f"filename_telemetry_{field}_mismatch")

    violations = round_payload.get("invariant_violations")
    if not isinstance(violations, list) or not all(
        isinstance(value, str) for value in violations
    ):
        errors.append("invariant_violations_invalid")
    else:
        actual_filename_violations = [
            value
            for value in violations
            if value.startswith("generated_filename_changed:")
        ]
        expected_filename_violations = (
            [
                "generated_filename_changed:"
                f"{candidate_filename}!={parent_filename}"
            ]
            if changed
            else []
        )
        if actual_filename_violations != expected_filename_violations:
            errors.append("filename_invariant_mismatch")
    return errors


def short_snapshot_errors(
    snapshot: Any,
    *,
    round_index: int,
    expected_schema: str,
) -> list[str]:
    """Reject malformed or future-contaminated trajectory-local snapshots."""

    errors: list[str] = []
    if not isinstance(snapshot, dict):
        return ["short_snapshot_missing"]
    if snapshot.get("schema_version") != expected_schema:
        errors.append("short_schema_mismatch")

    trajectory_key = snapshot.get("trajectory_key")
    trajectory_key_sha256 = snapshot.get("trajectory_key_sha256")
    if not isinstance(trajectory_key, dict):
        errors.append("short_trajectory_key_missing")
        trajectory_key_sha256 = None
    else:
        required_key_fields = {
            "benchmark_case_name": str,
            "sample_index": int,
            "seed": int,
            "mode": str,
        }
        for field, expected_type in required_key_fields.items():
            value = trajectory_key.get(field)
            if expected_type is int:
                valid = _exact_int(value)
            else:
                valid = isinstance(value, expected_type) and bool(value)
            if not valid:
                errors.append(f"short_trajectory_key_{field}_invalid")
        expected_key_sha = hashlib.sha256(
            json.dumps(
                trajectory_key,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if trajectory_key_sha256 != expected_key_sha:
            errors.append("short_trajectory_key_hash_mismatch")
    if snapshot.get("trajectory_key_explicit") is not True:
        errors.append("short_trajectory_key_not_explicit")

    max_entries = snapshot.get("max_entries")
    if not _exact_int(max_entries) or not 1 <= max_entries <= 5:
        errors.append("short_max_entries_invalid")
        max_entries = 5

    list_fields = (
        "entries",
        "candidate_archive",
        "signature_history",
        "code_sha_history",
    )
    values: dict[str, list[Any]] = {}
    for field in list_fields:
        value = snapshot.get(field)
        if not isinstance(value, list):
            errors.append(f"short_{field}_not_list")
            values[field] = []
        else:
            values[field] = value

    entries = values["entries"]
    archive = values["candidate_archive"]
    signatures = values["signature_history"]
    code_history = values["code_sha_history"]

    if not _exact_int(snapshot.get("entry_count")) or snapshot.get(
        "entry_count"
    ) != len(entries):
        errors.append("short_entry_count_mismatch")

    expected_archive_indices = list(range(round_index + 1))
    archive_indices: list[int] = []
    archive_ids: list[str] = []
    archive_by_id: dict[str, dict[str, Any]] = {}
    for value in archive:
        if not isinstance(value, dict):
            errors.append("short_candidate_record_invalid")
            continue
        index = value.get("round_index")
        candidate_id = value.get("candidate_id")
        if value.get("trajectory_key_sha256") != trajectory_key_sha256:
            errors.append("short_candidate_trajectory_key_mismatch")
        if not _exact_int(index):
            errors.append("short_candidate_round_invalid")
        else:
            archive_indices.append(index)
            if index < 0 or index > round_index:
                errors.append("short_candidate_from_future")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append("short_candidate_id_invalid")
        else:
            archive_ids.append(candidate_id)
            archive_by_id[candidate_id] = value
    if archive_indices != expected_archive_indices:
        errors.append("short_candidate_archive_rounds_mismatch")
    if len(archive_ids) != len(set(archive_ids)):
        errors.append("short_candidate_ids_not_unique")

    expected_entry_indices = list(
        range(max(0, round_index + 1 - int(max_entries)), round_index + 1)
    )
    entry_indices: list[int] = []
    for value in entries:
        if not isinstance(value, dict):
            errors.append("short_entry_invalid")
            continue
        index = value.get("round_index")
        if value.get("trajectory_key_sha256") != trajectory_key_sha256:
            errors.append("short_entry_trajectory_key_mismatch")
        if not _exact_int(index):
            errors.append("short_entry_round_invalid")
        else:
            entry_indices.append(index)
            if index < 0 or index > round_index:
                errors.append("short_entry_from_future")
        candidate_id = value.get("candidate_id")
        if candidate_id not in archive_by_id:
            errors.append("short_entry_candidate_unknown")
    if entry_indices != expected_entry_indices:
        errors.append("short_entry_rounds_mismatch")

    expected_history_length = round_index + 1
    if len(signatures) != expected_history_length:
        errors.append("short_signature_history_length_mismatch")
    if len(code_history) != expected_history_length:
        errors.append("short_code_history_length_mismatch")
    if code_history and len(archive) == len(code_history):
        for position, (code_sha, candidate) in enumerate(
            zip(code_history, archive)
        ):
            if (
                not isinstance(candidate, dict)
                or candidate.get("code_sha256") != code_sha
            ):
                errors.append(
                    f"short_code_history_archive_mismatch:{position}"
                )

    expected_signature_counts = dict(
        Counter(value for value in signatures if value is not None)
    )
    if snapshot.get("signature_counts") != expected_signature_counts:
        errors.append("short_signature_counts_mismatch")
    expected_code_counts = dict(Counter(code_history))
    if snapshot.get("code_sha_counts") != expected_code_counts:
        errors.append("short_code_counts_mismatch")

    best_round = snapshot.get("best_round_index")
    best_candidate = snapshot.get("best_candidate_id")
    if not _exact_int(best_round) or not 0 <= best_round <= round_index:
        errors.append("short_best_round_invalid")
    elif best_candidate not in archive_by_id:
        errors.append("short_best_candidate_unknown")
    else:
        record = archive_by_id[best_candidate]
        if record.get("round_index") != best_round:
            errors.append("short_best_round_candidate_mismatch")
        if snapshot.get("best_outcome") != record.get("outcome"):
            errors.append("short_best_outcome_mismatch")

    ledger = snapshot.get("negative_ledger", [])
    if not isinstance(ledger, list):
        errors.append("short_negative_ledger_not_list")
    else:
        for item in ledger:
            if not isinstance(item, dict):
                errors.append("short_negative_ledger_item_invalid")
                continue
            if item.get("trajectory_key_sha256") != trajectory_key_sha256:
                errors.append("short_negative_ledger_trajectory_key_mismatch")
            if item.get("scope_kind") not in {"source_failure_basin", "frontier_episode"}:
                errors.append("short_negative_ledger_scope_invalid")
            if item.get("family_kind") not in {"action_family", "root_family"}:
                errors.append("short_negative_ledger_family_invalid")
            if item.get("status") not in {"vetoed", "exhausted"}:
                errors.append("short_negative_ledger_status_invalid")

    cleanroom_round = snapshot.get("cleanroom_trigger_round")
    if cleanroom_round is not None and (
        not _exact_int(cleanroom_round)
        or cleanroom_round < 0
        or cleanroom_round > round_index
    ):
        errors.append("short_cleanroom_trigger_from_future")
    return errors


def independent_round_snapshot_errors(
    trajectory_path: Path,
    round_payload: dict[str, Any],
) -> list[str]:
    """Require the independently persisted round to equal the embedded round."""

    index = round_payload.get("round_index")
    if not _exact_int(index) or index < 0:
        return ["independent_round_index_invalid"]
    snapshot_path = (
        trajectory_path.parent / f"round__{index}" / "round.json"
    )
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"independent_round_snapshot_missing:{index}"]
    except (OSError, json.JSONDecodeError):
        return [f"independent_round_snapshot_invalid:{index}"]
    if snapshot != round_payload:
        return [f"independent_round_snapshot_mismatch:{index}"]
    return []


def completed_trajectory_contract_errors(
    trajectory_path: Path,
    payload: Any,
    *,
    expected_config_hash: str | None = None,
    expected_protocol_digest: str | None = None,
    expected_mode: str | None = None,
    expected_case_name: str | None = None,
    expected_sample_index: int | None = None,
    expected_seed: int | None = None,
    expected_rfl_schema: str | None = None,
    require_filename_telemetry: bool = True,
    require_round_files: bool = True,
) -> list[str]:
    """Validate every field that makes a terminal trajectory safe to resume."""

    if not isinstance(payload, dict):
        return ["trajectory_not_object"]
    errors: list[str] = []
    if payload.get("status") != "complete":
        errors.append("trajectory_status_not_complete")
    if payload.get("terminal") is not True:
        errors.append("trajectory_not_terminal")
    if (
        expected_config_hash is not None
        and payload.get("config_hash") != expected_config_hash
    ):
        errors.append("config_hash_mismatch")
    if (
        expected_protocol_digest is not None
        and payload.get("protocol_digest") != expected_protocol_digest
    ):
        errors.append("protocol_digest_mismatch")

    expectations = {
        "mode": expected_mode,
        "benchmark_case_name": expected_case_name,
        "sample_index": expected_sample_index,
        "seed": expected_seed,
    }
    for field, expected in expectations.items():
        if expected is not None and payload.get(field) != expected:
            errors.append(f"trajectory_{field}_mismatch")

    expected_trajectory_key_sha256: str | None = None
    if expected_rfl_schema is not None:
        if any(
            value is None
            for value in (
                expected_mode,
                expected_case_name,
                expected_sample_index,
                expected_seed,
            )
        ):
            errors.append("trajectory_key_expectation_incomplete")
        else:
            expected_trajectory_key = {
                "benchmark_case_name": expected_case_name,
                "sample_index": expected_sample_index,
                "seed": expected_seed,
                "mode": expected_mode,
            }
            expected_trajectory_key_sha256 = hashlib.sha256(
                json.dumps(
                    expected_trajectory_key,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if payload.get("trajectory_key") != expected_trajectory_key:
                errors.append("trajectory_key_mismatch")
            if payload.get("trajectory_key_sha256") != expected_trajectory_key_sha256:
                errors.append("trajectory_key_hash_mismatch")

    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return errors + ["trajectory_rounds_invalid"]
    if not all(isinstance(value, dict) for value in rounds):
        return errors + ["trajectory_round_not_object"]

    indices = [value.get("round_index") for value in rounds]
    if (
        not all(_exact_int(value) for value in indices)
        or indices != list(range(len(rounds)))
    ):
        errors.append("trajectory_round_indices_invalid")

    candidate_ids = [value.get("candidate_id") for value in rounds]
    if not all(isinstance(value, str) and value for value in candidate_ids):
        errors.append("trajectory_candidate_id_invalid")
    elif len(candidate_ids) != len(set(candidate_ids)):
        errors.append("trajectory_candidate_ids_not_unique")

    seen: set[str] = set()
    round_by_candidate: dict[str, dict[str, Any]] = {}
    for position, round_payload in enumerate(rounds):
        parent = round_payload.get("parent_candidate_id")
        if position == 0:
            if parent is not None:
                errors.append("round0_parent_present")
        elif parent not in seen:
            errors.append(f"round_parent_not_earlier:{position}")
        candidate_id = round_payload.get("candidate_id")
        if isinstance(candidate_id, str):
            seen.add(candidate_id)
            round_by_candidate[candidate_id] = round_payload

        if expected_rfl_schema is not None:
            if round_payload.get("trajectory_key_sha256") != expected_trajectory_key_sha256:
                errors.append(f"round_trajectory_key_mismatch:{position}")
        elif round_payload.get("trajectory_key_sha256") is not None:
            errors.append(f"round_trajectory_key_leaked:{position}")

        if (
            expected_protocol_digest is not None
            and round_payload.get("protocol_digest")
            != expected_protocol_digest
        ):
            errors.append(f"round_protocol_digest_mismatch:{position}")

        parent_round = (
            round_by_candidate.get(parent) if position else None
        )
        if require_filename_telemetry:
            errors.extend(
                f"round_{position}:{value}"
                for value in filename_telemetry_errors(
                    round_payload, parent_round
                )
            )

        snapshot = round_payload.get("rfl_state")
        if expected_rfl_schema is None:
            if snapshot is not None:
                errors.append(f"rfl_leaked:{position}")
        elif position == 0:
            if snapshot is not None:
                errors.append("round0_short_snapshot_present")
        else:
            errors.extend(
                f"round_{position}:{value}"
                for value in short_snapshot_errors(
                    snapshot,
                    round_index=position,
                    expected_schema=expected_rfl_schema,
                )
            )

        if require_round_files:
            errors.extend(
                independent_round_snapshot_errors(
                    trajectory_path, round_payload
                )
            )

    selected_index = payload.get("selected_round_index")
    if not _exact_int(selected_index) or not 0 <= selected_index < len(rounds):
        errors.append("selected_round_index_invalid")
        selected_round: dict[str, Any] | None = None
    else:
        selected_round = rounds[selected_index]
        if (
            payload.get("selected_candidate_id")
            != selected_round.get("candidate_id")
        ):
            errors.append("selected_candidate_id_mismatch")
        if payload.get("final_review") != selected_round.get("review"):
            errors.append("final_review_mismatch")
    if payload.get("last_candidate_id") != rounds[-1].get("candidate_id"):
        errors.append("last_candidate_id_mismatch")

    final_short = payload.get("rfl_state_final")
    last_index = len(rounds) - 1
    if expected_rfl_schema is None:
        if final_short is not None:
            errors.append("rfl_final_leaked")
    else:
        errors.extend(
            f"rfl_final:{value}"
            for value in short_snapshot_errors(
                final_short,
                round_index=last_index,
                expected_schema=expected_rfl_schema,
            )
        )
        if last_index > 0 and final_short != rounds[-1].get("rfl_state"):
            errors.append("rfl_final_snapshot_mismatch")
    return errors

