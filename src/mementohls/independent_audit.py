from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import atomic_write_json


METRIC_FLAGS = {
    "parse": "pass_parse",
    "compile": "pass_compile",
    "tb": "pass_tb",
    "synth": "pass_synth",
    "tb_and_synth": "pass_tb_and_synth",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_review(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # The full record comparison also protects candidate identity, logs and timeout state.
    return left == right


def _metric_rows(
    trajectories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, items in trajectories.items():
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_case[str(item["benchmark_case_name"])].append(item)
        for metric, flag in METRIC_FLAGS.items():
            values = [
                float(bool((item.get("final_review") or {}).get(flag)))
                for item in items
            ]
            p1 = sum(values) / len(values) if values else 0.0
            p5 = (
                sum(
                    any(bool((item.get("final_review") or {}).get(flag)) for item in case_items)
                    for case_items in by_case.values()
                )
                / len(by_case)
                if by_case
                else 0.0
            )
            rows.extend([
                {
                    "mode": mode,
                    "metric": f"{metric}_p@1",
                    "value": p1,
                    "case_count": len(by_case),
                    "trajectory_count": len(items),
                },
                {
                    "mode": mode,
                    "metric": f"{metric}_p@5",
                    "value": p5,
                    "case_count": len(by_case),
                    "trajectory_count": len(items),
                },
            ])
    return rows


def audit_run(
    run_root: Path,
    *,
    modes: Iterable[str],
    expected_tasks: int,
    expected_samples: int = 5,
    analysis_path: Path | None = None,
    require_analysis_strict_json: bool = False,
    runtime_deviation_manifests: Iterable[Path] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mode_list = list(modes)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    trajectories: dict[str, list[dict[str, Any]]] = {}
    unique_keys: set[tuple[str, int, int, str]] = set()
    round0: dict[tuple[str, int], dict[str, tuple[Any, Any, Any]]] = defaultdict(dict)
    source_hashes: set[str] = set()
    dataset_hashes: set[str] = set()
    dataset_kinds: set[str] = set()
    protocol_by_mode: dict[str, set[str]] = defaultdict(set)
    config_by_mode: dict[str, set[str]] = defaultdict(set)
    contract_by_mode: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    deployment_epochs: set[str] = set()
    strict_trajectory_json_count = 0
    strict_round_json_count = 0
    rfl_snapshot_count = 0
    reference_provenance_checks = 0
    code_sha_checks = 0
    selected_candidate_checks = 0

    def issue(category: str, detail: str, **context: Any) -> None:
        errors.append({"category": category, "detail": detail, **context})

    deviations: list[dict[str, Any]] = []
    recovery_policy: dict[str, Any] | None = None
    for path in runtime_deviation_manifests:
        try:
            payload = strict_json(path)
            deviations.append({
                "path": str(path),
                "sha256": _sha256_file(path),
                "payload": payload,
            })
            if (
                payload.get("status") == "frozen-before-recovery-inference"
                and payload.get("formal_source_modified") is False
                and payload.get("prior_resume_contracts_path")
                and payload.get("prior_resume_contracts_sha256")
                and payload.get("old_deployment_epoch")
                and payload.get("new_deployment_epoch")
            ):
                stable_identity = payload.get("stable_identity_checks") or {}
                stable_metadata = (
                    payload.get("stable_metadata_file_hash_checks") or {}
                )
                if not stable_identity or not stable_metadata:
                    raise ValueError(
                        "recovery amendment lacks stable identity/hash checks"
                    )
                if not all(
                    isinstance(row, dict)
                    and row.get("equal") is True
                    and row.get("old") == row.get("new")
                    for row in [*stable_identity.values(), *stable_metadata.values()]
                ):
                    raise ValueError(
                        "recovery amendment contains a non-identical semantic field"
                    )
                prior_path = Path(
                    str(payload["prior_resume_contracts_path"])
                ).resolve()
                if not prior_path.is_file():
                    portable_prior = (
                        path.resolve().parent
                        / Path(
                            str(payload["prior_resume_contracts_path"])
                        ).name
                    )
                    if portable_prior.is_file():
                        prior_path = portable_prior
                if (
                    not prior_path.is_file()
                    or _sha256_file(prior_path)
                    != str(payload["prior_resume_contracts_sha256"])
                ):
                    raise ValueError(
                        "recovery prior-contract allowlist hash mismatch"
                    )
                prior = strict_json(prior_path)
                if prior.get("status") != "frozen-before-recovery-inference":
                    raise ValueError(
                        "recovery prior-contract allowlist is not frozen"
                    )
                allowed_prior = {
                    (
                        str(row.get("mode")),
                        str(row.get("config_hash")),
                        str(row.get("protocol_digest")),
                        str(row.get("deployment_epoch")),
                        str(row.get("source_sha256")),
                        str(row.get("dataset_manifest_sha256")),
                    )
                    for row in prior.get("allowed_contracts", [])
                }
                candidate = {
                    "amendment_path": str(path),
                    "amendment_sha256": _sha256_file(path),
                    "prior_contracts_path": str(prior_path),
                    "prior_contracts_sha256": _sha256_file(prior_path),
                    "old_deployment_epoch": str(
                        payload["old_deployment_epoch"]
                    ),
                    "new_deployment_epoch": str(
                        payload["new_deployment_epoch"]
                    ),
                    "allowed_prior_contracts": allowed_prior,
                    "model_artifact_manifest_sha256": (
                        stable_identity.get(
                            "model_artifact_manifest_sha256", {}
                        ).get("new")
                    ),
                }
                if recovery_policy is not None:
                    raise ValueError(
                        "multiple recovery protocol amendments are ambiguous"
                    )
                recovery_policy = candidate
        except Exception as exc:
            issue(
                "validity",
                f"invalid runtime deviation manifest: {exc}",
                path=str(path),
            )

    expected_per_mode = expected_tasks * expected_samples
    for mode in mode_list:
        mode_root = run_root / mode
        run_errors_path = mode_root / "run_errors.json"
        try:
            run_errors = strict_json(run_errors_path)
            if run_errors != []:
                issue("completeness", "run_errors.json is non-empty", mode=mode)
        except Exception as exc:
            issue("validity", f"invalid run_errors.json: {exc}", mode=mode)

        paths = sorted(mode_root.glob("*/sample__*/trajectory.json"))
        if len(paths) != expected_per_mode:
            issue(
                "completeness",
                f"expected {expected_per_mode} trajectories, found {len(paths)}",
                mode=mode,
            )
        items: list[dict[str, Any]] = []
        for path in paths:
            try:
                item = strict_json(path)
                strict_trajectory_json_count += 1
            except Exception as exc:
                issue("validity", f"invalid trajectory JSON: {exc}", path=str(path))
                continue
            items.append(item)
            case = str(item.get("benchmark_case_name"))
            sample = int(item.get("sample_index", -1))
            seed = int(item.get("seed", -1))
            key = (case, sample, seed, str(item.get("mode")))
            if key in unique_keys:
                issue("uniqueness", "duplicate trajectory key", key=str(key))
            unique_keys.add(key)
            if key[3] != mode:
                issue("consistency", "trajectory mode differs from directory", key=str(key))
            if sample != seed:
                issue("consistency", "sample index and frozen seed differ", key=str(key))
            if item.get("terminal") is not True or item.get("status") != "complete":
                issue("completeness", "trajectory is not terminal complete", key=str(key))
            source_hashes.add(str(item.get("source_sha256")))
            dataset_hashes.add(str(item.get("dataset_manifest_sha256")))
            dataset_kinds.add(str(item.get("dataset_kind")))
            protocol = str(item.get("protocol_digest"))
            config = str(item.get("config_hash"))
            epoch = str(item.get("deployment_epoch"))
            source = str(item.get("source_sha256"))
            dataset = str(item.get("dataset_manifest_sha256"))
            protocol_by_mode[mode].add(protocol)
            config_by_mode[mode].add(config)
            deployment_epochs.add(epoch)
            contract_by_mode[mode].add(
                (mode, config, protocol, epoch, source, dataset)
            )

            rounds = item.get("rounds") or []
            if not rounds:
                issue("completeness", "trajectory has no rounds", key=str(key))
                continue
            if len(rounds) > 6 or (mode == "zero_shot" and len(rounds) != 1):
                issue("protocol", f"invalid round count {len(rounds)}", key=str(key))
            observed_indices = [int(round_item.get("round_index", -1)) for round_item in rounds]
            if observed_indices != list(range(len(rounds))):
                issue("consistency", "round indices are not contiguous from zero", key=str(key))

            first = rounds[0]
            round0[(case, sample)][mode] = (
                item.get("seed_prompt_sha256"),
                first.get("response_sha256"),
                first.get("code_sha256"),
            )
            for round_item in rounds:
                index = int(round_item.get("round_index", -1))
                review = round_item.get("review") or {}
                code = str(review.get("generated_code") or "")
                if code and round_item.get("code_sha256") != _sha256_text(code):
                    issue("integrity", "round code SHA mismatch", key=str(key), round=index)
                else:
                    code_sha_checks += 1
                provenance = round_item.get("prompt_provenance") or {}
                reference_provenance_checks += 1
                if provenance.get("reference_kernel_read_or_prompted") is not False:
                    issue(
                        "provenance",
                        "reference-kernel provenance is not explicitly false",
                        key=str(key),
                        round=index,
                    )
                disk_round_path = path.parent / f"round__{index}" / "round.json"
                try:
                    disk_round = strict_json(disk_round_path)
                    strict_round_json_count += 1
                except Exception as exc:
                    issue(
                        "validity",
                        f"invalid per-round JSON: {exc}",
                        key=str(key),
                        round=index,
                    )
                    continue
                if int(disk_round.get("round_index", -1)) != index:
                    issue("consistency", "per-round index mismatch", key=str(key), round=index)
                state = disk_round.get("rfl_state")
                if state:
                    rfl_snapshot_count += 1
                    entries = state.get("entries") or []
                    future = [
                        entry.get("round_index")
                        for entry in entries
                        if int(entry.get("round_index", -1)) > index
                    ]
                    if future:
                        issue(
                            "isolation",
                            f"RFL snapshot contains future rounds {future}",
                            key=str(key),
                            round=index,
                        )
                    if int(state.get("entry_count", len(entries))) != len(entries):
                        issue(
                            "consistency",
                            "RFL entry_count differs from entries length",
                            key=str(key),
                            round=index,
                        )
                    max_entries = int(state.get("max_entries") or 5)
                    if len(entries) > max_entries:
                        issue(
                            "protocol",
                            "RFL snapshot exceeds max_entries",
                            key=str(key),
                            round=index,
                        )
                memory_facts = disk_round.get("memory_facts") or {}
                if (
                    "reference_kernel_read_or_used" in memory_facts
                    and memory_facts["reference_kernel_read_or_used"] is not False
                ):
                    issue(
                        "provenance",
                        "memory facts report reference-kernel access",
                        key=str(key),
                        round=index,
                    )

            selected = int(item.get("selected_round_index", -1))
            if selected < 0 or selected >= len(rounds):
                issue("integrity", "selected round is out of range", key=str(key))
            else:
                selected_round = rounds[selected]
                if item.get("selected_candidate_id") != selected_round.get("candidate_id"):
                    issue("integrity", "selected candidate ID mismatch", key=str(key))
                if not _same_review(
                    item.get("final_review") or {},
                    selected_round.get("review") or {},
                ):
                    issue("integrity", "final review differs from selected round", key=str(key))
                selected_candidate_checks += 1
                selected_cpp = path.parent / "selected_candidate.cpp"
                expected_selected_sha = selected_round.get("code_sha256")
                if expected_selected_sha is None:
                    if selected_cpp.is_file():
                        issue(
                            "integrity",
                            "parse-invalid selected round unexpectedly has selected_candidate.cpp",
                            key=str(key),
                        )
                elif not selected_cpp.is_file():
                    issue("completeness", "selected_candidate.cpp is missing", key=str(key))
                else:
                    selected_sha = _sha256_text(selected_cpp.read_text(encoding="utf-8"))
                    if selected_sha != expected_selected_sha:
                        issue("integrity", "selected_candidate.cpp SHA mismatch", key=str(key))

            final_review = item.get("final_review") or {}
            conjunction = bool(final_review.get("pass_tb") and final_review.get("pass_synth"))
            if bool(final_review.get("pass_tb_and_synth")) != conjunction:
                issue("integrity", "TB&Synth is not a same-candidate conjunction", key=str(key))
            if bool(item.get("success")) != conjunction:
                issue("consistency", "trajectory success differs from final conjunction", key=str(key))
        trajectories[mode] = items

        case_counts: dict[str, int] = defaultdict(int)
        for item in items:
            case_counts[str(item.get("benchmark_case_name"))] += 1
        if len(case_counts) != expected_tasks:
            issue(
                "completeness",
                f"expected {expected_tasks} cases, found {len(case_counts)}",
                mode=mode,
            )
        for case, count in case_counts.items():
            if count != expected_samples:
                issue(
                    "completeness",
                    f"case has {count}, expected {expected_samples} trajectories",
                    mode=mode,
                    case=case,
                )

    for key, identities in round0.items():
        if set(identities) != set(mode_list):
            issue("completeness", "Round 0 identity group lacks modes", key=str(key))
        elif len(set(identities.values())) != 1:
            issue("integrity", "Round 0 prompt/response/code identity mismatch", key=str(key))

    if len(round0) != expected_per_mode:
        issue(
            "completeness",
            f"expected {expected_per_mode} Round 0 pairs, found {len(round0)}",
        )
    if len(source_hashes) != 1:
        issue("consistency", f"expected one source SHA, found {len(source_hashes)}")
    if len(dataset_hashes) != 1:
        issue("consistency", f"expected one dataset manifest SHA, found {len(dataset_hashes)}")
    if len(dataset_kinds) != 1:
        issue("consistency", f"expected one dataset kind, found {sorted(dataset_kinds)}")
    recovery_summary: dict[str, Any] | None = None
    if recovery_policy is None:
        for mode in mode_list:
            if len(protocol_by_mode[mode]) != 1:
                issue(
                    "consistency",
                    "protocol digest is not unique within mode",
                    mode=mode,
                )
            if len(config_by_mode[mode]) != 1:
                issue(
                    "consistency",
                    "config hash is not unique within mode",
                    mode=mode,
                )
    else:
        old_epoch = recovery_policy["old_deployment_epoch"]
        new_epoch = recovery_policy["new_deployment_epoch"]
        allowed_prior = recovery_policy["allowed_prior_contracts"]
        observed_old = 0
        observed_new = 0
        for mode in mode_list:
            new_contracts: set[tuple[str, ...]] = set()
            for contract in contract_by_mode[mode]:
                if contract[3] == old_epoch:
                    if contract not in allowed_prior:
                        issue(
                            "consistency",
                            "pre-recovery contract is absent from frozen allowlist",
                            mode=mode,
                            contract=str(contract),
                        )
                    else:
                        observed_old += sum(
                            1
                            for item in trajectories[mode]
                            if (
                                str(item.get("mode")),
                                str(item.get("config_hash")),
                                str(item.get("protocol_digest")),
                                str(item.get("deployment_epoch")),
                                str(item.get("source_sha256")),
                                str(item.get("dataset_manifest_sha256")),
                            )
                            == contract
                        )
                elif contract[3] == new_epoch:
                    new_contracts.add(contract)
                    observed_new += sum(
                        1
                        for item in trajectories[mode]
                        if (
                            str(item.get("mode")),
                            str(item.get("config_hash")),
                            str(item.get("protocol_digest")),
                            str(item.get("deployment_epoch")),
                            str(item.get("source_sha256")),
                            str(item.get("dataset_manifest_sha256")),
                        )
                        == contract
                    )
                else:
                    issue(
                        "consistency",
                        "trajectory uses an undeclared deployment epoch",
                        mode=mode,
                        contract=str(contract),
                    )
            if len(new_contracts) > 1:
                issue(
                    "consistency",
                    "recovery epoch has multiple contracts within mode",
                    mode=mode,
                )
            if len(protocol_by_mode[mode]) > 2:
                issue(
                    "consistency",
                    "more than two protocol digests within recovery mode",
                    mode=mode,
                )
            if len(config_by_mode[mode]) > 2:
                issue(
                    "consistency",
                    "more than two config hashes within recovery mode",
                    mode=mode,
                )
        if not ({old_epoch, new_epoch} >= deployment_epochs):
            issue(
                "consistency",
                "observed deployment epochs exceed declared recovery epochs",
            )
        recovery_summary = {
            key: value
            for key, value in recovery_policy.items()
            if key != "allowed_prior_contracts"
        }
        recovery_summary.update({
            "status": "validated_identical_model_recovery",
            "observed_old_epoch_trajectories": observed_old,
            "observed_new_epoch_trajectories": observed_new,
            "semantic_configuration_identity_passed": True,
        })
        warnings.append({
            "category": "runtime_recovery",
            "detail": (
                "A declared infrastructure recovery uses two process epochs; "
                "model artifacts, launch configuration and frozen source are identical."
            ),
            "old_deployment_epoch": old_epoch,
            "new_deployment_epoch": new_epoch,
        })

    metric_rows = _metric_rows(trajectories)
    stored_metric_max_abs_error: float | None = None
    analysis_nonfinite = 0
    if analysis_path is not None:
        raw = analysis_path.read_text(encoding="utf-8")
        nonfinite_constants: list[str] = []
        json.loads(
            raw,
            parse_constant=lambda value: nonfinite_constants.append(value),
        )
        analysis_nonfinite = len(nonfinite_constants)
        if require_analysis_strict_json and analysis_nonfinite:
            issue(
                "validity",
                f"analysis JSON contains {analysis_nonfinite} non-finite constants",
            )
        analysis = json.loads(raw)
        stored = {
            (mode, metric): float(value)
            for mode, summary in (analysis.get("summaries") or {}).items()
            for metric, value in (summary.get("metrics") or {}).items()
        }
        differences: list[float] = []
        for row in metric_rows:
            key = (str(row["mode"]), str(row["metric"]))
            if key not in stored:
                issue("completeness", "analysis lacks direct metric", key=str(key))
                continue
            differences.append(abs(float(row["value"]) - stored[key]))
        stored_metric_max_abs_error = max(differences, default=0.0)
        if stored_metric_max_abs_error > 1e-12:
            issue(
                "integrity",
                f"stored metric max abs error is {stored_metric_max_abs_error}",
            )

    category_counts: dict[str, int] = defaultdict(int)
    for row in errors:
        category_counts[str(row["category"])] += 1
    expected_total = expected_per_mode * len(mode_list)
    result = {
        "schema_version": 1,
        "passed": not errors,
        "status": (
            "pass_with_disclosed_runtime_deviation"
            if not errors and deviations
            else "pass"
            if not errors
            else "fail"
        ),
        "run_root": str(run_root),
        "expected": {
            "modes": mode_list,
            "task_count": expected_tasks,
            "samples_per_task": expected_samples,
            "trajectories_per_mode": expected_per_mode,
            "total_trajectories": expected_total,
        },
        "observed": {
            "total_trajectories": sum(len(items) for items in trajectories.values()),
            "unique_trajectory_keys": len(unique_keys),
            "round0_identity_pairs": len(round0),
            "strict_trajectory_json_count": strict_trajectory_json_count,
            "strict_round_json_count": strict_round_json_count,
            "rfl_snapshot_count": rfl_snapshot_count,
            "reference_provenance_checks": reference_provenance_checks,
            "code_sha_checks": code_sha_checks,
            "selected_candidate_checks": selected_candidate_checks,
            "analysis_nonfinite_constants": analysis_nonfinite,
            "stored_metric_max_abs_error": stored_metric_max_abs_error,
            "source_hashes": sorted(source_hashes),
            "dataset_manifest_hashes": sorted(dataset_hashes),
            "dataset_kinds": sorted(dataset_kinds),
            "protocol_digest_count_by_mode": {
                mode: len(protocol_by_mode[mode]) for mode in mode_list
            },
            "config_hash_count_by_mode": {
                mode: len(config_by_mode[mode]) for mode in mode_list
            },
            "deployment_epochs": sorted(deployment_epochs),
            "contract_count_by_mode": {
                mode: len(contract_by_mode[mode]) for mode in mode_list
            },
        },
        "data_quality": {
            "error_count": len(errors),
            "error_count_by_category": dict(sorted(category_counts.items())),
            "errors": errors[:1000],
            "warnings": warnings,
        },
        "runtime_protocol_deviations": deviations,
        "recovery_deployment_audit": recovery_summary,
        "protocol_fidelity": (
            "semantic_protocol_with_disclosed_runtime_deviations"
            if deviations
            else "frozen_protocol_no_recorded_runtime_deviation"
        ),
    }
    return result, metric_rows


def write_audit(
    result: dict[str, Any],
    metrics: list[dict[str, Any]],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "independent_audit.json", result)
    fields = ["mode", "metric", "value", "case_count", "trajectory_count"]
    with (output / "direct_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
