from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from collections import defaultdict
from typing import Any


TARGET_FORMULA_VERSION = "dac2027-target-15pp"
PROMOTION_POLICY_VERSION = "dac2027-audit-ordering-paired-ci"
REVIEW_METRICS = {
    "parse": "pass_parse",
    "compile": "pass_compile",
    "tb": "pass_tb",
    "synth": "pass_synth",
    "tb_and_synth": "pass_tb_and_synth",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



TRACKED_PYTHON_PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "psutil",
    "requests",
    "tomli",
    "matplotlib",
    "seaborn",
    "pytest",
    "torch",
)


def python_environment_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in TRACKED_PYTHON_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    value: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value

def stable_run_metadata_binding(metadata: dict[str, Any]) -> dict[str, Any]:
    """Project run metadata onto immutable/result-bearing fields only.

    Wall-clock timestamps and smoke latency are deliberately excluded so a
    validated terminal baseline remains resumable without weakening provenance.
    """
    vllm = metadata.get("vllm") or {}
    return {
        "case_count": metadata.get("case_count"),
        "iteration_id": metadata.get("iteration_id"),
        "test_informed_exploratory": metadata.get("test_informed_exploratory"),
        "config_hash": metadata.get("config_hash"),
        "source_sha256": metadata.get("source_sha256"),
        "feature_flags": metadata.get("feature_flags"),
        "config": metadata.get("config"),
        "rules": metadata.get("rules"),
        "rfl": metadata.get("rfl"),
        "dataset_manifest": metadata.get("dataset_manifest"),
        "hls_eval_source": metadata.get("hls_eval_source"),
        "freeze_manifest": metadata.get("freeze_manifest"),
        "deployment_binding": metadata.get("deployment_binding"),
        "vllm": {
            "models": vllm.get("models"),
            "context_limit": vllm.get("context_limit"),
            "run_metadata": vllm.get("run_metadata"),
            "deployment_epoch": vllm.get("deployment_epoch"),
        },
        "vitis": metadata.get("vitis"),
        "protocol_digest": metadata.get("protocol_digest"),
        "completed_trajectories": metadata.get("completed_trajectories"),
        "error_trajectories": metadata.get("error_trajectories"),
        "end_of_mode_provenance": metadata.get("end_of_mode_provenance"),
    }


def zero_shot_outcome_binding(
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trajectories:
        grouped[str(item["benchmark_case_name"])].append(item)
    cases: list[dict[str, Any]] = []
    for case_name in sorted(grouped):
        items = sorted(
            grouped[case_name], key=lambda item: int(item["sample_index"])
        )
        if [int(item["sample_index"]) for item in items] != list(range(5)):
            raise ValueError(
                f"Zero-shot samples are not exactly 0..4 for {case_name}"
            )
        cases.append(
            {
                "case_name": case_name,
                "seeds": [int(item["seed"]) for item in items],
                "sample_success_vectors": {
                    metric: [
                        int(bool(item["final_review"][review_key]))
                        for item in items
                    ]
                    for metric, review_key in REVIEW_METRICS.items()
                },
                "selected_code_sha256": [
                    item["rounds"][int(item["selected_round_index"])].get(
                        "code_sha256"
                    )
                    for item in items
                ],
            }
        )
    return {
        "task_ids": [entry["case_name"] for entry in cases],
        "cases": cases,
        "outcomes_sha256": canonical_sha256(cases),
    }
