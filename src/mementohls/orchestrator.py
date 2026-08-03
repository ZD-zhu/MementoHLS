from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .executor import ActionExecution, ContractActionExecutor
from .candidate_selection import (
    best_reviewed_round_index,
    last_parse_valid_round_index,
)
from .causal_replay import (
    CONTROL_MECHANISMS,
    CausalReplaySource,
    build_full_source_manifest,
    canonical_prompt_log,
)
from .compat import enable_python310_hls_eval
from .datasets import (
    DATASET_BENCH4HLS,
    DATASET_HLS_EVAL,
    build_seed_prompt,
    case_header_name,
    case_interface_contract,
    case_manifest_files,
    case_manifest_root,
    load_cases,
)
from .format_recovery import FormatRecoveryResult, recover_hls_envelope
from .llm_client import VLLMClient
from .ercl import (
    ExecutableRepairContractLibrary,
    MemoryRule,
    RetrievalResult,
    validate_diagnosis,
)
from .memory_facts import derive_memory_facts
from .models import (
    Diagnosis,
    FailureStage,
    FORMAL_RUN_MODES,
    HLS_EVAL_VERIFICATION_MODES,
    LLMCallResult,
    ReviewResult,
    RunMode,
    RFLPolicyConfig,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from .provenance import canonical_sha256, python_environment_manifest
from .prompts import (
    CLEAN_PROFILE_COUNT,
    CLEAN_RFL_BUDGETS,
    DIAGNOSER_SYSTEM,
    DIAGNOSIS_PROFILE_COUNT,
    DIAGNOSIS_RFL_BUDGETS,
    REPAIRER_SYSTEM,
    REPAIR_PROFILE_COUNT,
    REPAIR_RFL_BUDGETS,
    build_cleanroom_prompt,
    build_diagnosis_prompt,
    build_repair_prompt,
)
from .reviewer import HLSReviewer, ToolPools, failure_log
from .resource_gate import wait_for_capacity as wait_for_runtime_capacity
from .rfl import (
    RFL_SCHEMA_VERSION,
    ReviewerGroundedFalsificationLedger,
    TrajectoryKey,
    stage_vector,
)
from .trajectory_contracts import completed_trajectory_contract_errors


CONTROL_ABLATION_PROTOCOL = "1-independent-round0-paired"


def memory_active_for_round(
    mode: RunMode,
    repair_round: int,
    *,
    rfl_no_progress_streak: int = 0,
) -> bool:
    if repair_round < 1:
        raise ValueError("repair_round must be positive")
    if not (mode.uses_rfl or mode.uses_ercl):
        return False
    if repair_round == 1:
        return False
    if mode is RunMode.RFL and rfl_no_progress_streak >= 2:
        return False
    return True


def _validate_causal_replay_request(
    mode: RunMode,
    source: Path | None,
) -> Path | None:
    """Keep legacy replay opt-in while formal controls default to independent runs."""
    if source is not None and mode not in CONTROL_MECHANISMS:
        raise ValueError("--causal-replay-source is valid only for Full controls")
    return source


@dataclass(frozen=True)
class ExperimentConfig:
    mode: RunMode
    dataset_root: Path
    output_dir: Path
    hls_eval_root: Path
    repo_root: Path
    rules_path: Path
    vitis_dir: Path
    base_url: str
    model: str
    dataset_kind: str = DATASET_HLS_EVAL
    samples: int = 5
    temperature: float = 0.7
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_repair_rounds: int = 5
    max_tokens: int = 4096
    context_limit: int = 8192
    expected_case_count: int | None = 94
    freeze_manifest: Path | None = None
    deployment_binding: Path | None = None
    vllm_run_metadata: Path | None = None
    inference_server: str = "A100"
    fpga_part: str = "xczu9eg-ffvb1156-2-e"
    clock_ns: float = 5.0
    design_workers: int = 4
    llm_workers: int = 4
    csim_workers: int = 4
    synth_workers: int = 4
    llm_timeout: float = 600.0
    csim_timeout: float = 360.0
    synth_timeout: float = 360.0
    review_infrastructure_retries: int = 2
    runtime_start_cpu_max: float = 45.0
    runtime_start_load_max: float = 96.0
    runtime_pause_cpu_max: float = 70.0
    runtime_pause_load_max: float = 120.0
    runtime_start_memory_available_gib_min: float = 380.0
    runtime_pause_memory_available_gib_min: float = 350.0
    runtime_process_tree_rss_gib_max: float = 300.0
    runtime_capacity_poll_seconds: int = 15
    resume: bool = False
    round0_source: Path | None = None
    causal_replay_source: Path | None = None
    case_names: tuple[str, ...] = ()
    limit: int | None = None
    iteration_id: str = "development"
    exploratory: bool = True
    formal_protocol: bool = False
    block_id: int | None = None
    bench4hls_frozen_epoch: str = "@2027-01-01 00:00:00"


def _git_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(resolved), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    top_text = run("rev-parse", "--show-toplevel")
    try:
        top = Path(top_text).resolve()
        relative = resolved.relative_to(top).as_posix() or "."
    except Exception:
        top = resolved
        relative = "."
    # `all` expands every generated Vitis artifact. In v6 this embedded a
    # 90 MB status string in every mode summary. `normal` preserves dirty-root
    # evidence without recursively enumerating generated output trees.
    status = run(
        "status",
        "--short",
        "--branch",
        "--untracked-files=normal",
        "--",
        relative,
    )
    diff = run("diff", "--binary", "HEAD", "--", relative)
    status_lines = status.splitlines()
    preview_lines = status_lines[:100]
    preview = "\n".join(preview_lines)
    preview_was_byte_truncated = len(preview.encode("utf-8")) > 65536
    if preview_was_byte_truncated:
        preview = preview.encode("utf-8")[:65536].decode("utf-8", errors="ignore")
    return {
        "path": str(resolved),
        "git_toplevel": str(top),
        "scoped_path": relative,
        "commit": run("rev-parse", "HEAD"),
        "status_policy": "short-branch-untracked-normal",
        "status_sha256": _sha256_text(status),
        "status_bytes": len(status.encode("utf-8")),
        "status_line_count": len(status_lines),
        "status_preview": preview,
        "status_truncated": (
            len(preview_lines) < len(status_lines) or preview_was_byte_truncated
        ),
        "diff_sha256": _sha256_text(diff),
        "diff_bytes": len(diff.encode("utf-8")),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {"__pycache__", ".pytest_cache", ".git", "artifact", "docs"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if any(part in excluded for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo", ".log", ".orig"}:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hls_eval_source_manifest(root: Path) -> tuple[dict[str, Any], str]:
    """Freeze executable HLS-Eval package code without reading benchmark kernels."""
    package = root / "hls_eval"
    candidates = [path for path in package.rglob("*") if path.is_file()]
    for name in ("pyproject.toml", "uv.lock", "requirements.txt"):
        path = root / name
        if path.is_file():
            candidates.append(path)
    excluded = {"__pycache__", ".pytest_cache", ".git"}
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in sorted(set(candidates)):
        if any(part in excluded for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo", ".log", ".orig"}:
            continue
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")

    def git_output(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except Exception as exc:
            return f"unavailable: {exc}"

    commit = git_output("rev-parse", "HEAD").strip()
    status = git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "hls_eval",
        "pyproject.toml",
        "uv.lock",
        "requirements.txt",
    )
    diff = git_output("diff", "--binary", "HEAD", "--", "hls_eval", "pyproject.toml", "uv.lock")
    manifest = {
        "root": str(root.resolve()),
        "file_count": len(files),
        "files": files,
        "git_commit": commit,
        "git_status_sha256": _sha256_text(status),
        "git_diff_sha256": _sha256_text(diff),
        "reference_kernel_excluded": True,
    }
    source_sha = digest.hexdigest()
    manifest["source_tree_sha256"] = source_sha
    return manifest, source_sha


def _dataset_manifest(cases: list[Any]) -> tuple[dict[str, Any], str]:
    """Hash only immutable task inputs; never read or hash the reference kernel."""

    entries: list[dict[str, Any]] = []
    for case in cases:
        files = case_manifest_files(case)
        manifest_root = case_manifest_root(case)
        unique = sorted({path.resolve() for path in files}, key=str)
        entries.append(
            {
                "case_name": case.name,
                "dataset_id": str(
                    getattr(case, "dataset_id", DATASET_HLS_EVAL)
                ),
                "files": {
                    str(path.relative_to(manifest_root)): file_sha256(path)
                    for path in unique
                },
            }
        )
    manifest = {
        "case_count": len(entries),
        "reference_kernel_excluded": True,
        "cases": entries,
    }
    digest = _sha256_text(json.dumps(manifest, sort_keys=True))
    manifest["manifest_sha256"] = digest
    return manifest, digest


def _command_version(
    command: list[str], env: dict[str, str] | None = None
) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        return (result.stdout + result.stderr).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _load_vllm_deployment_manifest(
    path: Path | None,
    *,
    base_url: str,
    model: str,
    inference_server: str,
    formal_run: bool,
) -> dict[str, Any] | None:
    if path is None:
        if formal_run:
            raise ValueError("Formal 94-case runs require --vllm-run-metadata")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "inference_server",
        "endpoint",
        "served_model",
        "remote_hostname",
        "run_id",
        "selected_gpu",
        "selected_gpu_uuid",
        "tensor_parallel_size",
        "conda_environment",
        "model_path",
        "launch_command",
        "metadata_root",
        "metadata_files_sha256",
        "deployment_epoch",
        "model_listing_created_at_capture",
        "prometheus_process_start_time_seconds",
        "model_artifact_manifest_sha256",
        "model_artifact_entry_count",
        "model_weight_shard_count",
        "process_identity",
        "process_identity_sha256",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"vLLM deployment manifest misses {sorted(missing)}")
    if str(payload["endpoint"]).rstrip("/") != base_url.rstrip("/"):
        raise ValueError("vLLM deployment endpoint does not match --base-url")
    if payload["served_model"] != model:
        raise ValueError("vLLM served model does not match --model")
    if payload["inference_server"] != inference_server:
        raise ValueError("vLLM server label does not match --inference-server")
    if int(payload["tensor_parallel_size"]) != 1:
        raise ValueError("Single-GPU MementoHLS protocol requires tensor_parallel_size=1")
    gpu = int(payload["selected_gpu"])
    if not 0 <= gpu <= 7:
        raise ValueError("Selected physical GPU must be in 0..7")
    if not isinstance(payload["metadata_files_sha256"], dict) or not payload[
        "metadata_files_sha256"
    ]:
        raise ValueError("vLLM deployment manifest needs copied metadata hashes")
    metadata_root = Path(payload["metadata_root"]).resolve()
    for relative, expected_sha in payload["metadata_files_sha256"].items():
        candidate = (metadata_root / relative).resolve()
        if metadata_root not in candidate.parents:
            raise ValueError(f"Unsafe metadata path in vLLM manifest: {relative}")
        if not candidate.is_file() or file_sha256(candidate) != expected_sha:
            raise ValueError(f"Copied vLLM metadata hash mismatch: {candidate}")
    required_metadata = {
        "hostname.txt",
        "selected_gpu_and_free_mib.csv",
        f"launch_command_gpu{gpu}.txt",
        "models.json",
        "smoke.json",
        "model_config_sha256.txt",
        "model_file_inventory.csv",
        "vllm_transformers_versions.txt",
        "gpu_inventory_before.csv",
        "gpu_inventory_after.csv",
        "gpu_memory_utilization.txt",
        "conda_environment.txt",
        "started_at_utc.txt",
        "model_artifact_sha256.txt",
        "process_identity.json",
        "process_metrics_identity.prom",
        "process_start_time_seconds.txt",
        "vllm.pid",
    }
    missing_metadata = required_metadata - set(payload["metadata_files_sha256"])
    if missing_metadata:
        raise ValueError(f"vLLM manifest lacks required evidence files: {sorted(missing_metadata)}")
    if metadata_root.name != str(payload["run_id"]):
        raise ValueError("vLLM run_id does not equal copied metadata directory name")
    if str(payload["deployment_epoch"]) != str(payload["run_id"]):
        raise ValueError("vLLM deployment_epoch must equal the audited run_id")
    artifact_path = metadata_root / "model_artifact_sha256.txt"
    artifact_lines = [
        line for line in artifact_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        file_sha256(artifact_path) != payload["model_artifact_manifest_sha256"]
        or len(artifact_lines) != int(payload["model_artifact_entry_count"])
        or int(payload["model_weight_shard_count"]) < 1
        or sum(".safetensors" in line and "index.json" not in line for line in artifact_lines)
        != int(payload["model_weight_shard_count"])
    ):
        raise ValueError("vLLM cryptographic model artifact binding differs")
    process_path = metadata_root / "process_identity.json"
    process_identity = json.loads(process_path.read_text(encoding="utf-8"))
    if (
        file_sha256(process_path) != payload["process_identity_sha256"]
        or process_identity != payload["process_identity"]
        or int(process_identity.get("pid", -1))
        != int((metadata_root / "vllm.pid").read_text(encoding="utf-8").strip())
        or not process_identity.get("boot_id")
        or int(process_identity.get("process_start_ticks", 0)) <= 0
    ):
        raise ValueError("vLLM process identity evidence differs")
    hostname = (metadata_root / "hostname.txt").read_text(encoding="utf-8").strip()
    if hostname != str(payload["remote_hostname"]):
        raise ValueError("vLLM remote_hostname conflicts with hostname.txt")
    selected_row = (metadata_root / "selected_gpu_and_free_mib.csv").read_text(encoding="utf-8").strip().split(",")
    if not selected_row or int(selected_row[0]) != gpu:
        raise ValueError("vLLM selected_gpu conflicts with selected_gpu_and_free_mib.csv")
    launch_file = metadata_root / f"launch_command_gpu{gpu}.txt"
    launch = launch_file.read_text(encoding="utf-8").strip()
    if launch != str(payload["launch_command"]).strip():
        raise ValueError("vLLM launch_command conflicts with copied launch command")
    required_launch_tokens = [
        "--tensor-parallel-size 1",
        "--max-model-len 8192",
        "--max-num-seqs 8",
        "--gpu-memory-utilization 0.90",
        "--generation-config vllm",
        "--host 0.0.0.0",
        "--port 8000",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"--served-model-name {model}",
        str(payload["model_path"]),
    ]
    if any(token not in launch for token in required_launch_tokens):
        raise ValueError("vLLM launch command does not prove the frozen single-GPU protocol")
    models = json.loads((metadata_root / "models.json").read_text(encoding="utf-8"))
    model_entry = next(
        (item for item in models.get("data", []) if item.get("id") == model),
        None,
    )
    if model_entry is None:
        raise ValueError("vLLM models.json does not contain the served model")
    if int(model_entry.get("created", -1)) != int(
        payload["model_listing_created_at_capture"]
    ):
        raise ValueError("vLLM captured model listing conflicts with models.json")
    process_start_path = metadata_root / "process_start_time_seconds.txt"
    captured_process_start = float(
        process_start_path.read_text(encoding="utf-8").strip()
    )
    metrics_start_values = [
        float(line.split(None, 1)[1])
        for line in (metadata_root / "process_metrics_identity.prom")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("process_start_time_seconds ")
    ]
    if (
        len(metrics_start_values) != 1
        or abs(metrics_start_values[0] - captured_process_start) > 1e-6
        or abs(
            captured_process_start
            - float(payload["prometheus_process_start_time_seconds"])
        )
        > 1e-6
    ):
        raise ValueError("vLLM Prometheus process start identity differs")
    smoke = json.loads((metadata_root / "smoke.json").read_text(encoding="utf-8"))
    smoke_text = smoke["choices"][0]["message"]["content"].strip()
    if smoke_text != "HLS_SMOKE_OK":
        raise ValueError("vLLM smoke.json does not contain the exact smoke response")
    conda_environment = (metadata_root / "conda_environment.txt").read_text(encoding="utf-8").strip()
    if conda_environment != str(payload["conda_environment"]):
        raise ValueError("vLLM conda environment conflicts with copied evidence")
    expected_conda = {"A100": "vllm", "A6000": "SAGE"}.get(inference_server)
    if expected_conda and conda_environment != expected_conda:
        raise ValueError(
            f"{inference_server} formal service must use conda environment {expected_conda}"
        )
    memory_utilization = (
        metadata_root / "gpu_memory_utilization.txt"
    ).read_text(encoding="utf-8").strip()
    if memory_utilization != "0.90":
        raise ValueError("MementoHLS freezes gpu_memory_utilization at 0.90")
    before_rows = [
        [cell.strip() for cell in line.split(",")]
        for line in (metadata_root / "gpu_inventory_before.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible_before = [
        row for row in before_rows
        if 0 <= int(row[0]) <= 7 and int(row[5]) >= 30000
    ]
    if not eligible_before:
        raise ValueError("No >=30000 MiB single GPU was eligible before vLLM launch")
    highest_free = max(eligible_before, key=lambda row: (int(row[5]), -int(row[0])))
    if int(highest_free[0]) != gpu:
        raise ValueError("Selected GPU was not the highest-free eligible single GPU")
    if not (metadata_root / "model_config_sha256.txt").read_text(encoding="utf-8").strip():
        raise ValueError("vLLM model configuration checksum evidence is empty")
    versions = (metadata_root / "vllm_transformers_versions.txt").read_text(encoding="utf-8").lower()
    if "name: vllm" not in versions or "name: transformers" not in versions:
        raise ValueError("vLLM/Transformers version evidence is incomplete")
    inventory_rows = [
        [cell.strip() for cell in line.split(",")]
        for line in (metadata_root / "gpu_inventory_after.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_inventory = next((row for row in inventory_rows if int(row[0]) == gpu), None)
    if selected_inventory is None or selected_inventory[2] != str(payload["selected_gpu_uuid"]):
        raise ValueError("vLLM selected GPU UUID conflicts with GPU inventory")
    return payload


def _llm_to_dict(
    call: LLMCallResult | None,
    reused_from: str | None = None,
) -> dict[str, Any] | None:
    if call is None:
        return None
    value = asdict(call)
    value["physical_call"] = bool(call.physical_call) and reused_from is None
    if reused_from is not None:
        value["reused_from"] = reused_from
    return value


def _call_token_totals(*calls: LLMCallResult | None) -> tuple[int, int, int]:
    prompt = sum(int(call.prompt_tokens or 0) for call in calls if call)
    completion = sum(int(call.completion_tokens or 0) for call in calls if call)
    total = sum(int(call.total_tokens or 0) for call in calls if call)
    return prompt, completion, total


def _wrap_code(filename: str, code: str) -> str:
    return f'<OUTPUT_CODE name="{Path(filename).name}">\n{code.rstrip()}\n</OUTPUT_CODE>'


def _round0_source_manifest(root: Path) -> dict[str, Any]:
    """Hash stable Round-0 content, never mutable run timestamps or wall time."""
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for path in sorted(root.glob("*/sample__*/trajectory.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "complete"
            or payload.get("terminal") is not True
            or payload.get("mode") != RunMode.ZERO_SHOT.value
            or len(payload.get("rounds") or []) != 1
        ):
            raise ValueError(f"Round-0 source is not a complete zero-shot trajectory: {path}")
        key = (
            str(payload.get("benchmark_case_name")),
            int(payload.get("sample_index")),
            int(payload.get("seed")),
        )
        if key in seen:
            raise ValueError(f"Duplicate Round-0 source identity: {key}")
        seen.add(key)
        round0 = payload["rounds"][0]
        entries.append(
            {
                "benchmark_case_name": key[0],
                "sample_index": key[1],
                "seed": key[2],
                "iteration_id": payload.get("iteration_id"),
                "source_sha256": payload.get("source_sha256"),
                "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
                "dataset_kind": payload.get("dataset_kind"),
                "hls_eval_source_sha256": payload.get("hls_eval_source_sha256"),
                "seed_prompt_sha256": round0.get("seed_prompt_sha256"),
                "response_sha256": round0.get("response_sha256"),
                "code_sha256": round0.get("code_sha256"),
                "llm_call_sha256": _sha256_text(
                    json.dumps(
                        round0.get("llm") or {}, sort_keys=True, separators=(",", ":")
                    )
                ),
                "review_vector": {
                    key: bool((round0.get("review") or {}).get(key))
                    for key in (
                        "pass_parse",
                        "pass_compile",
                        "pass_tb",
                        "pass_synth",
                        "pass_tb_and_synth",
                    )
                },
                "failure_stage": (round0.get("review") or {}).get(
                    "failure_stage"
                ),
                "infrastructure_error": (round0.get("review") or {}).get(
                    "infrastructure_error"
                ),
            }
        )
    if not entries:
        raise ValueError(f"No completed Round-0 trajectories found below {root}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(root.resolve()),
        "trajectory_count": len(entries),
        "manifest_sha256": _sha256_text(canonical),
    }


def load_complete_trajectory(
    path: Path,
    expected_config_hash: str | None = None,
    expected_protocol_digest: str | None = None,
    *,
    expected_mode: str | None = None,
    expected_case_name: str | None = None,
    expected_sample_index: int | None = None,
    expected_seed: int | None = None,
    expected_rfl_schema: str | None = None,
    require_filename_telemetry: bool = True,
    require_round_files: bool = True,
) -> dict[str, Any] | None:
    """Return a terminal result only when every resume contract is intact."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if completed_trajectory_contract_errors(
        path,
        payload,
        expected_config_hash=expected_config_hash,
        expected_protocol_digest=expected_protocol_digest,
        expected_mode=expected_mode,
        expected_case_name=expected_case_name,
        expected_sample_index=expected_sample_index,
        expected_seed=expected_seed,
        expected_rfl_schema=expected_rfl_schema,
        require_filename_telemetry=require_filename_telemetry,
        require_round_files=require_round_files,
    ):
        return None
    return payload


def high_confidence_operator_rules(
    retrieved_rules: list[MemoryRule],
) -> list[MemoryRule]:
    """Deterministic Long authorization; LLM diagnosis cannot veto it."""
    return [
        rule
        for rule in retrieved_rules
        if rule.operator_id and rule.confidence == "high"
    ][:1]

def _ercl_rfl_match_history(
    rfl: ReviewerGroundedFalsificationLedger | None,
    stage: str,
) -> str:
    """Return deterministic Short evidence for Long-rule matching.

    The channel is independent of ReviewerGroundedFalsificationLedger.to_prompt: Round 1 stays free
    of model-visible Short repair history, while exact Long operators can require
    the observed Round 0 failure. Empty/Long-only trajectories fail closed.
    """
    return rfl.to_match_history(stage) if rfl is not None else ""


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        if config.samples != len(config.seeds):
            raise ValueError("samples must equal the number of explicit seeds")
        if not 0 <= config.max_repair_rounds <= 5:
            raise ValueError("max_repair_rounds must be in [0, 5]")
        enable_python310_hls_eval(config.hls_eval_root)
        from hls_eval.data import BenchmarkCase, find_benchmark_case_dirs
        from hls_eval.prompts import build_prompt_gen_zero_shot
        from hls_eval.tools import build_vitis_hls_env

        self.BenchmarkCase = BenchmarkCase
        self.find_benchmark_case_dirs = find_benchmark_case_dirs
        self.hls_eval_seed_prompt_builder = build_prompt_gen_zero_shot
        self.vitis_env = build_vitis_hls_env(config.vitis_dir)
        self.vitis_version_output = _command_version(
            [str(config.vitis_dir / "bin" / "vitis_hls"), "-version"],
            env=self.vitis_env,
        )
        expected_version = f"v{config.vitis_dir.name}"
        if expected_version not in self.vitis_version_output:
            raise ValueError(
                f"Vitis provenance failed: expected {expected_version!r} in "
                f"{self.vitis_version_output!r}"
            )
        self.ercl = (
            ExecutableRepairContractLibrary(config.rules_path)
            if config.mode.uses_ercl
            else None
        )
        if self.ercl is not None:
            expected_rules = (
                81
                if config.mode.uses_all_rules
                else 60
                if config.mode.uses_core_rules
                else 71
            )
            if len(self.ercl.rules) != expected_rules:
                raise ValueError(
                    f"{config.mode.value} requires exactly {expected_rules} ERCL contracts; "
                    f"found {len(self.ercl.rules)} in {config.rules_path}"
                )
            forbidden_tiers = (
                {"algorithm_specific", "exact_task_contract"}
                if config.mode.uses_core_rules
                else {"exact_task_contract"}
                if not config.mode.uses_all_rules
                else set()
            )
            forbidden_rules = [
                rule.rule_id
                for rule in self.ercl.rules
                if rule.evidence_tier in forbidden_tiers
            ]
            if forbidden_rules:
                raise ValueError(
                    f"{config.mode.value} contains forbidden ERCL evidence tiers: "
                    + ", ".join(forbidden_rules)
                )
        self.action_executor = (
            ContractActionExecutor() if config.mode.uses_action_executor else None
        )
        self.llm = VLLMClient(
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.llm_timeout,
            max_retries=3,
            max_concurrency=config.llm_workers,
            context_limit=config.context_limit,
        )
        self.pools = ToolPools(config.csim_workers, config.synth_workers)
        self.reviewer = HLSReviewer(
            hls_eval_root=config.hls_eval_root,
            vitis_dir=config.vitis_dir,
            fpga_part=config.fpga_part,
            clock_ns=config.clock_ns,
            csim_timeout=config.csim_timeout,
            synth_timeout=config.synth_timeout,
            pools=self.pools,
            dataset_kind=config.dataset_kind,
            bench4hls_frozen_epoch=config.bench4hls_frozen_epoch,
        )
        self.run_root = config.output_dir / config.mode.value
        self.all_cases = self._all_cases()
        self.cases = self._cases(self.all_cases)
        if not self.cases:
            raise ValueError("No benchmark cases selected")
        if config.formal_protocol:
            if config.expected_case_count is None:
                raise ValueError("Formal protocol requires expected_case_count")
            if len(self.all_cases) != config.expected_case_count:
                raise ValueError(
                    f"Formal {config.dataset_kind} protocol requires "
                    f"{config.expected_case_count} cases, found "
                    f"{len(self.all_cases)}"
                )
            if (
                not config.case_names
                and config.limit is None
                and len(self.cases) != config.expected_case_count
            ):
                raise ValueError(
                    "Formal unblocked run must select the complete frozen dataset"
                )
        elif config.expected_case_count is not None and len(self.cases) != config.expected_case_count:
            raise ValueError(
                f"Expected {config.expected_case_count} cases before any LLM call, "
                f"but selected {len(self.cases)}. Use expected_case_count=None "
                "only for explicit development smoke tests."
            )
        self.dataset_manifest, self.dataset_manifest_sha256 = _dataset_manifest(
            self.all_cases if config.formal_protocol else self.cases
        )
        self.hls_eval_source_manifest, self.hls_eval_source_sha256 = (
            _hls_eval_source_manifest(config.hls_eval_root)
        )
        self.source_hash = _tree_sha256(config.repo_root)
        self.rules_sha256 = file_sha256(config.rules_path)
        self.vitis_executable_sha256 = file_sha256(
            config.vitis_dir / "bin" / "vitis_hls"
        )
        self.vllm_run_metadata_sha256 = (
            file_sha256(config.vllm_run_metadata)
            if config.vllm_run_metadata is not None
            else None
        )
        self.python_environment = python_environment_manifest()
        formal_run = config.formal_protocol
        self.vllm_deployment_manifest = _load_vllm_deployment_manifest(
            config.vllm_run_metadata,
            base_url=config.base_url,
            model=config.model,
            inference_server=config.inference_server,
            formal_run=formal_run,
        )
        self.deployment_epoch = (
            str((self.vllm_deployment_manifest or {}).get("run_id") or "development")
        )
        self.llm.deployment_epoch = self.deployment_epoch
        self.causal_replay_source: CausalReplaySource | None = None
        causal_replay_path = _validate_causal_replay_request(
            config.mode,
            config.causal_replay_source,
        )
        if causal_replay_path is not None:
            self.causal_replay_source = CausalReplaySource(
                causal_replay_path,
                expected_trajectories=(
                    config.expected_case_count * config.samples
                    if config.expected_case_count is not None
                    else None
                ),
            )
            self.causal_replay_source.validate_binding(
                iteration_id=config.iteration_id,
                source_sha256=self.source_hash,
                dataset_manifest_sha256=self.dataset_manifest_sha256,
                hls_eval_source_sha256=self.hls_eval_source_sha256,
                deployment_epoch=self.deployment_epoch,
            )
        self.freeze_manifest = self._validate_freeze_manifest(formal_run)
        self.deployment_binding = self._validate_deployment_binding(formal_run)
        self.protocol_manifest = self._build_protocol_manifest(formal_run)
        self.protocol_digest = str(self.protocol_manifest["protocol_digest"])
        self.config_hash = self._config_hash()

    def _build_protocol_manifest(self, formal_run: bool) -> dict[str, Any]:
        target_path = self.config.output_dir / "protocol" / "locked_targets.json"
        target_binding: dict[str, Any]
        if self.config.mode is RunMode.ZERO_SHOT:
            target_binding = {"state": "pending-zero-shot", "sha256": None}
        elif target_path.is_file():
            target_value = json.loads(target_path.read_text(encoding="utf-8"))
            stable_target = dict(target_value)
            stable_target.pop("locked_at_unix", None)
            recorded_sha = stable_target.pop("lock_content_sha256", None)
            if recorded_sha != canonical_sha256(stable_target):
                raise ValueError("Target lock content digest is invalid")
            target_binding = {
                "state": "locked",
                "path": str(target_path.resolve()),
                "sha256": file_sha256(target_path),
                "content_sha256": recorded_sha,
            }
        elif formal_run:
            raise ValueError("Formal non-zero modes require locked_targets.json")
        else:
            target_binding = {"state": "development-unlocked", "sha256": None}
        value: dict[str, Any] = {
            "iteration_id": self.config.iteration_id,
            "mode": self.config.mode.value,
            "feature_flags": self.config.mode.feature_flags,
            "freeze_manifest_sha256": (
                file_sha256(self.config.freeze_manifest)
                if self.config.freeze_manifest else None
            ),
            "source_tree_sha256": self.source_hash,
            "rules_sha256": self.rules_sha256,
            "dataset_kind": self.config.dataset_kind,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "bench4hls_frozen_epoch": (
                self.config.bench4hls_frozen_epoch
                if self.config.dataset_kind == DATASET_BENCH4HLS
                else None
            ),
            "hls_eval_source_sha256": self.hls_eval_source_sha256,
            "vitis_executable_sha256": self.vitis_executable_sha256,
            "fpga_part": self.config.fpga_part,
            "clock_ns": self.config.clock_ns,
            "python_environment_sha256": self.python_environment["manifest_sha256"],
            "deployment_manifest_sha256": self.vllm_run_metadata_sha256,
            "deployment_binding_sha256": (
                file_sha256(self.config.deployment_binding)
                if self.config.deployment_binding else None
            ),
            "deployment_epoch": self.deployment_epoch,
            "target_lock": target_binding,
            "causal_replay_source_manifest_sha256": (
                self.causal_replay_source.manifest_sha256
                if self.causal_replay_source is not None
                else None
            ),
        }
        value["protocol_digest"] = canonical_sha256(value)
        return value

    def _validate_freeze_manifest(self, formal_run: bool) -> dict[str, Any] | None:
        path = self.config.freeze_manifest
        if path is None:
            if formal_run:
                raise ValueError("Formal DAC2027 runs require --freeze-manifest")
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError("Freeze manifest status is not pass")
        expected_rule_key = (
            "all_rules_sha256"
            if self.config.mode.uses_all_rules
            else "core_sha256"
            if self.config.mode.uses_core_rules
            else "primary_sha256"
        )
        fixed = payload.get("fixed_protocol") or {}
        checks = {
            "iteration_id": (payload.get("iteration_id"), self.config.iteration_id),
            "source_tree_sha256": (payload.get("source_tree_sha256"), self.source_hash),
            "rules_sha256": ((payload.get("ercl") or {}).get(expected_rule_key), self.rules_sha256),
            "dataset_kind": (
                (payload.get("dataset") or {}).get("kind"),
                self.config.dataset_kind,
            ),
            "dataset_case_count": (
                (payload.get("dataset") or {}).get("case_count"),
                self.config.expected_case_count,
            ),
            "dataset_manifest_sha256": ((payload.get("dataset") or {}).get("manifest_sha256"), self.dataset_manifest_sha256),
            "bench4hls_frozen_epoch": (
                fixed.get("bench4hls_frozen_epoch"),
                (
                    self.config.bench4hls_frozen_epoch
                    if self.config.dataset_kind == DATASET_BENCH4HLS
                    else None
                ),
            ),
            "hls_eval_source_sha256": ((payload.get("hls_eval") or {}).get("source_tree_sha256"), self.hls_eval_source_sha256),
            "hls_eval_git_commit": ((payload.get("hls_eval") or {}).get("git_commit"), self.hls_eval_source_manifest.get("git_commit")),
            "hls_eval_git_status_sha256": ((payload.get("hls_eval") or {}).get("git_status_sha256"), self.hls_eval_source_manifest.get("git_status_sha256")),
            "hls_eval_git_diff_sha256": ((payload.get("hls_eval") or {}).get("git_diff_sha256"), self.hls_eval_source_manifest.get("git_diff_sha256")),
            "python_environment_manifest_sha256": ((payload.get("python_environment") or {}).get("manifest_sha256"), self.python_environment.get("manifest_sha256")),
            "served_model": (fixed.get("served_model"), self.config.model),
            "context_limit": (fixed.get("context_limit"), self.config.context_limit),
            "samples": (fixed.get("samples"), self.config.samples),
            "seeds": (fixed.get("seeds"), list(self.config.seeds)),
            "temperature": (fixed.get("temperature"), self.config.temperature),
            "max_repair_rounds": (fixed.get("max_repair_rounds"), self.config.max_repair_rounds),
            "max_tokens": (fixed.get("max_tokens"), self.config.max_tokens),
            "modes": (
                fixed.get("modes"),
                [
                    item.value
                    for item in (
                        FORMAL_RUN_MODES
                        if self.config.dataset_kind == DATASET_BENCH4HLS
                        else HLS_EVAL_VERIFICATION_MODES
                    )
                ],
            ),
            "fpga_part": (fixed.get("fpga_part"), self.config.fpga_part),
            "clock_ns": (float(fixed.get("clock_ns", -1)), float(self.config.clock_ns)),
            "workers": (fixed.get("workers"), {
                "design": self.config.design_workers,
                "llm": self.config.llm_workers,
                "csim": self.config.csim_workers,
                "synth": self.config.synth_workers,
            }),
            "timeouts_seconds": (fixed.get("timeouts_seconds"), {
                "llm": self.config.llm_timeout,
                "csim": self.config.csim_timeout,
                "synth": self.config.synth_timeout,
            }),
        }
        mismatches = {
            key: {"frozen": frozen, "current": current}
            for key, (frozen, current) in checks.items()
            if frozen != current
        }
        vitis = payload.get("vitis") or {}
        if vitis.get("executable_sha256") != self.vitis_executable_sha256:
            mismatches["vitis_executable_sha256"] = {
                "frozen": vitis.get("executable_sha256"),
                "current": self.vitis_executable_sha256,
            }
        if vitis.get("version_output") != self.vitis_version_output:
            mismatches["vitis_version_output"] = {
                "frozen": vitis.get("version_output"),
                "current": self.vitis_version_output,
            }
        tests = payload.get("tests") or {}
        if formal_run and not (
            tests.get("unit_status") == "pass"
            and tests.get("vitis_smoke_status") == "pass"
            and tests.get("development_gate_status") == "pass"
            and tests.get("reference_kernel_read_or_prompted") is False
        ):
            mismatches["tests"] = {"frozen": tests, "current": "required passing evidence"}
        if mismatches:
            raise ValueError(
                "Frozen DAC2027 protocol mismatch; create a new development iteration: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return payload
    def _validate_deployment_binding(self, formal_run: bool) -> dict[str, Any] | None:
        path = self.config.deployment_binding
        if path is None:
            if formal_run:
                raise ValueError("Formal dataset runs require --deployment-binding")
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        deployment = self.vllm_deployment_manifest or {}
        expected = {
            "status": "bound-before-first-trajectory",
            "iteration_id": self.config.iteration_id,
            "manifest_path": str(self.config.vllm_run_metadata.resolve()),
            "manifest_sha256": file_sha256(self.config.vllm_run_metadata),
            "endpoint": self.config.base_url.rstrip("/"),
            "inference_server": self.config.inference_server,
            "served_model": self.config.model,
            "remote_hostname": deployment.get("remote_hostname"),
            "run_id": deployment.get("run_id"),
            "deployment_epoch": deployment.get("deployment_epoch"),
            "model_listing_created_at_capture": int(
                deployment.get("model_listing_created_at_capture", -1)
            ),
            "prometheus_process_start_time_seconds": deployment.get(
                "prometheus_process_start_time_seconds"
            ),
            "model_artifact_manifest_sha256": deployment.get(
                "model_artifact_manifest_sha256"
            ),
            "process_identity_sha256": deployment.get("process_identity_sha256"),
            "selected_gpu": int(deployment.get("selected_gpu", -1)),
            "selected_gpu_uuid": deployment.get("selected_gpu_uuid"),
            "tensor_parallel_size": 1,
            "conda_environment": deployment.get("conda_environment"),
            "model_path": deployment.get("model_path"),
            "launch_command": deployment.get("launch_command"),
        }
        mismatches = {
            key: {"bound": payload.get(key), "current": value}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Iteration deployment binding mismatch; create a new vN: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return payload

    def _config_hash(self) -> str:
        round0_provenance = (
            _round0_source_manifest(self.config.round0_source)
            if self.config.round0_source is not None
            else None
        )
        if (
            round0_provenance is not None
            and self.config.expected_case_count is not None
            and round0_provenance["trajectory_count"]
            != self.config.expected_case_count * self.config.samples
        ):
            raise ValueError(
                "Round-0 source count does not match the frozen paired design: "
                f"{round0_provenance['trajectory_count']}"
            )
        if (
            round0_provenance is not None
            and self.config.expected_case_count is not None
        ):
            errors_path = self.config.round0_source / "run_errors.json"
            if not errors_path.is_file() or json.loads(
                errors_path.read_text(encoding="utf-8")
            ) != []:
                raise ValueError(
                    "Formal Round-0 source requires an empty run_errors.json"
                )
        value = {
            "mode": self.config.mode.value,
            "features": self.config.mode.feature_flags,
            "protocol_digest": self.protocol_digest,
            "protocol_manifest": self.protocol_manifest,
            "dataset_kind": self.config.dataset_kind,
            "dataset_root": str(self.config.dataset_root.resolve()),
            "bench4hls_frozen_epoch": (
                self.config.bench4hls_frozen_epoch
                if self.config.dataset_kind == DATASET_BENCH4HLS
                else None
            ),
            "model": self.config.model,
            "inference_server": self.config.inference_server,
            "samples": self.config.samples,
            "temperature": self.config.temperature,
            "seeds": list(self.config.seeds),
            "max_repair_rounds": self.config.max_repair_rounds,
            "max_tokens": self.config.max_tokens,
            "context_limit": self.config.context_limit,
            "expected_case_count": self.config.expected_case_count,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "hls_eval_source_sha256": self.hls_eval_source_sha256,
            "hls_eval_git_commit": self.hls_eval_source_manifest.get("git_commit"),
            "hls_eval_git_status_sha256": self.hls_eval_source_manifest.get("git_status_sha256"),
            "hls_eval_git_diff_sha256": self.hls_eval_source_manifest.get("git_diff_sha256"),
            "freeze_manifest": (
                {
                    "path": str(self.config.freeze_manifest.resolve()),
                    "sha256": file_sha256(self.config.freeze_manifest),
                }
                if self.config.freeze_manifest
                else None
            ),
            "deployment_binding": (
                {
                    "path": str(self.config.deployment_binding.resolve()),
                    "sha256": file_sha256(self.config.deployment_binding),
                }
                if self.config.deployment_binding
                else None
            ),
            "vllm_run_metadata": (
                {
                    "path": str(self.config.vllm_run_metadata.resolve()),
                    "sha256": file_sha256(self.config.vllm_run_metadata),
                }
                if self.config.vllm_run_metadata
                else None
            ),
            "vitis_dir": str(self.config.vitis_dir.resolve()),
            "fpga_part": self.config.fpga_part,
            "clock_ns": self.config.clock_ns,
            "timeouts": {
                "llm": self.config.llm_timeout,
                "csim": self.config.csim_timeout,
                "synth": self.config.synth_timeout,
            },
            "review_infrastructure_retries": (
                self.config.review_infrastructure_retries
            ),
            "round0_source": round0_provenance,
            "causal_replay_source_manifest_sha256": (
                self.causal_replay_source.manifest_sha256
                if self.causal_replay_source is not None
                else None
            ),
            "workers": {
                "design": self.config.design_workers,
                "llm": self.config.llm_workers,
                "csim": self.config.csim_workers,
                "synth": self.config.synth_workers,
            },
            "iteration_id": self.config.iteration_id,
            "source_sha256": self.source_hash,
            "rules_sha256": (
                file_sha256(self.config.rules_path)
                if self.config.mode.uses_ercl
                else None
            ),
        }
        return _sha256_text(json.dumps(value, sort_keys=True))

    def _validate_end_of_mode_provenance(self) -> dict[str, Any]:
        """Recheck every frozen dependency after the long-running mode completes."""
        current_dataset, current_dataset_sha = _dataset_manifest(
            self.all_cases if self.config.formal_protocol else self.cases
        )
        current_hls_eval, current_hls_eval_sha = _hls_eval_source_manifest(
            self.config.hls_eval_root
        )
        current_python_environment = python_environment_manifest()
        target_path = self.config.output_dir / "protocol" / "locked_targets.json"
        expected_target_sha = (self.protocol_manifest.get("target_lock") or {}).get(
            "sha256"
        )
        current_target_sha = (
            file_sha256(target_path)
            if expected_target_sha is not None and target_path.is_file()
            else None
        )
        actual = {
            "source_tree_sha256": _tree_sha256(self.config.repo_root),
            "rules_sha256": file_sha256(self.config.rules_path),
            "dataset_manifest_sha256": current_dataset_sha,
            "hls_eval_source_sha256": current_hls_eval_sha,
            "hls_eval_git_status_sha256": current_hls_eval.get("git_status_sha256"),
            "hls_eval_git_diff_sha256": current_hls_eval.get("git_diff_sha256"),
            "vitis_executable_sha256": file_sha256(
                self.config.vitis_dir / "bin" / "vitis_hls"
            ),
            "vllm_run_metadata_sha256": (
                file_sha256(self.config.vllm_run_metadata)
                if self.config.vllm_run_metadata is not None
                else None
            ),
            "python_environment_sha256": current_python_environment.get(
                "manifest_sha256"
            ),
            "target_lock_sha256": current_target_sha,
            "causal_replay_source_manifest_sha256": (
                build_full_source_manifest(self.config.causal_replay_source).get(
                    "manifest_sha256"
                )
                if self.config.causal_replay_source is not None
                else None
            ),
        }
        expected = {
            "source_tree_sha256": self.source_hash,
            "rules_sha256": self.rules_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "hls_eval_source_sha256": self.hls_eval_source_sha256,
            "hls_eval_git_status_sha256": self.hls_eval_source_manifest.get(
                "git_status_sha256"
            ),
            "hls_eval_git_diff_sha256": self.hls_eval_source_manifest.get(
                "git_diff_sha256"
            ),
            "vitis_executable_sha256": self.vitis_executable_sha256,
            "vllm_run_metadata_sha256": self.vllm_run_metadata_sha256,
            "python_environment_sha256": self.python_environment.get(
                "manifest_sha256"
            ),
            "target_lock_sha256": expected_target_sha,
            "causal_replay_source_manifest_sha256": (
                self.causal_replay_source.manifest_sha256
                if self.causal_replay_source is not None
                else None
            ),
        }
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        if mismatches:
            raise RuntimeError(
                "End-of-mode frozen provenance changed: "
                + json.dumps(mismatches, sort_keys=True)
            )
        live_deployment_identity: dict[str, Any] | None = None
        if self.config.vllm_run_metadata is not None:
            deployment = _load_vllm_deployment_manifest(
                self.config.vllm_run_metadata,
                base_url=self.config.base_url,
                model=self.config.model,
                inference_server=self.config.inference_server,
                formal_run=True,
            ) or {}
            live_deployment_identity = self.llm.deployment_identity()
            expected_live_identity = {
                "prometheus_process_start_time_seconds": deployment.get(
                    "prometheus_process_start_time_seconds"
                ),
                "model_root": deployment.get("model_path"),
                "model_max_len": self.config.context_limit,
                "deployment_epoch": deployment.get("deployment_epoch"),
            }
            if live_deployment_identity != expected_live_identity:
                raise RuntimeError(
                    "End-of-mode live deployment epoch changed: "
                    + json.dumps(
                        {
                            "expected": expected_live_identity,
                            "actual": live_deployment_identity,
                        },
                        sort_keys=True,
                    )
                )
        return {
            "validated": True,
            "reference_kernel_read": False,
            "actual": actual,
            "live_deployment_identity": live_deployment_identity,
            "dataset_case_count": current_dataset.get("case_count"),
        }

    def close(self) -> None:
        self.pools.shutdown()

    def _all_cases(self) -> list[Any]:
        return load_cases(
            self.config.dataset_kind,
            self.config.dataset_root,
            self.BenchmarkCase,
            self.find_benchmark_case_dirs,
        )

    def _cases(self, all_cases: list[Any]) -> list[Any]:
        cases = list(all_cases)
        if self.config.case_names:
            requested = set(self.config.case_names)
            cases = [case for case in cases if case.name in requested]
            missing = requested - {case.name for case in cases}
            if missing:
                raise ValueError(f"Requested cases were not found: {sorted(missing)}")
        if self.config.limit is not None:
            cases = cases[: self.config.limit]
        return cases

    def _metadata(self, case_count: int, health: dict[str, Any]) -> dict[str, Any]:
        config_dict = asdict(self.config)
        config_dict["mode"] = self.config.mode.value
        for key, value in list(config_dict.items()):
            if isinstance(value, Path):
                config_dict[key] = str(value.resolve())
            elif isinstance(value, tuple):
                config_dict[key] = list(value)
        rules_metadata = {
            "enabled": self.config.mode.uses_ercl,
            "path": str(self.config.rules_path.resolve()),
            "sha256": file_sha256(self.config.rules_path),
            "version": self.ercl.version if self.ercl else None,
            "count": len(self.ercl.rules) if self.ercl else 0,
            "sources": self.ercl.sources if self.ercl else [],
            "retrieval_policy": (
                self.ercl.retrieval_policy if self.ercl else None
            ),
        }
        return {
            "started_at_unix": time.time(),
            "case_count": case_count,
            "iteration_id": self.config.iteration_id,
            "test_informed_exploratory": self.config.exploratory,
            "config_hash": self.config_hash,
            "source_sha256": self.source_hash,
            "protocol_digest": self.protocol_digest,
            "protocol_manifest": self.protocol_manifest,
            "python_environment": self.python_environment,
            "feature_flags": self.config.mode.feature_flags,
            "config": config_dict,
            "rules": rules_metadata,
            "rfl": {
                "schema_version": RFL_SCHEMA_VERSION,
                "max_entries": 5,
                "isolation": "one task/sample/seed trajectory",
                "cross_trajectory_sharing": False,
            },
            "causal_replay": (
                {
                    "enabled": True,
                    "source_root": str(self.causal_replay_source.root),
                    "source_mode": RunMode.MEMENTOHLS.value,
                    "version": self.causal_replay_source.manifest["version"],
                    "trajectory_count": self.causal_replay_source.manifest[
                        "trajectory_count"
                    ],
                    "manifest_sha256": self.causal_replay_source.manifest_sha256,
                }
                if self.causal_replay_source is not None
                else {"enabled": False}
            ),
            "dataset_manifest": {
                "sha256": self.dataset_manifest_sha256,
                "path": str((self.run_root / "dataset_manifest.json").resolve()),
                "reference_kernel_excluded": True,
            },
            "hls_eval_source": self.hls_eval_source_manifest,
            "freeze_manifest": (
                {
                    "path": str(self.config.freeze_manifest.resolve()),
                    "sha256": file_sha256(self.config.freeze_manifest),
                    "validated": True,
                }
                if self.config.freeze_manifest
                else None
            ),
            "deployment_binding": (
                {
                    "path": str(self.config.deployment_binding.resolve()),
                    "sha256": file_sha256(self.config.deployment_binding),
                    "validated": True,
                    "value": self.deployment_binding,
                }
                if self.config.deployment_binding
                else None
            ),
            "vllm": {
                **health,
                "context_limit": self.config.context_limit,
                "deployment_epoch": self.deployment_epoch,
                "run_metadata": (
                    {
                        "path": str(self.config.vllm_run_metadata.resolve()),
                        "sha256": file_sha256(self.config.vllm_run_metadata),
                    }
                    if self.config.vllm_run_metadata
                    else None
                ),
                "deployment_manifest": self.vllm_deployment_manifest,
                "launch_command": (
                    self.vllm_deployment_manifest.get("launch_command")
                    if self.vllm_deployment_manifest
                    else None
                ),
            },
            "vitis": {
                "directory": str(self.config.vitis_dir.resolve()),
                "version": self.config.vitis_dir.name,
                "version_output": self.vitis_version_output,
                "executable_sha256": file_sha256(
                    self.config.vitis_dir / "bin" / "vitis_hls"
                ),
                "fpga_part": self.config.fpga_part,
                "clock_ns": self.config.clock_ns,
                "flow_target": "vivado",
                "csim_total_timeout_seconds": self.config.csim_timeout,
                "csim_phase_timeout_seconds": self.config.csim_timeout / 2.0,
            },
            "implementation_hashes": {
                str(path.relative_to(self.config.repo_root)): file_sha256(path)
                for path in [
                    self.config.repo_root / "main.py",
                    *sorted((self.config.repo_root / "src" / "mementohls").glob("*.py")),
                    self.config.rules_path,
                ]
            },
            "repositories": {
                "mementohls": _git_metadata(self.config.repo_root),
                "hls_eval": _git_metadata(self.config.hls_eval_root),
            },
        }

    def _trajectory_path(self, case_name: str, sample_index: int) -> Path:
        return (
            self.run_root
            / case_name
            / f"sample__{sample_index}"
            / "trajectory.json"
        )

    def _load_reused_round0(
        self,
        case_name: str,
        sample_index: int,
        seed: int,
        prompt_sha256: str,
    ) -> tuple[LLMCallResult, str]:
        if self.config.round0_source is None:
            raise RuntimeError("round0 source was not configured")
        source = (
            self.config.round0_source
            / case_name
            / f"sample__{sample_index}"
            / "trajectory.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or not payload.get("rounds"):
            raise ValueError(f"Round-0 source is incomplete: {source}")
        expected_identity = {
            "mode": RunMode.ZERO_SHOT.value,
            "iteration_id": self.config.iteration_id,
            "benchmark_case_name": case_name,
            "sample_index": sample_index,
            "seed": seed,
            "source_sha256": self.source_hash,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_kind": self.config.dataset_kind,
            "hls_eval_source_sha256": self.hls_eval_source_sha256,
        }
        identity_mismatches = {
            key: {"source": payload.get(key), "expected": expected}
            for key, expected in expected_identity.items()
            if payload.get(key) != expected
        }
        if identity_mismatches:
            raise ValueError(
                f"Round-0 source identity mismatch: {source}: "
                + json.dumps(identity_mismatches, sort_keys=True)
            )
        round0 = payload["rounds"][0]
        if round0.get("seed_prompt_sha256") != prompt_sha256:
            raise ValueError(f"Round-0 prompt hash mismatch: {source}")
        prompt_provenance = round0.get("prompt_provenance")
        if not isinstance(prompt_provenance, dict):
            raise ValueError(
                f"Round-0 source prompt provenance is missing: {source}"
            )
        if prompt_provenance.get("user_prompt_sha256") != prompt_sha256:
            raise ValueError(
                f"Round-0 source user prompt hash mismatch: {source}"
            )
        llm_payload = round0.get("llm") or {}
        text = llm_payload.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Round-0 source has no LLM text: {source}")
        if round0.get("response_sha256") != _sha256_text(text):
            raise ValueError(f"Round-0 response hash mismatch: {source}")
        required_provenance = (
            "deployment_epoch",
            "prompt_sha256",
            "request_seed",
            "request_temperature",
            "finish_reason",
            "attempt_history",
        )
        missing = [
            field
            for field in required_provenance
            if llm_payload.get(field) is None
        ]
        if missing:
            raise ValueError(
                f"Round-0 source LLM provenance is incomplete: {source}: "
                + ", ".join(missing)
            )
        if (
            llm_payload.get("prompt_sha256")
            != prompt_provenance.get("message_prompt_sha256")
        ):
            raise ValueError(
                f"Round-0 source message prompt hash mismatch: {source}"
            )
        attempts = int(llm_payload.get("attempts") or 0)
        attempt_history = llm_payload.get("attempt_history")
        if (
            attempts < 1
            or not isinstance(attempt_history, list)
            or len(attempt_history) != attempts
        ):
            raise ValueError(
                f"Round-0 source LLM attempt history is inconsistent: {source}"
            )
        if llm_payload.get("request_seed") != seed:
            raise ValueError(f"Round-0 source LLM seed mismatch: {source}")
        if llm_payload.get("request_temperature") != self.config.temperature:
            raise ValueError(
                f"Round-0 source LLM temperature mismatch: {source}"
            )
        source_physical_call = llm_payload.get("physical_call")
        if source_physical_call is not True:
            raise ValueError(
                f"Round-0 source was not a physical LLM call: {source}"
            )
        call = LLMCallResult(
            text=text,
            raw_response=llm_payload.get("raw_response") or {},
            prompt_tokens=llm_payload.get("prompt_tokens"),
            completion_tokens=llm_payload.get("completion_tokens"),
            total_tokens=llm_payload.get("total_tokens"),
            elapsed_seconds=float(llm_payload.get("elapsed_seconds") or 0.0),
            attempts=attempts,
            prompt_variant_index=int(
                llm_payload.get("prompt_variant_index") or 0
            ),
            effective_user_prompt=llm_payload.get("effective_user_prompt"),
            request_seed=llm_payload.get("request_seed"),
            request_temperature=llm_payload.get("request_temperature"),
            requested_max_tokens=llm_payload.get("requested_max_tokens"),
            effective_max_tokens=llm_payload.get("effective_max_tokens"),
            finish_reason=llm_payload.get("finish_reason"),
            message_roles=list(llm_payload.get("message_roles") or []),
            prompt_sha256=llm_payload.get("prompt_sha256"),
            attempt_history=list(attempt_history),
            deployment_epoch=llm_payload.get("deployment_epoch"),
            physical_call=False,
        )
        return call, str(source)

    @staticmethod
    def _header_bundle(case: Any) -> str:
        # ERCL consumes either the immutable HLS-Eval header or the public
        # Bench4HLS instruction's interface contract. Golden TB code is hidden.
        return case_interface_contract(case)

    @staticmethod
    def _evidence_log(review: ReviewResult, raw_response: str) -> str:
        log = failure_log(review)
        if review.failure_stage == FailureStage.PARSE.value:
            log = f"{log}\n\nRaw model response:\n{raw_response}"
        return log

    @staticmethod
    def _compiler_error_symbols(log: str) -> set[str]:
        patterns = (
            r"(?:unknown type name|use of undeclared identifier)\s+[\'`‘’\"]?([A-Za-z_]\w*)",
            r"[\'`‘’\"]([A-Za-z_]\w*)[\'`‘’\"]\s+(?:was not declared|does not name a type)",
            r"(?:no member named|unknown identifier)\s+[\'`‘’\"]?([A-Za-z_]\w*)",
        )
        return {
            match.group(1)
            for pattern in patterns
            for match in re.finditer(pattern, log, flags=re.IGNORECASE)
        }

    def _matched_rules_with_provenance(
        self,
        *,
        review: ReviewResult,
        log: str,
        code: str,
        header: str,
        header_name: str,
        history: str,
        task: str = "",
        facts: str = "",
        veto_rule_ids: list[str] | None = None,
    ) -> RetrievalResult:
        if self.ercl is None:
            return RetrievalResult(
                rules=[],
                provenance={
                    "schema_version": "2",
                    "enabled": False,
                    "stage": review.failure_stage,
                    "selected_rule_ids": [],
                    "routing_decisions": [],
                },
            )
        # Re-derive bounded facts with the exact immutable header basename. This
        # makes action eligibility part of retrieval before ranking/top-k, rather
        # than deleting an unsafe top-1 rule after it has displaced a valid rule.
        retrieval_facts = derive_memory_facts(
            stage=review.failure_stage,
            log=log,
            code=code,
            header=header,
            task=task,
            header_name=header_name,
        )
        return self.ercl.match_with_provenance(
            review.failure_stage,
            log,
            code,
            header=header,
            history=history,
            task=task,
            facts=retrieval_facts,
            exclude_rule_ids=tuple(veto_rule_ids or ()),
        )

    def _matched_rules(self, **kwargs: Any) -> list[MemoryRule]:
        """Backward-compatible selected-rule interface for focused tests."""
        return self._matched_rules_with_provenance(**kwargs).rules

    def _review_candidate(
        self,
        case: Any,
        response_text: str,
        round_dir: Path,
        build_name: str,
    ) -> ReviewResult:
        errors: list[dict[str, Any]] = []
        attempts = self.config.review_infrastructure_retries + 1
        for attempt in range(1, attempts + 1):
            review = self.reviewer.review(
                case,
                response_text,
                round_dir / f"review_attempt__{attempt}",
                f"{build_name}__a{attempt}",
            )
            review.infrastructure_attempts = attempt
            if not review.infrastructure_error:
                review.infrastructure_retry_history = list(errors)
                if errors:
                    atomic_write_json(
                        round_dir / "infrastructure_retries.json", errors
                    )
                return review
            errors.append(
                {
                    "attempt": attempt,
                    "error": review.infrastructure_error,
                    "review": review.to_dict(),
                }
            )
        atomic_write_json(round_dir / "infrastructure_retries.json", errors)
        raise RuntimeError(
            f"Reviewer infrastructure failed after {attempts} attempts: "
            f"{errors[-1]['error']}"
        )

    @staticmethod
    def _normalized_top_signature(text: str, top_fn: str) -> str | None:
        pattern = re.compile(
            rf"(?ms)^[ \t]*(?:extern\s+\"C\"\s+)?"
            rf"([^#;{{}}]*?\b{re.escape(top_fn)}\s*\([^;{{}}]*\))"
            rf"\s*(?:;|\{{)"
        )
        match = pattern.search(text)
        if not match:
            return None
        value = re.sub(r"\s+", "", match.group(1))
        return value.replace('extern"C"', "")

    def _filename_telemetry(
        self,
        case: Any,
        parent_review: ReviewResult | None,
        review: ReviewResult,
    ) -> dict[str, Any]:
        """Record transport identity without treating a safe alias as an edit."""

        expected_filename = self.reviewer.infer_expected_filename(case)
        parent_filename = (
            parent_review.generated_filename
            if parent_review is not None and parent_review.pass_parse
            else None
        )
        candidate_filename = (
            review.generated_filename if review.pass_parse else None
        )
        changed_from_parent = bool(
            parent_filename
            and candidate_filename
            and candidate_filename != parent_filename
        )
        return {
            "inferred_expected_filename": expected_filename,
            "parent_generated_filename": parent_filename,
            "candidate_generated_filename": candidate_filename,
            "parent_filename_available": parent_filename is not None,
            "changed_from_parent": changed_from_parent,
            "candidate_matches_inferred": candidate_filename == expected_filename,
            "inherited_safe_alias": bool(
                parent_filename
                and candidate_filename == parent_filename
                and candidate_filename != expected_filename
            ),
            "credit_blocking": changed_from_parent,
        }

    def _invariant_violations(
        self,
        case: Any,
        parent_review: ReviewResult | None,
        review: ReviewResult,
    ) -> list[str]:
        violations: list[str] = []
        filename = self._filename_telemetry(case, parent_review, review)
        if filename["changed_from_parent"]:
            violations.append(
                "generated_filename_changed:"
                f"{filename['candidate_generated_filename']}!="
                f"{filename['parent_generated_filename']}"
            )
        code = review.generated_code or ""
        if case.top_fn and not re.search(rf"\b{re.escape(case.top_fn)}\s*\(", code):
            violations.append("top_function_missing")
        if case.top_fn != "main" and re.search(
            r"\b(?:int|void|bool)\s+main\s*\(", code
        ):
            violations.append("illegal_main_present")
        header_name = case_header_name(case)
        if header_name and not re.search(
            rf'#\s*include\s*"{re.escape(header_name)}"', code
        ):
            violations.append("exact_project_header_include_missing")
        if case.top_fn:
            header_signature = self._normalized_top_signature(
                case_interface_contract(case), case.top_fn
            )
            code_signature = self._normalized_top_signature(code, case.top_fn)
            if header_signature and code_signature and header_signature != code_signature:
                violations.append("top_function_signature_changed")
        return violations

    def _run_trajectory(
        self,
        case: Any,
        sample_index: int,
        seed: int,
    ) -> dict[str, Any]:
        trajectory_path = self._trajectory_path(case.name, sample_index)
        replay_plan = (
            self.causal_replay_source.plan_for(
                case_name=case.name,
                sample_index=sample_index,
                seed=seed,
                control_mode=self.config.mode,
            )
            if self.causal_replay_source is not None
            else None
        )
        if self.config.resume:
            existing = load_complete_trajectory(
                trajectory_path,
                expected_config_hash=self.config_hash,
                expected_protocol_digest=self.protocol_digest,
                expected_mode=self.config.mode.value,
                expected_case_name=case.name,
                expected_sample_index=sample_index,
                expected_seed=seed,
                expected_rfl_schema=(
                    RFL_SCHEMA_VERSION
                    if self.config.mode.uses_rfl
                    else None
                ),
            )
            if existing is not None:
                return existing

        trajectory_dir = trajectory_path.parent
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        seed_prompt = build_seed_prompt(
            case,
            self.config.dataset_kind,
            self.hls_eval_seed_prompt_builder,
        )
        seed_prompt_sha = _sha256_text(seed_prompt)
        atomic_write_text(trajectory_dir / "seed_prompt.txt", seed_prompt)
        payload: dict[str, Any] = {
            "status": "running",
            "terminal": False,
            "test_informed_exploratory": self.config.exploratory,
            "iteration_id": self.config.iteration_id,
            "config_hash": self.config_hash,
            "source_sha256": self.source_hash,
            "protocol_digest": self.protocol_digest,
            "deployment_epoch": self.deployment_epoch,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "hls_eval_source_sha256": self.hls_eval_source_sha256,
            "feature_flags": self.config.mode.feature_flags,
            "dataset_kind": self.config.dataset_kind,
            "short_policy_config": RFLPolicyConfig.for_mode(
                self.config.mode
            ).to_dict(),
            "mode": self.config.mode.value,
            "benchmark_case_name": case.name,
            "benchmark_case_tags": case.tags_all,
            "sample_index": sample_index,
            "seed": seed,
            "seed_prompt_sha256": seed_prompt_sha,
            "immutable_header_sha256": {
                path.name: file_sha256(path) for path in case.h_files
            },
            "causal_replay": replay_plan.provenance() if replay_plan else None,
            "rounds": [],
        }
        atomic_write_json(trajectory_path, payload)
        trajectory_key = TrajectoryKey(
            benchmark_case_name=case.name,
            sample_index=sample_index,
            seed=seed,
            mode=self.config.mode.value,
        )
        rfl = (
            ReviewerGroundedFalsificationLedger(max_entries=5, trajectory_key=trajectory_key)
            if self.config.mode.uses_rfl
            else None
        )
        if rfl is not None:
            payload["trajectory_key"] = asdict(trajectory_key)
            payload["trajectory_key_sha256"] = trajectory_key.sha256
            atomic_write_json(trajectory_path, payload)
        header_text = self._header_bundle(case)
        header_name = case_header_name(case)
        try:
            reused_from: str | None = None
            if self.config.round0_source is not None:
                llm_call, reused_from = self._load_reused_round0(
                    case.name, sample_index, seed, seed_prompt_sha
                )
            else:
                llm_call = self.llm.chat(
                    [{"role": "user", "content": seed_prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=seed,
                    minimum_completion_tokens=2500,
                )
            response_text = llm_call.text
            round0_dir = trajectory_dir / "round__0"
            current_review = self._review_candidate(
                case,
                response_text,
                round0_dir,
                f"{case.name}__s{sample_index}__r0",
            )
            candidate_id = (
                f"{self.config.iteration_id}:{self.config.mode.value}:"
                f"{case.name}:s{sample_index}:r0"
            )
            round0_payload = {
                "round_index": 0,
                "candidate_id": candidate_id,
                "trajectory_key_sha256": (
                    trajectory_key.sha256 if rfl is not None else None
                ),
                "protocol_digest": self.protocol_digest,
                "deployment_epoch": self.deployment_epoch,
                "parent_candidate_id": None,
                "action_origin": "round0",
                "seed_prompt_sha256": seed_prompt_sha,
                "prompt_provenance": {
                    "kind": "round0_generation",
                    "user_prompt_sha256": seed_prompt_sha,
                    "message_prompt_sha256": llm_call.prompt_sha256,
                    "message_roles": llm_call.message_roles,
                    "source_roles": (
                        ["kernel_description"]
                        if self.config.dataset_kind == DATASET_BENCH4HLS
                        else [
                            "kernel_description",
                            "immutable_testbench",
                            "immutable_project_header",
                        ]
                    ),
                    "reference_kernel_read_or_prompted": False,
                },
                "response_sha256": _sha256_text(response_text),
                "code_sha256": (
                    _sha256_text(current_review.generated_code)
                    if current_review.generated_code is not None
                    else None
                ),
                "llm": _llm_to_dict(llm_call, reused_from),
                "filename_telemetry": self._filename_telemetry(
                    case, None, current_review
                ),
                "invariant_violations": [],
                "review": current_review.to_dict(),
                "stage_vector": stage_vector(current_review),
                "diagnosis": None,
                "retrieved_rule_ids": [],
                "selected_rule_ids": [],
                "action_rule_ids": [],
                "matched_rule_ids": [],
                "matched_rules": [],
                "selected_rules": [],
                "action_rules": [],
                "operator": None,
                "cleanroom": {
                    "eligible": False,
                    "requested": False,
                    "triggered": False,
                    "authorization": "not_requested",
                    "authorization_rule_ids": [],
                    "reason": None,
                },
            }
            payload["rounds"].append(round0_payload)
            atomic_write_json(round0_dir / "round.json", round0_payload)
            atomic_write_json(trajectory_path, payload)

            if rfl is not None:
                rfl.observe_round0(
                    candidate_id=candidate_id,
                    code=current_review.generated_code or response_text,
                    review=current_review,
                    tool_log=self._evidence_log(current_review, response_text),
                    total_tokens=int(llm_call.total_tokens or 0),
                )

            if self.config.mode.allows_repair and not current_review.pass_tb_and_synth:
                for repair_round in range(1, self.config.max_repair_rounds + 1):
                    round_dir = trajectory_dir / f"round__{repair_round}"
                    round_dir.mkdir(parents=True, exist_ok=True)
                    # Every repair-capable mode shares the exact Feedback
                    # controller in round 1. Memory starts only after that
                    # candidate has been fully reviewed and archived.
                    memory_active = memory_active_for_round(
                        self.config.mode,
                        repair_round,
                        rfl_no_progress_streak=(
                            rfl.no_progress_streak if rfl is not None else 0
                        ),
                    )
                    rfl_feedback_fallback = bool(
                        self.config.mode is RunMode.RFL
                        and repair_round >= 2
                        and not memory_active
                    )
                    previous_candidate_id = candidate_id
                    previous_review = current_review
                    previous_response = response_text

                    # Snapshot the actual latest failed candidate before any
                    # frontier recovery. Clean-room eligibility and attribution
                    # must refer to this candidate, never to a recovered parent.
                    trigger_candidate_id = candidate_id
                    trigger_review = current_review
                    trigger_response = response_text
                    trigger_code = trigger_review.generated_code or ""
                    trigger_log = self._evidence_log(
                        trigger_review, trigger_response
                    )
                    trigger_signature = (
                        rfl.current_signature if rfl else None
                    )
                    trigger_outcome = stage_vector(trigger_review)
                    trigger_normalized_failure = (
                        rfl.entries[-1].normalized_failure_after
                        if rfl and rfl.entries
                        else ""
                    )
                    cleanroom_eligible = False
                    cleanroom_reason: str | None = None
                    if rfl is not None and memory_active:
                        cleanroom_eligible, cleanroom_reason = (
                            rfl.should_cleanroom(repair_round)
                        )
                    trigger_cleanroom = (
                        cleanroom_eligible and self.config.mode.uses_cleanroom
                    )

                    recovery_candidate_id: str | None = None
                    recovery_reason: str | None = None
                    # A clean-room round deliberately starts from the immutable
                    # task, so archive recovery is mutually exclusive with it.
                    if rfl is not None and memory_active and not cleanroom_eligible:
                        recovery_candidate_id, recovery_reason = (
                            rfl.recovery_source(
                                candidate_id, current_review
                            )
                        )
                    if recovery_candidate_id is not None:
                        source_round = next(
                            value
                            for value in payload["rounds"]
                            if value["candidate_id"] == recovery_candidate_id
                        )
                        previous_candidate_id = recovery_candidate_id
                        previous_review = ReviewResult(**source_round["review"])
                        current_code = previous_review.generated_code or ""
                        previous_response = _wrap_code(
                            previous_review.generated_filename
                            or self.reviewer.infer_expected_filename(case),
                            current_code,
                        )
                        rfl.mark_recovery()
                    else:
                        current_code = trigger_code
                    current_log = (
                        trigger_log
                        if previous_candidate_id == trigger_candidate_id
                        else self._evidence_log(previous_review, previous_response)
                    )
                    active_rfl_ledger = (
                        rfl.activate_source(previous_candidate_id)
                        if rfl is not None and memory_active
                        else []
                    )
                    if replay_plan is not None and replay_plan.should_replay(
                        "diagnosis", repair_round
                    ):
                        actual_parent_round = next(
                            value
                            for value in payload["rounds"]
                            if value["candidate_id"] == previous_candidate_id
                        )
                        replay_plan.assert_parent(
                            repair_round,
                            parent_round_index=int(actual_parent_round["round_index"]),
                            response_sha256=str(actual_parent_round["response_sha256"]),
                            code_sha256=actual_parent_round.get("code_sha256"),
                            review=previous_review.to_dict(),
                        )
                    prompt_log = canonical_prompt_log(current_log)
                    diagnosis_model_contracts = (
                        [
                            rfl.build_model_input_contract(
                                role="diagnoser",
                                compact_level=level,
                                budget_chars=budget,
                                chosen_source_stage=previous_review.failure_stage,
                            )
                            for level, budget in enumerate(DIAGNOSIS_RFL_BUDGETS)
                        ]
                        if rfl is not None and memory_active
                        else []
                    )
                    if any(not item.feasible for item in diagnosis_model_contracts):
                        raise RuntimeError(
                            "A configured Diagnoser RFL profile is infeasible"
                        )
                    diagnosis_short_views = [
                        item.json_text for item in diagnosis_model_contracts
                    ]
                    history_match = (
                        _ercl_rfl_match_history(rfl, previous_review.failure_stage)
                        if memory_active
                        else ""
                    )
                    veto_rule_ids = (
                        rfl.ercl_veto_rule_ids(previous_review.failure_stage)
                        if (
                            rfl is not None
                            and memory_active
                            and self.config.mode.uses_veto
                        )
                        else []
                    )
                    memory_facts = (
                        derive_memory_facts(
                            stage=previous_review.failure_stage,
                            log=current_log,
                            code=current_code,
                            header=header_text,
                            task=seed_prompt,
                            header_name=header_name,
                        )
                        if self.ercl is not None and memory_active
                        else ""
                    )
                    if memory_active:
                        retrieval_result = self._matched_rules_with_provenance(
                            review=previous_review,
                            log=current_log,
                            code=current_code,
                            header=header_text,
                            header_name=header_name,
                            history=history_match,
                            task=seed_prompt,
                            facts=memory_facts,
                            veto_rule_ids=veto_rule_ids,
                        )
                    else:
                        retrieval_result = RetrievalResult(
                            rules=[],
                            provenance={
                                "schema_version": "3",
                                "enabled": False,
                                "stage": previous_review.failure_stage,
                                "selected_rule_ids": [],
                                "routing_decisions": [],
                                "abstain_reason": "feedback_warm_start_round",
                            },
                        )
                    retrieved_rules = retrieval_result.rules
                    atomic_write_json(
                        round_dir / "ercl_retrieval.json",
                        retrieval_result.provenance,
                    )
                    retrieval_policy = (
                        self.ercl.retrieval_policy
                        if self.ercl is not None and memory_active
                        else None
                    )

                    diagnosis_prompt_variants = [
                        build_diagnosis_prompt(
                            previous_review.failure_stage,
                            prompt_log,
                            current_code,
                            retrieved_rules,
                            self.config.mode.uses_ercl and memory_active,
                            rfl=(
                                diagnosis_short_views[level]
                                if diagnosis_short_views
                                else None
                            ),
                            retrieval_policy=retrieval_policy,
                            compact_level=level,
                        )
                        for level in range(DIAGNOSIS_PROFILE_COUNT)
                    ]
                    atomic_write_text(
                        round_dir / "diagnosis_prompt_requested.txt",
                        diagnosis_prompt_variants[0],
                    )
                    diagnosis_reused_from: str | None = None
                    if replay_plan is not None and replay_plan.should_replay(
                        "diagnosis", repair_round
                    ):
                        replayed = replay_plan.replay_call(
                            role="diagnosis",
                            round_index=repair_round,
                            prompt_variants=diagnosis_prompt_variants,
                        )
                        diagnosis_call = replayed.call
                        diagnosis_reused_from = replayed.reused_from
                    else:
                        diagnosis_call = self.llm.chat(
                            [
                                {"role": "system", "content": DIAGNOSER_SYSTEM},
                                {
                                    "role": "user",
                                    "content": diagnosis_prompt_variants[0],
                                },
                            ],
                            temperature=0.0,
                            max_tokens=1600,
                            seed=seed + repair_round * 10000 + 1,
                            minimum_completion_tokens=800,
                            user_prompt_variants=diagnosis_prompt_variants[1:],
                        )
                    diagnosis_prompt = (
                        diagnosis_call.effective_user_prompt
                        or diagnosis_prompt_variants[0]
                    )
                    atomic_write_text(
                        round_dir / "diagnosis_prompt.txt", diagnosis_prompt
                    )
                    atomic_write_text(
                        round_dir / "diagnosis_response.txt",
                        diagnosis_call.text,
                    )
                    raw_diagnosis = validate_diagnosis(
                        diagnosis_call.text,
                        previous_review.failure_stage,
                        retrieved_rules,
                        current_code=current_code,
                        header=header_text,
                        header_name=header_name,
                    )
                    if (
                        rfl is not None
                        and memory_active
                        and self.config.mode.uses_veto
                    ):
                        diagnosis, diagnosis_novelty_gate = rfl.apply_novelty_gate(
                            raw_diagnosis,
                            stage=previous_review.failure_stage,
                            action_type="local_llm",
                            operator_id=None,
                            rule_ids=list(raw_diagnosis.rule_ids),
                        )
                    else:
                        diagnosis = raw_diagnosis
                        diagnosis_novelty_gate = {
                            "applied": False,
                            "active_hit_ids": [],
                            "llm_retry_added": False,
                        }
                    selected_rule_ids = set(diagnosis.rule_ids)
                    selected_rules = [
                        rule
                        for rule in retrieved_rules
                        if rule.rule_id in selected_rule_ids
                    ]
                    # An executable ERCL action is eligible only after the
                    # diagnosis and RFL novelty/veto gates agree on that rule.
                    # Otherwise the local RFL/Feedback repair path wins.
                    operator_rules = high_confidence_operator_rules(selected_rules)
                    ercl_rfl_resolution = (
                        "rfl_fallback_after_conflict"
                        if diagnosis_novelty_gate.get("active_hit_ids")
                        else (
                            "merged_agreement"
                            if self.config.mode is RunMode.MEMENTOHLS
                            and selected_rules
                            and rfl is not None
                            else (
                                "ercl_full_contract"
                                if selected_rules
                                else "feedback_or_rfl_path"
                            )
                        )
                    )
                    if replay_plan is not None and replay_plan.should_replay(
                        "diagnosis", repair_round
                    ):
                        replay_plan.assert_upstream(
                            repair_round,
                            diagnosis_prompt_sha256=_sha256_text(diagnosis_prompt),
                            diagnosis=diagnosis.to_dict(),
                            retrieved_rule_ids=[rule.rule_id for rule in retrieved_rules],
                            selected_rule_ids=[rule.rule_id for rule in selected_rules],
                        )
                    if self.ercl is not None and retrieved_rules:
                        global_veto = list(
                            self.ercl.retrieval_policy.get("global_veto", [])
                        )
                        diagnosis.forbidden_actions = list(
                            dict.fromkeys(diagnosis.forbidden_actions + global_veto)
                        )
                    cleanroom_requested = trigger_cleanroom
                    cleanroom_authorization_rules: list[MemoryRule] = []
                    if cleanroom_requested and self.ercl is not None:
                        cleanroom_authorization_rules = self.ercl.match_route(
                            "cleanroom_authorization",
                            previous_review.failure_stage,
                            current_log,
                            current_code,
                            header=header_text,
                            history=history_match,
                            task=seed_prompt,
                            facts=memory_facts,
                            limit=1,
                        )
                    if not cleanroom_requested:
                        cleanroom_authorization = "not_requested"
                    elif self.ercl is None:
                        cleanroom_authorization = "short_controller"
                    elif cleanroom_authorization_rules:
                        cleanroom_authorization = "long_rule_authorized"
                    else:
                        cleanroom_authorization = "missing_long_authorization"
                        trigger_cleanroom = False
                    action_rules = list(
                        {
                            rule.rule_id: rule
                            for rule in selected_rules + operator_rules + cleanroom_authorization_rules
                        }.values()
                    )
                    repair_model_contracts = (
                        [
                            rfl.build_model_input_contract(
                                role="repairer",
                                compact_level=level,
                                budget_chars=budget,
                                chosen_source_stage=previous_review.failure_stage,
                                effective_diagnosis=diagnosis,
                            )
                            for level, budget in enumerate(REPAIR_RFL_BUDGETS)
                        ]
                        if rfl is not None and memory_active
                        else []
                    )
                    clean_model_contracts = (
                        [
                            rfl.build_model_input_contract(
                                role="cleanroom",
                                compact_level=level,
                                budget_chars=budget,
                                chosen_source_stage=previous_review.failure_stage,
                                effective_diagnosis=diagnosis,
                            )
                            for level, budget in enumerate(CLEAN_RFL_BUDGETS)
                        ]
                        if rfl is not None and memory_active
                        else []
                    )
                    if any(
                        not item.feasible
                        for item in repair_model_contracts + clean_model_contracts
                    ):
                        raise RuntimeError(
                            "A configured Repair/Clean RFL profile is infeasible"
                        )
                    repair_short_views = [item.json_text for item in repair_model_contracts]
                    clean_short_views = [item.json_text for item in clean_model_contracts]

                    operator_result: ActionExecution | None = None
                    format_recovery_result: FormatRecoveryResult | None = None
                    repair_call: LLMCallResult | None = None
                    repair_reused_from: str | None = None
                    repair_prompt = ""
                    repair_prompt_variants: list[str] = []
                    action_origin = "local_llm"

                    if (
                        self.action_executor is not None
                        and memory_active
                        and operator_rules
                    ):
                        operator_result = self.action_executor.execute(
                            code=current_code,
                            header_name=header_name,
                            header_text=header_text,
                            matched_rules=operator_rules,
                            top_fn=case.top_fn,
                            failure_log=current_log,
                            task_text=seed_prompt,
                        )
                    if operator_result is not None and operator_result.changed:
                        trigger_cleanroom = False
                        action_origin = "operator"
                        response_text = _wrap_code(
                            previous_review.generated_filename
                            or self.reviewer.infer_expected_filename(case),
                            operator_result.output_code,
                        )
                    elif trigger_cleanroom:
                        if rfl is None:
                            raise RuntimeError("Clean-room requires RFL state")
                        rfl.mark_cleanroom(
                            cleanroom_reason or "stagnation",
                            repair_round,
                            trigger_candidate_id=trigger_candidate_id,
                            trigger_signature=trigger_signature,
                            trigger_stage=trigger_review.failure_stage,
                            trigger_outcome=trigger_outcome,
                        )
                        action_origin = "cleanroom_llm"
                        clean_diagnosis = Diagnosis(
                            failure_stage=previous_review.failure_stage,
                            evidence=list(diagnosis.evidence),
                            root_cause="abstain_rederive_from_evidence",
                            allowed_actions=list(
                                dict.fromkeys(
                                    action
                                    for rule in action_rules
                                    for action in rule.allowed_actions
                                )
                            ),
                            forbidden_actions=list(
                                dict.fromkeys(
                                    diagnosis.forbidden_actions
                                    + [
                                        "Do not repeat any active trajectory-local veto or exhausted action family"
                                    ]
                                )
                            ),
                            must_hold_invariants=list(diagnosis.must_hold_invariants),
                            rule_ids=[rule.rule_id for rule in action_rules],
                            fallback=False,
                            predicted_observation=(
                                "A newly derived candidate advances the objective Reviewer frontier"
                            ),
                            canonical_root_intent_id="stage_unknown",
                            canonical_action_intent_id="stage_generic_other",
                            canonical_action_target_id=None,
                            action_veto_eligible=False,
                            root_veto_eligible=False,
                            validation_notes=["cleanroom_diagnosis_sanitized"],
                        )
                        repair_prompt_variants = [
                            build_cleanroom_prompt(
                                seed_prompt,
                                previous_review.generated_filename
                                or self.reviewer.infer_expected_filename(case),
                                clean_diagnosis,
                                action_rules,
                                clean_short_views[level],
                                cleanroom_reason or "stagnation",
                                retrieval_policy=retrieval_policy,
                                compact_level=level,
                            )
                            for level in range(CLEAN_PROFILE_COUNT)
                        ]
                        repair_prompt = repair_prompt_variants[0]
                    elif self.action_executor is not None and memory_active:
                        if operator_result is None:
                            operator_result = self.action_executor.execute(
                                code=current_code,
                                header_name=header_name,
                                header_text=header_text,
                                matched_rules=selected_rules,
                                top_fn=case.top_fn,
                                failure_log=current_log,
                                task_text=seed_prompt,
                            )
                        if operator_result.changed:
                            action_origin = "operator"
                            response_text = _wrap_code(
                                previous_review.generated_filename
                                or self.reviewer.infer_expected_filename(case),
                                operator_result.output_code,
                            )
                        else:
                            repair_prompt_variants = [
                                build_repair_prompt(
                                    seed_prompt,
                                    previous_review.generated_filename
                                    or self.reviewer.infer_expected_filename(case),
                                    current_code or previous_response,
                                    diagnosis,
                                    operator_rules or selected_rules,
                                    (repair_short_views[level] if repair_short_views else None),
                                    retrieval_policy=retrieval_policy,
                                    compact_level=level,
                                )
                                for level in range(REPAIR_PROFILE_COUNT)
                            ]
                            repair_prompt = repair_prompt_variants[0]
                    else:
                        repair_prompt_variants = [
                            build_repair_prompt(
                                seed_prompt,
                                previous_review.generated_filename
                                or self.reviewer.infer_expected_filename(case),
                                current_code or previous_response,
                                diagnosis,
                                selected_rules,
                                (repair_short_views[level] if repair_short_views else None),
                                retrieval_policy=retrieval_policy,
                                compact_level=level,
                            )
                            for level in range(REPAIR_PROFILE_COUNT)
                        ]
                        repair_prompt = repair_prompt_variants[0]

                    if action_origin != "operator":
                        if not repair_prompt_variants:
                            raise RuntimeError("Repair prompt variants were not built")
                        atomic_write_text(
                            round_dir / "repair_prompt_requested.txt",
                            repair_prompt_variants[0],
                        )
                        if replay_plan is not None and replay_plan.should_replay(
                            "repair", repair_round
                        ):
                            replayed = replay_plan.replay_call(
                                role="repair",
                                round_index=repair_round,
                                prompt_variants=repair_prompt_variants,
                            )
                            repair_call = replayed.call
                            repair_reused_from = replayed.reused_from
                        else:
                            repair_call = self.llm.chat(
                                [
                                    {"role": "system", "content": REPAIRER_SYSTEM},
                                    {"role": "user", "content": repair_prompt_variants[0]},
                                ],
                                temperature=self.config.temperature,
                                max_tokens=min(self.config.max_tokens, 3800),
                                seed=seed + repair_round * 10000 + 2,
                                minimum_completion_tokens=2500,
                                user_prompt_variants=repair_prompt_variants[1:],
                            )
                        repair_prompt = (
                            repair_call.effective_user_prompt
                            or repair_prompt_variants[0]
                        )
                        atomic_write_text(
                            round_dir / "repair_prompt.txt", repair_prompt
                        )
                        raw_repair_response = repair_call.text
                        atomic_write_text(
                            round_dir / "repair_response_raw.txt", raw_repair_response
                        )
                        choices = list(
                            (repair_call.raw_response or {}).get("choices") or []
                        )
                        finish_reason = (
                            str(choices[0].get("finish_reason"))
                            if choices and choices[0].get("finish_reason") is not None
                            else None
                        )
                        if self.config.mode.uses_format_recovery and memory_active:
                            format_recovery_result = recover_hls_envelope(
                                raw_repair_response,
                                expected_filename=(
                                    previous_review.generated_filename
                                    or self.reviewer.infer_expected_filename(case)
                                ),
                                top_fn=case.top_fn,
                                finish_reason=finish_reason,
                                enabled=True,
                            )
                            response_text = format_recovery_result.effective_response
                            atomic_write_json(
                                round_dir / "format_recovery.json",
                                format_recovery_result.to_dict(),
                            )
                        else:
                            response_text = raw_repair_response
                        atomic_write_text(
                            round_dir / "repair_response.txt", response_text
                        )
                    else:
                        atomic_write_text(
                            round_dir / "operator_response.txt", response_text
                        )

                    current_review = self._review_candidate(
                        case,
                        response_text,
                        round_dir,
                        f"{case.name}__s{sample_index}__r{repair_round}",
                    )
                    candidate_id = (
                        f"{self.config.iteration_id}:{self.config.mode.value}:"
                        f"{case.name}:s{sample_index}:r{repair_round}"
                    )
                    prompt_tokens, completion_tokens, total_tokens = _call_token_totals(
                        diagnosis_call, repair_call
                    )
                    candidate_log = self._evidence_log(
                        current_review, response_text
                    )
                    filename_telemetry = self._filename_telemetry(
                        case, previous_review, current_review
                    )
                    invariant_violations = self._invariant_violations(
                        case, previous_review, current_review
                    )
                    if rfl is not None:
                        rfl.record_repair(
                            round_index=repair_round,
                            candidate_id=candidate_id,
                            source_code=(
                                None if trigger_cleanroom else current_code or previous_response
                            ),
                            candidate_code=current_review.generated_code
                            or response_text,
                            source_review=previous_review,
                            candidate_review=current_review,
                            source_log=current_log,
                            candidate_log=candidate_log,
                            diagnosis=diagnosis,
                            rule_ids=[rule.rule_id for rule in action_rules],
                            action_type=action_origin,
                            operator_id=(
                                operator_result.operator_id
                                if operator_result
                                and operator_result.changed
                                else None
                            ),
                            cleanroom_used=trigger_cleanroom,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            parent_candidate_id=previous_candidate_id,
                            base_origin=(
                                "immutable_original"
                                if trigger_cleanroom
                                else (
                                    "historical_best"
                                    if recovery_candidate_id is not None
                                    else "latest"
                                )
                            ),
                            invariant_violations=invariant_violations,
                            format_recovery=(
                                format_recovery_result.to_dict()
                                if format_recovery_result is not None
                                else None
                            ),
                        )

                    round_payload = {
                        "round_index": repair_round,
                        "feedback_warm_start": repair_round == 1,
                        "memory_active": memory_active,
                        "rfl_feedback_fallback": rfl_feedback_fallback,
                        "candidate_id": candidate_id,
                        "trajectory_key_sha256": (
                            trajectory_key.sha256 if rfl is not None else None
                        ),
                        "protocol_digest": self.protocol_digest,
                        "deployment_epoch": self.deployment_epoch,
                        "parent_candidate_id": previous_candidate_id,
                        "action_origin": action_origin,
                        "recovery": {
                            "triggered": recovery_candidate_id is not None,
                            "source_candidate_id": recovery_candidate_id,
                            "reason": recovery_reason,
                        },
                        "active_rfl_negative_ledger": [
                            asdict(item) for item in active_rfl_ledger
                        ],
                        "response_sha256": _sha256_text(response_text),
                        "raw_response_sha256": (
                            format_recovery_result.raw_response_sha256
                            if format_recovery_result is not None
                            else _sha256_text(response_text)
                        ),
                        "format_recovery": (
                            format_recovery_result.to_dict()
                            if format_recovery_result is not None
                            else None
                        ),
                        "code_sha256": (
                            _sha256_text(current_review.generated_code)
                            if current_review.generated_code is not None
                            else None
                        ),
                        "diagnosis_prompt_sha256": _sha256_text(diagnosis_prompt),
                        "diagnoser_llm": _llm_to_dict(
                            diagnosis_call, diagnosis_reused_from
                        ),
                        "raw_diagnosis": raw_diagnosis.to_dict(),
                        "diagnosis": diagnosis.to_dict(),
                        "diagnosis_novelty_gate": diagnosis_novelty_gate,
                        "ercl_rfl_resolution": ercl_rfl_resolution,
                        "model_input_contracts": {
                            "diagnoser": [
                                item.audit_dict() for item in diagnosis_model_contracts
                            ],
                            "repairer": [
                                item.audit_dict() for item in repair_model_contracts
                            ],
                            "cleanroom": [
                                item.audit_dict() for item in clean_model_contracts
                            ],
                        },
                        "memory_facts": (
                            json.loads(memory_facts) if memory_facts else None
                        ),
                        "ercl_retrieval": retrieval_result.provenance,
                        "retrieved_rule_ids": [
                            rule.rule_id for rule in retrieved_rules
                        ],
                        "selected_rule_ids": [
                            rule.rule_id for rule in selected_rules
                        ],
                        "action_rule_ids": [
                            rule.rule_id for rule in action_rules
                        ],
                        "matched_rule_ids": [
                            rule.rule_id for rule in retrieved_rules
                        ],
                        "matched_rules": [
                            rule.to_prompt_dict() for rule in retrieved_rules
                        ],
                        "selected_rules": [
                            rule.to_prompt_dict() for rule in selected_rules
                        ],
                        "action_rules": [
                            rule.to_prompt_dict() for rule in action_rules
                        ],
                        "repair_prompt": repair_prompt or None,
                        "repair_prompt_sha256": (
                            _sha256_text(repair_prompt) if repair_prompt else None
                        ),
                        "prompt_provenance": {
                            "diagnosis": {
                                "kind": "stage_diagnosis",
                                "user_prompt_sha256": _sha256_text(
                                    diagnosis_prompt
                                ),
                                "message_prompt_sha256": diagnosis_call.prompt_sha256,
                                "message_roles": diagnosis_call.message_roles,
                                "source_roles": (
                                    [
                                        "current_candidate_code",
                                        "reviewer_tool_feedback",
                                    ]
                                    + (
                                        ["ercl_matches"]
                                        if self.config.mode.uses_ercl
                                        and memory_active
                                        else []
                                    )
                                    + (
                                        ["rfl_summary"]
                                        if self.config.mode.uses_rfl
                                        and memory_active
                                        else []
                                    )
                                ),
                            },
                            "repair": {
                                "kind": action_origin,
                                "user_prompt_sha256": (
                                    _sha256_text(repair_prompt)
                                    if repair_prompt else None
                                ),
                                "message_prompt_sha256": (
                                    repair_call.prompt_sha256
                                    if repair_call else None
                                ),
                                "message_roles": (
                                    repair_call.message_roles
                                    if repair_call else []
                                ),
                                "source_roles": (
                                    ["immutable_round0_task"]
                                    + (
                                        []
                                        if trigger_cleanroom
                                        else ["current_candidate_code"]
                                    )
                                    + ["diagnosis"]
                                    + (
                                        ["ercl_matches"]
                                        if self.config.mode.uses_ercl
                                        and memory_active
                                        else []
                                    )
                                    + (
                                        ["rfl_summary"]
                                        if self.config.mode.uses_rfl
                                        and memory_active
                                        else []
                                    )
                                ),
                            },
                            "reference_kernel_read_or_prompted": False,
                        },
                        "llm": _llm_to_dict(repair_call, repair_reused_from),
                        "budget_contract": {
                            "logical_diagnoser_calls": 1,
                            "logical_action_generation_calls": (
                                0 if action_origin == "operator" else 1
                            ),
                            "reviewed_candidate_count": 1,
                            "operator_noop_fallback": bool(
                                operator_result is not None
                                and operator_result.attempted
                                and not operator_result.changed
                                and repair_call is not None
                            ),
                            "llm_resample_due_to_veto": 0,
                            "diagnoser_physical_http_attempts": int(
                                diagnosis_call.attempts
                            ),
                            "action_physical_http_attempts": int(
                                repair_call.attempts if repair_call else 0
                            ),
                        },
                        "operator": (
                            operator_result.to_dict() if operator_result else None
                        ),
                        "cleanroom": {
                            "eligible": cleanroom_eligible,
                            "requested": cleanroom_requested,
                            "triggered": trigger_cleanroom,
                            "authorization": cleanroom_authorization,
                            "authorization_rule_ids": [
                                rule.rule_id
                                for rule in cleanroom_authorization_rules
                            ],
                            "reason": cleanroom_reason,
                            "trigger_candidate_id": trigger_candidate_id,
                            "trigger_failure_signature": trigger_signature,
                            "trigger_failure_stage": trigger_review.failure_stage,
                            "failure_signature": trigger_signature,
                            "failure_stage": trigger_review.failure_stage,
                            "trigger_stage_vector": trigger_outcome,
                            "trigger_normalized_failure": (
                                trigger_normalized_failure
                            ),
                        },
                        "filename_telemetry": filename_telemetry,
                        "invariant_violations": list(invariant_violations),
                        "review": current_review.to_dict(),
                        "stage_vector": stage_vector(current_review),
                        "before_stage_vector": stage_vector(previous_review),
                        "rfl_state": (
                            rfl.audit_dict() if rfl else None
                        ),
                        "causal_replay": (
                            replay_plan.round_provenance(
                                repair_round,
                                diagnosis_reused_from=diagnosis_reused_from,
                                repair_reused_from=repair_reused_from,
                            )
                            if replay_plan is not None
                            else None
                        ),
                    }
                    if replay_plan is not None:
                        replay_plan.assert_completed_prefix_round(round_payload)
                        replay_plan.assert_activation_branch(
                            round_payload,
                            repair_reused=repair_reused_from is not None,
                        )
                    payload["rounds"].append(round_payload)
                    atomic_write_json(round_dir / "round.json", round_payload)
                    atomic_write_json(trajectory_path, payload)
                    if current_review.pass_tb_and_synth:
                        break

            last_round_index = len(payload["rounds"]) - 1
            if rfl is not None:
                if len(rfl.candidates) != len(payload["rounds"]):
                    raise RuntimeError(
                        "RFL archive must contain exactly one candidate per executed round"
                    )
                if len(rfl.candidates) > 1 + self.config.max_repair_rounds:
                    raise RuntimeError("RFL archive exceeded the candidate budget")
                if any(
                    candidate.trajectory_key_sha256
                    != rfl.trajectory_key_sha256
                    for candidate in rfl.candidates
                ):
                    raise RuntimeError("Cross-trajectory candidate detected")
                if any(
                    entry.trajectory_key_sha256
                    != rfl.trajectory_key_sha256
                    for entry in rfl.entries
                ):
                    raise RuntimeError("Cross-trajectory RFL entry detected")
                if any(
                    int(value.get("round_index") or 0) == 5
                    and bool((value.get("cleanroom") or {}).get("triggered"))
                    for value in payload["rounds"]
                ):
                    raise RuntimeError("Round 5 is reserved for local recovery")
            if rfl is not None and self.config.mode.uses_candidate_archive:
                selected_round_index = rfl.best_round_index
                selection_strategy = (
                    "trajectory_rfl_archive:"
                    "tb_and_synth>tb>synth>compile>parse>"
                    "compile_subfrontier>earlier>lower_cost"
                )
                parse_output_protection_activated = False
                discarded_parse_invalid_round_indices: list[int] = []
            elif self.config.mode.uses_ercl:
                selected_round_index = best_reviewed_round_index(payload["rounds"])
                selection_strategy = (
                    "trajectory_ercl_archive:"
                    "tb_and_synth>tb>synth>compile>parse>earlier"
                )
                parse_output_protection_activated = (
                    selected_round_index != last_round_index
                    and not bool(
                        payload["rounds"][last_round_index]["review"]["pass_parse"]
                    )
                )
                discarded_parse_invalid_round_indices = [
                    index
                    for index in range(selected_round_index + 1, last_round_index + 1)
                    if not bool(payload["rounds"][index]["review"]["pass_parse"])
                ]
            else:
                selected_round_index = last_parse_valid_round_index(payload["rounds"])
                selection_strategy = "last_parse_valid_candidate"
                parse_output_protection_activated = (
                    selected_round_index != last_round_index
                )
                discarded_parse_invalid_round_indices = list(
                    range(selected_round_index + 1, last_round_index + 1)
                )
            selected_round = payload["rounds"][selected_round_index]
            selected_review = selected_round["review"]
            payload["status"] = "complete"
            payload["terminal"] = True
            payload["selected_candidate_id"] = selected_round["candidate_id"]
            payload["selected_round_index"] = selected_round_index
            payload["last_candidate_id"] = payload["rounds"][-1]["candidate_id"]
            payload["selection_strategy"] = selection_strategy
            payload["archive_activated"] = bool(
                (
                    rfl is not None
                    and self.config.mode.uses_candidate_archive
                )
                or self.config.mode.uses_ercl
            ) and selected_round_index != last_round_index
            payload["parse_output_protection_activated"] = parse_output_protection_activated
            payload["discarded_parse_invalid_round_indices"] = discarded_parse_invalid_round_indices
            payload["success"] = bool(selected_review["pass_tb_and_synth"])
            payload["stop_reason"] = (
                "tb_and_synth_passed"
                if current_review.pass_tb_and_synth
                else (
                    "zero_shot_complete"
                    if self.config.mode is RunMode.ZERO_SHOT
                    else "repair_budget_exhausted"
                )
            )
            payload["final_review"] = selected_review
            payload["rfl_state_final"] = (
                rfl.audit_dict() if rfl else None
            )
            if replay_plan is not None:
                replay_plan.assert_terminal(payload)
            selected_code = selected_review.get("generated_code")
            if selected_code:
                atomic_write_text(
                    trajectory_dir / "selected_candidate.cpp", selected_code
                )
            atomic_write_json(trajectory_path, payload)
            return payload
        except Exception as exc:
            payload["status"] = "error"
            payload["terminal"] = True
            payload["error"] = repr(exc)
            payload["stop_reason"] = "unhandled_error"
            atomic_write_json(trajectory_path, payload)
            raise

    def run(self) -> list[dict[str, Any]]:
        cases = list(self.cases)
        health = self.llm.health()
        if self.vllm_deployment_manifest is not None:
            deployment = self.vllm_deployment_manifest
            live_identity = {
                "prometheus_process_start_time_seconds": health.get(
                    "prometheus_process_start_time_seconds"
                ),
                "model_root": health.get("model_root"),
                "model_max_len": health.get("model_max_len"),
                "deployment_epoch": health.get("deployment_epoch"),
            }
            expected_identity = {
                "prometheus_process_start_time_seconds": deployment.get(
                    "prometheus_process_start_time_seconds"
                ),
                "model_root": deployment.get("model_path"),
                "model_max_len": self.config.context_limit,
                "deployment_epoch": deployment.get("deployment_epoch"),
            }
            if live_identity != expected_identity:
                raise RuntimeError(
                    "Live vLLM deployment epoch differs from frozen manifest: "
                    + json.dumps(
                        {"expected": expected_identity, "actual": live_identity},
                        sort_keys=True,
                    )
                )
        self.run_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.run_root / "dataset_manifest.json", self.dataset_manifest
        )
        metadata_name = (
            "run_metadata.json" if self.config.block_id is None
            else f"run_metadata_block__{self.config.block_id}.json"
        )
        metadata_path = self.run_root / metadata_name
        metadata = self._metadata(len(cases), health)
        atomic_write_json(metadata_path, metadata)

        jobs = [
            (case, sample_index, seed)
            for case in cases
            for sample_index, seed in enumerate(self.config.seeds)
        ]
        futures: dict[Any, tuple[str, int, int]] = {}
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        run_started = time.time()
        completed_count = 0
        next_job = 0

        def submit_one(executor: ThreadPoolExecutor) -> None:
            nonlocal next_job
            case, sample_index, seed = jobs[next_job]
            next_job += 1
            future = executor.submit(
                self._run_trajectory, case, sample_index, seed
            )
            futures[future] = (case.name, sample_index, seed)

        def admit_replacement() -> None:
            wait_for_runtime_capacity(
                self.config.output_dir / "protocol" / "runtime_capacity.jsonl",
                stable_seconds=0,
                poll_seconds=self.config.runtime_capacity_poll_seconds,
                immediate_if_initially_allowed=True,
                continuation_gate=True,
                start_cpu_max=self.config.runtime_start_cpu_max,
                start_load_max=self.config.runtime_start_load_max,
                pause_cpu_max=self.config.runtime_pause_cpu_max,
                pause_load_max=self.config.runtime_pause_load_max,
                start_memory_available_gib_min=(
                    self.config.runtime_start_memory_available_gib_min
                ),
                pause_memory_available_gib_min=(
                    self.config.runtime_pause_memory_available_gib_min
                ),
                process_tree_rss_gib_max=(
                    self.config.runtime_process_tree_rss_gib_max
                ),
            )

        with ThreadPoolExecutor(
            max_workers=self.config.design_workers,
            thread_name_prefix="hls-design",
        ) as executor:
            for _ in range(min(self.config.design_workers, len(jobs))):
                submit_one(executor)
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    case_name, sample_index, seed = futures.pop(future)
                    completed_count += 1
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        errors.append(
                            {
                                "future_index": completed_count,
                                "benchmark_case_name": case_name,
                                "sample_index": sample_index,
                                "seed": seed,
                                "error": repr(exc),
                            }
                        )
                        print(
                            f"[{self.config.mode.value}] trajectory error: {exc!r}",
                            flush=True,
                        )
                    print(
                        f"[{self.config.mode.value}] collected "
                        f"{completed_count}/{len(jobs)} trajectories",
                        flush=True,
                    )
                if (
                    next_job < len(jobs)
                    and len(futures) < self.config.design_workers
                ):
                    admit_replacement()
                while (
                    next_job < len(jobs)
                    and len(futures) < self.config.design_workers
                ):
                    submit_one(executor)
        if not errors:
            try:
                metadata["end_of_mode_provenance"] = (
                    self._validate_end_of_mode_provenance()
                )
            except Exception as exc:
                errors.append(
                    {
                        "scope": "end_of_mode_provenance",
                        "error": repr(exc),
                    }
                )
        metadata["finished_at_unix"] = time.time()
        metadata["wall_seconds"] = metadata["finished_at_unix"] - run_started
        metadata["completed_trajectories"] = len(results)
        metadata["error_trajectories"] = len(errors)
        atomic_write_json(metadata_path, metadata)
        errors_name = (
            "run_errors.json" if self.config.block_id is None
            else f"run_errors_block__{self.config.block_id}.json"
        )
        atomic_write_json(self.run_root / errors_name, errors)
        if errors:
            raise RuntimeError(
                f"Mode {self.config.mode.value} finished collection with "
                f"{len(errors)} trajectory errors; rerun with --resume after fixing infrastructure"
            )
        return results
