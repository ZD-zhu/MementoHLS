from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .models import file_sha256


class DeploymentEvidenceError(ValueError):
    """Raised when copied deployment evidence is incomplete or inconsistent."""


START_REQUIRED_FILES = frozenset(
    {
        "hostname.txt",
        "selected_gpu_and_free_mib.csv",
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
        "vllm.pgid",
    }
)

STOP_FILES = (
    "compute_apps_after.txt",
    "compute_apps_before.txt",
    "gpu_before_stop.csv",
    "stop_manifest.json",
    "gpu_after_stop.csv",
    "process_group_after.txt",
    "process_group_before.txt",
    "health_down.txt",
    "recorded_pgid.txt",
    "stop_requested_at_utc.txt",
    "stopped_at_utc.txt",
)
STOP_HASHED_FILES = frozenset(STOP_FILES) - {"stop_manifest.json"}

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA_LINE = re.compile(r"^([0-9a-f]{64})[ \t]+(.+)$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")


def _root(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise DeploymentEvidenceError(f"{label} must not be a symlink: {path}")
    try:
        value = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DeploymentEvidenceError(f"{label} does not exist: {path}") from exc
    if not value.is_dir():
        raise DeploymentEvidenceError(f"{label} is not a directory: {value}")
    return value


def _file(root: Path, relative: str) -> Path:
    if Path(relative).name != relative:
        raise DeploymentEvidenceError(f"Unsafe evidence filename: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise DeploymentEvidenceError(
            f"Required evidence is missing or is a symlink: {path}"
        )
    resolved = path.resolve()
    if root != resolved.parent:
        raise DeploymentEvidenceError(f"Evidence escapes its run directory: {path}")
    return path


def _text(root: Path, relative: str, *, allow_empty: bool = False) -> str:
    try:
        value = _file(root, relative).read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise DeploymentEvidenceError(
            f"Evidence is not valid UTF-8: {relative}"
        ) from exc
    if not value and not allow_empty:
        raise DeploymentEvidenceError(f"Evidence is empty: {relative}")
    return value


def _json_object(root: Path, relative: str) -> dict[str, Any]:
    try:
        value = json.loads(_file(root, relative).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentEvidenceError(f"Invalid JSON evidence: {relative}") from exc
    if not isinstance(value, dict):
        raise DeploymentEvidenceError(f"JSON evidence must be an object: {relative}")
    return value


def _single_csv_row(root: Path, relative: str, fields: int) -> list[str]:
    rows = [
        [cell.strip() for cell in line.split(",")]
        for line in _file(root, relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or len(rows[0]) != fields:
        raise DeploymentEvidenceError(
            f"{relative} must contain exactly one {fields}-field CSV row"
        )
    return rows[0]


def _csv_rows(root: Path, relative: str, fields: int) -> list[list[str]]:
    rows = [
        [cell.strip() for cell in line.split(",")]
        for line in _file(root, relative).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(len(row) != fields for row in rows):
        raise DeploymentEvidenceError(
            f"{relative} must contain non-empty {fields}-field CSV rows"
        )
    return rows


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentEvidenceError(f"{label} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise DeploymentEvidenceError(f"{label} must be >= {minimum}")
    return parsed


def _number(value: object, label: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentEvidenceError(f"{label} must be numeric") from exc
    if positive and parsed <= 0:
        raise DeploymentEvidenceError(f"{label} must be positive")
    return parsed


def _timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DeploymentEvidenceError(f"{label} is not ISO-8601: {value}") from exc
    if parsed.tzinfo is None:
        raise DeploymentEvidenceError(f"{label} must include a timezone")
    return parsed


def _flag(tokens: list[str], name: str) -> str:
    positions = [index for index, token in enumerate(tokens) if token == name]
    if len(positions) != 1 or positions[0] + 1 >= len(tokens):
        raise DeploymentEvidenceError(
            f"Launch command must contain exactly one value for {name}"
        )
    value = tokens[positions[0] + 1]
    if value.startswith("--"):
        raise DeploymentEvidenceError(f"Launch command lacks a value for {name}")
    return value


def _validated_sha_lines(
    root: Path,
    relative: str,
    *,
    required_prefix: PurePosixPath | None = None,
) -> list[tuple[str, PurePosixPath]]:
    values: list[tuple[str, PurePosixPath]] = []
    seen: set[PurePosixPath] = set()
    for raw in _file(root, relative).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        match = _SHA_LINE.fullmatch(raw)
        if match is None:
            raise DeploymentEvidenceError(
                f"{relative} contains a malformed SHA256 line"
            )
        path = PurePosixPath(match.group(2))
        if not path.is_absolute() or ".." in path.parts:
            raise DeploymentEvidenceError(
                f"{relative} contains an unsafe artifact path: {path}"
            )
        if required_prefix is not None:
            try:
                path.relative_to(required_prefix)
            except ValueError as exc:
                raise DeploymentEvidenceError(
                    f"{relative} artifact is outside model_path: {path}"
                ) from exc
        if path in seen:
            raise DeploymentEvidenceError(
                f"{relative} contains a duplicate artifact: {path}"
            )
        seen.add(path)
        values.append((match.group(1), path))
    if not values:
        raise DeploymentEvidenceError(f"{relative} has no SHA256 entries")
    return values


def _metadata_hashes(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise DeploymentEvidenceError(
                f"Copied run metadata contains a symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise DeploymentEvidenceError(
                f"Copied run metadata contains a non-regular file: {path}"
            )
        values[path.name] = file_sha256(path)
    return values


def build_vllm_run_manifest(
    metadata_root: Path,
    *,
    endpoint: str,
    served_model: str,
    inference_server: str = "A100",
) -> dict[str, Any]:
    """Build the existing orchestrator-compatible manifest from local evidence."""

    root = _root(metadata_root, "Copied vLLM run metadata")
    run_id = root.name
    if not _RUN_ID.fullmatch(run_id):
        raise DeploymentEvidenceError(f"Unsafe vLLM run_id: {run_id!r}")
    if not endpoint.rstrip("/").endswith("/v1"):
        raise DeploymentEvidenceError("The OpenAI-compatible endpoint must end in /v1")
    if inference_server not in {"A100", "A6000"}:
        raise DeploymentEvidenceError("inference_server must be A100 or A6000")
    if not served_model.strip():
        raise DeploymentEvidenceError("served_model must not be empty")

    hashes = _metadata_hashes(root)
    missing = START_REQUIRED_FILES - set(hashes)
    if missing:
        raise DeploymentEvidenceError(
            f"Copied vLLM run metadata misses {sorted(missing)}"
        )

    selected = _single_csv_row(root, "selected_gpu_and_free_mib.csv", 2)
    gpu = _integer(selected[0], "selected GPU")
    _integer(selected[1], "selected GPU free MiB", minimum=0)
    if not 0 <= gpu <= 7:
        raise DeploymentEvidenceError("selected GPU must be in physical range 0..7")

    launch_name = f"launch_command_gpu{gpu}.txt"
    launch_path = _file(root, launch_name)
    hashes[launch_name] = file_sha256(launch_path)
    launch = launch_path.read_text(encoding="utf-8").strip()
    try:
        tokens = shlex.split(launch)
    except ValueError as exc:
        raise DeploymentEvidenceError("Launch command is not valid shell quoting") from exc
    model_path = _flag(tokens, "--model")
    if _flag(tokens, "--served-model-name") != served_model:
        raise DeploymentEvidenceError(
            "Launch command served model differs from requested served_model"
        )
    if _integer(_flag(tokens, "--tensor-parallel-size"), "tensor parallel") != 1:
        raise DeploymentEvidenceError("Only a single-GPU deployment is admissible")
    if _integer(_flag(tokens, "--max-model-len"), "max model length") != 8192:
        raise DeploymentEvidenceError("Formal protocol requires max model length 8192")
    if _integer(_flag(tokens, "--max-num-seqs"), "max num seqs") != 8:
        raise DeploymentEvidenceError("Formal protocol requires max num seqs 8")
    if _flag(tokens, "--gpu-memory-utilization") != "0.90":
        raise DeploymentEvidenceError(
            "Formal protocol requires GPU memory utilization 0.90"
        )
    if _flag(tokens, "--generation-config") != "vllm":
        raise DeploymentEvidenceError("Formal protocol requires vLLM generation config")
    if _flag(tokens, "--host") != "0.0.0.0" or _flag(tokens, "--port") != "8000":
        raise DeploymentEvidenceError("Formal protocol requires host 0.0.0.0 port 8000")
    visible = [token for token in tokens if token.startswith("CUDA_VISIBLE_DEVICES=")]
    if visible != [f"CUDA_VISIBLE_DEVICES={gpu}"]:
        raise DeploymentEvidenceError(
            "Launch command must bind exactly the selected single physical GPU"
        )

    hostname = _text(root, "hostname.txt")
    conda_environment = _text(root, "conda_environment.txt")
    expected_conda = {"A100": "vllm", "A6000": "SAGE"}[inference_server]
    if conda_environment != expected_conda:
        raise DeploymentEvidenceError(
            f"{inference_server} evidence must use conda environment {expected_conda}"
        )
    if _text(root, "gpu_memory_utilization.txt") != "0.90":
        raise DeploymentEvidenceError("gpu_memory_utilization.txt must be 0.90")
    _timestamp(_text(root, "started_at_utc.txt"), "started_at_utc")

    process = _json_object(root, "process_identity.json")
    process_keys = {"pid", "process_start_ticks", "boot_id"}
    if set(process) != process_keys:
        raise DeploymentEvidenceError(
            "process_identity.json must contain only pid/start_ticks/boot_id"
        )
    pid = _integer(process["pid"], "process identity pid", minimum=1)
    _integer(process["process_start_ticks"], "process start ticks", minimum=1)
    if not str(process["boot_id"]).strip():
        raise DeploymentEvidenceError("process identity boot_id is empty")
    if _integer(_text(root, "vllm.pid"), "vllm.pid", minimum=1) != pid:
        raise DeploymentEvidenceError("vllm.pid differs from process_identity.json")
    if _integer(_text(root, "vllm.pgid"), "vllm.pgid", minimum=1) != pid:
        raise DeploymentEvidenceError(
            "vllm.pgid must equal the setsid leader recorded as process pid"
        )

    model_root = PurePosixPath(model_path)
    artifacts = _validated_sha_lines(
        root,
        "model_artifact_sha256.txt",
        required_prefix=model_root,
    )
    weight_shards = [
        path
        for _, path in artifacts
        if path.name.endswith(".safetensors")
        and path.name != "model.safetensors.index.json"
    ]
    if not weight_shards:
        raise DeploymentEvidenceError("Model artifact evidence contains no weight shard")
    _validated_sha_lines(
        root,
        "model_config_sha256.txt",
        required_prefix=model_root,
    )

    models = _json_object(root, "models.json")
    entries = [
        item
        for item in models.get("data", [])
        if isinstance(item, dict) and item.get("id") == served_model
    ]
    if len(entries) != 1:
        raise DeploymentEvidenceError(
            "models.json must contain exactly one requested served model"
        )
    model_created = _integer(
        entries[0].get("created"), "models.json created", minimum=1
    )
    smoke = _json_object(root, "smoke.json")
    try:
        smoke_text = smoke["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise DeploymentEvidenceError("smoke.json lacks an assistant message") from exc
    if smoke_text != "HLS_SMOKE_OK":
        raise DeploymentEvidenceError("smoke.json does not contain exact HLS_SMOKE_OK")

    metrics_values = []
    for line in _file(root, "process_metrics_identity.prom").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith("process_start_time_seconds "):
            metrics_values.append(
                _number(
                    line.split(None, 1)[1],
                    "Prometheus process_start_time_seconds",
                    positive=True,
                )
            )
    if len(metrics_values) != 1:
        raise DeploymentEvidenceError(
            "Metrics evidence must contain one process_start_time_seconds"
        )
    process_start = _number(
        _text(root, "process_start_time_seconds.txt"),
        "process_start_time_seconds.txt",
        positive=True,
    )
    if abs(metrics_values[0] - process_start) > 1e-6:
        raise DeploymentEvidenceError(
            "Prometheus process start differs from captured process start"
        )

    after_rows = _csv_rows(root, "gpu_inventory_after.csv", 7)
    selected_after = [row for row in after_rows if _integer(row[0], "GPU index") == gpu]
    if len(selected_after) != 1:
        raise DeploymentEvidenceError(
            "gpu_inventory_after.csv must contain the selected GPU exactly once"
        )
    gpu_uuid = selected_after[0][2]
    if not _GPU_UUID.fullmatch(gpu_uuid):
        raise DeploymentEvidenceError("Selected GPU UUID is malformed")

    payload: dict[str, Any] = {
        "inference_server": inference_server,
        "endpoint": endpoint.rstrip("/"),
        "served_model": served_model,
        "remote_hostname": hostname,
        "run_id": run_id,
        "selected_gpu": gpu,
        "selected_gpu_uuid": gpu_uuid,
        "tensor_parallel_size": 1,
        "conda_environment": conda_environment,
        "model_path": model_path,
        "launch_command": launch,
        "metadata_root": str(root),
        "metadata_files_sha256": dict(sorted(hashes.items())),
        "deployment_epoch": run_id,
        "model_artifact_manifest_sha256": file_sha256(
            root / "model_artifact_sha256.txt"
        ),
        "model_artifact_entry_count": len(artifacts),
        "model_weight_shard_count": len(weight_shards),
        "process_identity": process,
        "process_identity_sha256": file_sha256(root / "process_identity.json"),
        "model_listing_created_at_capture": model_created,
        "prometheus_process_start_time_seconds": process_start,
        "identity_semantics": {
            "deployment_epoch": "audited run directory and PID/start-ticks/boot-id",
            "live_process_identity": (
                "Prometheus process_start_time_seconds; "
                "/v1/models.created is response-time dynamic"
            ),
            "model_identity": (
                "SHA256 of all safetensors, index, tokenizer and config artifacts"
            ),
        },
    }
    return payload


def validate_vllm_run_manifest(path: Path) -> dict[str, Any]:
    """Run the authoritative orchestrator validator against an existing manifest."""

    if path.is_symlink() or not path.is_file():
        raise DeploymentEvidenceError(f"Manifest is missing or is a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentEvidenceError(f"Invalid vLLM manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DeploymentEvidenceError("vLLM manifest must be a JSON object")
    from .orchestrator import _load_vllm_deployment_manifest

    try:
        loaded = _load_vllm_deployment_manifest(
            path,
            base_url=str(payload.get("endpoint", "")),
            model=str(payload.get("served_model", "")),
            inference_server=str(payload.get("inference_server", "")),
            formal_run=True,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentEvidenceError(str(exc)) from exc
    if loaded != payload:
        raise DeploymentEvidenceError("Authoritative loader changed manifest payload")
    return payload


def write_vllm_run_manifest(
    metadata_root: Path,
    output: Path,
    *,
    endpoint: str,
    served_model: str,
    inference_server: str = "A100",
) -> dict[str, Any]:
    root = _root(metadata_root, "Copied vLLM run metadata")
    output = output.absolute()
    try:
        output.resolve(strict=False).relative_to(root)
    except ValueError:
        pass
    else:
        raise DeploymentEvidenceError(
            "Manifest output must be outside the hash-bound run metadata directory"
        )
    payload = build_vllm_run_manifest(
        root,
        endpoint=endpoint,
        served_model=served_model,
        inference_server=inference_server,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = validate_vllm_run_manifest(output)
        if existing != payload:
            raise DeploymentEvidenceError(
                "Refusing to overwrite a different vLLM run manifest"
            )
        return existing

    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=str(output.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        validate_vllm_run_manifest(tmp)
        try:
            os.link(tmp, output)
        except FileExistsError:
            existing = validate_vllm_run_manifest(output)
            if existing != payload:
                raise DeploymentEvidenceError(
                    "Another writer created a different vLLM run manifest"
                )
        return validate_vllm_run_manifest(output)
    finally:
        tmp.unlink(missing_ok=True)


def _gpu_stop_row(root: Path, relative: str) -> tuple[int, str, int, int, int]:
    row = _single_csv_row(root, relative, 5)
    return (
        _integer(row[0], f"{relative} GPU"),
        row[1],
        _integer(row[2], f"{relative} total MiB", minimum=1),
        _integer(row[3], f"{relative} used MiB", minimum=0),
        _integer(row[4], f"{relative} free MiB", minimum=0),
    )


def _process_group_members(root: Path, recorded_pgid: int) -> dict[int, str]:
    members: dict[int, str] = {}
    for line in _text(root, "process_group_before.txt").splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            raise DeploymentEvidenceError(
                "process_group_before.txt contains a malformed ps row"
            )
        pid = _integer(fields[0], "process-group PID", minimum=1)
        pgid = _integer(fields[1], "process-group PGID", minimum=1)
        if pgid != recorded_pgid:
            raise DeploymentEvidenceError(
                "process_group_before.txt contains a process outside the recorded PGID"
            )
        if pid in members:
            raise DeploymentEvidenceError(
                "process_group_before.txt contains a duplicate PID"
            )
        members[pid] = fields[3]
    if not members:
        raise DeploymentEvidenceError(
            "process_group_before.txt contains no recorded PGID members"
        )
    return members


def _compute_apps(root: Path, relative: str) -> list[tuple[int, str, int]]:
    text = _text(root, relative, allow_empty=True)
    if not text:
        return []
    rows: list[tuple[int, str, int]] = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not _GPU_UUID.fullmatch(fields[1]):
            raise DeploymentEvidenceError(
                f"{relative} contains a malformed compute-app row"
            )
        rows.append(
            (
                _integer(fields[0], f"{relative} PID", minimum=1),
                fields[1],
                _integer(fields[2], f"{relative} used MiB", minimum=0),
            )
        )
    return rows


def validate_stop_evidence(
    stop_root: Path,
    start_manifest_path: Path,
) -> dict[str, Any]:
    root = _root(stop_root, "vLLM stop evidence")
    start = validate_vllm_run_manifest(start_manifest_path)
    for name in STOP_FILES:
        _file(root, name)

    stop = _json_object(root, "stop_manifest.json")
    if stop.get("status") != "vllm-stopped-and-verified":
        raise DeploymentEvidenceError("Stop status is not vllm-stopped-and-verified")
    if stop.get("process_group_empty") is not True:
        raise DeploymentEvidenceError("Stop evidence does not prove an empty PGID")
    if stop.get("health_down") is not True:
        raise DeploymentEvidenceError("Stop evidence does not prove health is down")

    expected_pid = _integer(
        (start.get("process_identity") or {}).get("pid"),
        "start process identity pid",
        minimum=1,
    )
    recorded = _integer(stop.get("recorded_pgid"), "stop recorded_pgid", minimum=1)
    recorded_file = _integer(
        _text(root, "recorded_pgid.txt"), "recorded_pgid.txt", minimum=1
    )
    if recorded != expected_pid or recorded_file != expected_pid:
        raise DeploymentEvidenceError(
            "Stop PGID does not match the audited start-process identity"
        )

    hashes = stop.get("evidence_files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != STOP_HASHED_FILES:
        raise DeploymentEvidenceError(
            "Stop hash inventory is incomplete or contains unknown files"
        )
    for name in sorted(STOP_HASHED_FILES):
        expected = hashes[name]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise DeploymentEvidenceError(f"Malformed stop SHA256 for {name}")
        if file_sha256(_file(root, name)) != expected:
            raise DeploymentEvidenceError(f"Stop evidence SHA256 mismatch: {name}")

    before = _gpu_stop_row(root, "gpu_before_stop.csv")
    after = _gpu_stop_row(root, "gpu_after_stop.csv")
    if before[:3] != after[:3]:
        raise DeploymentEvidenceError("Before/after stop GPU identity or size differs")
    if before[0] != _integer(start.get("selected_gpu"), "start selected GPU"):
        raise DeploymentEvidenceError("Stop GPU differs from start manifest")
    if before[1] != start.get("selected_gpu_uuid"):
        raise DeploymentEvidenceError("Stop GPU UUID differs from start manifest")
    used_before = _integer(stop.get("gpu_used_before_mib"), "stop used before", minimum=0)
    used_after = _integer(stop.get("gpu_used_after_mib"), "stop used after", minimum=0)
    released = _integer(stop.get("gpu_memory_released_mib"), "stop released", minimum=1)
    if used_before != before[3] or used_after != after[3]:
        raise DeploymentEvidenceError("Stop manifest GPU use differs from CSV evidence")
    if released != used_before - used_after or used_after >= used_before:
        raise DeploymentEvidenceError("GPU memory release arithmetic is inconsistent")

    group_members = _process_group_members(root, recorded)
    if not any(
        "vllm.entrypoints.openai.api_server" in command
        for command in group_members.values()
    ):
        raise DeploymentEvidenceError(
            "Pre-stop process group does not prove the recorded vLLM server"
        )
    if _text(root, "process_group_after.txt", allow_empty=True):
        raise DeploymentEvidenceError("Recorded process group is non-empty after stop")
    if _text(root, "health_down.txt") != "health endpoint unreachable as required":
        raise DeploymentEvidenceError("health_down.txt has an unexpected attestation")

    compute_before = _compute_apps(root, "compute_apps_before.txt")
    compute_after = _compute_apps(root, "compute_apps_after.txt")
    experiment_compute_pids = {
        pid
        for pid, gpu_uuid, _ in compute_before
        if gpu_uuid == before[1] and pid in group_members
    }
    if not experiment_compute_pids:
        raise DeploymentEvidenceError(
            "Pre-stop evidence has no selected-GPU compute PID in the recorded PGID"
        )
    remaining_compute_pids = {pid for pid, _, _ in compute_after}
    leaked = sorted(experiment_compute_pids & remaining_compute_pids)
    if leaked:
        raise DeploymentEvidenceError(
            "Experiment compute PIDs remain after stop: "
            + ", ".join(str(pid) for pid in leaked)
        )

    started = _timestamp(
        _text(Path(start["metadata_root"]), "started_at_utc.txt"),
        "started_at_utc",
    )
    requested = _timestamp(
        _text(root, "stop_requested_at_utc.txt"), "stop_requested_at_utc"
    )
    stopped = _timestamp(_text(root, "stopped_at_utc.txt"), "stopped_at_utc")
    if not started <= requested <= stopped:
        raise DeploymentEvidenceError(
            "Deployment start/stop timestamps are not monotonic"
        )

    return {
        "status": "validated",
        "run_id": start["run_id"],
        "deployment_epoch": start["deployment_epoch"],
        "selected_gpu": before[0],
        "selected_gpu_uuid": before[1],
        "recorded_pgid": recorded,
        "gpu_memory_released_mib": released,
        "start_manifest_sha256": file_sha256(start_manifest_path),
        "stop_manifest_sha256": file_sha256(root / "stop_manifest.json"),
        "evidence_files_sha256": dict(sorted(hashes.items())),
    }


def import_stop_evidence(
    source: Path,
    destination: Path,
    start_manifest_path: Path,
) -> dict[str, Any]:
    source_root = _root(source, "Source vLLM stop evidence")
    source_attestation = validate_stop_evidence(source_root, start_manifest_path)
    destination = destination.absolute()

    if destination.exists():
        existing = validate_stop_evidence(destination, start_manifest_path)
        if existing != source_attestation:
            raise DeploymentEvidenceError(
                "Refusing to overwrite different imported stop evidence"
            )
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.import.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        lock_fd = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise DeploymentEvidenceError(f"Another stop import holds {lock}") from exc
    os.close(lock_fd)

    temp = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        for name in STOP_FILES:
            source_path = _file(source_root, name)
            target = temp / name
            with source_path.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
        copied = validate_stop_evidence(temp, start_manifest_path)
        if destination.exists():
            existing = validate_stop_evidence(destination, start_manifest_path)
            if existing != copied:
                raise DeploymentEvidenceError(
                    "Another writer imported different stop evidence"
                )
            return existing
        os.rename(temp, destination)
        return validate_stop_evidence(destination, start_manifest_path)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
        lock.unlink(missing_ok=True)
