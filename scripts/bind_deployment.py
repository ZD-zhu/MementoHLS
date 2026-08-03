#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mementohls.models import file_sha256
from mementohls.orchestrator import _load_vllm_deployment_manifest


def bind_or_validate(
    *,
    iteration_root: Path,
    manifest_path: Path,
    endpoint: str,
    server: str,
    model: str,
) -> Path:
    iteration_root = iteration_root.resolve()
    manifest_path = manifest_path.resolve()
    deployment = _load_vllm_deployment_manifest(
        manifest_path,
        base_url=endpoint,
        model=model,
        inference_server=server,
        formal_run=True,
    )
    protocol = iteration_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    binding_path = protocol / "deployment_binding.json"
    stable = {
        "schema_version": 1,
        "status": "bound-before-first-trajectory",
        "iteration_id": iteration_root.name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "endpoint": endpoint.rstrip("/"),
        "inference_server": server,
        "served_model": model,
        "remote_hostname": deployment["remote_hostname"],
        "run_id": deployment["run_id"],
        "deployment_epoch": deployment["deployment_epoch"],
        "model_listing_created_at_capture": int(
            deployment["model_listing_created_at_capture"]
        ),
        "prometheus_process_start_time_seconds": deployment[
            "prometheus_process_start_time_seconds"
        ],
        "model_artifact_manifest_sha256": deployment[
            "model_artifact_manifest_sha256"
        ],
        "process_identity_sha256": deployment["process_identity_sha256"],
        "selected_gpu": int(deployment["selected_gpu"]),
        "selected_gpu_uuid": deployment["selected_gpu_uuid"],
        "tensor_parallel_size": int(deployment["tensor_parallel_size"]),
        "conda_environment": deployment["conda_environment"],
        "model_path": deployment["model_path"],
        "launch_command": deployment["launch_command"],
    }
    if binding_path.exists():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in stable}
        if comparable != stable:
            raise RuntimeError(
                "Iteration is already bound to a different vLLM deployment; create a new vN: "
                + json.dumps({"existing": comparable, "requested": stable}, sort_keys=True)
            )
        return binding_path
    value = {**stable, "bound_at_unix": time.time(), "bound_by_pid": os.getpid()}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(binding_path, flags, 0o644)
    except FileExistsError:
        return bind_or_validate(
            iteration_root=iteration_root,
            manifest_path=manifest_path,
            endpoint=endpoint,
            server=server,
            model=model,
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return binding_path


def main() -> None:
    parser = argparse.ArgumentParser("Atomically bind one MementoHLS iteration to one vLLM deployment")
    parser.add_argument("--iteration-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    path = bind_or_validate(
        iteration_root=args.iteration_root,
        manifest_path=args.manifest,
        endpoint=args.endpoint,
        server=args.server,
        model=args.model,
    )
    print(path)


if __name__ == "__main__":
    main()
