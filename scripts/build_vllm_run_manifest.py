#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mementohls.deployment_evidence import (
    DeploymentEvidenceError,
    validate_vllm_run_manifest,
    write_vllm_run_manifest,
)
from mementohls.models import file_sha256


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Build or validate a DAC2027 vllm_run_manifest.json using only A100/A6000 "
            "run metadata already copied onto V80. This command performs no SSH, "
            "network, GPU, vLLM, or Vitis operation."
        )
    )
    commands = value.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build atomically, then validate")
    build.add_argument("--metadata-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--endpoint", required=True)
    build.add_argument(
        "--served-model",
        default="meta-llama/Llama-3-8b-chat-hf",
    )
    build.add_argument(
        "--inference-server",
        choices=("A100", "A6000"),
        default="A100",
    )

    validate = commands.add_parser(
        "validate", help="strictly validate an existing manifest and its files"
    )
    validate.add_argument("--manifest", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "build":
            payload = write_vllm_run_manifest(
                args.metadata_root,
                args.output,
                endpoint=args.endpoint,
                served_model=args.served_model,
                inference_server=args.inference_server,
            )
            path = args.output.resolve()
            action = "built-and-validated"
        else:
            payload = validate_vllm_run_manifest(args.manifest)
            path = args.manifest.resolve()
            action = "validated"
    except DeploymentEvidenceError as exc:
        raise SystemExit(f"deployment evidence rejected: {exc}") from exc

    print(
        json.dumps(
            {
                "status": action,
                "manifest": str(path),
                "manifest_sha256": file_sha256(path),
                "run_id": payload["run_id"],
                "deployment_epoch": payload["deployment_epoch"],
                "inference_server": payload["inference_server"],
                "endpoint": payload["endpoint"],
                "selected_gpu": payload["selected_gpu"],
                "selected_gpu_uuid": payload["selected_gpu_uuid"],
                "tensor_parallel_size": payload["tensor_parallel_size"],
                "metadata_root": payload["metadata_root"],
                "metadata_file_count": len(payload["metadata_files_sha256"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
