#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${MEMENTOHLS_ITERATION_ROOT:?set the frozen iteration root}"
: "${MEMENTOHLS_DEPLOYMENT_BINDING:?set deployment_binding.json}"
: "${MEMENTOHLS_VLLM_MANIFEST:?set vllm_run_manifest.json}"
: "${MEMENTOHLS_BASE_URL:?set the live OpenAI-compatible endpoint}"

exec "${REPO_ROOT}/scripts/run_full.sh"

