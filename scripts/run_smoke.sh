#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
RESULT_DIR="${RESULT_DIR:-/home/xjzhu/HLS/DAC_test/result/artifact_smoke}"
mkdir -p "${RESULT_DIR}"

python -m pytest -q --junitxml="${RESULT_DIR}/unit_tests.xml"

if [[ "${RUN_VITIS_INTEGRATION:-0}" == "1" ]]; then
  nice -n 10 ionice -c2 -n7 env RUN_VITIS_INTEGRATION=1 \
    python -m pytest -q "${REPO_ROOT}/tests/test_vitis_integration.py" \
    --junitxml="${RESULT_DIR}/vitis_smoke.xml"
fi

echo "Smoke evidence written to ${RESULT_DIR}"
