#!/usr/bin/env bash
set -euo pipefail

REPO="${MEMENTOHLS_REPO:-/home/xjzhu/HLS/DAC_test/MementoHLS}"
ITERATION_ROOT="${MEMENTOHLS_ITERATION_ROOT:?set MEMENTOHLS_ITERATION_ROOT}"
SELECTION="${MEMENTOHLS_SELECTION:?set MEMENTOHLS_SELECTION}"
SELECTION_KEY="${MEMENTOHLS_SELECTION_KEY:-smoke_cases}"
DATASET_KIND="${MEMENTOHLS_DATASET_KIND:-bench4hls}"
BASE_URL="${MEMENTOHLS_BASE_URL:-http://172.16.120.87:8000/v1}"
MODEL="${MEMENTOHLS_MODEL:-meta-llama/Llama-3-8b-chat-hf}"
VLLM_MANIFEST="${MEMENTOHLS_VLLM_MANIFEST:?set MEMENTOHLS_VLLM_MANIFEST}"
DEPLOYMENT_BINDING="${MEMENTOHLS_DEPLOYMENT_BINDING:-${ITERATION_ROOT}/protocol/deployment_binding.json}"
PYTHON="${MEMENTOHLS_PYTHON:-/home/xjzhu/.conda/envs/Bench4HLS/bin/python}"
SEEDS_TEXT="${MEMENTOHLS_SEEDS:-0}"
MAX_REPAIR_ROUNDS="${MEMENTOHLS_MAX_REPAIR_ROUNDS:-5}"
STABLE_SECONDS="${MEMENTOHLS_GATE_STABLE_SECONDS:-30}"

if [[ "${DATASET_KIND}" == "bench4hls" ]]; then
  DATASET_ROOT="${MEMENTOHLS_DATASET_ROOT:-/home/xjzhu/HLS/DAC_test/datasets/Bench4HLS/benchmark}"
  DEFAULT_MODES="feedback rfl ercl mementohls mementohls_core_rules mementohls_all_rules"
else
  DATASET_ROOT="${MEMENTOHLS_DATASET_ROOT:-/home/xjzhu/HLS/DAC_test/datasets/Hls-Eval/hls_eval_data}"
  DEFAULT_MODES="feedback rfl ercl mementohls"
fi
MODES_TEXT="${MEMENTOHLS_MODES:-${DEFAULT_MODES}}"

export PYTHONPATH="${REPO}/src:${REPO}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "${ITERATION_ROOT}/protocol" "${ITERATION_ROOT}/logs"

exec 9>"${ITERATION_ROOT}/protocol/development_driver.lock"
if ! flock -n 9; then
  echo "Another development driver owns ${ITERATION_ROOT}" >&2
  exit 1
fi

mapfile -t cases < <("${PYTHON}" - "${SELECTION}" "${SELECTION_KEY}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
items = payload.get(sys.argv[2])
if not isinstance(items, list) or not items:
    raise SystemExit(f"selection key {sys.argv[2]!r} is missing or empty")
for item in items:
    print(item["case"] if isinstance(item, dict) else item)
PY
)
read -r -a seeds <<<"${SEEDS_TEXT}"
read -r -a modes <<<"${MODES_TEXT}"
case_args=()
for case_name in "${cases[@]}"; do
  case_args+=(--case "${case_name}")
done

common=(
  --output-dir "${ITERATION_ROOT}"
  --workspace-root /home/xjzhu/HLS/DAC_test
  --protected-source-root /home/xjzhu/HLS/DAC2027
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --inference-server A100
  --deployment-binding "${DEPLOYMENT_BINDING}"
  --vllm-run-metadata "${VLLM_MANIFEST}"
  --dataset-kind "${DATASET_KIND}"
  --dataset-root "${DATASET_ROOT}"
  --expected-case-count 0
  --samples "${#seeds[@]}"
  --seeds "${seeds[@]}"
  --temperature 0.7
  --max-repair-rounds "${MAX_REPAIR_ROUNDS}"
  --context-limit 8192
  --max-tokens 4096
  --design-workers 16
  --llm-workers 8
  --csim-workers 12
  --synth-workers 8
  --runtime-start-cpu-max 45
  --runtime-start-load-max 96
  --runtime-pause-cpu-max 70
  --runtime-pause-load-max 120
  --runtime-start-memory-available-gib-min 380
  --runtime-pause-memory-available-gib-min 350
  --runtime-process-tree-rss-gib-max 300
  --llm-timeout 600
  --csim-timeout 360
  --synth-timeout 360
  --review-infrastructure-retries 2
  --iteration-id "$(basename "${ITERATION_ROOT}")"
  --skip-summary
  --resume
  "${case_args[@]}"
)
gate=(
  --start-cpu-max 45
  --start-load-max 96
  --pause-cpu-max 70
  --pause-load-max 120
  --start-memory-available-gib-min 380
  --pause-memory-available-gib-min 350
  --process-tree-rss-gib-max 300
)
runner=(
  /home/xjzhu/anaconda3/bin/conda run -n Bench4HLS
  env "PYTHONPATH=${PYTHONPATH}" "PYTHONDONTWRITEBYTECODE=1"
  "OMP_NUM_THREADS=2" "OPENBLAS_NUM_THREADS=1" "MKL_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
  nice -n 10 ionice -c2 -n7 taskset -c 0-79
  python "${REPO}/main.py"
)

"${PYTHON}" "${REPO}/scripts/wait_for_v80_capacity.py" \
  --log "${ITERATION_ROOT}/protocol/v80_capacity.jsonl" \
  --stable-seconds "${STABLE_SECONDS}" "${gate[@]}"
"${runner[@]}" --mode zero_shot "${common[@]}" \
  >"${ITERATION_ROOT}/logs/zero_shot.log" 2>&1

for mode in "${modes[@]}"; do
  "${PYTHON}" "${REPO}/scripts/wait_for_v80_capacity.py" \
    --log "${ITERATION_ROOT}/protocol/v80_capacity.jsonl" \
    --stable-seconds "${STABLE_SECONDS}" --continuation-gate "${gate[@]}"
  "${runner[@]}" --mode "${mode}" \
    --round0-source "${ITERATION_ROOT}/zero_shot" "${common[@]}" \
    >"${ITERATION_ROOT}/logs/${mode}.log" 2>&1
done

/home/xjzhu/anaconda3/bin/conda run -n Bench4HLS \
  env "PYTHONPATH=${PYTHONPATH}" "PYTHONDONTWRITEBYTECODE=1" \
  python "${REPO}/scripts/analyze_development_matrix.py" \
  --root "${ITERATION_ROOT}" \
  --modes zero_shot "${modes[@]}" \
  --output "${ITERATION_ROOT}/analysis/development_analysis.json"
