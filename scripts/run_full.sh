#!/usr/bin/env bash
set -euo pipefail

REPO="${MEMENTOHLS_REPO:-/home/xjzhu/HLS/DAC_test/MementoHLS}"
ITERATION_ROOT="${MEMENTOHLS_ITERATION_ROOT:?set MEMENTOHLS_ITERATION_ROOT}"
FREEZE_MANIFEST="${MEMENTOHLS_FREEZE_MANIFEST:-${ITERATION_ROOT}/protocol/freeze_manifest.json}"
DEPLOYMENT_BINDING="${MEMENTOHLS_DEPLOYMENT_BINDING:?set MEMENTOHLS_DEPLOYMENT_BINDING}"
VLLM_MANIFEST="${MEMENTOHLS_VLLM_MANIFEST:?set MEMENTOHLS_VLLM_MANIFEST}"
SCHEDULE="${MEMENTOHLS_SCHEDULE:-/home/xjzhu/HLS/DAC_test/preregistration/${MEMENTOHLS_DATASET_KIND:-hls_eval}/schedule.json}"
BASE_URL="${MEMENTOHLS_BASE_URL:-http://172.16.120.87:8000/v1}"
MODEL="${MEMENTOHLS_MODEL:-meta-llama/Llama-3-8b-chat-hf}"
DATASET_KIND="${MEMENTOHLS_DATASET_KIND:-hls_eval}"
if [[ "${DATASET_KIND}" == "bench4hls" ]]; then
  DATASET_ROOT="${MEMENTOHLS_DATASET_ROOT:-/home/xjzhu/HLS/DAC_test/datasets/Bench4HLS/benchmark}"
  EXPECTED_CASE_COUNT="${MEMENTOHLS_EXPECTED_CASE_COUNT:-170}"
else
  DATASET_ROOT="${MEMENTOHLS_DATASET_ROOT:-/home/xjzhu/HLS/DAC_test/datasets/Hls-Eval/hls_eval_data}"
  EXPECTED_CASE_COUNT="${MEMENTOHLS_EXPECTED_CASE_COUNT:-94}"
fi
BENCH4HLS_FROZEN_EPOCH="${MEMENTOHLS_BENCH4HLS_FROZEN_EPOCH:-@2027-01-01 00:00:00}"
CAPACITY_LOG="${ITERATION_ROOT}/protocol/v80_capacity.jsonl"
STABLE_SECONDS="${MEMENTOHLS_GATE_STABLE_SECONDS:-300}"
START_CPU_MAX="${MEMENTOHLS_START_CPU_MAX:-45}"
START_LOAD_MAX="${MEMENTOHLS_START_LOAD_MAX:-96}"
PAUSE_CPU_MAX="${MEMENTOHLS_PAUSE_CPU_MAX:-70}"
PAUSE_LOAD_MAX="${MEMENTOHLS_PAUSE_LOAD_MAX:-120}"
START_MEMORY_GIB_MIN="${MEMENTOHLS_START_MEMORY_GIB_MIN:-380}"
PAUSE_MEMORY_GIB_MIN="${MEMENTOHLS_PAUSE_MEMORY_GIB_MIN:-350}"
PROCESS_RSS_GIB_MAX="${MEMENTOHLS_PROCESS_RSS_GIB_MAX:-300}"
PREVALIDATED_CAPACITY_LOG="${MEMENTOHLS_PREVALIDATED_CAPACITY_LOG:-}"
PREVALIDATED_MAX_AGE="${MEMENTOHLS_PREVALIDATED_MAX_AGE:-600}"
export PYTHONPATH="${REPO}/src:${REPO}"
gate=(--start-cpu-max "${START_CPU_MAX}" --start-load-max "${START_LOAD_MAX}" --pause-cpu-max "${PAUSE_CPU_MAX}" --pause-load-max "${PAUSE_LOAD_MAX}" --start-memory-available-gib-min "${START_MEMORY_GIB_MIN}" --pause-memory-available-gib-min "${PAUSE_MEMORY_GIB_MIN}" --process-tree-rss-gib-max "${PROCESS_RSS_GIB_MAX}")
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${ITERATION_ROOT}/protocol"
exec 9>"${ITERATION_ROOT}/protocol/formal_driver.lock"
if ! flock -n 9; then
  echo "Another formal driver owns ${ITERATION_ROOT}" >&2
  exit 1
fi

common=(
  --output-dir "${ITERATION_ROOT}"
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --inference-server A100
  --freeze-manifest "${FREEZE_MANIFEST}"
  --deployment-binding "${DEPLOYMENT_BINDING}"
  --vllm-run-metadata "${VLLM_MANIFEST}"
  --iteration-id "$(basename "${ITERATION_ROOT}")"
  --dataset-kind "${DATASET_KIND}"
  --dataset-root "${DATASET_ROOT}"
  --expected-case-count "${EXPECTED_CASE_COUNT}"
  --bench4hls-frozen-epoch "${BENCH4HLS_FROZEN_EPOCH}"
  --formal --resume
  --samples 5 --seeds 0 1 2 3 4 --temperature 0.7
  --max-repair-rounds 5 --context-limit 8192 --max-tokens 4096
  --design-workers 16 --llm-workers 8 --csim-workers 12 --synth-workers 8
  --runtime-start-cpu-max "${START_CPU_MAX}"
  --runtime-start-load-max "${START_LOAD_MAX}"
  --runtime-pause-cpu-max "${PAUSE_CPU_MAX}"
  --runtime-pause-load-max "${PAUSE_LOAD_MAX}"
  --runtime-start-memory-available-gib-min "${START_MEMORY_GIB_MIN}"
  --runtime-pause-memory-available-gib-min "${PAUSE_MEMORY_GIB_MIN}"
  --runtime-process-tree-rss-gib-max "${PROCESS_RSS_GIB_MAX}"
  --llm-timeout 600 --csim-timeout 360 --synth-timeout 360
  --review-infrastructure-retries 2
)

if [[ -n "${PREVALIDATED_CAPACITY_LOG}" ]]; then
  python "${REPO}/scripts/validate_capacity_evidence.py" --log "${PREVALIDATED_CAPACITY_LOG}" --stable-seconds "${STABLE_SECONDS}" --max-age-seconds "${PREVALIDATED_MAX_AGE}" --start-cpu-max "${START_CPU_MAX}" --start-load-max "${START_LOAD_MAX}"
  python "${REPO}/scripts/wait_for_v80_capacity.py" --log "${CAPACITY_LOG}" --stable-seconds "${STABLE_SECONDS}" --continuation-gate "${gate[@]}"
else
  python "${REPO}/scripts/wait_for_v80_capacity.py" --log "${CAPACITY_LOG}" --stable-seconds "${STABLE_SECONDS}" "${gate[@]}"
fi
nice -n 10 ionice -c2 -n7 taskset -c 0-79 python "${REPO}/main.py" --mode zero_shot "${common[@]}"
python "${REPO}/scripts/lock_targets.py" --iteration-root "${ITERATION_ROOT}"

mapfile -t blocks < <(python - "${SCHEDULE}" <<'PY'
import json,sys
schedule=json.load(open(sys.argv[1]))
for item in schedule['blocks']:
    print(item['block_id'])
PY
)
for block in "${blocks[@]}"; do
  mapfile -t cases < <(python - "${SCHEDULE}" "${block}" <<'PY'
import json,sys
schedule=json.load(open(sys.argv[1]))
entry=next(item for item in schedule['blocks'] if int(item['block_id']) == int(sys.argv[2]))
for item in entry['cases']:
    print(item['case'])
PY
)
  mapfile -t modes < <(python - "${SCHEDULE}" "${block}" <<'PY'
import json,sys
schedule=json.load(open(sys.argv[1]))
entry=next(item for item in schedule['blocks'] if int(item['block_id']) == int(sys.argv[2]))
for mode in entry['repair_mode_order']:
    print(mode)
PY
)
  case_args=()
  for case_name in "${cases[@]}"; do case_args+=(--case "${case_name}"); done
  for mode in "${modes[@]}"; do
    python "${REPO}/scripts/wait_for_v80_capacity.py" --log "${CAPACITY_LOG}" --stable-seconds "${STABLE_SECONDS}" --continuation-gate "${gate[@]}"
    nice -n 10 ionice -c2 -n7 taskset -c 0-79 python "${REPO}/main.py" --mode "${mode}" --block-id "${block}" "${case_args[@]}" --round0-source "${ITERATION_ROOT}/zero_shot" "${common[@]}"
  done
done

python "${REPO}/scripts/finalize_blocks.py" --iteration-root "${ITERATION_ROOT}" --schedule "${SCHEDULE}"
python "${REPO}/main.py" --analyze-only --output-dir "${ITERATION_ROOT}"
