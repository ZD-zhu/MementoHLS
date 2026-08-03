#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${HLS_MODEL_PATH:-/mnt/sdb/llm_models/Meta-Llama-3-8B-Instruct}"
SERVED_MODEL="${HLS_SERVED_MODEL:-meta-llama/Llama-3-8b-chat-hf}"
CONDA_ENV="${HLS_VLLM_CONDA_ENV:-vllm}"
CONDA_BIN="${HLS_CONDA_BIN:-}"
if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" "/opt/conda/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONDA_BIN}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi
[[ -x "${CONDA_BIN}" ]] || {
  echo "Unable to locate an executable conda; set HLS_CONDA_BIN" >&2
  exit 1
}

PORT="${HLS_VLLM_PORT:-8000}"
RUN_ROOT="${HLS_VLLM_RUN_ROOT:-/home/xjzhu/HLS/DAC_test/vllm_runs}"
CACHE_ROOT="${HLS_CACHE_ROOT:-/home/xjzhu/HLS/DAC_test/cache}"
MIN_FREE_MIB="${HLS_MIN_FREE_MIB:-30000}"
GPU_MEMORY_UTILIZATION="${HLS_GPU_MEMORY_UTILIZATION:-0.90}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${CACHE_ROOT}"/{vllm,torchinductor,huggingface,cuda,triton,torch,tmp}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if find "${MODEL_PATH}" -type f -name '*.incomplete' -print -quit | grep -q .; then
  echo "Model download is incomplete; refusing to start vLLM" >&2
  exit 1
fi
MODEL_INDEX="${MODEL_PATH}/model.safetensors.index.json"
[[ -f "${MODEL_INDEX}" ]] || {
  echo "Missing model.safetensors.index.json: ${MODEL_INDEX}" >&2
  exit 1
}
python3 -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); data=json.load(open(sys.argv[2], encoding="utf-8")); shards=sorted(set(data.get("weight_map", {}).values())); assert shards, "empty weight_map"; missing=[name for name in shards if not (root/name).is_file() or (root/name).stat().st_size == 0]; assert not missing, f"missing/empty shards: {missing}"' \
  "${MODEL_PATH}" "${MODEL_INDEX}"
if ! [[ "${MIN_FREE_MIB}" =~ ^[0-9]+$ ]]; then
  echo "HLS_MIN_FREE_MIB must be an integer MiB value" >&2
  exit 1
fi
if curl --silent --fail --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already serves a healthy process; stop and audit it first" >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.free \
  --format=csv,noheader,nounits >"${RUN_DIR}/gpu_inventory_before.csv"

requested_gpu="${HLS_VLLM_GPU:-}"
if [[ -n "${requested_gpu}" ]]; then
  [[ "${requested_gpu}" =~ ^[0-7]$ ]] || {
    echo "HLS_VLLM_GPU must be one physical GPU index from 0 through 7" >&2
    exit 1
  }
  gpu_candidates=("${requested_gpu}")
else
  mapfile -t gpu_candidates < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 >= 0 && $1 <= 7) print $1 "," $2}' |
      sort -t, -k2,2nr |
      cut -d, -f1
  )
fi
[[ "${#gpu_candidates[@]}" -gt 0 ]] || {
  echo "No GPU in the allowed physical index range 0-7 was detected" >&2
  exit 1
}

ACTIVE_PID=""
COMMITTED=0
cleanup_active() {
  if [[ "${COMMITTED}" -eq 0 && -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -TERM -- "-${ACTIVE_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${ACTIVE_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${ACTIVE_PID}" 2>/dev/null || true
  fi
}
trap cleanup_active EXIT INT TERM

launch_on_gpu() {
  local gpu="$1"
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU ${gpu} has ${free_mib} MiB free, below required ${MIN_FREE_MIB} MiB" >&2
    return 1
  fi
  local log="${RUN_DIR}/vllm_gpu${gpu}.log"
  local -a command=(
    "${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output
    python -m vllm.entrypoints.openai.api_server
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --max-model-len 8192
    --max-num-seqs 8
    --tensor-parallel-size 1
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --generation-config vllm
    --host 0.0.0.0
    --port "${PORT}"
  )
  local -a envs=(
    "XDG_CACHE_HOME=${CACHE_ROOT}"
    "VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm"
    "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor"
    "HF_HOME=${CACHE_ROOT}/huggingface"
    "CUDA_CACHE_PATH=${CACHE_ROOT}/cuda"
    "TRITON_CACHE_DIR=${CACHE_ROOT}/triton"
    "TORCH_HOME=${CACHE_ROOT}/torch"
    "TMPDIR=${CACHE_ROOT}/tmp"
    "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    "CUDA_VISIBLE_DEVICES=${gpu}"
  )
  printf '%q ' env "${envs[@]}" "${command[@]}" >"${RUN_DIR}/launch_command_gpu${gpu}.txt"
  printf '\n' >>"${RUN_DIR}/launch_command_gpu${gpu}.txt"

  setsid env "${envs[@]}" "${command[@]}" >"${log}" 2>&1 < /dev/null &
  ACTIVE_PID=$!
  echo "${ACTIVE_PID}" >"${RUN_DIR}/vllm.pgid"
  echo "${ACTIVE_PID}" >"${RUN_DIR}/vllm.pid"
  printf '%s,%s\n' "${gpu}" "${free_mib}" >"${RUN_DIR}/selected_gpu_and_free_mib.csv"

  for _ in $(seq 1 180); do
    if ! kill -0 "${ACTIVE_PID}" 2>/dev/null; then
      wait "${ACTIVE_PID}" || true
      ACTIVE_PID=""
      return 1
    fi
    if curl --silent --fail --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  cleanup_active
  ACTIVE_PID=""
  return 1
}

selected=""
for gpu in "${gpu_candidates[@]}"; do
  if launch_on_gpu "${gpu}"; then
    selected="${gpu}"
    break
  fi
  echo "GPU ${gpu} unavailable; trying the next highest-free allowed GPU" >&2
done
[[ -n "${selected}" ]] || {
  echo "No single allowed GPU had enough free memory or launched successfully" >&2
  exit 1
}

curl --silent --show-error --fail --max-time 10 \
  "http://127.0.0.1:${PORT}/health" >"${RUN_DIR}/health.txt"
curl --silent --show-error --fail --max-time 10 \
  "http://127.0.0.1:${PORT}/v1/models" >"${RUN_DIR}/models.json"
curl --silent --show-error --fail --max-time 120 \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: HLS_SMOKE_OK\"}],\"temperature\":0,\"max_tokens\":16,\"seed\":0}" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" >"${RUN_DIR}/smoke.json"
python3 -c 'import json,sys; models=json.load(open(sys.argv[1], encoding="utf-8")); smoke=json.load(open(sys.argv[2], encoding="utf-8")); assert any(item.get("id")==sys.argv[3] for item in models.get("data", [])); content=smoke["choices"][0]["message"]["content"].strip(); assert content=="HLS_SMOKE_OK", repr(content)' \
  "${RUN_DIR}/models.json" "${RUN_DIR}/smoke.json" "${SERVED_MODEL}"
curl --silent --show-error --fail --max-time 10 \
  "http://127.0.0.1:${PORT}/metrics" >"${RUN_DIR}/process_metrics_identity.prom"
awk '$1 == "process_start_time_seconds" {print $2}' \
  "${RUN_DIR}/process_metrics_identity.prom" \
  >"${RUN_DIR}/process_start_time_seconds.txt"
[[ -s "${RUN_DIR}/process_start_time_seconds.txt" ]] || {
  echo "vLLM metrics lacks process_start_time_seconds" >&2
  exit 1
}

date -u +%Y-%m-%dT%H:%M:%SZ >"${RUN_DIR}/started_at_utc.txt"
printf '%s\n' "${CONDA_ENV}" >"${RUN_DIR}/conda_environment.txt"
printf '%s\n' "${CONDA_BIN}" >"${RUN_DIR}/conda_executable.txt"
printf '%s\n' "${GPU_MEMORY_UTILIZATION}" >"${RUN_DIR}/gpu_memory_utilization.txt"
hostname -f >"${RUN_DIR}/hostname.txt"
nvidia-smi -q >"${RUN_DIR}/nvidia_smi_q.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits >"${RUN_DIR}/gpu_inventory_after.csv"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip show vllm transformers >"${RUN_DIR}/vllm_transformers_versions.txt"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m pip freeze >"${RUN_DIR}/environment_pip_freeze.txt"
find "${MODEL_PATH}" -maxdepth 1 -type f \
  \( -name 'config*.json' -o -name 'generation_config.json' -o -name 'tokenizer*.json' -o -name 'model*.json' \) \
  -print0 | sort -z | xargs -0 -r sha256sum >"${RUN_DIR}/model_config_sha256.txt"
find "${MODEL_PATH}" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name 'model.safetensors.index.json' \
     -o -name 'config*.json' -o -name 'generation_config.json' \
     -o -name 'tokenizer*' -o -name 'special_tokens_map.json' \) \
  -print0 | sort -z | xargs -0 -r sha256sum \
  >"${RUN_DIR}/model_artifact_sha256.txt"
python3 -c 'import json,pathlib,sys; pid=int(sys.argv[1]); stat=pathlib.Path(f"/proc/{pid}/stat").read_text().split(); value={"pid":pid,"process_start_ticks":int(stat[21]),"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()}; pathlib.Path(sys.argv[2]).write_text(json.dumps(value,indent=2))' \
  "${ACTIVE_PID}" "${RUN_DIR}/process_identity.json"
find "${MODEL_PATH}" -type f -printf '%P,%s,%T@\n' | sort \
  >"${RUN_DIR}/model_file_inventory.csv"

ln -sfn "${RUN_DIR}" "${RUN_ROOT}/latest"
ln -sfn "${RUN_DIR}" "${RUN_ROOT}/current"
COMMITTED=1
trap - EXIT INT TERM
echo "vLLM ready on one physical GPU ${selected}; PID/PGID ${ACTIVE_PID}; metadata ${RUN_DIR}"
