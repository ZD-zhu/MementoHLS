#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${HLS_VLLM_RUN_ROOT:-$HOME/HLS-skill/DAC2027/vllm_runs}"
PORT="${HLS_VLLM_PORT:-8000}"
target="${1:-${RUN_ROOT}/current}"
if [[ ! -e "${target}" ]]; then
  echo "No active vLLM run metadata at ${target}; nothing to stop"
  exit 0
fi
RUN_DIR="$(readlink -f "${target}")"
pid_file="${RUN_DIR}/vllm.pgid"
[[ -f "${pid_file}" ]] || {
  echo "Missing ${pid_file}; refusing to guess a process" >&2
  exit 1
}
pgid="$(tr -d '[:space:]' <"${pid_file}")"
[[ "${pgid}" =~ ^[0-9]+$ ]] || {
  echo "Invalid process group id in ${pid_file}" >&2
  exit 1
}
EVIDENCE_DIR="${RUN_DIR}/stop_evidence"
mkdir -p "${EVIDENCE_DIR}"
date -u +%Y-%m-%dT%H:%M:%SZ >"${EVIDENCE_DIR}/stop_requested_at_utc.txt"
printf '%s
' "${pgid}" >"${EVIDENCE_DIR}/recorded_pgid.txt"
selected_gpu="$(cut -d, -f1 "${RUN_DIR}/selected_gpu_and_free_mib.csv" | tr -d '[:space:]')"
[[ "${selected_gpu}" =~ ^[0-7]$ ]] || {
  echo "Invalid selected GPU evidence" >&2
  exit 1
}
ps -eo pid=,pgid=,stat=,args= | awk -v target="${pgid}" '$2 == target'   >"${EVIDENCE_DIR}/process_group_before.txt"
nvidia-smi -i "${selected_gpu}"   --query-gpu=index,uuid,memory.total,memory.used,memory.free   --format=csv,noheader,nounits >"${EVIDENCE_DIR}/gpu_before_stop.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits   >"${EVIDENCE_DIR}/compute_apps_before.txt" 2>/dev/null || true

if awk 'NF {found=1} END {exit !found}' "${EVIDENCE_DIR}/process_group_before.txt"; then
  if ! grep -Fq 'vllm.entrypoints.openai.api_server' "${EVIDENCE_DIR}/process_group_before.txt"; then
    echo "Process group ${pgid} is not the recorded vLLM server; refusing to kill it" >&2
    exit 1
  fi
  kill -TERM -- "-${pgid}"
  for _ in $(seq 1 60); do
    if ! ps -eo pgid= | awk -v target="${pgid}" '$1 == target {found=1} END {exit !found}'; then
      break
    fi
    sleep 1
  done
  if ps -eo pgid= | awk -v target="${pgid}" '$1 == target {found=1} END {exit !found}'; then
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
fi

for _ in $(seq 1 30); do
  ps -eo pid=,pgid=,stat=,args= | awk -v target="${pgid}" '$2 == target'     >"${EVIDENCE_DIR}/process_group_after.txt"
  if [[ ! -s "${EVIDENCE_DIR}/process_group_after.txt" ]]; then
    break
  fi
  sleep 1
done
if [[ -s "${EVIDENCE_DIR}/process_group_after.txt" ]]; then
  echo "Recorded vLLM process group ${pgid} still has members" >&2
  exit 1
fi

for _ in $(seq 1 20); do
  curl --silent --fail --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || break
  sleep 1
done
if curl --silent --fail --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} remains healthy; audit manually" >&2
  exit 1
fi
printf '%s
' "health endpoint unreachable as required"   >"${EVIDENCE_DIR}/health_down.txt"
nvidia-smi -i "${selected_gpu}"   --query-gpu=index,uuid,memory.total,memory.used,memory.free   --format=csv,noheader,nounits >"${EVIDENCE_DIR}/gpu_after_stop.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits   >"${EVIDENCE_DIR}/compute_apps_after.txt" 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >"${EVIDENCE_DIR}/stopped_at_utc.txt"
cp "${EVIDENCE_DIR}/stopped_at_utc.txt" "${RUN_DIR}/stopped_at_utc.txt"
python3 -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); before=[x.strip() for x in (root/"gpu_before_stop.csv").read_text().split(",")]; after=[x.strip() for x in (root/"gpu_after_stop.csv").read_text().split(",")]; files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file() and p.name!="stop_manifest.json"}; value={"status":"vllm-stopped-and-verified","recorded_pgid":int((root/"recorded_pgid.txt").read_text()),"process_group_empty":not (root/"process_group_after.txt").read_text().strip(),"health_down":True,"selected_gpu":int(after[0]),"selected_gpu_uuid":after[1],"gpu_used_before_mib":int(before[3]),"gpu_used_after_mib":int(after[3]),"gpu_memory_released_mib":int(before[3])-int(after[3]),"evidence_files_sha256":files}; (root/"stop_manifest.json").write_text(json.dumps(value,indent=2))'   "${EVIDENCE_DIR}"
cat "${EVIDENCE_DIR}/stop_manifest.json"
echo "Stopped vLLM process group ${pgid}; endpoint is down and GPU release evidence is saved"
