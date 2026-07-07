#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR=${LOG_DIR:-logs/train}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29500}
PYTHON=${PYTHON:-.venv_sft/bin/python}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-auto}

mkdir -p "$LOG_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/run_sft_list.sh CONFIG.yaml [CONFIG2.yaml ...]

Examples:
  bash scripts/train/run_sft_list.sh \
    config/sft/qwen3_sft-v8.yaml \
    config/sft/qwen3_sft-v9.yaml \
    config/sft/qwen3_sft-v10.yaml

Optional env:
  CUDA_VISIBLE_DEVICES=0,1,2,3
  NPROC_PER_NODE=4
  PYTHON=.venv_sft/bin/python
  LOG_DIR=logs/train
  MASTER_PORT_BASE=29500
  RESUME_FROM_CHECKPOINT=auto|none|/path/to/checkpoint
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

port_is_free() {
  local port="$1"
  ! timeout 1 bash -c "</dev/tcp/${MASTER_ADDR}/${port}" >/dev/null 2>&1
}

next_free_port() {
  local port="$1"
  while ! port_is_free "$port"; do
    port="$((port + 1))"
  done
  echo "$port"
}

run_one() {
  local idx="$1"
  local config="$2"
  local stem
  local log_path
  local master_port

  if [[ ! -f "$config" ]]; then
    echo "[FAIL] config not found: ${config}"
    return 2
  fi

  stem="$(basename "$config")"
  stem="${stem%.*}"
  stem="${stem// /_}"
  log_path="${LOG_DIR}/sft_${idx}_${stem}.log"
  master_port="$(next_free_port "$((MASTER_PORT_BASE + idx - 1))")"

  echo "[START] ${idx}: ${config}"
  echo "[LOG] ${log_path}"
  echo "[MASTER_PORT] ${master_port}"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$master_port" \
  PYTHON="$PYTHON" \
  CONFIG="$config" \
  RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
  bash scripts/train/run_sft.sh 2>&1 | tee "$log_path"

  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] ${config} exited with status ${rc}"
    echo "[STOP] remaining SFT runs will not start. Check ${log_path}"
    return "$rc"
  fi

  echo "[DONE] ${config}"
}

idx=0
for config in "$@"; do
  idx="$((idx + 1))"
  run_one "$idx" "$config" || exit $?
done

echo "[ALL DONE] SFT runs finished."
