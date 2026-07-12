#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=8
export DECORD_EOF_RETRY_MAX=2048001
export SWANLAB_DIR=${SWANLAB_DIR:-swanlog}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
PYTHON=${PYTHON:-.venv_sft/bin/python}
CONFIG=${CONFIG:-config/sft/qwen3_sft.yaml}

port_is_free() {
  local port="$1"
  ! timeout 1 bash -c "</dev/tcp/${MASTER_ADDR}/${port}" >/dev/null 2>&1
}

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT=29500
  while ! port_is_free "$MASTER_PORT"; do
    MASTER_PORT="$((MASTER_PORT + 1))"
  done
fi

# Resume controls:
#   RESUME_FROM_CHECKPOINT=auto  -> let LLaMA-Factory find the latest checkpoint
#   RESUME_FROM_CHECKPOINT=none  -> start from scratch; requires overwrite_output_dir=true
#   RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N -> resume from an explicit checkpoint
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-auto}

EXTRA_ARGS=()
case "${RESUME_FROM_CHECKPOINT}" in
  auto)
    EXTRA_ARGS+=("overwrite_output_dir=false")
    ;;
  none)
    EXTRA_ARGS+=("overwrite_output_dir=true")
    ;;
  *)
    EXTRA_ARGS+=("resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
    EXTRA_ARGS+=("overwrite_output_dir=false")
    ;;
esac

echo "[MASTER_PORT] ${MASTER_PORT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONPATH="${REPO_ROOT}/LLaMA-Factory/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON}" -m torch.distributed.run \
  --nnodes 1 \
  --node_rank 0 \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  LLaMA-Factory/src/llamafactory/launcher.py \
  "${CONFIG}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
