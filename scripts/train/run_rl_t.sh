#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=8
export DECORD_EOF_RETRY_MAX=2048001
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-decord}
export SWANLAB_DIR=${SWANLAB_DIR:-swanlog}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PYTHON=${PYTHON:-.venv_rl/bin/python}
CONFIG=${CONFIG:-config/rl/qwen3_rl_t.yaml}

export PYTHONPATH="${REPO_ROOT}/EasyR1${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON}" -m verl.trainer.main \
    config="${CONFIG}" \
    "$@"
