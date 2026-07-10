#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG=${CONFIG:-config/rl/qwen3_rl_bs16.yaml}
LOG_DIR=${LOG_DIR:-logs/train}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PYTHON=${PYTHON:-.venv_rl/bin/python}

mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local adv_estimator="$2"
  local ckpt_path="$3"
  shift 3
  local log_path="${LOG_DIR}/${name}.log"

  echo "[START] ${name}"
  echo "[CONFIG] ${CONFIG}"
  echo "[CKPT] ${ckpt_path}"
  echo "[LOG] ${log_path}"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHON="$PYTHON" \
  CONFIG="$CONFIG" \
  bash scripts/train/run_rl.sh \
    "algorithm.adv_estimator=${adv_estimator}" \
    "trainer.experiment_name=${name}" \
    "trainer.save_checkpoint_path=${ckpt_path}" \
    "trainer.find_last_checkpoint=false" \
    "$@" \
    2>&1 | tee "$log_path"

  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] ${name} exited with status ${rc}"
    echo "[STOP] second run will not start. Check ${log_path}"
    return "$rc"
  fi

  echo "[DONE] ${name}"
}

# run_one \
#   "rl_4b-bs16-van-100" \
#   "grpo" \
#   "models/TimeThinker-4B-RL-bs16-van-100" \
#   "algorithm.temporal=false" || exit $?

run_one \
  "rl_4b-bs16-tgrpo-diagnose-100" \
  "grpo" \
  "models/TimeThinker-4B-RL-bs16-tgrpo-diagnose-100" \
  "algorithm.temporal=true" \
  "algorithm.shuffled_rollout_ratio=0.5" \
  "algorithm.temporal_reward=0.3" \
  "algorithm.temporal_compare_ratio=0.8" || exit $?

run_one \
  "rl_4b-bs16-tgrpo-strict-r01-100" \
  "grpo" \
  "models/TimeThinker-4B-RL-bs16-tgrpo-strict-r01-100" \
  "algorithm.temporal=true" \
  "algorithm.shuffled_rollout_ratio=0.5" \
  "algorithm.temporal_reward=0.1" \
  "algorithm.temporal_compare_ratio=1.0" || exit $?

run_one \
  "rl_4b-bs16-ema-100" \
  "ema_grpo" \
  "models/TimeThinker-4B-RL-bs16-ema-100" \
  "algorithm.temporal=false" || exit $?

echo "[ALL DONE] runs finished."
