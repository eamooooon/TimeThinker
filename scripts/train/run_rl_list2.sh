#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG=${CONFIG:-config/rl/qwen3_rl_bs16.yaml}
LOG_DIR=${LOG_DIR:-logs/train}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PYTHON=${PYTHON:-.venv_rl/bin/python}
BASE_MODEL=${BASE_MODEL:-models/TimeThinker-4B-SFT-v9-10k-3ep}

mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local adv_estimator="$2"
  local ckpt_path="$3"
  shift 3
  local log_path="${LOG_DIR}/${name}.log"

  echo "[START] ${name}"
  echo "[CONFIG] ${CONFIG}"
  echo "[BASE_MODEL] ${BASE_MODEL}"
  echo "[CKPT] ${ckpt_path}"
  echo "[LOG] ${log_path}"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHON="$PYTHON" \
  CONFIG="$CONFIG" \
  bash scripts/train/run_rl.sh \
    "worker.actor.model.model_path=${BASE_MODEL}" \
    "algorithm.adv_estimator=${adv_estimator}" \
    "trainer.experiment_name=${name}" \
    "trainer.save_checkpoint_path=${ckpt_path}" \
    "trainer.find_last_checkpoint=false" \
    "$@" \
    2>&1 | tee "$log_path"

  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] ${name} exited with status ${rc}"
    echo "[STOP] remaining runs will not start. Check ${log_path}"
    return "$rc"
  fi

  echo "[DONE] ${name}"
}

# Clean SFT-v9 initialization baseline. This is the most important missing run:
# previous bs16 RL directories were named like v9, but their experiment_config
# still points to Qwen/Qwen3-VL-4B-Instruct.
run_one \
  "rl_4b-sftv9-bs16-grpo-100" \
  "grpo" \
  "models/TimeThinker-4B-RL-sftv9-bs16-grpo-100" \
  "algorithm.temporal=false" || exit $?

# Same SFT-v9 base, EMA advantage. This checks whether EMA-GRPO is still better
# once the cold-start model is already strong.
run_one \
  "rl_4b-sftv9-bs16-ema-100" \
  "ema_grpo" \
  "models/TimeThinker-4B-RL-sftv9-bs16-ema-100" \
  "algorithm.temporal=false" || exit $?

# Low-strength temporal reward. Earlier temporal_reward=0.3 helped some slices
# but was not stable, so this keeps the signal while reducing reward hijacking.
run_one \
  "rl_4b-sftv9-bs16-tgrpo-r01-strict-100" \
  "grpo" \
  "models/TimeThinker-4B-RL-sftv9-bs16-tgrpo-r01-strict-100" \
  "algorithm.temporal=true" \
  "algorithm.shuffled_rollout_ratio=0.5" \
  "algorithm.temporal_reward=0.1" \
  "algorithm.temporal_compare_ratio=1.0" || exit $?

# Conservative GRPO from SFT-v9. If normal RL damages prompt-sensitive SFT
# behavior, this lower-LR/higher-KL run is the direct control.
# run_one \
#   "rl_4b-sftv9-bs16-grpo-conservative-100" \
#   "grpo" \
#   "models/TimeThinker-4B-RL-sftv9-bs16-grpo-conservative-100" \
#   "algorithm.temporal=false" \
#   "algorithm.kl_coef=8.0e-2" \
#   "worker.actor.optim.lr=5.0e-7" || exit $?

echo "[ALL DONE] runs finished."
