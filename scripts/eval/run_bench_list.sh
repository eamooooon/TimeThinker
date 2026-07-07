#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR=${LOG_DIR:-logs}
LIST_LOG_DIR=${LIST_LOG_DIR:-$LOG_DIR}
RUN_BENCH=${RUN_BENCH:-scripts/eval/run_bench.sh}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

mkdir -p "$LIST_LOG_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval/run_bench_list.sh MODEL_PATH [MODEL_PATH2 ...]

Examples:
  bash scripts/eval/run_bench_list.sh \
    models/TimeThinker-4B-SFT-v3-10000 \
    models/TimeThinker-4B-RL-Zero-100-grpo-v2/global_step_100/actor/huggingface \
    models/TimeThinker-4B-RL-Zero-100-ema-v2/global_step_100/actor/huggingface

Optional env forwarded to run_bench.sh:
  DATASETS=eval_mvbench.json,eval_videomathqa.json
  MAX_FRAMES=16
  BATCH_SIZE=64
  MAX_SAMPLES=100
  RUN_PARALLEL=1
  EVAL_GPUS=0,1,2,3
  FRAME_CACHE_DIR=Evaluation/data/.cache/eval_frames
  DISABLE_FRAME_CACHE=0
  CUDA_VISIBLE_DEVICES=0,1,2,3
  PYTHON=.venv_eval/bin/python
  OUT_ROOT_BASE=Evaluation/results
  LOG_DIR=logs

Optional env for this list runner:
  LIST_LOG_DIR=logs
  RUN_BENCH=scripts/eval/run_bench.sh
  CONTINUE_ON_ERROR=0|1

Notes:
  Models are evaluated one by one. Dataset parallelism inside each model run is
  still controlled by RUN_PARALLEL in scripts/eval/run_bench.sh.
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

if [[ ! -f "$RUN_BENCH" ]]; then
  echo "[FAIL] run bench script not found: ${RUN_BENCH}"
  exit 2
fi

normalize_model_path() {
  local model="$1"
  if [[ "$model" == "$REPO_ROOT/"* ]]; then
    echo "${model#"$REPO_ROOT/"}"
  else
    echo "$model"
  fi
}

model_tag_for() {
  local model="$1"
  local model_tag
  model="$(normalize_model_path "$model")"
  IFS='/' read -r -a parts <<< "$model"
  if [[ "${#parts[@]}" -ge 2 && -n "${parts[1]}" ]]; then
    model_tag="${parts[1]}"
  else
    model_tag="$(basename "$model")"
  fi
  model_tag="${model_tag//[^[:alnum:]._-]/_}"
  echo "$model_tag"
}

warn_if_local_model_missing() {
  local model="$1"
  if [[ "$model" == /* || "$model" == ./* || "$model" == ../* || "$model" == models/* ]]; then
    if [[ ! -e "$model" ]]; then
      echo "[WARN] local model path does not exist yet: ${model}"
    fi
  fi
}

run_one() {
  local idx="$1"
  local raw_model="$2"
  local model
  local tag
  local log_path

  model="$(normalize_model_path "$raw_model")"
  tag="$(model_tag_for "$model")"
  log_path="${LIST_LOG_DIR%/}/eval_list_${idx}_${tag}.log"

  warn_if_local_model_missing "$model"

  echo "[START] ${idx}: ${model}"
  echo "[LOG] ${log_path}"
  echo "[RUN_BENCH] ${RUN_BENCH}"
  echo "[RUN_PARALLEL] ${RUN_PARALLEL:-1}"

  MODEL_PATH="$model" bash "$RUN_BENCH" 2>&1 | tee "$log_path"

  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] ${model} exited with status ${rc}"
    echo "[LOG] ${log_path}"
    return "$rc"
  fi

  echo "[DONE] ${model}"
}

idx=0
status=0
for model in "$@"; do
  idx="$((idx + 1))"
  run_one "$idx" "$model"
  rc="$?"
  if [[ "$rc" -ne 0 ]]; then
    status="$rc"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "[STOP] remaining model eval runs will not start."
      exit "$status"
    fi
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[DONE WITH FAILURES] model eval list finished with status ${status}."
  exit "$status"
fi

echo "[ALL DONE] model eval list finished."
