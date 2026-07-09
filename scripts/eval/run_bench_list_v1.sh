#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR=${LOG_DIR:-logs/v1}
LIST_LOG_DIR=${LIST_LOG_DIR:-$LOG_DIR}
RUN_BENCH=${RUN_BENCH:-scripts/eval/run_bench.sh}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

mkdir -p "$LIST_LOG_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval/run_bench_list_v1.sh MODEL_PATH [MODEL_PATH2 ...]

Examples:
  bash scripts/eval/run_bench_list_v1.sh \
    Qwen/Qwen3-VL-4B-Instruct \
    models/TimeThinker-4B-SFT-v3-10000-1ep

Defaults:
  EVAL_BENCH_SCRIPT=Evaluation/Eval/eval_bench_v1.py
  OUT_ROOT_BASE=Evaluation/results-v1-rerun
  FRAME_CACHE_DIR=Evaluation/data/.cache/eval_frames
  VIDEO_READER=auto

Notes:
  Frame cache only caches decoded video frames. Model outputs are resumed only
  when eval_*.json already exists under OUT_ROOT_BASE/<model>/framesN.

Useful env:
  OUT_ROOT_BASE=Evaluation/results-v1-rerun
  DATASETS=eval_mvbench.json,eval_tempcompass.json
  MAX_FRAMES=16
  BATCH_SIZE=64
  RUN_PARALLEL=1
  EVAL_GPUS=0,1,2,3
  EVAL_SCHEDULE=balanced|listed
  CONTINUE_ON_ERROR=0|1
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
  log_path="${LIST_LOG_DIR%/}/eval_list_v1_${idx}_${tag}.log"

  warn_if_local_model_missing "$model"

  echo "[START][v1] ${idx}: ${model}"
  echo "[LOG] ${log_path}"
  echo "[RUN_BENCH] ${RUN_BENCH}"
  echo "[OUT_ROOT_BASE] ${OUT_ROOT_BASE:-Evaluation/results-v1-rerun}"
  echo "[FRAME_CACHE_DIR] ${FRAME_CACHE_DIR:-${DATASET_PREFIX:-$REPO_ROOT/Evaluation/data}/.cache/eval_frames}"

  MODEL_PATH="$model" \
  EVAL_BENCH_SCRIPT="${EVAL_BENCH_SCRIPT:-Evaluation/Eval/eval_bench_v1.py}" \
  EVAL_BENCH_SUPPORTS_FRAME_CACHE="${EVAL_BENCH_SUPPORTS_FRAME_CACHE:-1}" \
  OUT_ROOT_BASE="${OUT_ROOT_BASE:-Evaluation/results-v1-rerun}" \
  VIDEO_READER="${VIDEO_READER:-auto}" \
  bash "$RUN_BENCH" 2>&1 | tee "$log_path"

  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] ${model} exited with status ${rc}"
    echo "[LOG] ${log_path}"
    return "$rc"
  fi

  echo "[DONE][v1] ${model}"
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
      echo "[STOP] remaining v1 eval runs will not start."
      exit "$status"
    fi
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[DONE WITH FAILURES] v1 model eval list finished with status ${status}."
  exit "$status"
fi

echo "[ALL DONE] v1 model eval list finished."
