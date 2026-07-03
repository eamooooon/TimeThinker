#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Unified evaluation entry for this project. Keep DATASETS to video QA /
# math-style QA benchmarks that match the current model and metric script.

# MODEL_PATHS=(
#   "${MODEL_PATH:-models/TimeThinker-4B-SFT-v2-10000}"
# )
MODEL_PATHS=(
  "${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
)

# Unified benchmark list.
# VideoMME is skipped by default because its media is not downloaded locally.
BENCHMARK_DATASETS=(
  # "eval_mvbench.json"
  "eval_tempcompass.json"
  "eval_videommmu.json"
  "eval_vsibench.json"
  "eval_mmvu.json"
  # "eval_longvideoreason.json"
  # "eval_videomathqa.json"
)

if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a BENCHMARK_DATASETS <<< "$DATASETS"
fi

DATASET_PREFIX=${DATASET_PREFIX:-$REPO_ROOT/Evaluation/data}
OUT_ROOT_BASE=${OUT_ROOT_BASE:-Evaluation/results}

DATE_SUFFIX=${DATE_SUFFIX:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-logs}

MAX_PIXELS_VIDEO=${MAX_PIXELS_VIDEO:-$((256*28*28))}
MAX_PIXELS_IMAGE=${MAX_PIXELS_IMAGE:-$((1024*28*28))}
MAX_FRAMES=${MAX_FRAMES:-16}
FPS=${FPS:-2}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0.01}
TOP_P=${TOP_P:-0.001}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_SAMPLES=${MAX_SAMPLES:-}
VIDEO_READER=${VIDEO_READER:-auto}
RUN_PARALLEL=${RUN_PARALLEL:-1}
EVAL_GPUS=${EVAL_GPUS:-0,1,2,3}
PYTHON=${PYTHON:-$REPO_ROOT/.venv_eval/bin/python}

export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-2048001}
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-decord}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}
export OPENCV_NUM_THREADS=${OPENCV_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

detect_gpus() {
  if [[ -n "$EVAL_GPUS" ]]; then
    echo "$EVAL_GPUS"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -
  else
    echo "${CUDA_VISIBLE_DEVICES:-0}"
  fi
}

model_tag_for() {
  local model="$1"
  local model_tag
  model_tag="$(basename "$model")"
  model_tag="${model_tag//[^[:alnum:]._-]/_}"
  echo "$model_tag"
}

result_suffix_for() {
  local model="$1"
  echo "_$(model_tag_for "$model")_frames${MAX_FRAMES}"
}

log_path_for() {
  local model="$1"
  local ds_name="$2"
  local stem="${ds_name%.json}"
  echo "${LOG_DIR%/}/eval_$(model_tag_for "$model")_frames${MAX_FRAMES}_${stem}.log"
}

run_one_dataset() {
  local dataset_prefix="$1"
  local model="$2"
  local ds_name="$3"
  local out_dir="$4"

  local ds_path="${dataset_prefix%/}/${ds_name}"
  if [[ ! -f "$ds_path" ]]; then
    echo "[SKIP] ${ds_name}: not found at ${ds_path}"
    return 0
  fi

  echo "[RUN] ${ds_name}"
  local extra_args=()
  local result_suffix
  result_suffix="$(result_suffix_for "$model")"
  echo "[RESULT_SUFFIX] ${result_suffix}"
  if [[ -n "$MAX_SAMPLES" ]]; then
    extra_args+=(--max_samples "$MAX_SAMPLES")
  fi

  "$PYTHON" -u Evaluation/Eval/eval_bench.py \
    --model_path "$model" \
    --input_json "$ds_path" \
    --base_prefix "$dataset_prefix" \
    --out_dir "$out_dir" \
    --suffix "$result_suffix" \
    --batch_size "$BATCH_SIZE" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_pixels_video "$MAX_PIXELS_VIDEO" \
    --max_pixels_image "$MAX_PIXELS_IMAGE" \
    --max_frames "$MAX_FRAMES" \
    --fps "$FPS" \
    --video_reader "$VIDEO_READER" \
    "${extra_args[@]}"
}

run_datasets() {
  local dataset_prefix="$1"
  shift 1
  local datasets=("$@")

  local out_dir="${OUT_ROOT_BASE%/}"
  mkdir -p "$out_dir"
  mkdir -p "$LOG_DIR"

  echo "[RUN_ID] DATE_SUFFIX=${DATE_SUFFIX} MAX_FRAMES=${MAX_FRAMES} LOG_DIR=${LOG_DIR}"

  if [[ "$RUN_PARALLEL" == "1" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$(detect_gpus)"
    if [[ "${#gpu_ids[@]}" -eq 0 || -z "${gpu_ids[0]}" ]]; then
      gpu_ids=(0)
    fi

    echo "[PARALLEL] GPUs=${gpu_ids[*]}"

    pids=()
    active=0
    status=0
    for model in "${MODEL_PATHS[@]}"; do
      for ds_name in "${datasets[@]}"; do
        local ds_path="${dataset_prefix%/}/${ds_name}"
        if [[ ! -f "$ds_path" ]]; then
          echo "[SKIP] ${ds_name}: not found at ${ds_path}"
          continue
        fi

        local gpu="${gpu_ids[$active]}"
        local log_path
        log_path="$(log_path_for "$model" "$ds_name")"
        echo "[RUN][GPU ${gpu}] ${ds_name} -> ${log_path}"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          run_one_dataset "$dataset_prefix" "$model" "$ds_name" "$out_dir"
        ) >"$log_path" 2>&1 &
        pids+=("$!")
        active=$((active + 1))

        if [[ "$active" -ge "${#gpu_ids[@]}" ]]; then
          for pid in "${pids[@]}"; do
            wait "$pid" || status=1
          done
          pids=()
          active=0
        fi
      done
    done

    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    return "$status"
  fi

  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  for model in "${MODEL_PATHS[@]}"; do
    for ds_name in "${datasets[@]}"; do
      local log_path
      log_path="$(log_path_for "$model" "$ds_name")"
      echo "[RUN] ${ds_name} -> ${log_path}"
      run_one_dataset "$dataset_prefix" "$model" "$ds_name" "$out_dir" 2>&1 | tee "$log_path"
    done
  done
}

run_datasets "$DATASET_PREFIX" "${BENCHMARK_DATASETS[@]}"
