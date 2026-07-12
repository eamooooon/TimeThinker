#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Unified evaluation entry for this project. Keep DATASETS to video QA /
# math-style QA benchmarks that match the current model and metric script.

MODEL_PATHS=(
  "${MODEL_PATH:-models/TimeThinker-4B-SFT-v10-50k-canonical/checkpoint-1000}"
)
# MODEL_PATHS=(
#   "${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
# )

# Unified benchmark list.
BENCHMARK_DATASETS=(
  "eval_longvideoreason.json"
  "eval_videommmu.json"
  "eval_mvbench.json"
  "eval_tempcompass.json"
  "eval_videomathqa.json"
  "eval_mmvu.json"
  "eval_videomme.json"
  "eval_vsibench.json"
)

if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a BENCHMARK_DATASETS <<< "$DATASETS"
fi

DATASET_PREFIX=${DATASET_PREFIX:-$REPO_ROOT/Evaluation/data}
OUT_ROOT_BASE=${OUT_ROOT_BASE:-Evaluation/results}

DATE_SUFFIX=${DATE_SUFFIX:-$(date +%Y%m%d_%H%M%S)}
RUN_START_EPOCH=${RUN_START_EPOCH:-$(date +%s)}
LOG_DIR=${LOG_DIR:-logs}
TERMINAL_PROGRESS=${TERMINAL_PROGRESS:-1}
SUMMARY_RESULTS=${SUMMARY_RESULTS:-1}
SUMMARY_PATTERN=${SUMMARY_PATTERN:-}

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
FRAME_CACHE_DIR=${FRAME_CACHE_DIR:-${DATASET_PREFIX%/}/.cache/eval_frames}
DISABLE_FRAME_CACHE=${DISABLE_FRAME_CACHE:-0}
RUN_PARALLEL=${RUN_PARALLEL:-1}
EVAL_GPUS=${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
EVAL_SCHEDULE=${EVAL_SCHEDULE:-balanced}  # balanced|listed
PYTHON=${PYTHON:-$REPO_ROOT/.venv_eval/bin/python}
EVAL_BENCH_SCRIPT=${EVAL_BENCH_SCRIPT:-Evaluation/Eval/eval_bench.py}
if [[ -z "${EVAL_BENCH_SUPPORTS_FRAME_CACHE:-}" ]]; then
  EVAL_BENCH_SUPPORTS_FRAME_CACHE=1
fi

case "$EVAL_SCHEDULE" in
  balanced|listed) ;;
  *)
    echo "[FAIL] unsupported EVAL_SCHEDULE=${EVAL_SCHEDULE}; expected balanced|listed"
    exit 2
    ;;
esac

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
  IFS='/' read -r -a parts <<< "$model"
  if [[ "${#parts[@]}" -ge 2 && -n "${parts[1]}" ]]; then
    model_tag="${parts[1]}"
  else
    model_tag="$(basename "$model")"
  fi
  model_tag="${model_tag//[^[:alnum:]._-]/_}"
  echo "$model_tag"
}

result_suffix_for() {
  echo "${RESULT_SUFFIX:-}"
}

result_dir_for() {
  local model="$1"
  echo "${OUT_ROOT_BASE%/}/$(model_tag_for "$model")/frames${MAX_FRAMES}"
}

log_path_for() {
  local model="$1"
  local ds_name="$2"
  local stem="${ds_name%.json}"
  echo "${LOG_DIR%/}/eval_$(model_tag_for "$model")_frames${MAX_FRAMES}_${stem}.log"
}

video_reader_for() {
  echo "$VIDEO_READER"
}

benchmark_estimate_seconds() {
  # Cache-hit full-run estimates from recent Qwen3-VL 4B evals.
  # They are used only to order parallel jobs; eval results still record actual time.
  case "$(basename "$1")" in
    eval_mvbench.json) echo 1540 ;;
    eval_tempcompass.json) echo 1072 ;;
    eval_videomme.json) echo 931 ;;
    eval_longvideoreason.json) echo 875 ;;
    eval_vsibench.json) echo 788 ;;
    eval_videommmu.json) echo 672 ;;
    eval_videomathqa.json) echo 435 ;;
    eval_mmvu.json) echo 261 ;;
    *) echo 1 ;;
  esac
}

schedule_datasets_for_parallel() {
  local mode="$1"
  shift 1

  if [[ "$mode" == "listed" ]]; then
    printf '%s\n' "$@"
    return 0
  fi

  if [[ "$mode" != "balanced" ]]; then
    echo "[FAIL] unsupported EVAL_SCHEDULE=${mode}; expected balanced|listed" >&2
    return 2
  fi

  local idx=0
  local ds
  for ds in "$@"; do
    printf '%s\t%06d\t%s\n' "$(benchmark_estimate_seconds "$ds")" "$idx" "$ds"
    idx=$((idx + 1))
  done | sort -s -t $'\t' -k1,1nr -k2,2n | cut -f3-
}

print_dataset_stats() {
  local ds_path="$1"
  "$PYTHON" - "$ds_path" "$BATCH_SIZE" "${MAX_SAMPLES:-}" <<'PY'
import json
import math
import sys

path, batch_size_raw, max_samples_raw = sys.argv[1:4]
batch_size = int(batch_size_raw)
max_samples = int(max_samples_raw) if max_samples_raw else -1

if path.endswith(".jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())
else:
    with open(path, "r", encoding="utf-8") as f:
        total = len(json.load(f))

samples = min(total, max_samples) if max_samples > 0 else total
batches = math.ceil(samples / batch_size) if samples else 0
print(f"[DATASET] samples={samples} batch_size={batch_size} batches={batches}")
PY
}

stream_progress() {
  local label="$1"
  "$PYTHON" -u -c '
import os
import re
import sys

label = sys.argv[1]
keep = (
    "[RUN]",
    "[RESULT_SUFFIX]",
    "[Done]",
    "[Metrics]",
    "/acc:",
    "Traceback",
    "Error",
    "Exception",
    "KeyboardInterrupt",
    "fatal:",
)
buf = ""

def emit(line):
    line = line.strip()
    line = re.sub(r"\[[A-Za-z0-9_]+ @ 0x[0-9a-fA-F]+\].*$", "", line).strip()
    if not line:
        return
    if "batches:" in line.lower():
        pass
    elif not any(token in line for token in keep):
        return
    print(f"[{label}] {line}", flush=True)

while True:
    chunk = os.read(0, 4096)
    if not chunk:
        break
    buf += chunk.decode(errors="replace")
    while True:
        match = re.search(r"[\r\n]", buf)
        if not match:
            break
        emit(buf[:match.start()])
        buf = buf[match.end():]

emit(buf)
' "$label"
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

  mkdir -p "$out_dir"
  echo "[RUN] ${ds_name}"
  echo "[OUT_DIR] ${out_dir}"
  local extra_args=()
  local result_suffix
  local video_reader
  result_suffix="$(result_suffix_for "$model")"
  video_reader="$(video_reader_for "$ds_name")"
  echo "[RESULT_SUFFIX] ${result_suffix}"
  echo "[VIDEO_READER] ${video_reader}"
  echo "[FRAME_CACHE_DIR] ${FRAME_CACHE_DIR}"
  print_dataset_stats "$ds_path"
  if [[ -n "$MAX_SAMPLES" ]]; then
    extra_args+=(--max_samples "$MAX_SAMPLES")
  fi
  if [[ "$DISABLE_FRAME_CACHE" == "1" ]]; then
    if [[ "$EVAL_BENCH_SUPPORTS_FRAME_CACHE" == "1" ]]; then
      extra_args+=(--disable_frame_cache)
    fi
  fi
  if [[ "$EVAL_BENCH_SUPPORTS_FRAME_CACHE" == "1" ]]; then
    extra_args+=(--frame_cache_dir "$FRAME_CACHE_DIR")
  fi

  "$PYTHON" -u "$EVAL_BENCH_SCRIPT" \
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
    --video_reader "$video_reader" \
    "${extra_args[@]}"
}

run_datasets() {
  local dataset_prefix="$1"
  shift 1
  local datasets=("$@")

  mkdir -p "$OUT_ROOT_BASE"
  mkdir -p "$LOG_DIR"

  echo "[RUN_ID] DATE_SUFFIX=${DATE_SUFFIX} MAX_FRAMES=${MAX_FRAMES} LOG_DIR=${LOG_DIR}"

  if [[ "$RUN_PARALLEL" == "1" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$(detect_gpus)"
    if [[ "${#gpu_ids[@]}" -eq 0 || -z "${gpu_ids[0]}" ]]; then
      gpu_ids=(0)
    fi

    echo "[PARALLEL] GPUs=${gpu_ids[*]}"
    echo "[SCHEDULE] mode=${EVAL_SCHEDULE}"

    scheduled_datasets=()
    mapfile -t scheduled_datasets < <(schedule_datasets_for_parallel "$EVAL_SCHEDULE" "${datasets[@]}")
    echo "[SCHEDULE] datasets=${scheduled_datasets[*]}"

    task_models=()
    task_datasets=()
    for model in "${MODEL_PATHS[@]}"; do
      for ds_name in "${scheduled_datasets[@]}"; do
        local ds_path="${dataset_prefix%/}/${ds_name}"
        if [[ ! -f "$ds_path" ]]; then
          echo "[SKIP] ${ds_name}: not found at ${ds_path}"
          continue
        fi

        task_models+=("$model")
        task_datasets+=("$ds_name")
      done
    done

    if [[ "${#task_models[@]}" -eq 0 ]]; then
      echo "[PARALLEL] no runnable datasets"
      return 0
    fi

    active_pids=()
    declare -A pid_to_gpu=()
    declare -A pid_to_task=()
    next_task=0
    status=0

    start_parallel_job() {
      local gpu="$1"
      local model="${task_models[$next_task]}"
      local ds_name="${task_datasets[$next_task]}"
      local log_path
      local label
      local out_dir
      local pid

      out_dir="$(result_dir_for "$model")"
      log_path="$(log_path_for "$model" "$ds_name")"
      label="$(model_tag_for "$model")/${ds_name%.json}/GPU${gpu}"
      echo "[RUN][GPU ${gpu}] ${ds_name} -> ${log_path}"

      if [[ "$TERMINAL_PROGRESS" == "1" ]]; then
        (
          set +e
          (
            export CUDA_VISIBLE_DEVICES="$gpu"
            run_one_dataset "$dataset_prefix" "$model" "$ds_name" "$out_dir"
          ) 2>&1 | tee "$log_path" | stream_progress "$label"
          exit "${PIPESTATUS[0]}"
        ) &
      else
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          run_one_dataset "$dataset_prefix" "$model" "$ds_name" "$out_dir"
        ) >"$log_path" 2>&1 &
      fi

      pid="$!"
      active_pids+=("$pid")
      pid_to_gpu["$pid"]="$gpu"
      pid_to_task["$pid"]="$(model_tag_for "$model")/${ds_name%.json}"
      next_task=$((next_task + 1))
    }

    remove_active_pid() {
      local done_pid="$1"
      local kept=()
      local pid
      for pid in "${active_pids[@]}"; do
        if [[ "$pid" != "$done_pid" ]]; then
          kept+=("$pid")
        fi
      done
      active_pids=("${kept[@]}")
    }

    for gpu in "${gpu_ids[@]}"; do
      if [[ "$next_task" -ge "${#task_models[@]}" ]]; then
        break
      fi
      start_parallel_job "$gpu"
    done

    while [[ "${#active_pids[@]}" -gt 0 ]]; do
      local done_pid
      local wait_rc
      done_pid=""
      set +e +u
      wait -n -p done_pid "${active_pids[@]}"
      wait_rc="$?"
      set -e -u

      if [[ -z "${done_pid:-}" ]]; then
        # Bash can leave -p unset when the last child is reaped after the
        # active list was already drained by job-control bookkeeping. Treat
        # this as an orderly loop exit instead of failing under set -u.
        if [[ "${#active_pids[@]}" -eq 0 || "$wait_rc" -eq 127 ]]; then
          break
        fi
        for pid in "${active_pids[@]}"; do
          if ! kill -0 "$pid" 2>/dev/null; then
            done_pid="$pid"
            break
          fi
        done
        done_pid="${done_pid:-${active_pids[0]}}"
      fi

      local freed_gpu="${pid_to_gpu[$done_pid]:-}"
      local done_task="${pid_to_task[$done_pid]:-unknown}"
      remove_active_pid "$done_pid"
      unset "pid_to_gpu[$done_pid]"
      unset "pid_to_task[$done_pid]"

      if [[ "$wait_rc" -ne 0 ]]; then
        echo "[FAIL][GPU ${freed_gpu}] ${done_task} exited with status ${wait_rc}"
        status=1
      else
        echo "[DONE][GPU ${freed_gpu}] ${done_task}"
      fi

      if [[ -n "$freed_gpu" && "$next_task" -lt "${#task_models[@]}" ]]; then
        start_parallel_job "$freed_gpu"
      fi
    done
    return "$status"
  fi

  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(detect_gpus)}
  echo "[SERIAL] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  for model in "${MODEL_PATHS[@]}"; do
    for ds_name in "${datasets[@]}"; do
      local log_path
      local out_dir
      local label
      out_dir="$(result_dir_for "$model")"
      log_path="$(log_path_for "$model" "$ds_name")"
      label="$(model_tag_for "$model")/${ds_name%.json}/serial"
      echo "[RUN] ${ds_name} -> ${log_path}"
      if [[ "$TERMINAL_PROGRESS" == "1" ]]; then
        set +e
        run_one_dataset "$dataset_prefix" "$model" "$ds_name" "$out_dir" 2>&1 | tee "$log_path" | stream_progress "$label"
        local pipe_status="${PIPESTATUS[0]}"
        set -e
        if [[ "$pipe_status" -ne 0 ]]; then
          return "$pipe_status"
        fi
      else
        run_one_dataset "$dataset_prefix" "$model" "$ds_name" "$out_dir" 2>&1 | tee "$log_path"
      fi
    done
  done
}

run_datasets "$DATASET_PREFIX" "${BENCHMARK_DATASETS[@]}"
RUN_END_EPOCH=${RUN_END_EPOCH:-$(date +%s)}

if [[ "$SUMMARY_RESULTS" == "1" ]]; then
  summary_dirs=()
  for model in "${MODEL_PATHS[@]}"; do
    summary_dirs+=("$(result_dir_for "$model")")
  done

  if [[ -z "$SUMMARY_PATTERN" ]]; then
    if [[ -n "${RESULT_SUFFIX:-}" ]]; then
      SUMMARY_PATTERN="eval_*${RESULT_SUFFIX}.json"
    else
      SUMMARY_PATTERN="eval_*.json"
    fi
  fi

  echo "[SUMMARY] pattern=${SUMMARY_PATTERN}"
  "$PYTHON" scripts/eval/summarize_results.py \
    --pattern "$SUMMARY_PATTERN" \
    --wall_start_epoch "$RUN_START_EPOCH" \
    --wall_end_epoch "$RUN_END_EPOCH" \
    --run_date_suffix "$DATE_SUFFIX" \
    --eval_schedule "$EVAL_SCHEDULE" \
    "${summary_dirs[@]}"
fi
