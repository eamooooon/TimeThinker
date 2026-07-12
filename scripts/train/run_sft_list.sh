#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG=${CONFIG:-config/sft/qwen3_sft.yaml}
LOG_DIR=${LOG_DIR:-logs/train}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29600}
PYTHON=${PYTHON:-.venv_sft/bin/python}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-none}
OVERWRITE_CACHE=${OVERWRITE_CACHE:-true}

mkdir -p "$LOG_DIR"

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
  local name="$2"
  shift 2

  local log_path="${LOG_DIR}/${name}.log"
  local master_port
  master_port="$(next_free_port "$((MASTER_PORT_BASE + idx - 1))")"

  echo "[START] ${name}"
  echo "[CONFIG] ${CONFIG}"
  echo "[LOG] ${log_path}"
  echo "[MASTER_PORT] ${master_port}"
  echo "[RESUME_FROM_CHECKPOINT] ${RESUME_FROM_CHECKPOINT}"
  echo "[OVERWRITE_CACHE] ${OVERWRITE_CACHE}"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$master_port" \
  PYTHON="$PYTHON" \
  CONFIG="$CONFIG" \
  RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
  bash scripts/train/run_sft.sh \
    "overwrite_cache=${OVERWRITE_CACHE}" \
    "$@" \
    2>&1 | tee "$log_path"

  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] ${name} exited with status ${rc}"
    echo "[STOP] remaining SFT runs will not start. Check ${log_path}"
    return "$rc"
  fi

  echo "[DONE] ${name}"
}


# Replicate SFT v5 config: 10k mixed, higher image resolution, lr=5e-6.
# run_one 4 "sft_4b-v5-10k-canonical" \
#   "image_max_pixels=200704" \
#   "tokenized_path=LLaMA-Factory/cache/timethinker_sft_10k_img2_canonical" \
#   "output_dir=models/TimeThinker-4B-SFT-v5-10k-canonical" \
#   "swanlab_run_name=sft_4b-v5-10k-canonical" || exit $?

# Replicate SFT v6 config: 10k mixed, projector frozen, lr=5e-6.
run_one 1 "sft_4b-v6-10k-canonical" \
  "use_reentrant_gc=false" \
  "freeze_vision_tower=false" \
  "freeze_multi_modal_projector=true" \
  "tokenized_path=LLaMA-Factory/cache/timethinker_sft_10k_v6_canonical" \
  "output_dir=models/TimeThinker-4B-SFT-v6-10k-canonical" \
  "swanlab_run_name=sft_4b-v6-10k-canonical" || exit $?

# Replicate SFT v7 config: 10k mixed, full multimodal tuning, lr=1e-5.
run_one 2 "sft_4b-v7-10k-canonical" \
  "use_reentrant_gc=false" \
  "freeze_vision_tower=false" \
  "freeze_multi_modal_projector=false" \
  "tokenized_path=LLaMA-Factory/cache/timethinker_sft_10k_v7_canonical" \
  "output_dir=models/TimeThinker-4B-SFT-v7-10k-canonical" \
  "swanlab_run_name=sft_4b-v7-10k-canonical" || exit $?

# Replicate SFT v8 config: video-only, max_samples=20000, lr=5e-6.
run_one 5 "sft_4b-v8-video-canonical" \
  "dataset=timethinker_sft_video" \
  "max_samples=20000" \
  "tokenized_path=LLaMA-Factory/cache/timethinker_sft_10k_video_canonical" \
  "output_dir=models/TimeThinker-4B-SFT-v8-10k-canonical" \
  "swanlab_run_name=sft_4b-v8-video-canonical" || exit $?

# Replicate SFT v9 config: 10k mixed, vision tower frozen, 3 epochs.
# run_one 3 "sft_4b-v9-10k-canonical" \
#   "tokenized_path=LLaMA-Factory/cache/timethinker_sft_10k_canonical" \
#   "output_dir=models/TimeThinker-4B-SFT-v9-10k-canonical" \
#   "num_train_epochs=3.0" \
#   "swanlab_run_name=sft_4b-v9-10k-canonical" || exit $?

# Replicate SFT v10 config: 50k mixed, lr=5e-6.
run_one 6 "sft_4b-v10-50k-canonical" \
  "max_samples=50000" \
  "tokenized_path=LLaMA-Factory/cache/timethinker_sft_50k_canonical" \
  "output_dir=models/TimeThinker-4B-SFT-v10-50k-canonical" \
  "swanlab_run_name=sft_4b-v10-50k-canonical" || exit $?

echo "[ALL DONE] SFT canonical reruns finished."
