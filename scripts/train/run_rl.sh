#!/usr/bin/env bash
set -ex

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export DECORD_EOF_RETRY_MAX=2048001

PROJECT_NAME=${PROJECT_NAME:-EasyR1-timethinker-rl}
EXP_NAME=${EXP_NAME:-qwen3_vl_timethinker_4b_rl}

MODEL_PATH=${MODEL_PATH:-models/TimeThinker-4B-SFT}
TRAIN_FILE=${TRAIN_FILE:-EasyR1/data/timethinker_rl_train.json}
TEST_FILE=${TEST_FILE:-${TRAIN_FILE}}
IMAGE_DIR=${IMAGE_DIR:-.}

# Video-R1 trains for about 1.2k RL steps. Set MAX_STEPS=null to run by epoch.
MAX_STEPS=${MAX_STEPS:-1200}
SAVE_FREQ=${SAVE_FREQ:-100}

ROLLOUT_BS=${ROLLOUT_BS:-32}
GLOBAL_BS=${GLOBAL_BS:-32}
MB_PER_UPDATE=${MB_PER_UPDATE:-1}
MB_PER_EXP=${MB_PER_EXP:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}

TP_SIZE=${TP_SIZE:-4}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
MIN_PIXELS=${MIN_PIXELS:-3136}
MAX_PIXELS=${MAX_PIXELS:-401408}
VIDEO_FPS=${VIDEO_FPS:-2.0}

LEARNING_RATE=${LEARNING_RATE:-1.0e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-1.0e-2}
KL_COEF=${KL_COEF:-4.0e-2}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-5.0}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
LOGGER=${LOGGER:-'["file"]'}

PYTHON=${PYTHON:-.venv_rl/bin/python}

export PYTHONPATH="${REPO_ROOT}/EasyR1${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m verl.trainer.main \
    config=EasyR1/examples/config_ema_grpo_64.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_dir="${IMAGE_DIR}" \
    data.format_prompt=null \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    data.min_pixels="${MIN_PIXELS}" \
    data.max_pixels="${MAX_PIXELS}" \
    data.video_fps="${VIDEO_FPS}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.max_grad_norm="${MAX_GRAD_NORM}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr="${LEARNING_RATE}" \
    worker.actor.optim.weight_decay="${WEIGHT_DECAY}" \
    worker.rollout.n="${NUM_GENERATIONS}" \
    worker.rollout.temperature="${TEMPERATURE}" \
    worker.rollout.top_p="${TOP_P}" \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    algorithm.kl_coef="${KL_COEF}" \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_checkpoint_path=models/TimeThinker-4B-RL
