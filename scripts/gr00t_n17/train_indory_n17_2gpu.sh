#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GROOT_DIR="${GROOT_DIR:-$(cd "${LEROBOT_DIR}/.." && pwd)/Isaac-GR00T-N1.7}"
DATASET_PATH="${DATASET_PATH:-${LEROBOT_DIR}/data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${SCRIPT_DIR}/indory_xlerobot_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${LEROBOT_DIR}/outputs/train/indory_xlerobot_groot_n17_86ep10hz_100k}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-indory_xlerobot_groot_n17_86ep10hz_100k}"
WANDB_PROJECT="${WANDB_PROJECT:-indory_lerobot}"

NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29517}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_STEPS="${SAVE_STEPS:-50000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-128}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-3}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
TRANSFORMERS_CACHE_DIR="${TRANSFORMERS_CACHE_DIR:-}"
TRANSFORMERS_LOCAL_FILES_ONLY="${TRANSFORMERS_LOCAL_FILES_ONLY:-0}"

cd "$GROOT_DIR"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [ "$SKIP_PREFLIGHT" != "1" ]; then
  "${SCRIPT_DIR}/preflight_indory_n17.sh"
fi

TRANSFORMERS_ARGS=()
if [ -n "$TRANSFORMERS_CACHE_DIR" ]; then
  TRANSFORMERS_ARGS+=(--transformers-cache-dir "$TRANSFORMERS_CACHE_DIR")
fi
if [ "$TRANSFORMERS_LOCAL_FILES_ONLY" = "1" ]; then
  TRANSFORMERS_ARGS+=(--transformers-local-files-only)
fi

exec uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --shard-size 1024 \
  --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH" \
  --episode-sampling-rate 1.0 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --deepspeed-stage "$DEEPSPEED_STAGE" \
  --gradient-checkpointing \
  --logging-steps 10 \
  --use-wandb \
  "${TRANSFORMERS_ARGS[@]}"
