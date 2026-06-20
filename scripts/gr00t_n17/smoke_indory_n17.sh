#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GROOT_DIR="${GROOT_DIR:-$(cd "${LEROBOT_DIR}/.." && pwd)/Isaac-GR00T-N1.7}"
DATASET_PATH="${DATASET_PATH:-${LEROBOT_DIR}/data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${SCRIPT_DIR}/indory_xlerobot_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/indory_groot_n17_smoke}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
TRANSFORMERS_CACHE_DIR="${TRANSFORMERS_CACHE_DIR:-}"
TRANSFORMERS_LOCAL_FILES_ONLY="${TRANSFORMERS_LOCAL_FILES_ONLY:-0}"

cd "$GROOT_DIR"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

exec uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 1 \
  --save-steps 1 \
  --save-total-limit 1 \
  --global-batch-size 1 \
  --dataloader-num-workers 0 \
  --num-shards-per-epoch 1 \
  --episode-sampling-rate 1.0 \
  --no-use-wandb \
  --skip-weight-loading \
  --save-only-model \
  "${TRANSFORMERS_ARGS[@]}"
