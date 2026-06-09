#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${INDORY_DATASET_REPO_ID:?Set INDORY_DATASET_REPO_ID, for example user/indory-demo}"

detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

detect_visible_gpu_count() {
  if [[ -z "${INDORY_CUDA_VISIBLE_DEVICES:-}" ]]; then
    detect_gpu_count
    return
  fi
  local visible="${INDORY_CUDA_VISIBLE_DEVICES// /}"
  if [[ -z "${visible}" ]]; then
    printf '0'
    return
  fi
  local comma_count="${visible//[^,]/}"
  printf '%s' "$(( ${#comma_count} + 1 ))"
}

default_policy_preset() {
  case "$1" in
    pi05) printf 'pi05_so101_fold_towel' ;;
    *) printf '' ;;
  esac
}

default_pretrained_path() {
  case "${INDORY_POLICY_PRESET:-$(default_policy_preset "$1")}" in
    pi05_so101_fold_towel) printf 'CoRL2026-CSI/pi05_teleop_fold_towel'; return ;;
    groot_so101_multitask) printf 'CoRL2026-CSI/Gr00t_n1.5-IsaacLab-SO101-Multi_Task-30fps_8epoch'; return ;;
  esac
  case "$1" in
    pi0) printf 'lerobot/pi0_base' ;;
    pi05) printf 'lerobot/pi05_base' ;;
    groot) printf 'nvidia/GR00T-N1.5-3B' ;;
    *) printf '' ;;
  esac
}

default_policy_batch_size() {
  case "$1" in
    pi0 | pi05 | groot) printf '1' ;;
    *) printf '8' ;;
  esac
}

policy_type="${INDORY_POLICY_TYPE:-pi05}"
output_dir="${INDORY_OUTPUT_DIR:-outputs/train/indory_xlerobot_${policy_type}}"
job_name="${INDORY_JOB_NAME:-indory_xlerobot_${policy_type}}"
batch_size="${INDORY_BATCH_SIZE:-$(default_policy_batch_size "${policy_type}")}"
steps="${INDORY_TRAIN_STEPS:-20000}"
num_workers="${INDORY_NUM_WORKERS:-4}"
save_freq="${INDORY_SAVE_FREQ:-5000}"
log_freq="${INDORY_LOG_FREQ:-100}"
policy_push_to_hub="${INDORY_POLICY_PUSH_TO_HUB:-false}"
dataset_video_backend="${INDORY_DATASET_VIDEO_BACKEND:-pyav}"
num_gpus="${INDORY_NUM_GPUS:-$(detect_visible_gpu_count)}"
use_accelerate="${INDORY_USE_ACCELERATE:-auto}"
mixed_precision="${INDORY_MIXED_PRECISION:-bf16}"
pretrained_path="${INDORY_POLICY_PRETRAINED_PATH:-$(default_pretrained_path "${policy_type}")}"

if [[ -n "${INDORY_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${INDORY_CUDA_VISIBLE_DEVICES}"
fi

cd "${repo_root}"

cmd=(
  --dataset.repo_id="${INDORY_DATASET_REPO_ID}"
  --dataset.video_backend="${dataset_video_backend}"
  --output_dir="${output_dir}"
  --job_name="${job_name}"
  --batch_size="${batch_size}"
  --steps="${steps}"
  --num_workers="${num_workers}"
  --save_freq="${save_freq}"
  --log_freq="${log_freq}"
  --policy.push_to_hub="${policy_push_to_hub}"
)

if [[ -n "${INDORY_DATASET_ROOT:-}" ]]; then
  cmd+=(--dataset.root="${INDORY_DATASET_ROOT}")
fi

if [[ -n "${INDORY_POLICY_PATH:-}" ]]; then
  cmd+=(--policy.path="${INDORY_POLICY_PATH}")
else
  cmd+=(--policy.type="${policy_type}")
  if [[ -n "${pretrained_path}" ]]; then
    cmd+=(--policy.pretrained_path="${pretrained_path}")
  fi
fi

case "${policy_type}" in
  pi0 | pi05)
    cmd+=(--policy.dtype="${INDORY_POLICY_DTYPE:-bfloat16}")
    cmd+=(--policy.gradient_checkpointing="${INDORY_GRADIENT_CHECKPOINTING:-true}")
    cmd+=(--policy.compile_model="${INDORY_COMPILE_MODEL:-false}")
    cmd+=(--policy.freeze_vision_encoder="${INDORY_FREEZE_VISION_ENCODER:-false}")
    cmd+=(--policy.train_expert_only="${INDORY_TRAIN_EXPERT_ONLY:-true}")
    ;;
  groot)
    cmd+=(--policy.chunk_size="${INDORY_CHUNK_SIZE:-16}")
    cmd+=(--policy.n_action_steps="${INDORY_N_ACTION_STEPS:-16}")
    cmd+=(--policy.tune_llm="${INDORY_GROOT_TUNE_LLM:-false}")
    cmd+=(--policy.tune_visual="${INDORY_GROOT_TUNE_VISUAL:-false}")
    cmd+=(--policy.tune_projector="${INDORY_GROOT_TUNE_PROJECTOR:-true}")
    cmd+=(--policy.tune_diffusion_model="${INDORY_GROOT_TUNE_DIFFUSION_MODEL:-true}")
    cmd+=(--policy.use_bf16="${INDORY_GROOT_USE_BF16:-true}")
    ;;
esac

if [[ -n "${INDORY_POLICY_REPO_ID:-}" ]]; then
  cmd+=(--policy.repo_id="${INDORY_POLICY_REPO_ID}")
fi

if [[ "${INDORY_WANDB_ENABLE:-false}" == "true" ]]; then
  cmd+=(--wandb.enable=true)
  if [[ -n "${INDORY_WANDB_PROJECT:-}" ]]; then
    cmd+=(--wandb.project="${INDORY_WANDB_PROJECT}")
  fi
fi

cmd+=("$@")

launch_with_accelerate=false
if [[ "${use_accelerate}" == "true" ]]; then
  launch_with_accelerate=true
elif [[ "${use_accelerate}" == "auto" && "${num_gpus}" =~ ^[0-9]+$ && "${num_gpus}" -gt 1 ]]; then
  launch_with_accelerate=true
fi

if [[ "${launch_with_accelerate}" == "true" ]]; then
  if [[ "${INDORY_DRY_RUN:-false}" == "true" ]]; then
    printf '%q ' accelerate launch --multi_gpu --num_processes="${num_gpus}" --mixed_precision="${mixed_precision}" -m lerobot.scripts.lerobot_train "${cmd[@]}"
    printf '\n'
    exit 0
  fi
  exec accelerate launch \
    --multi_gpu \
    --num_processes="${num_gpus}" \
    --mixed_precision="${mixed_precision}" \
    -m lerobot.scripts.lerobot_train \
    "${cmd[@]}"
fi

if [[ "${INDORY_DRY_RUN:-false}" == "true" ]]; then
  printf '%q ' python -m lerobot.scripts.lerobot_train "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec python -m lerobot.scripts.lerobot_train "${cmd[@]}"
