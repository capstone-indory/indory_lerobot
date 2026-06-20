#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GROOT_DIR="${GROOT_DIR:-$(cd "${LEROBOT_DIR}/.." && pwd)/Isaac-GR00T-N1.7}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
REMOTE_IP="${REMOTE_IP:-${1:-}}"

if [[ -z "$REMOTE_IP" ]]; then
  echo "REMOTE_IP is required. Example: REMOTE_IP=<pi-ip> $0" >&2
  exit 2
fi

SEND="${SEND:-0}"
ALLOW_BASE_MOTION="${ALLOW_BASE_MOTION:-0}"
ALLOW_MISSING_CAMERAS="${ALLOW_MISSING_CAMERAS:-0}"
FPS="${FPS:-1}"
DURATION_S="${DURATION_S:-0}"
DEVICE="${DEVICE:-cuda:0}"
CAMERA_PORT="${CAMERA_PORT:-8866}"
CAMERA_PORT_TIMEOUT_S="${CAMERA_PORT_TIMEOUT_S:-1}"
CHECKPOINT="${CHECKPOINT:-$LEROBOT_DIR/outputs/train/indory_xlerobot_groot_n17_86ep10hz_20k/indory_xlerobot_groot_n17_86ep10hz_20k/checkpoint-10000}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"
if [[ -z "$MAX_RELATIVE_TARGET" ]]; then
  # Match the Indory recording wrapper default. The model predicts the
  # leader-derived absolute target; this cap is only the per-command follower
  # target clamp, not the learned action scale.
  MAX_RELATIVE_TARGET="10.0"
fi

args=(
  "$LEROBOT_DIR/scripts/gr00t_n17/indory_n17_robot_smoke.py"
  --remote-ip "$REMOTE_IP"
  --checkpoint "$CHECKPOINT"
  --fps "$FPS"
  --duration-s "$DURATION_S"
  --max-relative-target "$MAX_RELATIVE_TARGET"
  --device "$DEVICE"
  --camera-port "$CAMERA_PORT"
  --camera-port-timeout-s "$CAMERA_PORT_TIMEOUT_S"
)

if [[ -n "${TASK:-}" ]]; then
  args+=(--task "$TASK")
fi
if [[ "$SEND" == "1" ]]; then
  args+=(--send)
fi
if [[ "$ALLOW_BASE_MOTION" == "1" ]]; then
  args+=(--allow-base-motion)
fi
if [[ "$ALLOW_MISSING_CAMERAS" == "1" ]]; then
  args+=(--allow-missing-cameras)
fi

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GROOT_HF_LOCAL_FIRST="${GROOT_HF_LOCAL_FIRST:-1}"
export GROOT_PATCH_MISTRAL="${GROOT_PATCH_MISTRAL:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$LEROBOT_DIR/src:$GROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cmd=(conda run -n "$CONDA_ENV" python "${args[@]}")
if [[ "${PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

cd "$LEROBOT_DIR"
exec "${cmd[@]}"
