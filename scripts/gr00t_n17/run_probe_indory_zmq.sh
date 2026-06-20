#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
REMOTE_IP="${REMOTE_IP:-${1:-}}"
TIMEOUT_S="${TIMEOUT_S:-3}"
ROBOT_ID="${ROBOT_ID:-0}"
SKIP_CAMERAS="${SKIP_CAMERAS:-0}"

if [[ -z "$REMOTE_IP" ]]; then
  echo "REMOTE_IP is required. Example: REMOTE_IP=<adapter-host> $0" >&2
  exit 2
fi

args=(
  "$LEROBOT_DIR/scripts/gr00t_n17/probe_indory_zmq.py"
  --remote-ip "$REMOTE_IP"
  --robot-id "$ROBOT_ID"
  --timeout-s "$TIMEOUT_S"
)

if [[ "$SKIP_CAMERAS" == "1" ]]; then
  args+=(--skip-cameras)
fi

export PYTHONPATH="$LEROBOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

cmd=(conda run -n "$CONDA_ENV" python "${args[@]}")
if [[ "${PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

cd "$LEROBOT_DIR"
exec "${cmd[@]}"
