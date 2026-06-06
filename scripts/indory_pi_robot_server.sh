#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

export INDORY_ZMQ_HOST="${INDORY_ZMQ_HOST:-127.0.0.1}"
export INDORY_ZMQ_CMD_PORT="${INDORY_ZMQ_CMD_PORT:-8856}"
export INDORY_ZMQ_OBSERVATIONS_PORT="${INDORY_ZMQ_OBSERVATIONS_PORT:-8855}"
export INDORY_ZMQ_RPC_PORT="${INDORY_ZMQ_RPC_PORT:-8857}"
export INDORY_ZMQ_CAMERAS_PORT="${INDORY_ZMQ_CAMERAS_PORT:-8866}"
export INDORY_ZMQ_ROBOT_ID="${INDORY_ZMQ_ROBOT_ID:-0}"
export INDORY_ZMQ_SOURCE_ID="${INDORY_ZMQ_SOURCE_ID:-indory_lerobot_pi_proxy}"
export INDORY_ZMQ_SOURCE_ROLE="${INDORY_ZMQ_SOURCE_ROLE:-proxy}"

cd "${repo_root}"
exec python -m lerobot.robots.xlerobot.xlerobot_host
