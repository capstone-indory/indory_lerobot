#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${INDORY_ZMQ_HOST:?Set INDORY_ZMQ_HOST to the Raspberry Pi address}"
: "${INDORY_LEFT_LEADER_PORT:?Set INDORY_LEFT_LEADER_PORT to the left leader arm serial port}"
: "${INDORY_RIGHT_LEADER_PORT:?Set INDORY_RIGHT_LEADER_PORT to the right leader arm serial port}"

robot_id="${INDORY_ROBOT_ID:-indory_xlerobot}"
fps="${INDORY_FPS:-30}"
display_data="${INDORY_DISPLAY_DATA:-true}"

cd "${repo_root}"
exec python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=xlerobot_client \
  --robot.id="${robot_id}" \
  --robot.remote_ip="${INDORY_ZMQ_HOST}" \
  --robot.robot_id="${INDORY_ZMQ_ROBOT_ID:-0}" \
  --robot.source_id="${INDORY_ZMQ_SOURCE_ID:-mac_xlerobot_teleop}" \
  --robot.source_role="${INDORY_ZMQ_SOURCE_ROLE:-teleop}" \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port="${INDORY_LEFT_LEADER_PORT}" \
  --teleop.right_arm_config.port="${INDORY_RIGHT_LEADER_PORT}" \
  --fps="${fps}" \
  --display_data="${display_data}"
