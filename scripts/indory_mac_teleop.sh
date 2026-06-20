#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${INDORY_ZMQ_HOST:?Set INDORY_ZMQ_HOST to the indory_server adapter host}"
: "${INDORY_LEFT_LEADER_PORT:?Set INDORY_LEFT_LEADER_PORT to the left leader arm serial port}"
: "${INDORY_RIGHT_LEADER_PORT:?Set INDORY_RIGHT_LEADER_PORT to the right leader arm serial port}"

robot_id="${INDORY_ROBOT_ID:-indory_xlerobot}"
teleop_id="${INDORY_TELEOP_ID:-xlerobot_bi_so101_leader}"
fps="${INDORY_FPS:-15}"
display_data="${INDORY_DISPLAY_DATA:-true}"
max_relative_target="${INDORY_MAX_RELATIVE_TARGET:-10.0}"
head_step_rad="${INDORY_HEAD_STEP_RAD:-0.05}"
head_pan_sign="${INDORY_HEAD_PAN_SIGN:-1.0}"
head_tilt_sign="${INDORY_HEAD_TILT_SIGN:-1.0}"

cd "${repo_root}"
exec python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=xlerobot_client \
  --robot.id="${robot_id}" \
  --robot.remote_ip="${INDORY_ZMQ_HOST}" \
  --robot.robot_id="${INDORY_ZMQ_ROBOT_ID:-0}" \
  --robot.leader_action_units=degrees \
  --robot.max_relative_target="${max_relative_target}" \
  --robot.head_step_rad="${head_step_rad}" \
  --robot.head_pan_sign="${head_pan_sign}" \
  --robot.head_tilt_sign="${head_tilt_sign}" \
  --robot.source_id="${INDORY_ZMQ_SOURCE_ID:-mac_xlerobot_teleop}" \
  --robot.source_role="${INDORY_ZMQ_SOURCE_ROLE:-teleop}" \
  --teleop.type=bi_so_leader \
  --teleop.id="${teleop_id}" \
  --teleop.left_arm_config.port="${INDORY_LEFT_LEADER_PORT}" \
  --teleop.left_arm_config.use_degrees=true \
  --teleop.right_arm_config.port="${INDORY_RIGHT_LEADER_PORT}" \
  --teleop.right_arm_config.use_degrees=true \
  --fps="${fps}" \
  --display_data="${display_data}"
