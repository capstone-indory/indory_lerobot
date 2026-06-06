#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${INDORY_ZMQ_HOST:?Set INDORY_ZMQ_HOST to the Raspberry Pi address}"
: "${INDORY_LEFT_LEADER_PORT:?Set INDORY_LEFT_LEADER_PORT to the left leader arm serial port}"
: "${INDORY_RIGHT_LEADER_PORT:?Set INDORY_RIGHT_LEADER_PORT to the right leader arm serial port}"
: "${INDORY_DATASET_REPO_ID:?Set INDORY_DATASET_REPO_ID, for example user/indory-demo}"
: "${INDORY_TASK:?Set INDORY_TASK to the task description saved in the dataset}"

robot_id="${INDORY_ROBOT_ID:-indory_xlerobot}"
teleop_id="${INDORY_TELEOP_ID:-xlerobot_bi_so101_leader}"
fps="${INDORY_FPS:-30}"
episodes="${INDORY_NUM_EPISODES:-10}"
episode_time_s="${INDORY_EPISODE_TIME_S:-60}"
reset_time_s="${INDORY_RESET_TIME_S:-10}"
max_relative_target="${INDORY_MAX_RELATIVE_TARGET:-10.0}"
head_step_rad="${INDORY_HEAD_STEP_RAD:-0.025}"
head_pan_sign="${INDORY_HEAD_PAN_SIGN:--1.0}"
head_tilt_sign="${INDORY_HEAD_TILT_SIGN:--1.0}"

cd "${repo_root}"
exec python -m lerobot.scripts.lerobot_record \
  --robot.type=xlerobot_client \
  --robot.id="${robot_id}" \
  --robot.remote_ip="${INDORY_ZMQ_HOST}" \
  --robot.robot_id="${INDORY_ZMQ_ROBOT_ID:-0}" \
  --robot.leader_action_units=degrees \
  --robot.max_relative_target="${max_relative_target}" \
  --robot.head_step_rad="${head_step_rad}" \
  --robot.head_pan_sign="${head_pan_sign}" \
  --robot.head_tilt_sign="${head_tilt_sign}" \
  --robot.source_id="${INDORY_ZMQ_SOURCE_ID:-mac_xlerobot_record}" \
  --robot.source_role="${INDORY_ZMQ_SOURCE_ROLE:-record}" \
  --teleop.type=bi_so_leader \
  --teleop.id="${teleop_id}" \
  --teleop.left_arm_config.port="${INDORY_LEFT_LEADER_PORT}" \
  --teleop.left_arm_config.use_degrees=true \
  --teleop.right_arm_config.port="${INDORY_RIGHT_LEADER_PORT}" \
  --teleop.right_arm_config.use_degrees=true \
  --dataset.repo_id="${INDORY_DATASET_REPO_ID}" \
  --dataset.single_task="${INDORY_TASK}" \
  --dataset.fps="${fps}" \
  --dataset.num_episodes="${episodes}" \
  --dataset.episode_time_s="${episode_time_s}" \
  --dataset.reset_time_s="${reset_time_s}"
