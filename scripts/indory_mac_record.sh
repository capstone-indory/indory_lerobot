#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${INDORY_ZMQ_HOST:?Set INDORY_ZMQ_HOST to the indory_server adapter host}"
: "${INDORY_LEFT_LEADER_PORT:?Set INDORY_LEFT_LEADER_PORT to the left leader arm serial port}"
: "${INDORY_RIGHT_LEADER_PORT:?Set INDORY_RIGHT_LEADER_PORT to the right leader arm serial port}"
: "${INDORY_DATASET_REPO_ID:?Set INDORY_DATASET_REPO_ID, for example user/indory-demo}"
: "${INDORY_TASK:?Set INDORY_TASK to the task description saved in the dataset}"

robot_id="${INDORY_ROBOT_ID:-indory_xlerobot}"
teleop_id="${INDORY_TELEOP_ID:-xlerobot_bi_so101_leader}"
fps="${INDORY_FPS:-10}"
episodes="${INDORY_NUM_EPISODES:-10}"
episode_time_s="${INDORY_EPISODE_TIME_S:-60}"
reset_time_s="${INDORY_RESET_TIME_S:-10}"
max_relative_target="${INDORY_MAX_RELATIVE_TARGET:-10.0}"
head_step_rad="${INDORY_HEAD_STEP_RAD:-0.05}"
head_pan_sign="${INDORY_HEAD_PAN_SIGN:-1.0}"
head_tilt_sign="${INDORY_HEAD_TILT_SIGN:-1.0}"
enable_rgbd="${INDORY_ENABLE_RGBD:-false}"
rgbd_port="${INDORY_RGBD_PORT:-8867}"
rgbd_topic="${INDORY_RGBD_TOPIC:-/xlerobot/head/rgbd}"
rgbd_depth_width="${INDORY_RGBD_DEPTH_WIDTH:-640}"
rgbd_depth_height="${INDORY_RGBD_DEPTH_HEIGHT:-480}"
camera_transport="${INDORY_CAMERA_TRANSPORT:-zmq}"
cam_bridge_base_url="${INDORY_CAM_BRIDGE_BASE_URL:-ws://127.0.0.1:8870}"
rtp_udp_bind_ip="${INDORY_RTP_UDP_BIND_IP:-0.0.0.0}"
rtp_udp_payload_type="${INDORY_RTP_UDP_PAYLOAD_TYPE:-96}"
rtp_udp_front_port="${INDORY_RTP_UDP_FRONT_PORT:-5600}"
rtp_udp_wrist_left_port="${INDORY_RTP_UDP_WRIST_LEFT_PORT:-5602}"
rtp_udp_wrist_right_port="${INDORY_RTP_UDP_WRIST_RIGHT_PORT:-5604}"
rtp_udp_depth_port="${INDORY_RTP_UDP_DEPTH_PORT:-5610}"
rtp_udp_ffmpeg_path="${INDORY_RTP_UDP_FFMPEG_PATH:-}"
vcodec="${INDORY_VCODEC:-auto}"
streaming_encoding="${INDORY_STREAMING_ENCODING:-false}"
display_data="${INDORY_DISPLAY_DATA:-false}"
resume="${INDORY_RESUME:-false}"

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
  --robot.camera_transport="${camera_transport}" \
  --robot.cam_bridge_base_url="${cam_bridge_base_url}" \
  --robot.rtp_udp_bind_ip="${rtp_udp_bind_ip}" \
  --robot.rtp_udp_payload_type="${rtp_udp_payload_type}" \
  --robot.rtp_udp_front_port="${rtp_udp_front_port}" \
  --robot.rtp_udp_wrist_left_port="${rtp_udp_wrist_left_port}" \
  --robot.rtp_udp_wrist_right_port="${rtp_udp_wrist_right_port}" \
  --robot.rtp_udp_depth_port="${rtp_udp_depth_port}" \
  --robot.rtp_udp_ffmpeg_path="${rtp_udp_ffmpeg_path}" \
  --robot.enable_rgbd="${enable_rgbd}" \
  --robot.port_zmq_rgbd="${rgbd_port}" \
  --robot.rgbd_topic="${rgbd_topic}" \
  --robot.rgbd_depth_width="${rgbd_depth_width}" \
  --robot.rgbd_depth_height="${rgbd_depth_height}" \
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
  --dataset.reset_time_s="${reset_time_s}" \
  --dataset.vcodec="${vcodec}" \
  --dataset.streaming_encoding="${streaming_encoding}" \
  --display_data="${display_data}" \
  --resume="${resume}"
