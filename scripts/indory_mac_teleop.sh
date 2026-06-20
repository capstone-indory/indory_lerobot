#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${INDORY_ZMQ_HOST:?Set INDORY_ZMQ_HOST to the indory_server adapter host}"

robot_id="${INDORY_ROBOT_ID:-indory_xlerobot}"
fps="${INDORY_FPS:-15}"
display_data="${INDORY_DISPLAY_DATA:-true}"
max_relative_target="${INDORY_MAX_RELATIVE_TARGET:-10.0}"
head_step_rad="${INDORY_HEAD_STEP_RAD:-0.05}"
head_pan_sign="${INDORY_HEAD_PAN_SIGN:-1.0}"
head_tilt_sign="${INDORY_HEAD_TILT_SIGN:-1.0}"
camera_transport="${INDORY_CAMERA_TRANSPORT:-zmq}"
cam_bridge_base_url="${INDORY_CAM_BRIDGE_BASE_URL:-ws://127.0.0.1:8870}"
rtp_udp_bind_ip="${INDORY_RTP_UDP_BIND_IP:-0.0.0.0}"
rtp_udp_payload_type="${INDORY_RTP_UDP_PAYLOAD_TYPE:-96}"
rtp_udp_front_port="${INDORY_RTP_UDP_FRONT_PORT:-5600}"
rtp_udp_wrist_left_port="${INDORY_RTP_UDP_WRIST_LEFT_PORT:-5602}"
rtp_udp_wrist_right_port="${INDORY_RTP_UDP_WRIST_RIGHT_PORT:-5604}"
rtp_udp_depth_port="${INDORY_RTP_UDP_DEPTH_PORT:-5610}"
rtp_udp_ffmpeg_path="${INDORY_RTP_UDP_FFMPEG_PATH:-}"
left_leader_port="${INDORY_LEFT_LEADER_PORT:-}"
right_leader_port="${INDORY_RIGHT_LEADER_PORT:-}"
teleop_arm_mode="${INDORY_TELEOP_ARM_MODE:-auto}"

camera_args=(
  --robot.camera_transport="${camera_transport}"
  --robot.cam_bridge_base_url="${cam_bridge_base_url}"
  --robot.rtp_udp_bind_ip="${rtp_udp_bind_ip}"
  --robot.rtp_udp_payload_type="${rtp_udp_payload_type}"
  --robot.rtp_udp_front_port="${rtp_udp_front_port}"
  --robot.rtp_udp_wrist_left_port="${rtp_udp_wrist_left_port}"
  --robot.rtp_udp_wrist_right_port="${rtp_udp_wrist_right_port}"
  --robot.rtp_udp_depth_port="${rtp_udp_depth_port}"
)
if [[ -n "${rtp_udp_ffmpeg_path}" ]]; then
  camera_args+=(--robot.rtp_udp_ffmpeg_path="${rtp_udp_ffmpeg_path}")
fi

if [[ "${teleop_arm_mode}" == "auto" ]]; then
  if [[ -n "${left_leader_port}" && -n "${right_leader_port}" ]]; then
    teleop_arm_mode="bimanual"
  elif [[ -n "${right_leader_port}" ]]; then
    teleop_arm_mode="right_only"
  else
    echo "Set INDORY_RIGHT_LEADER_PORT, or set both INDORY_LEFT_LEADER_PORT and INDORY_RIGHT_LEADER_PORT." >&2
    exit 2
  fi
fi

teleop_args=()
case "${teleop_arm_mode}" in
  bimanual|bi)
    : "${left_leader_port:?Set INDORY_LEFT_LEADER_PORT for bimanual teleop}"
    : "${right_leader_port:?Set INDORY_RIGHT_LEADER_PORT for bimanual teleop}"
    teleop_id="${INDORY_TELEOP_ID:-xlerobot_bi_so101_leader}"
    teleop_args=(
      --teleop.type=bi_so_leader
      --teleop.id="${teleop_id}"
      --teleop.left_arm_config.port="${left_leader_port}"
      --teleop.left_arm_config.use_degrees=true
      --teleop.right_arm_config.port="${right_leader_port}"
      --teleop.right_arm_config.use_degrees=true
    )
    ;;
  right|right_only)
    : "${right_leader_port:?Set INDORY_RIGHT_LEADER_PORT for right-only teleop}"
    teleop_id="${INDORY_TELEOP_ID:-xlerobot_right_so101_leader}"
    single_leader_type="${INDORY_SINGLE_LEADER_TYPE:-so101_leader}"
    teleop_args=(
      --teleop.type="${single_leader_type}"
      --teleop.id="${teleop_id}"
      --teleop.port="${right_leader_port}"
      --teleop.use_degrees=true
    )
    ;;
  *)
    echo "Unsupported INDORY_TELEOP_ARM_MODE=${teleop_arm_mode}. Use auto, bimanual, or right_only." >&2
    exit 2
    ;;
esac

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
  "${camera_args[@]}" \
  --robot.source_id="${INDORY_ZMQ_SOURCE_ID:-mac_xlerobot_teleop}" \
  --robot.source_role="${INDORY_ZMQ_SOURCE_ROLE:-teleop}" \
  "${teleop_args[@]}" \
  --fps="${fps}" \
  --display_data="${display_data}"
