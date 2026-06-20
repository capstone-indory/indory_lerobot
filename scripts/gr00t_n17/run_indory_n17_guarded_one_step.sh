#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REMOTE_IP="${REMOTE_IP:-${1:-}}"
TIMEOUT_S="${TIMEOUT_S:-3}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10.0}"
DURATION_S="${DURATION_S:-0}"
FPS="${FPS:-1}"
REQUEST_SEND="${SEND:-0}"
CONFIRM_OPERATOR_READY="${CONFIRM_OPERATOR_READY:-0}"

if [[ -z "$REMOTE_IP" ]]; then
  echo "REMOTE_IP is required. Example: REMOTE_IP=<adapter-host> $0" >&2
  exit 2
fi

if [[ "$REQUEST_SEND" == "1" && "$CONFIRM_OPERATOR_READY" != "1" ]]; then
  echo "Refusing SEND=1 without CONFIRM_OPERATOR_READY=1." >&2
  exit 2
fi

if [[ "${PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' "REMOTE_IP=$REMOTE_IP" "TIMEOUT_S=$TIMEOUT_S" bash \
    "$LEROBOT_DIR/scripts/gr00t_n17/run_probe_indory_zmq.sh"
  printf '\n'
  printf '%q ' "REMOTE_IP=$REMOTE_IP" "SEND=0" "MAX_RELATIVE_TARGET=$MAX_RELATIVE_TARGET" \
    "DURATION_S=$DURATION_S" "FPS=$FPS" bash \
    "$LEROBOT_DIR/scripts/gr00t_n17/run_indory_n17_robot_smoke.sh"
  printf '\n'
  printf '%q ' "REMOTE_IP=$REMOTE_IP" "TIMEOUT_S=$TIMEOUT_S" bash \
    "$LEROBOT_DIR/scripts/gr00t_n17/run_probe_indory_zmq.sh"
  printf '\n'
  if [[ "$REQUEST_SEND" == "1" ]]; then
    printf '%q ' "REMOTE_IP=$REMOTE_IP" "SEND=1" "MAX_RELATIVE_TARGET=$MAX_RELATIVE_TARGET" \
      "DURATION_S=$DURATION_S" "FPS=$FPS" bash \
      "$LEROBOT_DIR/scripts/gr00t_n17/run_indory_n17_robot_smoke.sh"
    printf '\n'
    printf '%q ' "REMOTE_IP=$REMOTE_IP" "TIMEOUT_S=$TIMEOUT_S" bash \
      "$LEROBOT_DIR/scripts/gr00t_n17/run_probe_indory_zmq.sh"
    printf '\n'
  fi
  exit 0
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

accepted_commands() {
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    report = json.load(f)
value = report.get("rpc", {}).get("health", {}).get("accepted_commands")
print("" if value is None else value)
PY
}

expected_send_delta() {
  python - "$DURATION_S" "$FPS" <<'PY'
import sys

duration_s = float(sys.argv[1])
fps = float(sys.argv[2])
n_steps = 1 if duration_s <= 0 else max(1, int(round(duration_s * fps)))
print(n_steps + 1)
PY
}

echo "== pre-probe =="
REMOTE_IP="$REMOTE_IP" TIMEOUT_S="$TIMEOUT_S" \
  bash "$LEROBOT_DIR/scripts/gr00t_n17/run_probe_indory_zmq.sh" | tee "$tmpdir/pre.json"
pre_count="$(accepted_commands "$tmpdir/pre.json")"

echo "== dry-run, no command payload =="
REMOTE_IP="$REMOTE_IP" SEND=0 MAX_RELATIVE_TARGET="$MAX_RELATIVE_TARGET" DURATION_S="$DURATION_S" FPS="$FPS" \
  bash "$LEROBOT_DIR/scripts/gr00t_n17/run_indory_n17_robot_smoke.sh"

echo "== mid-probe, verify dry-run sent no commands =="
REMOTE_IP="$REMOTE_IP" TIMEOUT_S="$TIMEOUT_S" \
  bash "$LEROBOT_DIR/scripts/gr00t_n17/run_probe_indory_zmq.sh" | tee "$tmpdir/mid.json"
mid_count="$(accepted_commands "$tmpdir/mid.json")"
if [[ -z "$pre_count" || -z "$mid_count" ]]; then
  echo "Could not read accepted_commands from pre/mid probe." >&2
  exit 3
fi
if [[ "$mid_count" != "$pre_count" ]]; then
  echo "Dry-run changed accepted_commands ($pre_count -> $mid_count); refusing to send." >&2
  exit 4
fi

if [[ "$REQUEST_SEND" != "1" ]]; then
  echo "Dry-run complete. Set SEND=1 CONFIRM_OPERATOR_READY=1 to send the capped rollout."
  echo "accepted_commands_pre=$pre_count"
  echo "accepted_commands_mid=$mid_count"
  exit 0
fi

echo "== SEND=1 capped rollout =="
REMOTE_IP="$REMOTE_IP" SEND=1 MAX_RELATIVE_TARGET="$MAX_RELATIVE_TARGET" DURATION_S="$DURATION_S" FPS="$FPS" \
  bash "$LEROBOT_DIR/scripts/gr00t_n17/run_indory_n17_robot_smoke.sh"

echo "== post-probe =="
REMOTE_IP="$REMOTE_IP" TIMEOUT_S="$TIMEOUT_S" \
  bash "$LEROBOT_DIR/scripts/gr00t_n17/run_probe_indory_zmq.sh" | tee "$tmpdir/post.json"
post_count="$(accepted_commands "$tmpdir/post.json")"

if [[ -z "$post_count" ]]; then
  echo "Could not read accepted_commands from post probe." >&2
  exit 3
fi
expected_delta="$(expected_send_delta)"
actual_delta=$((post_count - mid_count))
if [[ "$actual_delta" -ne "$expected_delta" ]]; then
  echo "Unexpected accepted_commands delta after send: expected $expected_delta, got $actual_delta ($mid_count -> $post_count)." >&2
  exit 5
fi

echo "accepted_commands_pre=$pre_count"
echo "accepted_commands_mid=$mid_count"
echo "accepted_commands_post=$post_count"
echo "accepted_commands_delta=$actual_delta"
