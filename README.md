# indory_lerobot

LeRobot companion fork for the Indoory XLeRobot ZMQ stack.

This repository is the Mac/Ubuntu-side LeRobot workspace. It does not replace
the Raspberry Pi hardware server. The Pi must run `/home/pi/indory_zmq`, and
this fork connects to that server with `robot.type=xlerobot_client`.

## What This Fork Does

- Registers `robot.type=xlerobot_client`.
- Connects to the Pi fast ZMQ state, command, RPC, and RGB camera ports.
- Keeps robot hardware ownership on the Pi.
- Lets Mac teleop combine local SO101/biSO101 leader arms with keyboard base
  control.
- Lets Mac active recording command the robot and save LeRobot datasets.
- Provides a default Ubuntu training wrapper for datasets collected through this
  path.

## What This Fork Does Not Do

- It does not open XLeRobot follower motor serial ports on the Mac or Ubuntu
  host.
- It does not own Pi camera, lidar, or RealSense devices.
- It does not provide command authentication for `8856`.
- It does not yet provide a passive observer-only recorder.
- The included training wrapper is a default ACT recipe, not a final production
  policy contract.

## Runtime Roles

| Machine | Role | Repository |
| --- | --- | --- |
| Raspberry Pi | Own motors, lidar, cameras, and live ZMQ ports | `/home/pi/indory_zmq` |
| Mac | Teleop with leader arms and keyboard | this repo |
| Mac | Active record while controlling the robot | this repo |
| Ubuntu | Train on recorded datasets | this repo |

The Pi must be started first:

```bash
cd /home/pi/indory_zmq
./scripts/indory_live_stack.sh restart
./scripts/indory_live_stack.sh status
```

## ZMQ Contract

| Port | Direction | Purpose |
| --- | --- | --- |
| `8855` | Pi to clients | State topics: `proprio`, `joint_states`, `odom`, `tf.links`, `scan` |
| `8856` | Clients to Pi | Command `PUSH/PULL` socket |
| `8857` | Client to Pi | RPC: `health`, `calibration`, `command_status`, `topic_list` |
| `8866` | Pi to clients | RGB topics: `rgb.front.0`, `rgb.wrist_left.0`, `rgb.wrist_right.0` |
| `8867` | Pi to clients | Optional RGB-D stream |

Commands sent to `8856` are MessagePack dictionaries using
`schema: xlerobot_v1.1`. Base commands use `frame: body` and
`base_cmd_vel: [vx, vy, wz]`.

## Key Files

- `INDORY_ZMQ_ROLES.md`: detailed role split and known missing work.
- `scripts/indory_mac_teleop.sh`: Mac teleop wrapper.
- `scripts/indory_mac_record.sh`: Mac active recording wrapper.
- `scripts/indory_ubuntu_train.sh`: default training wrapper.
- `src/lerobot/robots/xlerobot/config_xlerobot.py`: `xlerobot_client`
  configuration registration.
- `src/lerobot/robots/xlerobot/xlerobot_client.py`: ZMQ robot client.
- `src/lerobot/scripts/lerobot_teleoperate.py`: keyboard plus arm teleop merge.
- `src/lerobot/scripts/lerobot_record.py`: active record loop integration.

## Local Setup

Use this checkout directly, or install it editable in the environment you use
for LeRobot:

```bash
cd /home/pi/indory_lerobot
python -m pip install -e .
```

The role scripts also export `PYTHONPATH="$repo_root/src:$PYTHONPATH"`, so they
can run from the checkout even before a full editable install.

Verify the Indory client is the one being imported:

```bash
cd /home/pi/indory_lerobot
PYTHONPATH="$PWD/src:${PYTHONPATH:-}" python - <<'PY'
import inspect
from lerobot.robots.xlerobot import XLerobotClient
print(inspect.getfile(XLerobotClient))
PY
```

Expected path:

```text
/home/pi/indory_lerobot/src/lerobot/robots/xlerobot/xlerobot_client.py
```

## Mac Teleop

Teleop uses:

- remote Pi follower through `xlerobot_client`
- local `bi_so_leader` leader arms
- local keyboard base control

Run:

```bash
cd /home/pi/indory_lerobot
INDORY_ZMQ_HOST=<pi-ip> \
INDORY_LEFT_LEADER_PORT=<left-leader-port> \
INDORY_RIGHT_LEADER_PORT=<right-leader-port> \
./scripts/indory_mac_teleop.sh
```

Common optional variables:

```bash
INDORY_ROBOT_ID=indory_xlerobot
INDORY_ZMQ_ROBOT_ID=0
INDORY_ZMQ_SOURCE_ID=mac_xlerobot_teleop
INDORY_ZMQ_SOURCE_ROLE=teleop
INDORY_FPS=30
INDORY_DISPLAY_DATA=true
```

Keyboard base controls:

| Key | Motion |
| --- | --- |
| `i` | forward |
| `k` | backward |
| `j` | strafe left |
| `l` | strafe right |
| `u` | rotate left |
| `o` | rotate right |
| `n` | speed up |
| `m` | speed down |

The client sends arm targets and base velocity through the same robot object, so
`indory_zmq` sees a single command source lease instead of competing clients.

## Mac Active Recording

This record path controls the robot while saving data. It is active recording,
not passive observation.

Run:

```bash
cd /home/pi/indory_lerobot
INDORY_ZMQ_HOST=<pi-ip> \
INDORY_LEFT_LEADER_PORT=<left-leader-port> \
INDORY_RIGHT_LEADER_PORT=<right-leader-port> \
INDORY_DATASET_REPO_ID=<user>/<dataset-name> \
INDORY_TASK="<task description>" \
./scripts/indory_mac_record.sh
```

Common optional variables:

```bash
INDORY_NUM_EPISODES=10
INDORY_EPISODE_TIME_S=60
INDORY_RESET_TIME_S=10
INDORY_FPS=30
INDORY_ZMQ_SOURCE_ID=mac_xlerobot_record
INDORY_ZMQ_SOURCE_ROLE=record
```

The saved action is the action returned by `robot.send_action()`. That means the
dataset records what this client attempted to send through the ZMQ command path.

Do not use this script for shadow logging beside another controller. A passive
observer recorder should subscribe to `8855`, `8866`, and optionally `8867` only
and must not connect to `8856`; that tool is not implemented yet.

## Ubuntu Training

The training wrapper is a default ACT training entrypoint:

```bash
cd /home/pi/indory_lerobot
INDORY_DATASET_REPO_ID=<user>/<dataset-name> \
./scripts/indory_ubuntu_train.sh
```

Optional variables:

```bash
INDORY_POLICY_TYPE=act
INDORY_OUTPUT_DIR=outputs/train/indory_xlerobot_act
INDORY_JOB_NAME=indory_xlerobot_act
INDORY_BATCH_SIZE=8
INDORY_TRAIN_STEPS=20000
```

Before treating a trained policy as production-ready, define and verify:

- dataset feature schema
- camera topic availability
- state/action units
- normalization and calibration assumptions
- validation task and success gate

## Observation And Action Shape

`XLerobotClient` exposes state features for:

```text
left_arm_shoulder_pan
left_arm_shoulder_lift
left_arm_elbow_flex
left_arm_wrist_flex
left_arm_wrist_roll
left_arm_gripper
right_arm_shoulder_pan
right_arm_shoulder_lift
right_arm_elbow_flex
right_arm_wrist_flex
right_arm_wrist_roll
right_arm_gripper
head_motor_1
head_motor_2
x.vel
y.vel
theta.vel
```

Camera observations:

```text
head
left_wrist
right_wrist
```

The remote RGB topics are:

```text
rgb.front.<robot_id>
rgb.wrist_left.<robot_id>
rgb.wrist_right.<robot_id>
```

`xlerobot_client` decodes `jpeg` and `h264_fmp4` RGB payloads from `8866`.
H.264/fMP4 payloads need the init bytes supplied by the camera stream.

## Safety

The command port `8856` is unauthenticated. Only run teleop and active recording
on a trusted network, private tunnel, or firewall-restricted path.

Always verify:

- the Pi stack is healthy before teleop
- leader arm ports are correct
- camera feeds are side-specific
- no other command client is driving the robot
- there is a reachable physical stop or power procedure

## Troubleshooting

`xlerobot_client` cannot connect:

```bash
cd /home/pi/indory_zmq
./tools/fast_robot_client.py --host <pi-ip> health
./tools/fast_robot_client.py --host <pi-ip> topics
```

No images in display or dataset:

- Confirm the Pi publishes RGB on `8866`, not `8855`.
- Confirm topics are `rgb.front.0`, `rgb.wrist_left.0`, and
  `rgb.wrist_right.0`.
- Confirm PyAV is installed if the stream uses `h264_fmp4`.

Wrong LeRobot import:

```bash
cd /home/pi/indory_lerobot
PYTHONPATH="$PWD/src:${PYTHONPATH:-}" python - <<'PY'
import inspect
from lerobot.robots.xlerobot import XLerobotClient
print(inspect.getfile(XLerobotClient))
PY
```

Leader arm not found:

- Run `lerobot-find-port` on the Mac.
- Use the discovered ports for `INDORY_LEFT_LEADER_PORT` and
  `INDORY_RIGHT_LEADER_PORT`.

Record accidentally moves the robot:

- That is expected for `indory_mac_record.sh`; it is active recording.
- Use or implement a passive observer-only recorder for non-commanding logs.

## Upstream LeRobot

This fork is based on Hugging Face LeRobot. General documentation remains useful
for datasets, policies, processors, and training concepts:

- https://huggingface.co/docs/lerobot
- https://github.com/huggingface/lerobot
