# indory_lerobot

LeRobot companion fork for the Indoory XLeRobot adapter stack.

This repository is the Mac/Ubuntu-side LeRobot workspace. It does not connect
to robot hardware directly. It connects to the
`indory_server` adapter endpoint with `robot.type=xlerobot_client`; the adapter
owns the downstream hardware path.

## What This Fork Does

- Registers `robot.type=xlerobot_client`.
- Connects to the `indory_server` adapter state, command, RPC, and RGB camera
  ports.
- Keeps robot hardware ownership behind the `indory_server` adapter.
- Lets Mac teleop combine local SO101/biSO101 leader arms with keyboard base
  control.
- Lets Mac active recording command the robot and save LeRobot datasets.
- Provides a default Ubuntu training wrapper for datasets collected through this
  path.

## What This Fork Does Not Do

- It does not open XLeRobot follower motor serial ports on the Mac or Ubuntu
  host.
- It does not own robot camera, lidar, RealSense, or hardware-side processes.
- It does not provide command authentication for `8856`.
- It does not yet provide a passive observer-only recorder.
- The included training wrapper is a default ACT recipe, not a final production
  policy contract.

## Runtime Roles

| Component               | Role                                                  | Repository                            |
| ----------------------- | ----------------------------------------------------- | ------------------------------------- |
| Robot hardware          | Motors, lidar, and camera devices                     | managed downstream of `indory_server` |
| `indory_server` adapter | Expose live ZMQ state, command, RPC, and camera ports | `indory_server`                       |
| Mac                     | Teleop with leader arms and keyboard                  | this repo                             |
| Mac                     | Active record while controlling the robot             | this repo                             |
| Ubuntu                  | Train on recorded datasets                            | this repo                             |

The `indory_server` adapter must be started first using that repository's
runtime procedure. `indory_lerobot` should only need the adapter host/IP and the
ZMQ ports below.

## ZMQ Contract

| Port   | Direction          | Purpose                                                             |
| ------ | ------------------ | ------------------------------------------------------------------- |
| `8855` | Adapter to clients | State topics: `proprio`, `joint_states`, `odom`, `tf.links`, `scan` |
| `8856` | Clients to adapter | Command `PUSH/PULL` socket                                          |
| `8857` | Client to adapter  | RPC: `health`, `calibration`, `command_status`, `topic_list`        |
| `8866` | Adapter to clients | RGB topics: `rgb.front.0`, `rgb.wrist_left.0`, `rgb.wrist_right.0`  |
| `8867` | Adapter to clients | Optional RGB-D stream                                               |

Commands sent to `8856` are MessagePack dictionaries using
`schema: xlerobot_v1.1`. Base commands use `frame: body` and
`base_cmd_vel: [vx, vy, wz]`.

## Key Files

- `INDORY_ZMQ_ROLES.md`: detailed role split and known missing work.
- `INDORY_CONTROL_API.md`: north-bound ZMQ REQ/REP + PUB/SUB contract for
  `indory_server` <-> `indory_lerobot` daemon mode (replaces subprocess
  stdout capture). REQ/REP on port 8891, events PUB on port 8893.
- `scripts/indory_mac_teleop.sh`: adapter-backed Mac teleop wrapper for standalone/manual use.
- `scripts/indory_mac_leader_publisher.py`: Mac-side DROP leader action publisher. It does not connect to the adapter command socket.
- `scripts/indory_mac_record.sh`: Mac active recording wrapper.
- `scripts/indory_act_live.py`: guarded ACT live runner with success-detector hook.
- `scripts/indory_drop_supervised_teleop.py`: super-side DROP runner that gates Mac leader actions through a success detector before forwarding to the adapter.
- `scripts/indory_success_watch.py`: observation-only success detector watcher for
  standalone success-detector checks.
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
git clone https://github.com/capstone-indory/indory_lerobot.git
cd indory_lerobot
python -m pip install -U pip
python -m pip install -e .
```

The role scripts also export `PYTHONPATH="$repo_root/src:$PYTHONPATH"`, so they
can run from the checkout even before a full editable install.

For the Ubuntu NVIDIA training server, create the shared conda environment from
the repo instead of relying on an ad hoc local environment:

```bash
conda env create -f environment.yml
conda activate lerobot
python -m pip install -U pip
python -m pip install -e ".[pi]"
python -m pip install --no-build-isolation "flash-attn==2.7.4.post1"
python -m pip install -e ".[groot]"
```

`flash-attn` is installed after the conda environment is created because it must
see the already-installed PyTorch/CUDA stack during build or wheel selection.
The `groot` extra is intentionally not part of `.[all]` in upstream LeRobot, so
install it explicitly before running `INDORY_POLICY_TYPE=groot`.

Verify the Indory client is the one being imported:

```bash
cd /path/to/indory_lerobot
PYTHONPATH="$PWD/src:${PYTHONPATH:-}" python - <<'PY'
import inspect
from lerobot.robots.xlerobot import XLerobotClient
print(inspect.getfile(XLerobotClient))
PY
```

Expected path:

```text
/path/to/indory_lerobot/src/lerobot/robots/xlerobot/xlerobot_client.py
```

## Mac Teleop

Teleop uses:

- remote adapter-backed follower through `xlerobot_client`
- local `bi_so_leader` leader arms, or a single right SO101 leader arm
- local keyboard base and head control

Run with both leader arms:

```bash
cd /path/to/indory_lerobot
INDORY_ZMQ_HOST=<adapter-host> \
INDORY_LEFT_LEADER_PORT=<left-leader-port> \
INDORY_RIGHT_LEADER_PORT=<right-leader-port> \
./scripts/indory_mac_teleop.sh
```

Run with only the right leader arm connected:

```bash
cd /path/to/indory_lerobot
INDORY_ZMQ_HOST=<adapter-host> \
INDORY_RIGHT_LEADER_PORT=<right-leader-port> \
./scripts/indory_mac_teleop.sh
```

In right-only mode, the single SO leader action is interpreted as the right arm
target. The left arm is not included in outgoing targets, so it stays at the
current/initial follower position.

For server-managed DROP, do not run the direct Mac teleop wrapper. The Mac runs
`scripts/indory_mac_leader_publisher.py`, which publishes local leader arm
actions to super and never opens the adapter command socket. Super runs
`scripts/indory_drop_supervised_teleop.py`, reads those leader actions, runs the
DROP success detector, and only forwards fresh actions to the adapter while
success is false. Use `success_region=upper` in `sky-blue-parcel-color` kwargs
to make the same line test work as the inverse of the GRAB detector.

Start the super-side gated DROP runner without sending robot commands:

```bash
python scripts/indory_drop_supervised_teleop.py \
  --remote-ip <adapter-host> \
  --leader-bind-host 0.0.0.0 \
  --leader-bind-port 8892 \
  --camera-transport cam_bridge \
  --cam-bridge-base-url ws://127.0.0.1:8870 \
  --success-detector sky-blue-parcel-color \
  --success-detector-kwargs '{"hsv_lower":[82,70,80],"hsv_upper":[125,255,255],"bottom_y_ratio":0.65,"success_region":"upper","min_area_ratio":0.02,"max_area_ratio":0.45,"position_metric":"centroid","required_consecutive":15}'
```

Add `--send` only when this super process should actually forward gated leader
actions to the adapter.

Start the Mac leader publisher:

```bash
python scripts/indory_mac_leader_publisher.py \
  --server-url tcp://super:8892 \
  --right-leader-port <right-leader-port> \
  --fps 15
```

Common optional variables:

```bash
INDORY_ROBOT_ID=indory_xlerobot
INDORY_ZMQ_ROBOT_ID=0
INDORY_ZMQ_SOURCE_ID=mac_xlerobot_teleop
INDORY_ZMQ_SOURCE_ROLE=teleop
INDORY_FPS=15
INDORY_DISPLAY_DATA=true
INDORY_TELEOP_ARM_MODE=auto
INDORY_SINGLE_LEADER_TYPE=so101_leader
INDORY_HEAD_STEP_RAD=0.05
INDORY_HEAD_PAN_SIGN=1.0
INDORY_HEAD_TILT_SIGN=1.0
```

Keyboard base controls:

| Key | Motion       |
| --- | ------------ |
| `w` | forward      |
| `s` | backward     |
| `a` | strafe left  |
| `d` | strafe right |
| `q` | rotate left  |
| `e` | rotate right |
| `n` | speed up     |
| `m` | speed down   |

Keyboard head controls:

| Key | Motion        |
| --- | ------------- |
| `i` | tilt up       |
| `k` | tilt down     |
| `j` | pan left      |
| `l` | pan right     |
| `h` | recenter head |

The client sends arm targets and base velocity through the same robot object, so
the `indory_server` adapter sees a single command source lease instead of
competing clients.

## Mac Active Recording

This record path controls the robot while saving data. It is active recording,
not passive observation.

Run:

```bash
cd /path/to/indory_lerobot
INDORY_ZMQ_HOST=<adapter-host> \
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
INDORY_FPS=15
INDORY_RESUME=false
INDORY_VCODEC=auto
INDORY_STREAMING_ENCODING=false
INDORY_DISPLAY_DATA=false
INDORY_ZMQ_SOURCE_ID=mac_xlerobot_record
INDORY_ZMQ_SOURCE_ROLE=record
INDORY_HEAD_STEP_RAD=0.05
INDORY_HEAD_PAN_SIGN=1.0
INDORY_HEAD_TILT_SIGN=1.0
```

The saved action is the action returned by `robot.send_action()`. That means the
dataset records what this client attempted to send through the ZMQ command path.
Head actions are stored as raw `head_motor_1` / `head_motor_2` tick targets:
keyboard head moves are converted to target ticks, and frames without a head move
record the current head ticks as hold targets.

RGB camera packets from `8866` are archived during each episode and decoded into
the LeRobot dataset at episode save time. This keeps teleop/record control
responsive even when real-time H.264 preview decoding falls behind.

Do not use this script for shadow logging beside another controller. A passive
observer recorder should subscribe to `8855`, `8866`, and optionally `8867` only
and must not connect to `8856`; that tool is not implemented yet.

## Ubuntu Training

On an AI server, clone this repository, create the conda environment, then run
the training wrapper against a Hugging Face dataset repo:

```bash
git clone https://github.com/capstone-indory/indory_lerobot.git
cd indory_lerobot
conda env create -f environment.yml
conda activate lerobot
python -m pip install -U pip
python -m pip install -e ".[pi]"
python -m pip install --no-build-isolation "flash-attn==2.7.4.post1"
python -m pip install -e ".[groot]"
```

The training wrapper defaults to a dual-arm SO-101 Pi0.5 fine-tuned prior and
uses Hugging Face Accelerate automatically when multiple CUDA GPUs are visible:

```bash
INDORY_DATASET_REPO_ID=<user>/<dataset-name> \
./scripts/indory_ubuntu_train.sh
```

On the 2-GPU 3090 / 3090 Ti server, the default launch is equivalent to a
2-process bf16 DDP run with per-GPU batch size 1. Increase
`INDORY_BATCH_SIZE` only after a short run confirms memory headroom.

Optional variables:

```bash
INDORY_POLICY_TYPE=pi05
INDORY_POLICY_PRESET=pi05_so101_fold_towel
INDORY_POLICY_PRETRAINED_PATH=CoRL2026-CSI/pi05_teleop_fold_towel
INDORY_POLICY_PATH=<exact-policy-checkpoint-config>
INDORY_DATASET_ROOT=<local-patched-dataset-root>
INDORY_OUTPUT_DIR=outputs/train/indory_xlerobot_pi05
INDORY_JOB_NAME=indory_xlerobot_pi05
INDORY_BATCH_SIZE=1
INDORY_TRAIN_STEPS=20000
INDORY_NUM_GPUS=2
INDORY_USE_ACCELERATE=auto
INDORY_MIXED_PRECISION=bf16
INDORY_DATASET_VIDEO_BACKEND=pyav
INDORY_TRAIN_EXPERT_ONLY=true
INDORY_GRADIENT_CHECKPOINTING=true
INDORY_POLICY_PUSH_TO_HUB=false
INDORY_DRY_RUN=false
```

Use `INDORY_POLICY_PRETRAINED_PATH` for adaptation: the policy feature schema is
rebuilt from the XLerobot dataset and compatible weights are loaded from the
pretrained model. If a SO-101 checkpoint has a 12D state/action head but
XLerobot needs 17D, matching backbone/action-expert tensors are reused and the
changed state/action projection tensors are trained from scratch. Use
`INDORY_POLICY_PATH` only when you intentionally want to load a saved policy
config exactly as-is, for example resuming a previous XLerobot run.

Pretrained model presets:

```bash
# SO-101 dual-arm Pi0.5 fine-tune, current default
INDORY_POLICY_TYPE=pi05
INDORY_POLICY_PRESET=pi05_so101_fold_towel

# Pi0.5 base
INDORY_POLICY_TYPE=pi05
INDORY_POLICY_PRETRAINED_PATH=lerobot/pi05_base

# Pi0 base
INDORY_POLICY_TYPE=pi0
INDORY_POLICY_PRETRAINED_PATH=lerobot/pi0_base

# GR00T N1.5 base or SO-101 multi-task fine-tune
INDORY_POLICY_TYPE=groot
INDORY_POLICY_PRETRAINED_PATH=nvidia/GR00T-N1.5-3B
INDORY_POLICY_PRESET=groot_so101_multitask
```

GR00T requires the `groot` extras and Flash Attention in the training
environment. If `flash_attn` is not importable, start with Pi0.5 first or install
the GR00T dependency stack before launching `INDORY_POLICY_TYPE=groot`.

Before launching multi-GPU Pi0.5 training for the first time, pre-cache the
policy assets in a single process. This avoids multiple DDP ranks competing for
the same Hugging Face cache locks:

```bash
HF_HUB_DISABLE_XET=1 \
python scripts/indory_precache_policy.py \
  --policy-type pi05 \
  --pretrained-path CoRL2026-CSI/pi05_teleop_fold_towel
```

You can also launch the recommended Pi0.5 SO-101-to-XLerobot run directly from
Python. The launcher pre-caches assets by default, uses 2-GPU Accelerate DDP
when both GPUs are visible, and enables W&B:

```bash
python -m wandb login
python scripts/indory_train_pi05_so101.py \
  --wandb-mode online
```

If `--wandb-mode online` is requested without a configured W&B API key, the
launcher fails before loading the model. Use `--wandb-mode offline` to keep local
W&B logs and sync later.

For long runs, detach the launcher and watch the generated log:

```bash
python scripts/indory_train_pi05_so101.py \
  --detach \
  --wandb-mode offline \
  --no-precache

tail -f outputs/train/_logs/<run-log>.log
```

For a one-step smoke test without W&B upload:

```bash
python scripts/indory_train_pi05_so101.py \
  --smoke \
  --wandb-mode offline \
  --no-precache
```

To inspect the exact command without starting model download or training:

```bash
INDORY_DRY_RUN=true \
INDORY_DATASET_REPO_ID=<user>/<dataset-name> \
./scripts/indory_ubuntu_train.sh
```

Run preflight before a long job:

```bash
python scripts/indory_train_preflight.py \
  --root data/indory_xlerobot_pick_delivery_head_patched \
  --policy-type pi05 \
  --scan-actions
```

For older Indory recordings, preflight should show nonzero patched head actions
but may show zero base actions. Such datasets are usable for arms/head
adaptation, but not for learning base motion.

For datasets recorded before head actions were stored in the action vector,
create a local patched dataset root before training:

```bash
python scripts/indory_patch_head_actions.py \
  --repo-id <user>/<dataset-name> \
  --output-root data/indory_xlerobot_pick_delivery_head_patched

INDORY_DATASET_REPO_ID=<user>/<dataset-name> \
INDORY_DATASET_ROOT=data/indory_xlerobot_pick_delivery_head_patched \
./scripts/indory_ubuntu_train.sh
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

- the `indory_server` adapter and downstream robot stack are healthy before teleop
- leader arm ports are correct
- camera feeds are side-specific
- no other command client is driving the robot
- there is a reachable physical stop or power procedure

## Troubleshooting

`xlerobot_client` cannot connect:

```bash
cd /path/to/indory_server
./tools/fast_robot_client.py --host <adapter-host> health
./tools/fast_robot_client.py --host <adapter-host> topics
```

No images in display or dataset:

- Confirm the adapter publishes RGB on `8866`, not `8855`.
- Confirm topics are `rgb.front.0`, `rgb.wrist_left.0`, and
  `rgb.wrist_right.0`.
- Confirm PyAV is installed if the stream uses `h264_fmp4`.
- For GR00T N1.7 live smoke, run the no-command preflight first:

```bash
REMOTE_IP=<adapter-host> \
bash scripts/gr00t_n17/run_probe_indory_zmq.sh
```

If RPC and state are healthy but the probe reports no camera topics, or TCP
connect to `8866` is refused, the adapter is reachable but the adapter camera
publisher is not serving RGB. Restart the `indory_server` adapter camera path
before running a policy dry-run or any `SEND=1` command.

Wrong LeRobot import:

```bash
cd /path/to/indory_lerobot
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
