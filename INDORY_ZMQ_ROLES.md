# Indory Adapter Role Split

This repository can participate in the Indory stack, but the current target
architecture is not "LeRobot opens robot hardware directly".

The hardware owner is outside this repository. LeRobot clients connect only to
the `indory_server` adapter endpoint and must not connect directly to the
Raspberry Pi or robot hardware.

## Target Runtime

| Component               | Role             | Expected behavior                                                                                               |
| ----------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
| Robot/Pi hardware       | Hardware runtime | Own motors, lidar, and cameras behind `indory_server`.                                                          |
| `indory_server` adapter | ZMQ adapter      | Expose state, command, RPC, and camera sockets to LeRobot clients.                                              |
| Mac                     | Teleop client    | Use `xlerobot_client` to send commands to the adapter; use local leader arms and keyboard as input devices.     |
| Mac                     | Record client    | Use LeRobot active record: read teleop input, call `send_action()`, and save observations/actions to a dataset. |
| Ubuntu                  | Train host       | Use standard `lerobot-train` against datasets collected from the Mac/robot setup.                               |

## ZMQ Contract

The client path uses the Indory adapter ZMQ ports:

| Port   | Direction          | Purpose                                                                              |
| ------ | ------------------ | ------------------------------------------------------------------------------------ |
| `8855` | Adapter to clients | Robot state topics such as `proprio`, `joint_states`, and `odom`.                    |
| `8856` | Clients to adapter | Robot command PUSH/PULL socket.                                                      |
| `8857` | Client to adapter  | RPC, including `health` and `calibration`.                                           |
| `8866` | Adapter to clients | Optimized RGB camera topics: `rgb.front.0`, `rgb.wrist_left.0`, `rgb.wrist_right.0`. |

Commands on `8856` are MessagePack dictionaries using schema
`xlerobot_v1.1`. The client includes `source_id` and `source_role`, so teleop
and record clients can be distinguished by the `indory_server` adapter.

## Implemented

- `src/lerobot/robots/xlerobot/xlerobot_client.py` is the aligned robot class
  for the Indory architecture. It connects to `8855`, `8856`, `8857`, and
  `8866` on the adapter endpoint, and it does not open local motor serial ports
  or local cameras.
- `src/lerobot/robots/xlerobot/config_xlerobot.py` registers
  `robot.type=xlerobot_client`.
- `lerobot-teleoperate` supports `xlerobot_client` plus a leader arm and
  automatically adds `KeyboardTeleop`, so arm joints and base keyboard commands
  can be sent together.
- `lerobot-record` has the same `xlerobot_client` plus leader arm and keyboard
  path. It is an active controller/recorder: it calls `send_action()` and saves
  the returned action with the observation.
- `lerobot-train` is available through the normal LeRobot CLI.

## Role Commands

There is no LeRobot-side robot server in this architecture. Start the
`indory_server` adapter first, then run Mac and Ubuntu clients against its ZMQ
ports.

Mac teleop client:

```bash
INDORY_ZMQ_HOST=<adapter-host> \
INDORY_LEFT_LEADER_PORT=<left-leader-port> \
INDORY_RIGHT_LEADER_PORT=<right-leader-port> \
./scripts/indory_mac_teleop.sh
```

Mac active record client:

```bash
INDORY_ZMQ_HOST=<adapter-host> \
INDORY_LEFT_LEADER_PORT=<left-leader-port> \
INDORY_RIGHT_LEADER_PORT=<right-leader-port> \
INDORY_DATASET_REPO_ID=<user>/<dataset-name> \
INDORY_TASK="<task>" \
./scripts/indory_mac_record.sh
```

Ubuntu training host:

```bash
INDORY_DATASET_REPO_ID=<user>/<dataset-name> ./scripts/indory_ubuntu_train.sh
```

## Not Yet Aligned Or Missing

- The hardware path still needs the external `indory_server` adapter/runtime to
  own hardware. This repository does not replace that process.
- Command authorization is not implemented in this repository. If `8856` is
  reachable, any process that can format a valid MessagePack command can send
  motion commands unless network-level controls are applied.
- The included Ubuntu training wrapper is a default ACT recipe. The canonical
  production policy, hyperparameters, and dataset validation gate are not yet
  defined.

## Current Mac Teleop Shape

Example shape, with ports adjusted for the Mac leader arms:

```bash
lerobot-teleoperate \
  --robot.type=xlerobot_client \
  --robot.id=indory_xlerobot \
  --robot.remote_ip=<adapter-host> \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=<left-leader-port> \
  --teleop.right_arm_config.port=<right-leader-port> \
  --fps=15 \
  --display_data=true
```

With `robot.type=xlerobot_client`, this path sends arm targets and keyboard base
commands through the `indory_server` adapter rather than opening robot hardware
locally.

## Current Record Shape

The record path is active teleop-plus-record. It is allowed to command the robot:

```bash
lerobot-record \
  --robot.type=xlerobot_client \
  --robot.id=indory_xlerobot \
  --robot.remote_ip=<adapter-host> \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=<left-leader-port> \
  --teleop.right_arm_config.port=<right-leader-port> \
  --dataset.repo_id=<user>/<dataset-name> \
  --dataset.single_task="<task>" \
  --dataset.fps=15
```

This records the action returned by `robot.send_action()`. It is not a passive
observer; it is the controller while recording.

## Required Next Implementation Work

1. Add network safety guidance or tooling for restricting `8856`.
2. Define the canonical training command and dataset feature contract for
   Indory XLeRobot datasets.
3. Optionally add a separate passive observer if we later need non-commanding
   diagnostics or shadow recording.
