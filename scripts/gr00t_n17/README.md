# GR00T N1.7 Indory Setup

This directory contains the Indory-specific adapters and guarded smoke-test
wrappers for NVIDIA Isaac-GR00T N1.7. The scripts assume two adjacent worktrees
by default:

```text
<workspace>/
  indory_lerobot/
  Isaac-GR00T-N1.7/
```

Override paths with environment variables when your layout differs:

```bash
export LEROBOT_DIR=/path/to/indory_lerobot
export GROOT_DIR=/path/to/Isaac-GR00T-N1.7
export DATASET_PATH=/path/to/converted/lerobot_v21_dataset
```

## Required Local Assets

The live and offline GR00T scripts need local assets that are intentionally not
tracked in this repository:

- Converted GR00T LeRobot v2.1 dataset:
  `data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz`
- A trained GR00T N1.7 checkpoint under `outputs/train/...`
- Isaac-GR00T N1.7 source checkout
- Hugging Face access for the required GR00T/Cosmos model repositories

The tracked files here provide the Indory modality config, ZMQ probe, guarded
live policy wrapper, and tests for the wrapper logic.

## Dataset Shape

The expected converted dataset has:

- 10 Hz recordings
- 3 camera streams per episode: `head`, `left_wrist`, `right_wrist`
- 17D state/action split into `left_arm`, `left_gripper`, `right_arm`,
  `right_gripper`, `head`, and `base_velocity`
- `meta/modality.json` and `meta/relative_stats.json`

## Commands

Run the offline finetune pipeline smoke:

```bash
bash scripts/gr00t_n17/smoke_indory_n17.sh
```

Run the no-command live robot preflight:

```bash
REMOTE_IP=<pi-ip> bash scripts/gr00t_n17/run_probe_indory_zmq.sh
```

Run a live policy dry-run. This connects to the robot and checks inference, but
does not send command payloads:

```bash
REMOTE_IP=<pi-ip> bash scripts/gr00t_n17/run_indory_n17_robot_smoke.sh
```

The probe must show RPC health, `proprio.0`, and camera messages for
`rgb.front.0`, `rgb.wrist_left.0`, and `rgb.wrist_right.0` before running a live
policy smoke. If camera TCP connect to `8866` is refused or all camera topics are
missing, restart the Pi camera publisher before continuing.

For a guarded one-step sequence:

```bash
REMOTE_IP=<pi-ip> \
bash scripts/gr00t_n17/run_indory_n17_guarded_one_step.sh

REMOTE_IP=<pi-ip> SEND=1 CONFIRM_OPERATOR_READY=1 \
bash scripts/gr00t_n17/run_indory_n17_guarded_one_step.sh
```

For a short guarded rollout:

```bash
REMOTE_IP=<pi-ip> DURATION_S=3 FPS=1 \
bash scripts/gr00t_n17/run_indory_n17_guarded_one_step.sh

REMOTE_IP=<pi-ip> DURATION_S=3 FPS=1 SEND=1 CONFIRM_OPERATOR_READY=1 \
bash scripts/gr00t_n17/run_indory_n17_guarded_one_step.sh
```

Base motion is disabled unless `ALLOW_BASE_MOTION=1` is set explicitly.

Record-matched live inference settings:

```bash
REMOTE_IP=<pi-ip> DURATION_S=45 FPS=10 MAX_RELATIVE_TARGET=10.0 \
  SEND=1 CONFIRM_OPERATOR_READY=1 \
bash scripts/gr00t_n17/run_indory_n17_guarded_one_step.sh
```

Keep `MAX_RELATIVE_TARGET=10.0` unless intentionally testing a more conservative
safety clamp. Smaller values truncate the policy target before the follower can
chase it and can make the robot appear stationary.

## Offline Evaluation

Dataset-grounded offline evaluation compares GR00T-predicted action chunks
against dataset ground-truth actions and a hold-current-state baseline:

```bash
PYTHONPATH="$PWD/src:/path/to/Isaac-GR00T-N1.7" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run -n lerobot python scripts/gr00t_n17/eval_indory_n17_offline_gt.py \
  --dataset-root data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz \
  --checkpoint outputs/train/<run>/<checkpoint> \
  --episodes 0:10 \
  --samples-per-episode 2 \
  --output-json outputs/eval/indory_n17_offline_gt/latest.json
```

## Validation

The unit tests below validate the wrapper and action mapping logic without
requiring a live robot:

```bash
pytest -q \
  tests/scripts/test_indory_n17_guarded_wrapper.py \
  tests/scripts/test_indory_n17_robot_smoke.py \
  tests/robots/test_xlerobot_command_builder.py
```

Dataset/checkpoint alignment tests are skipped automatically when the local
dataset or checkpoint is absent:

```bash
pytest -q tests/scripts/test_indory_n17_pipeline_alignment.py
```

## Training

Full 2-GPU training:

```bash
bash scripts/gr00t_n17/train_indory_n17_2gpu.sh
```

The training script runs preflight first. To bypass it after a manual check:

```bash
SKIP_PREFLIGHT=1 bash scripts/gr00t_n17/train_indory_n17_2gpu.sh
```

To force cached-only operation after all model files are downloaded:

```bash
TRANSFORMERS_LOCAL_FILES_ONLY=1 \
bash scripts/gr00t_n17/train_indory_n17_2gpu.sh
```

For a shorter first run:

```bash
MAX_STEPS=1000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=1 \
bash scripts/gr00t_n17/train_indory_n17_2gpu.sh
```
