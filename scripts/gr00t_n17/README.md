# GR00T N1.7 INDORY Setup

## Prepared Paths

- Isaac-GR00T N1.7 worktree: `/home/hanbin5/Research/Capstone-Design/Isaac-GR00T-N1.7`
- Converted GR00T LeRobot v2.1 dataset: `/home/hanbin5/Research/Capstone-Design/indory_lerobot/data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz`
- Modality config: `/home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/indory_xlerobot_config.py`
- Dataset modality metadata: `/home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/modality.json`

## Current Blocker

`nvidia/GR00T-N1.7-3B` is public for this account, but the N1.7 code also loads
`nvidia/Cosmos-Reason2-2B`. That repo is gated. Request access on HuggingFace and
rerun the smoke command after the request is approved.

If a browser login uses a different token than the CLI, export it before running
preflight/training:

```bash
export HF_TOKEN=<your_huggingface_token>
```

Check access:

```bash
bash /home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/preflight_indory_n17.sh
```

## Verified Without Model Weights

The dataset was converted from LeRobot v3.0 to v2.1 and now has:

- 86 episodes
- 58,167 frames
- 10 Hz
- 258 videos, 3 cameras per episode
- 17D state/action split into `left_arm`, `left_gripper`, `right_arm`, `right_gripper`, `head`, `base_velocity`
- `meta/relative_stats.json` for relative `left_arm`, `right_arm`, and `head` actions

## Commands

Smoke after Cosmos access is approved:

```bash
bash /home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/smoke_indory_n17.sh
```

Full 2-GPU training:

```bash
bash /home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/train_indory_n17_2gpu.sh
```

The training script runs preflight first. To bypass it after a manual check:

```bash
SKIP_PREFLIGHT=1 bash /home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/train_indory_n17_2gpu.sh
```

To force cached-only operation after all model files are downloaded:

```bash
TRANSFORMERS_LOCAL_FILES_ONLY=1 \
bash /home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/train_indory_n17_2gpu.sh
```

For a shorter first run:

```bash
MAX_STEPS=1000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=1 \
bash /home/hanbin5/Research/Capstone-Design/indory_lerobot/scripts/gr00t_n17/train_indory_n17_2gpu.sh
```
