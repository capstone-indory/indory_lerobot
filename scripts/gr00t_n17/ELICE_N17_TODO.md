# GR00T N1.7 Elice A100 Handoff

Date: 2026-06-09 UTC

## Current State

- Remote SSH target: `elicer@central-02.tcp.tunnel.elice.io`, port `23550`.
- Remote host: `443c75017cb8`.
- Remote GPU: `NVIDIA A100 80GB PCIe`.
- Remote RAM: `192Gi`.
- Remote base directory: `/home/elicer/Research/Capstone-Design`.
- Remote GR00T repo: `/home/elicer/Research/Capstone-Design/Isaac-GR00T-N1.7`.
- Remote LeRobot repo/workdir: `/home/elicer/Research/Capstone-Design/indory_lerobot`.
- GR00T source was cloned directly from GitHub and checked out at:
  `626af89d3e914ec92eab5323e23b9ed44a7b26c8`.
- The local N1.7 CLI patch was applied on remote to:
  - `gr00t/configs/finetune_config.py`
  - `gr00t/experiment/launch_finetune.py`
- Remote GR00T training venv exists at:
  `/home/elicer/Research/Capstone-Design/Isaac-GR00T-N1.7/.venv`.
- Use `uv pip install -e .` style setup on this repo. `uv sync` failed because the repo contains aarch64 wheel files as Git LFS pointers, and `uv sync` tries to resolve both x86_64 and aarch64 required environments.
- User-level `ffmpeg` is installed through `imageio-ffmpeg` and linked at `~/.local/bin/ffmpeg`.

## Dataset State

Remote dataset path:

```bash
/home/elicer/Research/Capstone-Design/indory_lerobot/data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz
```

Dataset was downloaded directly from Hugging Face repo:

```bash
hanbin5/indory_xlerobot_pick_delivery
```

Converted dataset status:

- `codebase_version`: `v2.1`
- episodes: `86`
- frames: `58167`
- fps: `10`
- videos: `258`
- `meta/modality.json`: present
- `meta/relative_stats.json`: missing

`relative_stats.json` must be generated before real GR00T relative-action training.

## Local Machine State

A local N1.7 20k training job is still running unless manually stopped:

```bash
cd /home/hanbin5/Research/Capstone-Design/indory_lerobot
ps -p "$(cat outputs/train/_logs/indory_xlerobot_groot_n17_86ep10hz_20k.pid)" -o pid,stat,etime,cmd
```

Local run details:

- PID: `3053103`
- W&B run id: `olsg7mfn`
- W&B URL: `https://wandb.ai/hanbin5-chung-ang-university/indory_lerobot/runs/olsg7mfn`
- log: `outputs/train/_logs/indory_xlerobot_groot_n17_86ep10hz_20k.log`
- output: `outputs/train/indory_xlerobot_groot_n17_86ep10hz_20k`

Only stop it intentionally:

```bash
kill "$(cat outputs/train/_logs/indory_xlerobot_groot_n17_86ep10hz_20k.pid)"
```

## SSH

From the local machine:

```bash
ssh -i /home/hanbin5/.codex/attachments/26a4c078-dfb2-451a-bb68-c9f24c7c4102/elice-cloud-ondemand-efbfa767-bc2c-4eef-b9ec-1f91fc28add3.pem \
  -p 23550 \
  -o StrictHostKeyChecking=accept-new \
  elicer@central-02.tcp.tunnel.elice.io
```

Do not print or copy the private key contents.

## Remote Environment Setup

Run after SSH login:

```bash
export PATH="$HOME/.local/bin:$PATH"
export BASE="$HOME/Research/Capstone-Design"
export GROOT_DIR="$BASE/Isaac-GR00T-N1.7"
export LEROBOT_DIR="$BASE/indory_lerobot"
export DATASET_PATH="$LEROBOT_DIR/data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz"
export MODALITY_CONFIG_PATH="$LEROBOT_DIR/scripts/gr00t_n17/indory_xlerobot_config.py"
export CUDA_VISIBLE_DEVICES=0
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_DISABLE_XET=1
cd "$GROOT_DIR"
```

Check GPU and environment:

```bash
nvidia-smi
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
PY
```

## Auth Checks

Remote may not have Hugging Face or W&B auth yet.

```bash
cd "$GROOT_DIR"
uv run huggingface-cli whoami || uv run huggingface-cli login
uv run wandb login
```

The Hugging Face account/token needs access to:

- `nvidia/GR00T-N1.7-3B`
- `nvidia/Cosmos-Reason2-2B`
- `hanbin5/indory_xlerobot_pick_delivery` if the dataset is private

## Generate Relative Stats

Run this before smoke or training:

```bash
cd "$GROOT_DIR"
uv run python gr00t/data/stats.py \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG_PATH"
```

Verify:

```bash
test -f "$DATASET_PATH/meta/relative_stats.json" && echo "relative_stats OK"
```

## Dataset Verification

```bash
python3 - <<PY
import json
from pathlib import Path
root = Path("$DATASET_PATH")
info = json.loads((root / "meta/info.json").read_text())
print("dataset", root)
print("version", info.get("codebase_version"))
print("episodes", info.get("total_episodes"))
print("frames", info.get("total_frames"))
print("fps", info.get("fps"))
print("videos", len(list((root / "videos").glob("**/*.mp4"))))
print("modality", (root / "meta/modality.json").exists())
print("relative_stats", (root / "meta/relative_stats.json").exists())
PY
```

Expected:

```text
version v2.1
episodes 86
frames 58167
fps 10
videos 258
modality True
relative_stats True
```

## Preflight

The training script has local `/home/hanbin5/...` defaults, so always pass Elice path overrides.

```bash
GROOT_DIR="$GROOT_DIR" \
DATASET_PATH="$DATASET_PATH" \
MODALITY_CONFIG_PATH="$MODALITY_CONFIG_PATH" \
bash "$LEROBOT_DIR/scripts/gr00t_n17/preflight_indory_n17.sh"
```

If this fails on model access, fix Hugging Face login/token first.

## Fast Smoke Run

This fast smoke does not load all pretrained weights because it uses `--skip-weight-loading`. It checks dataset/config/wiring quickly.

```bash
rm -rf "$LEROBOT_DIR/outputs/train/_smoke_n17_a100_fast"
SKIP_PREFLIGHT=1 \
GROOT_DIR="$GROOT_DIR" \
DATASET_PATH="$DATASET_PATH" \
MODALITY_CONFIG_PATH="$MODALITY_CONFIG_PATH" \
OUTPUT_DIR="$LEROBOT_DIR/outputs/train/_smoke_n17_a100_fast" \
CUDA_VISIBLE_DEVICES=0 \
bash "$LEROBOT_DIR/scripts/gr00t_n17/smoke_indory_n17.sh"
```

## Full One-Step Smoke

Run one real step that loads the pretrained GR00T N1.7 weights. This is the best check before launching 20k.

```bash
rm -rf "$LEROBOT_DIR/outputs/train/_smoke_n17_a100_full"
cd "$GROOT_DIR"
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --num-gpus 1 \
  --output-dir "$LEROBOT_DIR/outputs/train/_smoke_n17_a100_full" \
  --experiment-name indory_xlerobot_groot_n17_a100_full_smoke \
  --wandb-project indory_lerobot \
  --max-steps 1 \
  --save-steps 1 \
  --save-total-limit 1 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --dataloader-num-workers 4 \
  --shard-size 1024 \
  --num-shards-per-epoch 8 \
  --episode-sampling-rate 1.0 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --deepspeed-stage 2 \
  --gradient-checkpointing \
  --logging-steps 1 \
  --no-use-wandb \
  --save-only-model
```

## Recommended A100 20k Training

Goal: checkpoint at 10k and 20k only.

Recommended first A100 attempt:

- `NUM_GPUS=1`
- `DEEPSPEED_STAGE=2`
- `GLOBAL_BATCH_SIZE=4`
- `GRADIENT_ACCUMULATION_STEPS=2`
- `DATALOADER_NUM_WORKERS=8`
- `MAX_STEPS=20000`
- `SAVE_STEPS=10000`
- `SAVE_TOTAL_LIMIT=2`

Start in background:

```bash
mkdir -p "$LEROBOT_DIR/outputs/train/_logs"
LOG="$LEROBOT_DIR/outputs/train/_logs/indory_xlerobot_groot_n17_86ep10hz_20k_a100.log"
PIDFILE="$LEROBOT_DIR/outputs/train/_logs/indory_xlerobot_groot_n17_86ep10hz_20k_a100.pid"

nohup env \
  PATH="$PATH" \
  CUDA_VISIBLE_DEVICES=0 \
  NO_ALBUMENTATIONS_UPDATE=1 \
  HF_HUB_DISABLE_XET=1 \
  WANDB_PROJECT=indory_lerobot \
  WANDB_MODE=online \
  SKIP_PREFLIGHT=1 \
  GROOT_DIR="$GROOT_DIR" \
  DATASET_PATH="$DATASET_PATH" \
  MODALITY_CONFIG_PATH="$MODALITY_CONFIG_PATH" \
  OUTPUT_DIR="$LEROBOT_DIR/outputs/train/indory_xlerobot_groot_n17_86ep10hz_20k_a100" \
  EXPERIMENT_NAME=indory_xlerobot_groot_n17_86ep10hz_20k_a100 \
  NUM_GPUS=1 \
  MASTER_PORT=29517 \
  GLOBAL_BATCH_SIZE=4 \
  GRADIENT_ACCUMULATION_STEPS=2 \
  MAX_STEPS=20000 \
  SAVE_STEPS=10000 \
  SAVE_TOTAL_LIMIT=2 \
  DATALOADER_NUM_WORKERS=8 \
  NUM_SHARDS_PER_EPOCH=128 \
  DEEPSPEED_STAGE=2 \
  bash "$LEROBOT_DIR/scripts/gr00t_n17/train_indory_n17_2gpu.sh" \
  > "$LOG" 2>&1 &

echo $! | tee "$PIDFILE"
echo "$LOG"
```

## Monitor Training

```bash
tail -f "$LEROBOT_DIR/outputs/train/_logs/indory_xlerobot_groot_n17_86ep10hz_20k_a100.log"
```

In another SSH terminal:

```bash
watch -n 10 nvidia-smi
```

Check process:

```bash
ps -p "$(cat "$LEROBOT_DIR/outputs/train/_logs/indory_xlerobot_groot_n17_86ep10hz_20k_a100.pid")" -o pid,stat,etime,cmd
```

Check checkpoints:

```bash
find "$LEROBOT_DIR/outputs/train/indory_xlerobot_groot_n17_86ep10hz_20k_a100" \
  -maxdepth 2 -type d -name "checkpoint-*" | sort
```

## If A100 OOMs

Retry with these safer values:

```bash
GLOBAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
DEEPSPEED_STAGE=3
DATALOADER_NUM_WORKERS=4
```

Keep:

```bash
MAX_STEPS=20000
SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=2
```

## If It Is Too Slow

After a successful full smoke and a few hundred training steps, inspect step time in the log.

Safe speed knobs on A100 80GB / 192Gi RAM:

- Try `DATALOADER_NUM_WORKERS=12` if GPU utilization is low and CPU/RAM are idle.
- Try `GLOBAL_BATCH_SIZE=6` or `8` only after confirming memory headroom in `nvidia-smi`.
- Keep `DEEPSPEED_STAGE=2` if it fits; stage 3 usually saves memory but can be slower.

Do not change checkpoint cadence if the goal is exactly 10k and 20k checkpoints.

## Known Gotchas

- Do not run plain `uv sync` in the remote GR00T repo unless the aarch64 LFS wheel pointers are fixed or required environments are changed.
- `train_indory_n17_2gpu.sh` and `smoke_indory_n17.sh` have `/home/hanbin5/...` defaults or hardcoded preflight paths. On Elice, pass env overrides and set `SKIP_PREFLIGHT=1` after manually running preflight.
- W&B will fail or go offline if `uv run wandb login` has not been run on remote.
- Hugging Face model loading will fail if remote token does not have gated access to NVIDIA GR00T/Cosmos repos.
- `relative_stats.json` is required because N1.7 config uses relative actions.
