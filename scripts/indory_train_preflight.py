#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

BASE_NAMES = ("x.vel", "y.vel", "theta.vel")
HEAD_NAMES = ("head_motor_1", "head_motor_2")
SO101_PI05 = "CoRL2026-CSI/pi05_teleop_fold_towel"
SO101_GROOT = "CoRL2026-CSI/Gr00t_n1.5-IsaacLab-SO101-Multi_Task-30fps_8epoch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for Indory XLerobot Pi/GR00T training.")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("INDORY_DATASET_REPO_ID", "capstone-indory/indory_xlerobot_pick_delivery"),
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--policy-type", choices=("pi05", "pi0", "groot"), default="pi05")
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument(
        "--scan-actions", action="store_true", help="Scan parquet action columns for zero labels."
    )
    return parser.parse_args()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def read_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    return json.loads(info_path.read_text())


def check_gpu() -> None:
    print("GPU")
    print(f"  cuda_available: {torch.cuda.is_available()}")
    print(f"  cuda_device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  bf16_supported: {torch.cuda.is_bf16_supported()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_gib = props.total_memory / 1024**3
            print(f"  gpu{index}: {props.name}, {total_gib:.1f} GiB, cc={props.major}.{props.minor}")


def check_deps(policy_type: str) -> None:
    deps = ["accelerate", "transformers", "peft", "safetensors", "av", "torchvision"]
    if policy_type == "groot":
        deps.extend(["flash_attn", "decord"])
    print("Dependencies")
    for dep in deps:
        print(f"  {dep}: {module_available(dep)}")


def check_dataset(root: Path, *, scan_actions: bool) -> None:
    info = read_info(root)
    features = info["features"]
    state = features["observation.state"]
    action = features["action"]
    camera_keys = [key for key in features if key.startswith("observation.images.")]

    print("Dataset")
    print(f"  root: {root}")
    print(f"  robot_type: {info.get('robot_type')}")
    print(f"  fps: {info.get('fps')}")
    print(f"  frames: {info.get('total_frames')}")
    print(f"  episodes: {info.get('total_episodes')}")
    print(f"  state_dim: {state['shape'][0]} names={state.get('names')}")
    print(f"  action_dim: {action['shape'][0]} names={action.get('names')}")
    print(f"  cameras: {camera_keys}")

    expected_cameras = {
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    missing_cameras = expected_cameras.difference(camera_keys)
    if missing_cameras:
        print(f"  warning: missing expected XLerobot cameras: {sorted(missing_cameras)}")

    if scan_actions:
        scan_action_columns(root, action["names"])


def scan_action_columns(root: Path, action_names: list[str]) -> None:
    action_indices = {name: action_names.index(name) for name in (*HEAD_NAMES, *BASE_NAMES)}
    values: list[np.ndarray] = []
    for path in sorted((root / "data").glob("*/*.parquet")):
        df = pq.read_table(path, columns=["action"]).to_pandas()
        actions = np.stack(df["action"].to_numpy())
        values.append(actions[:, list(action_indices.values())])
    if not values:
        print("  warning: no parquet action files found")
        return

    arr = np.concatenate(values)
    print("Action label scan")
    for index, name in enumerate(action_indices):
        col = arr[:, index]
        nonzero = int(np.count_nonzero(np.abs(col) > 1e-9))
        print(
            f"  {name}: min={float(col.min()):.6g}, max={float(col.max()):.6g}, "
            f"std={float(col.std()):.6g}, nonzero={nonzero}/{len(col)}"
        )


def check_strategy(policy_type: str, pretrained_path: str | None) -> None:
    if pretrained_path is None:
        pretrained_path = {
            "pi05": SO101_PI05,
            "pi0": "lerobot/pi0_base",
            "groot": SO101_GROOT,
        }[policy_type]

    print("Training Strategy")
    print(f"  policy_type: {policy_type}")
    print(f"  pretrained_path: {pretrained_path}")
    if policy_type in {"pi0", "pi05"}:
        print("  adaptation: load compatible pretrained weights; train XLerobot-shaped state/action heads")
        print("  precision: bf16 + accelerate DDP recommended on 3090/3090Ti")
        print("  memory_default: train_expert_only=true, gradient_checkpointing=true, batch_size=1/GPU")
    elif policy_type == "groot":
        print("  adaptation: GR00T max_state_dim/max_action_dim padding supports XLerobot 17D action")
        print("  requirement: flash_attn must import before GR00T training")


def main() -> None:
    args = parse_args()
    root = args.root
    if root is None:
        root = Path("data") / args.repo_id.split("/", 1)[-1]
    root = root.expanduser()

    check_gpu()
    check_deps(args.policy_type)
    check_dataset(root, scan_actions=args.scan_actions)
    check_strategy(args.policy_type, args.pretrained_path)


if __name__ == "__main__":
    main()
