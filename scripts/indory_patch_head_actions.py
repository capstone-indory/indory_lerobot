#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

STATS_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch an Indory XLeRobot LeRobot dataset so action head_motor_1/2 "
            "use pseudo-labels derived from observation.state."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("INDORY_DATASET_REPO_ID", "capstone-indory/indory_xlerobot_pick_delivery"),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--root", type=Path, default=None, help="Existing local dataset root to patch.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination root. If --root is omitted, the dataset is downloaded here and patched.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Patch --root in place. Without this, --root is copied to --output-root first.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("next-observation", "current-observation"),
        default="next-observation",
        help="Pseudo-label source for head action slots.",
    )
    parser.add_argument(
        "--include-camera-archives",
        action="store_true",
        help="Download camera_archives/ when --root is omitted. Archives are not needed for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_root(args)
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing {stats_path}")

    info = json.loads(info_path.read_text())
    action_names = info["features"]["action"]["names"]
    state_names = info["features"]["observation.state"]["names"]
    action_head_indices = [action_names.index("head_motor_1"), action_names.index("head_motor_2")]
    state_head_indices = [state_names.index("head_motor_1"), state_names.index("head_motor_2")]

    patched_actions_by_episode: dict[int, np.ndarray] = {}
    patched_rows = 0
    for data_path in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pq.read_table(data_path).to_pandas()
        actions = np.stack([np.asarray(row, dtype=np.float32) for row in df["action"]])
        states = np.stack([np.asarray(row, dtype=np.float32) for row in df["observation.state"]])
        episodes = df["episode_index"].to_numpy()

        for episode_index in sorted({int(ep) for ep in episodes}):
            row_indices = np.flatnonzero(episodes == episode_index)
            head_labels = states[row_indices][:, state_head_indices].copy()
            if args.label_mode == "next-observation" and len(row_indices) > 1:
                head_labels[:-1] = head_labels[1:]
            actions[row_indices[:, None], action_head_indices] = head_labels
            patched_actions_by_episode[episode_index] = actions[row_indices].copy()

        df["action"] = [np.asarray(row, dtype=np.float32) for row in actions]
        write_parquet(df, data_path)
        patched_rows += len(df)

    if not patched_actions_by_episode:
        raise RuntimeError(f"No data parquet files found under {root / 'data'}")

    patch_global_action_stats(stats_path, patched_actions_by_episode)
    patch_episode_action_stats(root, patched_actions_by_episode)
    print(f"Patched {patched_rows} rows in {root}")


def resolve_root(args: argparse.Namespace) -> Path:
    if args.root is None:
        if args.output_root is None:
            raise ValueError("Set --output-root when --root is not provided.")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output_root,
            ignore_patterns=None if args.include_camera_archives else "camera_archives/",
        )
        return args.output_root

    if args.in_place:
        return args.root

    if args.output_root is None:
        raise ValueError("Set --output-root, or pass --in-place to patch --root directly.")
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    shutil.copytree(args.root, args.output_root)
    return args.output_root


def patch_global_action_stats(
    stats_path: Path,
    patched_actions_by_episode: dict[int, np.ndarray],
) -> None:
    stats = json.loads(stats_path.read_text())
    all_actions = np.concatenate([patched_actions_by_episode[k] for k in sorted(patched_actions_by_episode)])
    stats["action"] = vector_stats(all_actions)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")


def patch_episode_action_stats(
    root: Path,
    patched_actions_by_episode: dict[int, np.ndarray],
) -> None:
    episode_stats = {
        episode_index: vector_stats(actions)
        for episode_index, actions in patched_actions_by_episode.items()
    }
    for meta_path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        df = pq.read_table(meta_path).to_pandas()
        if "episode_index" not in df:
            continue
        for key in STATS_KEYS:
            column = f"stats/action/{key}"
            if column in df.columns:
                df[column] = [episode_stats[int(ep)][key] for ep in df["episode_index"]]
        write_parquet(df, meta_path)


def vector_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    out = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for key, quantile in QUANTILES.items():
        out[key] = np.quantile(values, quantile, axis=0).tolist()
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


if __name__ == "__main__":
    main()
