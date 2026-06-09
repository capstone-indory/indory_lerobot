#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


GROUPS = {
    "left_arm": list(range(0, 6)),
    "right_arm": list(range(6, 12)),
    "head": list(range(12, 14)),
    "base": list(range(14, 17)),
    "nonbase_14d": list(range(0, 14)),
    "all_17d": list(range(17)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GT action evaluation for Indory GR00T checkpoints.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="hanbin5/indory_xlerobot_pick_delivery")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth frame.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of selected frames.")
    parser.add_argument("--samples-per-episode", type=int, default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--processor-stats",
        choices=("checkpoint", "eval_dataset"),
        default="checkpoint",
        help="Use checkpoint-saved stats for deployment-like inference, or eval dataset stats.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def select_indices(ds: LeRobotDataset, stride: int, limit: int | None, samples_per_episode: int | None) -> list[int]:
    if samples_per_episode is None:
        indices = list(range(0, ds.num_frames, stride))
    else:
        episode_indices = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
        indices = []
        for ep_idx in range(ds.num_episodes):
            ep_frame_indices = np.flatnonzero(episode_indices == ep_idx)
            if ep_frame_indices.size == 0:
                continue
            rel = np.linspace(0, ep_frame_indices.size - 1, samples_per_episode + 2, dtype=np.int64)[1:-1]
            indices.extend(int(ep_frame_indices[i]) for i in rel)
    if limit is not None:
        indices = indices[:limit]
    return indices


def new_accumulator() -> dict[str, Any]:
    return {"count": 0, "sum_abs": 0.0, "sum_sq": 0.0}


def update_acc(acc: dict[str, Any], values: torch.Tensor) -> None:
    values = values.detach().float().cpu()
    acc["count"] += int(values.numel())
    acc["sum_abs"] += float(values.sum())
    acc["sum_sq"] += float(values.square().sum())


def finalize_acc(acc: dict[str, Any]) -> dict[str, float]:
    count = max(1, int(acc["count"]))
    return {
        "mae": acc["sum_abs"] / count,
        "rmse": (acc["sum_sq"] / count) ** 0.5,
    }


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, video_backend=args.video_backend)
    selected_indices = select_indices(ds, args.stride, args.limit, args.samples_per_episode)
    if not selected_indices:
        raise RuntimeError("No frames selected for evaluation.")

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    dataset_stats = ds.meta.stats if args.processor_stats == "eval_dataset" else None
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=dataset_stats,
    )
    policy.eval()

    loader = DataLoader(
        Subset(ds, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    action_names = list(ds.meta.features["action"]["names"])
    action_min = torch.as_tensor(ds.meta.stats["action"]["min"]).float()
    action_max = torch.as_tensor(ds.meta.stats["action"]["max"]).float()
    action_range = action_max - action_min

    group_acc = {name: new_accumulator() for name in GROUPS}
    axis_acc = [new_accumulator() for _ in action_names]
    task_acc: dict[str, dict[str, Any]] = defaultdict(new_accumulator)
    episode_acc: dict[str, dict[str, Any]] = defaultdict(new_accumulator)
    abs_errors: list[torch.Tensor] = []
    worst_rows: list[dict[str, Any]] = []

    seen = 0
    for batch in loader:
        obs = {k: v for k, v in batch.items() if k.startswith("observation.") or k == "task"}
        with torch.inference_mode():
            policy.reset()
            pred = post(policy.select_action(pre(obs)))

        gt = batch["action"].detach().cpu().float()
        pred = pred.detach().cpu().float()
        abs_err = (pred - gt).abs()
        sq_err = (pred - gt).square()
        abs_errors.append(abs_err)

        for group, cols in GROUPS.items():
            update_acc(group_acc[group], abs_err[:, cols])
        for col in range(len(action_names)):
            update_acc(axis_acc[col], abs_err[:, col])

        nonbase_mae = abs_err[:, GROUPS["nonbase_14d"]].mean(dim=1)
        tasks = batch["task"]
        episode_indices = batch["episode_index"].detach().cpu().tolist()
        frame_indices = batch["frame_index"].detach().cpu().tolist()
        dataset_indices = batch["index"].detach().cpu().tolist()
        for row_idx, row_mae in enumerate(nonbase_mae.tolist()):
            task = tasks[row_idx]
            update_acc(task_acc[task], abs_err[row_idx, GROUPS["nonbase_14d"]])
            update_acc(episode_acc[str(int(episode_indices[row_idx]))], abs_err[row_idx, GROUPS["nonbase_14d"]])
            worst_rows.append(
                {
                    "dataset_index": int(dataset_indices[row_idx]),
                    "episode_index": int(episode_indices[row_idx]),
                    "frame_index": int(frame_indices[row_idx]),
                    "task": task,
                    "mae_nonbase_14d": float(row_mae),
                    "mae_all_17d": float(abs_err[row_idx].mean()),
                }
            )
        worst_rows = sorted(worst_rows, key=lambda row: row["mae_nonbase_14d"], reverse=True)[:20]

        seen += int(gt.shape[0])
        if seen == int(gt.shape[0]) or seen % args.progress_every < int(gt.shape[0]) or seen >= len(selected_indices):
            elapsed = time.perf_counter() - start
            fps = seen / elapsed if elapsed > 0 else 0.0
            print(
                f"eval {seen}/{len(selected_indices)} frames "
                f"elapsed={elapsed:.1f}s rate={fps:.2f} frame/s",
                flush=True,
            )

    all_abs = torch.cat(abs_errors, dim=0)
    metrics = {name: finalize_acc(acc) for name, acc in group_acc.items()}
    for name, cols in GROUPS.items():
        vals = all_abs[:, cols].flatten()
        metrics[name].update(
            {
                "median_abs": float(vals.median()),
                "p90_abs": float(torch.quantile(vals, 0.90)),
                "max_abs": float(vals.max()),
            }
        )
        norm_vals = []
        for col in cols:
            rng = float(action_range[col])
            if rng > 1e-9:
                norm_vals.append(float(all_abs[:, col].mean()) / rng)
        metrics[name]["norm_mae_frac_of_eval_range"] = None if not norm_vals else float(np.mean(norm_vals))

    per_axis = []
    for col, name in enumerate(action_names):
        vals = all_abs[:, col]
        axis = finalize_acc(axis_acc[col])
        rng = float(action_range[col])
        axis.update(
            {
                "index": col,
                "name": name,
                "median_abs": float(vals.median()),
                "p90_abs": float(torch.quantile(vals, 0.90)),
                "max_abs": float(vals.max()),
                "range": rng,
                "norm_mae_frac_of_eval_range": None if rng <= 1e-9 else axis["mae"] / rng,
            }
        )
        per_axis.append(axis)

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "dataset_frames": int(ds.num_frames),
        "dataset_episodes": int(ds.num_episodes),
        "selected_frames": int(len(selected_indices)),
        "stride": args.stride,
        "limit": args.limit,
        "samples_per_episode": args.samples_per_episode,
        "batch_size": args.batch_size,
        "processor_stats": args.processor_stats,
        "device": cfg.device,
        "elapsed_sec": float(time.perf_counter() - start),
        "metrics": metrics,
        "per_axis": per_axis,
        "per_task_nonbase_14d": {task: finalize_acc(acc) for task, acc in sorted(task_acc.items())},
        "per_episode_nonbase_14d": {
            ep: finalize_acc(acc) for ep, acc in sorted(episode_acc.items(), key=lambda kv: int(kv[0]))
        },
        "worst_samples_by_nonbase_mae": worst_rows,
    }

    with args.output_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"output_json": str(args.output_json), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
