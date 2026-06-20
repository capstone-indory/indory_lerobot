#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
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
from lerobot.utils.constants import ACTION

GROUPS = {
    "left_arm_6d": list(range(0, 6)),
    "right_arm_6d": list(range(6, 12)),
    "head_2d": list(range(12, 14)),
    "base_3d": list(range(14, 17)),
    "nonbase_14d": list(range(0, 14)),
    "all_17d": list(range(17)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline chunk GT evaluation for Indory ACT checkpoints.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/indory_xlerobot_pick_delivery_head_patched"),
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("INDORY_DATASET_REPO_ID", "capstone-indory/indory_xlerobot_pick_delivery"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train/act_indory_100k/checkpoints/080000/pretrained_model"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/eval/act_indory_chunk_gt/latest.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-episode", type=int, default=2)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def select_indices(ds: LeRobotDataset, samples_per_episode: int | None, limit: int | None) -> list[int]:
    if samples_per_episode is None:
        indices = list(range(ds.num_frames))
    else:
        episode_indices = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
        indices = []
        for ep_idx in range(ds.num_episodes):
            ep_frame_indices = np.flatnonzero(episode_indices == ep_idx)
            if ep_frame_indices.size == 0:
                continue
            rel = np.linspace(0, ep_frame_indices.size - 1, samples_per_episode + 2, dtype=np.int64)[1:-1]
            indices.extend(int(ep_frame_indices[i]) for i in rel)
    return indices if limit is None else indices[:limit]


def new_accumulator() -> dict[str, Any]:
    return {"count": 0, "sum_abs": 0.0, "sum_sq": 0.0}


def update_acc(acc: dict[str, Any], abs_err: torch.Tensor, mask: torch.Tensor) -> None:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if abs_err.ndim == mask.ndim + 1:
        mask = mask.unsqueeze(-1).expand_as(abs_err)
    values = abs_err[mask].detach().float().cpu()
    if values.numel() == 0:
        return
    acc["count"] += int(values.numel())
    acc["sum_abs"] += float(values.sum())
    acc["sum_sq"] += float(values.square().sum())


def finalize_acc(acc: dict[str, Any]) -> dict[str, Any]:
    count = max(1, int(acc["count"]))
    return {
        "count": int(acc["count"]),
        "mae": acc["sum_abs"] / count,
        "rmse": (acc["sum_sq"] / count) ** 0.5,
    }


def add_distribution(metrics: dict[str, Any], values: list[torch.Tensor]) -> None:
    if not values:
        metrics.update({"median_abs": None, "p90_abs": None, "max_abs": None})
        return
    vals = torch.cat([v.flatten().detach().float().cpu() for v in values])
    if vals.numel() == 0:
        metrics.update({"median_abs": None, "p90_abs": None, "max_abs": None})
        return
    metrics.update(
        {
            "median_abs": float(vals.median()),
            "p90_abs": float(torch.quantile(vals, 0.90)),
            "max_abs": float(vals.max()),
        }
    )


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if cfg.type != "act":
        raise ValueError(f"Expected an ACT checkpoint, got policy type {cfg.type!r}.")
    cfg.pretrained_path = args.checkpoint
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    probe_ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, video_backend=args.video_backend)
    horizon = min(int(cfg.chunk_size), int(cfg.n_action_steps))
    action_delta_timestamps = [i / probe_ds.fps for i in range(horizon)]
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
        delta_timestamps={ACTION: action_delta_timestamps},
    )
    selected_indices = select_indices(ds, args.samples_per_episode, args.limit)
    if not selected_indices:
        raise RuntimeError("No frames selected for evaluation.")

    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=cfg.pretrained_path)
    policy.eval()

    loader = DataLoader(
        Subset(ds, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model_acc = {name: new_accumulator() for name in GROUPS}
    hold_acc = {name: new_accumulator() for name in GROUPS}
    first_model_acc = {name: new_accumulator() for name in GROUPS}
    first_hold_acc = {name: new_accumulator() for name in GROUPS}
    model_dist: dict[str, list[torch.Tensor]] = defaultdict(list)
    hold_dist: dict[str, list[torch.Tensor]] = defaultdict(list)
    worst_rows: list[dict[str, Any]] = []

    start = time.perf_counter()
    seen = 0
    for batch in loader:
        obs = {k: v for k, v in batch.items() if k.startswith("observation.") or k == "task"}
        with torch.inference_mode():
            proc_obs = pre(obs)
            policy.reset()
            pred_steps = [post(policy.select_action(proc_obs)).detach().cpu().float() for _ in range(horizon)]
        pred = torch.stack(pred_steps, dim=1)

        gt = batch[ACTION][:, :horizon].detach().cpu().float()
        valid = ~batch["action_is_pad"][:, :horizon].detach().cpu().bool()
        state = batch["observation.state"].detach().cpu().float()
        hold = state[:, None, :].expand_as(gt)

        model_abs = (pred - gt).abs()
        hold_abs = (hold - gt).abs()
        first_valid = valid[:, :1]

        for group, cols in GROUPS.items():
            update_acc(model_acc[group], model_abs[:, :, cols], valid)
            update_acc(hold_acc[group], hold_abs[:, :, cols], valid)
            update_acc(first_model_acc[group], model_abs[:, :1, cols], first_valid)
            update_acc(first_hold_acc[group], hold_abs[:, :1, cols], first_valid)
            group_mask = valid.unsqueeze(-1).expand_as(model_abs[:, :, cols])
            model_dist[group].append(model_abs[:, :, cols][group_mask].cpu())
            hold_dist[group].append(hold_abs[:, :, cols][group_mask].cpu())

        nonbase = model_abs[:, :, GROUPS["nonbase_14d"]]
        nonbase_mask = valid.unsqueeze(-1).expand_as(nonbase)
        per_row = nonbase.masked_fill(~nonbase_mask, 0).sum(dim=(1, 2)) / nonbase_mask.sum(
            dim=(1, 2)
        ).clamp_min(1)
        for row_idx, row_mae in enumerate(per_row.tolist()):
            worst_rows.append(
                {
                    "dataset_index": int(batch["index"][row_idx]),
                    "episode_index": int(batch["episode_index"][row_idx]),
                    "frame_index": int(batch["frame_index"][row_idx]),
                    "task": batch["task"][row_idx],
                    "model_chunk_mae_nonbase_14d": float(row_mae),
                }
            )
        worst_rows = sorted(worst_rows, key=lambda row: row["model_chunk_mae_nonbase_14d"], reverse=True)[:20]

        seen += int(gt.shape[0])
        if (
            seen == int(gt.shape[0])
            or seen % args.progress_every < int(gt.shape[0])
            or seen >= len(selected_indices)
        ):
            elapsed = time.perf_counter() - start
            rate = seen / elapsed if elapsed > 0 else 0.0
            print(
                f"eval {seen}/{len(selected_indices)} samples elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
            )

    metrics = {}
    first_step_metrics = {}
    for group in GROUPS:
        model = finalize_acc(model_acc[group])
        hold = finalize_acc(hold_acc[group])
        add_distribution(model, model_dist[group])
        add_distribution(hold, hold_dist[group])
        model["mae_vs_hold_ratio"] = None if hold["mae"] == 0 else model["mae"] / hold["mae"]
        metrics[group] = {"model": model, "hold_current": hold}

        first_model = finalize_acc(first_model_acc[group])
        first_hold = finalize_acc(first_hold_acc[group])
        first_model["mae_vs_hold_ratio"] = (
            None if first_hold["mae"] == 0 else first_model["mae"] / first_hold["mae"]
        )
        first_step_metrics[group] = {"model": first_model, "hold_current": first_hold}

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "dataset_frames": int(ds.num_frames),
        "dataset_episodes": int(ds.num_episodes),
        "dataset_fps": int(ds.fps),
        "selected_samples": int(len(selected_indices)),
        "horizon": int(horizon),
        "batch_size": int(args.batch_size),
        "device": cfg.device,
        "elapsed_sec": float(time.perf_counter() - start),
        "metrics": metrics,
        "first_step_metrics": first_step_metrics,
        "worst_samples_by_model_nonbase_chunk_mae": worst_rows,
    }

    with args.output_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"output_json": str(args.output_json), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
