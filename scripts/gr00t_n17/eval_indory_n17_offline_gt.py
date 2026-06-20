#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")

import numpy as np
import torch
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/train/indory_xlerobot_groot_n17_86ep10hz_20k"
    / "indory_xlerobot_groot_n17_86ep10hz_20k/checkpoint-10000"
)
DEFAULT_OUTPUT_JSON = ROOT / "outputs/eval/indory_n17_offline_gt/latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GT eval for Indory GR00T N1.7 checkpoints.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--episodes",
        default="0:5",
        help="Episode selector, e.g. '0:5', '0,3,7', or 'all'. Default evaluates episodes 0..4.",
    )
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--video-backend", default="auto")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def parse_episodes(spec: str, total: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total))
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid episode range: {spec!r}")
        start = 0 if parts[0] == "" else int(parts[0])
        stop = total if parts[1] == "" else int(parts[1])
        return [idx for idx in range(start, stop) if 0 <= idx < total]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if 0 <= idx < total:
            out.append(idx)
    return out


def sample_steps(length: int, horizon: int, samples_per_episode: int) -> list[int]:
    max_start = length - horizon
    if max_start < 0:
        return []
    if samples_per_episode <= 0:
        return list(range(max_start + 1))
    if samples_per_episode == 1:
        return [max_start // 2]
    return sorted({int(x) for x in np.linspace(0, max_start, samples_per_episode)})


def make_loader(
    dataset_root: Path,
    modality_config: dict[str, Any],
    video_backend: str,
    first_episode: int | None,
) -> LeRobotEpisodeLoader:
    candidates = ["torchcodec", "decord", "ffmpeg", "opencv"] if video_backend == "auto" else [video_backend]
    errors: list[str] = []
    for candidate in candidates:
        try:
            loader = LeRobotEpisodeLoader(
                dataset_path=dataset_root,
                modality_configs=modality_config,
                video_backend=candidate,
                video_backend_kwargs=None,
            )
            if first_episode is not None:
                _ = loader[first_episode].iloc[0]
            print(f"video_backend={candidate}", flush=True)
            return loader
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No video backend could load the dataset:\n" + "\n".join(errors))


def gr00t_observation(step_data: Any, language_key: str) -> dict[str, Any]:
    return {
        "video": {key: np.asarray(value, dtype=np.uint8)[None, :] for key, value in step_data.images.items()},
        "state": {key: np.asarray(value, dtype=np.float32)[None, :] for key, value in step_data.states.items()},
        "language": {language_key: [[step_data.text]]},
    }


def unbatch_action(action_chunk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for key, value in action_chunk.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != 1:
            raise ValueError(f"Expected action {key!r} shape (1,T,D), got {arr.shape}")
        out[key] = arr[0]
    return out


def concat_groups(values: dict[str, np.ndarray], keys: Iterable[str]) -> np.ndarray:
    return np.concatenate([np.atleast_2d(values[key]) for key in keys], axis=-1).astype(np.float32)


def hold_baseline(step_data: Any, action_keys: list[str], horizon: int) -> dict[str, np.ndarray]:
    baseline = {}
    for key in action_keys:
        ref_key = key
        state = np.asarray(step_data.states[ref_key], dtype=np.float32)
        baseline[key] = np.repeat(state[-1: ], horizon, axis=0)
    return baseline


def update_group_acc(acc: dict[str, dict[str, float]], group: str, abs_err: np.ndarray) -> None:
    vals = np.asarray(abs_err, dtype=np.float64).reshape(-1)
    bucket = acc[group]
    bucket["count"] += int(vals.size)
    bucket["sum_abs"] += float(vals.sum())
    bucket["sum_sq"] += float(np.square(vals).sum())
    bucket["max_abs"] = max(bucket["max_abs"], float(vals.max(initial=0.0)))
    bucket["values"].extend(float(v) for v in vals)


def finalize(bucket: dict[str, Any]) -> dict[str, float]:
    count = max(1, int(bucket["count"]))
    vals = np.asarray(bucket["values"], dtype=np.float64)
    return {
        "mae": bucket["sum_abs"] / count,
        "rmse": (bucket["sum_sq"] / count) ** 0.5,
        "median_abs": float(np.median(vals)) if vals.size else 0.0,
        "p90_abs": float(np.quantile(vals, 0.9)) if vals.size else 0.0,
        "max_abs": float(bucket["max_abs"]),
    }


def new_acc() -> dict[str, Any]:
    return {"count": 0, "sum_abs": 0.0, "sum_sq": 0.0, "max_abs": 0.0, "values": []}


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)

    start = time.perf_counter()
    tag = EmbodimentTag.resolve(args.embodiment_tag)
    device = args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(args.checkpoint), device=device, strict=True)
    modality_config = policy.get_modality_config()
    action_keys = list(modality_config["action"].modality_keys)
    action_horizon = len(modality_config["action"].delta_indices)

    probe_loader = LeRobotEpisodeLoader(args.dataset_root, modality_config, video_backend="opencv")
    episodes = parse_episodes(args.episodes, len(probe_loader))
    if not episodes:
        raise RuntimeError("No episodes selected.")
    loader = make_loader(args.dataset_root, modality_config, args.video_backend, episodes[0])

    model_acc: dict[str, dict[str, Any]] = defaultdict(new_acc)
    hold_acc: dict[str, dict[str, Any]] = defaultdict(new_acc)
    first_step_model_acc: dict[str, dict[str, Any]] = defaultdict(new_acc)
    first_step_hold_acc: dict[str, dict[str, Any]] = defaultdict(new_acc)
    worst: list[dict[str, Any]] = []

    selected = 0
    for episode_index in episodes:
        traj = loader[episode_index]
        steps = sample_steps(len(traj), action_horizon, args.samples_per_episode)
        for step_index in steps:
            if args.max_samples is not None and selected >= args.max_samples:
                break
            step_data = extract_step_data(
                traj,
                step_index,
                modality_config,
                tag,
                allow_padding=False,
            )
            obs = gr00t_observation(step_data, policy.language_key)
            with torch.inference_mode():
                pred_chunk, _ = policy.get_action(obs)
            pred = unbatch_action(pred_chunk)
            gt = {key: np.asarray(step_data.actions[key], dtype=np.float32) for key in action_keys}
            hold = hold_baseline(step_data, action_keys, action_horizon)

            group_errors = {}
            for key in action_keys:
                model_abs = np.abs(pred[key] - gt[key])
                hold_abs = np.abs(hold[key] - gt[key])
                update_group_acc(model_acc, key, model_abs)
                update_group_acc(hold_acc, key, hold_abs)
                update_group_acc(first_step_model_acc, key, model_abs[:1])
                update_group_acc(first_step_hold_acc, key, hold_abs[:1])
                group_errors[key] = float(model_abs.mean())

            model_all = concat_groups(pred, action_keys)
            gt_all = concat_groups(gt, action_keys)
            hold_all = concat_groups(hold, action_keys)
            model_abs_all = np.abs(model_all - gt_all)
            hold_abs_all = np.abs(hold_all - gt_all)
            update_group_acc(model_acc, "all", model_abs_all)
            update_group_acc(hold_acc, "all", hold_abs_all)
            update_group_acc(first_step_model_acc, "all", model_abs_all[:1])
            update_group_acc(first_step_hold_acc, "all", hold_abs_all[:1])

            row = {
                "episode_index": int(episode_index),
                "step_index": int(step_index),
                "task": str(step_data.text),
                "model_mae_all": float(model_abs_all.mean()),
                "hold_mae_all": float(hold_abs_all.mean()),
                "group_model_mae": group_errors,
            }
            worst.append(row)
            worst = sorted(worst, key=lambda x: x["model_mae_all"], reverse=True)[:20]

            selected += 1
            if selected == 1 or selected % args.progress_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"eval samples={selected} episode={episode_index} step={step_index} "
                    f"model_mae_all={row['model_mae_all']:.3f} hold_mae_all={row['hold_mae_all']:.3f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
        if args.max_samples is not None and selected >= args.max_samples:
            break

    metrics = {key: finalize(bucket) for key, bucket in sorted(model_acc.items())}
    hold_metrics = {key: finalize(bucket) for key, bucket in sorted(hold_acc.items())}
    first_metrics = {key: finalize(bucket) for key, bucket in sorted(first_step_model_acc.items())}
    first_hold_metrics = {key: finalize(bucket) for key, bucket in sorted(first_step_hold_acc.items())}
    for key, values in metrics.items():
        hold_mae = hold_metrics.get(key, {}).get("mae", 0.0)
        values["mae_vs_hold_ratio"] = None if hold_mae <= 1e-12 else values["mae"] / hold_mae
    for key, values in first_metrics.items():
        hold_mae = first_hold_metrics.get(key, {}).get("mae", 0.0)
        values["mae_vs_hold_ratio"] = None if hold_mae <= 1e-12 else values["mae"] / hold_mae

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "device": device,
        "video_backend": loader.video_backend,
        "episodes": episodes,
        "samples_per_episode": args.samples_per_episode,
        "selected_samples": selected,
        "action_horizon": action_horizon,
        "action_keys": action_keys,
        "elapsed_sec": time.perf_counter() - start,
        "model_metrics": metrics,
        "hold_baseline_metrics": hold_metrics,
        "first_step_model_metrics": first_metrics,
        "first_step_hold_baseline_metrics": first_hold_metrics,
        "worst_samples_by_model_mae_all": worst,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"output_json": str(args.output_json), "selected_samples": selected, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
