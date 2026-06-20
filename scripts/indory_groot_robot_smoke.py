#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, cast

import msgpack
import numpy as np
import torch
import zmq

import lerobot.policies.groot.configuration_groot  # noqa: F401 - register config subclass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.robots.xlerobot.xlerobot_constants import (
    CANONICAL_MOTORS,
    HEAD_MOTORS,
    LEFT_MOTORS,
    RIGHT_MOTORS,
)
from lerobot.robots.xlerobot.xlerobot_leader_kinematics import cap_raw_targets_to_current
from lerobot.utils.robot_utils import precise_sleep

STATE_NAMES = (
    *LEFT_MOTORS,
    *RIGHT_MOTORS,
    *HEAD_MOTORS,
    "x.vel",
    "y.vel",
    "theta.vel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short guarded GR00T smoke inference on Indory XLerobot.")
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Local LeRobot dataset root for feature metadata."
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("INDORY_DATASET_REPO_ID", "capstone-indory/indory_xlerobot_pick_delivery"),
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means predict one step only.")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--max-relative-target", type=float, default=3.0)
    parser.add_argument("--command-lease-ms", type=int, default=300)
    parser.add_argument("--connect-timeout-s", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--send", action="store_true", help="Actually send actions to the robot.")
    parser.add_argument(
        "--allow-base-motion",
        action="store_true",
        help="Allow policy-predicted base velocities. By default base velocity commands are forced to zero.",
    )
    return parser.parse_args()


def observation_frame(raw_obs: dict[str, object]) -> dict[str, np.ndarray]:
    state = np.asarray(
        [float(cast(Any, raw_obs.get(name, 0.0)) or 0.0) for name in STATE_NAMES], dtype=np.float32
    )
    return {
        "observation.state": state,
        "observation.images.head": np.asarray(raw_obs["head"], dtype=np.uint8),
        "observation.images.left_wrist": np.asarray(raw_obs["left_wrist"], dtype=np.uint8),
        "observation.images.right_wrist": np.asarray(raw_obs["right_wrist"], dtype=np.uint8),
    }


def print_action_summary(action: dict[str, float], raw_obs: dict[str, object], *, prefix: str) -> None:
    print(prefix, flush=True)
    for name in STATE_NAMES:
        pred = float(action.get(name, 0.0) or 0.0)
        cur = float(cast(Any, raw_obs.get(name, 0.0)) or 0.0)
        delta = pred - cur
        print(f"  {name:24s} pred={pred:9.3f} current={cur:9.3f} delta={delta:9.3f}", flush=True)


def capped_raw_policy_action(
    robot: XLerobotClient,
    action: dict[str, float],
    *,
    allow_base_motion: bool = False,
) -> dict[str, float]:
    current = robot.command_builder.current_canonical_ticks(robot.robot_id)
    if current is None:
        raise RuntimeError("Cannot send policy action before receiving proprio joint positions.")

    targets = list(current)
    for idx, motor in enumerate(RIGHT_MOTORS):
        targets[idx] = float(action[motor])
    for idx, motor in enumerate(LEFT_MOTORS):
        targets[6 + idx] = float(action[motor])
    for idx, motor in enumerate(HEAD_MOTORS):
        targets[12 + idx] = float(action[motor])

    capped = cap_raw_targets_to_current(
        targets,
        current,
        CANONICAL_MOTORS,
        robot.follower_calibration,
        robot.config.max_relative_target,
        robot.config.leader_action_units,
    )

    sent: dict[str, float] = dict.fromkeys(STATE_NAMES, 0.0)
    for idx, motor in enumerate(RIGHT_MOTORS):
        sent[motor] = float(capped[idx])
    for idx, motor in enumerate(LEFT_MOTORS):
        sent[motor] = float(capped[6 + idx])
    for idx, motor in enumerate(HEAD_MOTORS):
        sent[motor] = float(capped[12 + idx])
    if allow_base_motion:
        sent["x.vel"] = float(action.get("x.vel", 0.0) or 0.0)
        sent["y.vel"] = float(action.get("y.vel", 0.0) or 0.0)
        sent["theta.vel"] = float(action.get("theta.vel", 0.0) or 0.0)
    else:
        sent["x.vel"] = 0.0
        sent["y.vel"] = 0.0
        sent["theta.vel"] = 0.0
    return sent


def send_capped_raw_policy_action(
    robot: XLerobotClient,
    action: dict[str, float],
    *,
    allow_base_motion: bool = False,
) -> dict[str, float]:
    sent = capped_raw_policy_action(robot, action, allow_base_motion=allow_base_motion)
    canonical = [sent[motor] for motor in CANONICAL_MOTORS]
    payload = robot.command_builder._base_payload(
        robot._seq,
        robot.source_id,
        robot.source_role,
        robot.command_lease_ms,
    )
    payload["arm_joint_pos_target"] = canonical
    payload["arm_joint_pos_target_units"] = "feetech_ticks"
    base_cmd = [sent["x.vel"], sent["y.vel"], sent["theta.vel"]]
    if any(abs(value) > 1e-9 for value in base_cmd):
        payload["base_cmd_vel"] = base_cmd
    robot._seq += 1
    robot.zmq_cmd_socket.send(msgpack.packb(payload, use_bin_type=True), flags=zmq.NOBLOCK)
    return sent


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    # Use checkpoint-saved GR00T processor stats. Passing eval/runtime dataset stats here would
    # make real inference depend on the evaluation dataset distribution.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=None,
    )
    policy.eval()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    robot_cfg = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        id="indory_xlerobot_groot_smoke",
        robot_id=args.robot_id,
        source_id="groot060000_smoke",
        source_role="policy_eval",
        max_relative_target=args.max_relative_target,
        command_lease_ms=args.command_lease_ms,
        connect_timeout_s=args.connect_timeout_s,
    )
    robot = XLerobotClient(robot_cfg)

    print(
        f"Connecting to {args.remote_ip}; send={args.send}; fps={args.fps}; "
        f"duration_s={args.duration_s}; max_relative_target={args.max_relative_target}; "
        f"allow_base_motion={args.allow_base_motion}",
        flush=True,
    )
    robot.connect()
    try:
        n_steps = 1 if args.duration_s <= 0 else max(1, int(round(args.duration_s * args.fps)))
        for step in range(n_steps):
            start_t = time.perf_counter()
            raw_obs = robot.get_observation()
            obs = observation_frame(raw_obs)
            obs = prepare_observation_for_inference(
                obs,
                torch.device(cfg.device),
                task=args.task,
                robot_type=robot.robot_type,
            )
            with torch.inference_mode():
                action = postprocessor(policy.select_action(preprocessor(obs)))
            robot_action = make_robot_action(action, ds_meta.features)
            if step == 0 or not args.send:
                print_action_summary(robot_action, raw_obs, prefix=f"step={step} predicted action")
            if args.send:
                sent = send_capped_raw_policy_action(
                    robot,
                    robot_action,
                    allow_base_motion=args.allow_base_motion,
                )
                if step == 0:
                    print_action_summary(sent, raw_obs, prefix=f"step={step} sent/capped action")
            elif step == 0:
                sent = capped_raw_policy_action(
                    robot,
                    robot_action,
                    allow_base_motion=args.allow_base_motion,
                )
                print_action_summary(sent, raw_obs, prefix=f"step={step} would-send/capped action")
            precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - start_t)))
        print("done", flush=True)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
