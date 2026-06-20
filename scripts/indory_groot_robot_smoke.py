#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import lerobot.policies.groot.configuration_groot  # noqa: F401 - register config subclass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.robots.xlerobot.xlerobot_constants import HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS
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
    parser.add_argument("--camera-transport", choices=("zmq", "rtp_udp", "cam_bridge"), default="zmq")
    parser.add_argument("--cam-bridge-base-url", default="ws://127.0.0.1:8870")
    parser.add_argument("--rtp-udp-bind-ip", default="0.0.0.0")  # nosec B104 - robot LAN UDP receiver.
    parser.add_argument("--rtp-udp-payload-type", type=int, default=96)
    parser.add_argument("--rtp-udp-front-port", type=int, default=5600)
    parser.add_argument("--rtp-udp-wrist-left-port", type=int, default=5602)
    parser.add_argument("--rtp-udp-wrist-right-port", type=int, default=5604)
    parser.add_argument("--rtp-udp-depth-port", type=int, default=5610)
    parser.add_argument("--rtp-udp-ffmpeg-path", default=None)
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
    return robot.capped_canonical_action(action, allow_base_action=allow_base_motion)


def send_capped_raw_policy_action(
    robot: XLerobotClient,
    action: dict[str, float],
    *,
    allow_base_motion: bool = False,
) -> dict[str, float]:
    sent = capped_raw_policy_action(robot, action, allow_base_motion=allow_base_motion)
    return robot.send_canonical_action(sent, include_zero_base=False)


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
        camera_transport=args.camera_transport,
        cam_bridge_base_url=args.cam_bridge_base_url,
        rtp_udp_bind_ip=args.rtp_udp_bind_ip,
        rtp_udp_payload_type=args.rtp_udp_payload_type,
        rtp_udp_front_port=args.rtp_udp_front_port,
        rtp_udp_wrist_left_port=args.rtp_udp_wrist_left_port,
        rtp_udp_wrist_right_port=args.rtp_udp_wrist_right_port,
        rtp_udp_depth_port=args.rtp_udp_depth_port,
        rtp_udp_ffmpeg_path=args.rtp_udp_ffmpeg_path,
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
