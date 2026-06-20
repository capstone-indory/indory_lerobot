#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")

import msgpack
import numpy as np
import torch
import zmq

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

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path(
    ROOT
    / "outputs/train/indory_xlerobot_groot_n17_86ep10hz_20k"
    / "indory_xlerobot_groot_n17_86ep10hz_20k/checkpoint-10000"
)
DEFAULT_TASK = "pick up the blue parcel bag and place it in the basket"

LEFT_ARM_MOTORS = LEFT_MOTORS[:5]
LEFT_GRIPPER_MOTORS = LEFT_MOTORS[5:]
RIGHT_ARM_MOTORS = RIGHT_MOTORS[:5]
RIGHT_GRIPPER_MOTORS = RIGHT_MOTORS[5:]
BASE_KEYS = ("x.vel", "y.vel", "theta.vel")
STATE_NAMES = (*LEFT_MOTORS, *RIGHT_MOTORS, *HEAD_MOTORS, *BASE_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded live GR00T N1.7 smoke test on Indory XLerobot.")
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means infer one control step only.")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--max-relative-target", type=float, default=1.0)
    parser.add_argument("--command-lease-ms", type=int, default=200)
    parser.add_argument("--connect-timeout-s", type=int, default=5)
    parser.add_argument("--camera-port", type=int, default=8866)
    parser.add_argument("--camera-port-timeout-s", type=float, default=1.0)
    parser.add_argument("--observation-timeout-s", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--send", action="store_true", help="Actually send capped actions to the robot.")
    parser.add_argument(
        "--allow-missing-cameras",
        action="store_true",
        help="Continue with zero-filled camera frames. By default missing cameras abort before inference.",
    )
    parser.add_argument(
        "--allow-base-motion",
        action="store_true",
        help="Allow policy-predicted base velocities. By default base velocity commands are forced to zero.",
    )
    return parser.parse_args()


def finite_float(value: Any, *, key: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} is not a numeric value: {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{key} is not finite: {out!r}")
    return out


def state_array(raw_obs: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    values = [finite_float(raw_obs.get(name, 0.0) or 0.0, key=name) for name in names]
    return np.asarray(values, dtype=np.float32)[None, None, :]


def image_array(raw_obs: dict[str, Any], name: str) -> np.ndarray:
    image = np.asarray(raw_obs[name])
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} image must be HxWx3 uint8, got shape={image.shape} dtype={image.dtype}")
    return image[None, None, ...]


def gr00t_observation(raw_obs: dict[str, Any], task: str, language_key: str) -> dict[str, Any]:
    return {
        "video": {
            "head": image_array(raw_obs, "head"),
            "left_wrist": image_array(raw_obs, "left_wrist"),
            "right_wrist": image_array(raw_obs, "right_wrist"),
        },
        "state": {
            "left_arm": state_array(raw_obs, LEFT_ARM_MOTORS),
            "left_gripper": state_array(raw_obs, LEFT_GRIPPER_MOTORS),
            "right_arm": state_array(raw_obs, RIGHT_ARM_MOTORS),
            "right_gripper": state_array(raw_obs, RIGHT_GRIPPER_MOTORS),
            "head": state_array(raw_obs, HEAD_MOTORS),
            "base_velocity": state_array(raw_obs, BASE_KEYS),
        },
        "language": {
            language_key: [[task]],
        },
    }


def action_step(action_chunk: dict[str, np.ndarray], key: str, step: int) -> np.ndarray:
    if key not in action_chunk:
        raise KeyError(f"Policy action is missing key {key!r}")
    values = np.asarray(action_chunk[key])
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for {key}, got shape={values.shape}")
        if step >= values.shape[1]:
            raise ValueError(f"Action step {step} is out of range for {key} with shape={values.shape}")
        return values[0, step]
    if values.ndim == 2:
        if step >= values.shape[0]:
            raise ValueError(f"Action step {step} is out of range for {key} with shape={values.shape}")
        return values[step]
    if values.ndim == 1:
        if step != 0:
            raise ValueError(f"Action step {step} is out of range for 1D {key} action")
        return values
    raise ValueError(f"Unsupported action shape for {key}: {values.shape}")


def fill_action_values(
    out: dict[str, float],
    action_chunk: dict[str, np.ndarray],
    key: str,
    names: tuple[str, ...],
    step: int,
) -> None:
    values = action_step(action_chunk, key, step)
    if values.shape[-1] != len(names):
        raise ValueError(f"Action {key} dim {values.shape[-1]} does not match {len(names)} motors")
    for name, value in zip(names, values, strict=True):
        out[name] = finite_float(value, key=f"action.{key}.{name}")


def robot_action_from_chunk(
    action_chunk: dict[str, np.ndarray],
    *,
    step: int = 0,
    allow_base_motion: bool = False,
) -> dict[str, float]:
    action = dict.fromkeys(STATE_NAMES, 0.0)
    fill_action_values(action, action_chunk, "left_arm", LEFT_ARM_MOTORS, step)
    fill_action_values(action, action_chunk, "left_gripper", LEFT_GRIPPER_MOTORS, step)
    fill_action_values(action, action_chunk, "right_arm", RIGHT_ARM_MOTORS, step)
    fill_action_values(action, action_chunk, "right_gripper", RIGHT_GRIPPER_MOTORS, step)
    fill_action_values(action, action_chunk, "head", HEAD_MOTORS, step)
    if allow_base_motion:
        fill_action_values(action, action_chunk, "base_velocity", BASE_KEYS, step)
    return action


def print_action_summary(action: dict[str, float], raw_obs: dict[str, Any], *, prefix: str) -> None:
    print(prefix, flush=True)
    for name in STATE_NAMES:
        pred = float(action.get(name, 0.0) or 0.0)
        cur = float(raw_obs.get(name, 0.0) or 0.0)
        print(f"  {name:24s} target={pred:9.3f} current={cur:9.3f} delta={pred - cur:9.3f}", flush=True)


def capped_raw_policy_action(
    robot: XLerobotClient,
    action: dict[str, float],
    *,
    allow_base_motion: bool = False,
) -> dict[str, float]:
    current = robot.command_builder.current_canonical_ticks(robot.robot_id)
    if current is None:
        raise RuntimeError("Cannot cap policy action before receiving proprio joint positions.")

    targets = list(current)
    for idx, motor in enumerate(RIGHT_MOTORS):
        targets[idx] = action[motor]
    for idx, motor in enumerate(LEFT_MOTORS):
        targets[6 + idx] = action[motor]
    for idx, motor in enumerate(HEAD_MOTORS):
        targets[12 + idx] = action[motor]

    capped = cap_raw_targets_to_current(
        targets,
        current,
        CANONICAL_MOTORS,
        robot.follower_calibration,
        robot.config.max_relative_target,
        robot.config.leader_action_units,
    )

    sent = dict.fromkeys(STATE_NAMES, 0.0)
    for idx, motor in enumerate(RIGHT_MOTORS):
        sent[motor] = float(capped[idx])
    for idx, motor in enumerate(LEFT_MOTORS):
        sent[motor] = float(capped[6 + idx])
    for idx, motor in enumerate(HEAD_MOTORS):
        sent[motor] = float(capped[12 + idx])
    if allow_base_motion:
        for key in BASE_KEYS:
            sent[key] = action[key]
    return sent


def send_capped_raw_policy_action(
    robot: XLerobotClient,
    action: dict[str, float],
    *,
    allow_base_motion: bool = False,
) -> dict[str, float]:
    sent = capped_raw_policy_action(robot, action, allow_base_motion=allow_base_motion)
    payload = robot.command_builder._base_payload(
        robot._seq,
        robot.source_id,
        robot.source_role,
        robot.command_lease_ms,
    )
    payload["arm_joint_pos_target"] = [sent[motor] for motor in CANONICAL_MOTORS]
    payload["arm_joint_pos_target_units"] = "feetech_ticks"
    base_cmd = [sent[key] for key in BASE_KEYS]
    if any(abs(value) > 1e-9 for value in base_cmd):
        payload["base_cmd_vel"] = base_cmd
    robot._seq += 1
    robot.zmq_cmd_socket.send(msgpack.packb(payload, use_bin_type=True), flags=zmq.NOBLOCK)
    return sent


def wait_for_observation(robot: XLerobotClient, timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_obs: dict[str, Any] | None = None
    while True:
        last_obs = robot.get_observation()
        if robot.command_builder.current_canonical_ticks(robot.robot_id) is not None:
            return last_obs
        if time.perf_counter() >= deadline:
            raise RuntimeError("Timed out waiting for proprio joint positions from the robot.")
        precise_sleep(0.05)


def warn_zero_images(raw_obs: dict[str, Any]) -> list[str]:
    zero_names = []
    for name in ("head", "left_wrist", "right_wrist"):
        image = np.asarray(raw_obs.get(name))
        if image.size == 0 or not np.any(image):
            zero_names.append(name)
            print(f"warning: {name} frame is empty/zero-filled", flush=True)
    return zero_names


def ensure_tcp_connectable(remote_ip: str, port: int, timeout_s: float, *, label: str) -> None:
    try:
        with socket.create_connection((remote_ip, int(port)), timeout=max(0.1, float(timeout_s))):
            return
    except OSError as exc:
        raise RuntimeError(
            f"{label} port {remote_ip}:{port} is not reachable ({exc}). "
            "Start the Pi camera publisher before running GR00T policy smoke, "
            "or pass --allow-missing-cameras for control-path debugging only."
        ) from exc


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def close_robot(robot: XLerobotClient, *, send_stop: bool) -> None:
    if not robot.is_connected:
        return
    if send_stop:
        robot.disconnect()
        return
    robot.disconnect_sockets()
    robot._is_connected = False


def main() -> None:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint}")
    if not args.allow_missing_cameras:
        ensure_tcp_connectable(
            args.remote_ip,
            args.camera_port,
            args.camera_port_timeout_s,
            label="Camera",
        )

    robot_cfg = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        id="indory_xlerobot_groot_n17_smoke",
        robot_id=args.robot_id,
        port_zmq_cameras=args.camera_port,
        source_id="groot_n17_smoke",
        source_role="policy_smoke",
        max_relative_target=args.max_relative_target,
        command_lease_ms=args.command_lease_ms,
        connect_timeout_s=args.connect_timeout_s,
    )
    robot = XLerobotClient(robot_cfg)

    print(
        f"Connecting to {args.remote_ip}; fps={args.fps}; duration_s={args.duration_s}; "
        f"max_relative_target={args.max_relative_target}",
        flush=True,
    )
    robot.connect()
    try:
        n_steps = 1 if args.duration_s <= 0 else max(1, int(round(args.duration_s * args.fps)))
        first_raw_obs = wait_for_observation(robot, args.observation_timeout_s)
        zero_images = warn_zero_images(first_raw_obs)
        if zero_images and not args.allow_missing_cameras:
            raise RuntimeError(
                "Missing camera frames: "
                + ", ".join(zero_images)
                + ". Start the Pi camera publisher or pass --allow-missing-cameras for debugging only."
            )

        device = resolve_device(args.device)
        print(
            f"Loading GR00T N1.7 policy from {args.checkpoint}; device={device}; "
            f"send={args.send}; allow_base_motion={args.allow_base_motion}",
            flush=True,
        )
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
            model_path=str(args.checkpoint),
            device=device,
            strict=True,
        )

        for step in range(n_steps):
            start_t = time.perf_counter()
            raw_obs = first_raw_obs if step == 0 else wait_for_observation(robot, args.observation_timeout_s)
            if step > 0:
                zero_images = warn_zero_images(raw_obs)
                if zero_images and not args.allow_missing_cameras:
                    raise RuntimeError(
                        "Missing camera frames: "
                        + ", ".join(zero_images)
                        + ". Start the Pi camera publisher or pass --allow-missing-cameras for debugging only."
                    )
            obs = gr00t_observation(raw_obs, args.task, policy.language_key)
            with torch.inference_mode():
                action_chunk, _ = policy.get_action(obs)
            robot_action = robot_action_from_chunk(
                action_chunk,
                step=0,
                allow_base_motion=args.allow_base_motion,
            )
            print_action_summary(robot_action, raw_obs, prefix=f"step={step} predicted first action")
            if args.send:
                sent = send_capped_raw_policy_action(
                    robot,
                    robot_action,
                    allow_base_motion=args.allow_base_motion,
                )
                print_action_summary(sent, raw_obs, prefix=f"step={step} sent/capped action")
            else:
                sent = capped_raw_policy_action(
                    robot,
                    robot_action,
                    allow_base_motion=args.allow_base_motion,
                )
                print_action_summary(sent, raw_obs, prefix=f"step={step} would-send/capped action")
            precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - start_t)))
        print("done", flush=True)
    finally:
        close_robot(robot, send_stop=args.send)


if __name__ == "__main__":
    main()
