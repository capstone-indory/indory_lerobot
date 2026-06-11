#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import lerobot.policies  # noqa: F401 - register policy config subclasses, including ACT.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.indory.arm_recenter import (
    ARM_RECENTER_MOTORS,
    GRIPPER_MOTORS,
    apply_arm_recenter_targets,
    closed_grippers,
    moderate_gripper_open_targets,
    parse_arm_recenter_targets,
    target_position_error,
    targets_within_tolerance,
)
from lerobot.indory.live_debug_view import LiveDebugWebView, cli_debug_summary
from lerobot.indory.success_detector import make_success_detector, parse_detector_kwargs
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
JOINT_NAMES = (*LEFT_MOTORS, *RIGHT_MOTORS, *HEAD_MOTORS)
RECENTER_SETTLE_MOTORS = (*ARM_RECENTER_MOTORS, *HEAD_MOTORS)
RECENTER_GRIPPER_SETTLE_MOTORS = (*GRIPPER_MOTORS, *HEAD_MOTORS)


class GripperLatch:
    """Keep grippers firmly squeezed once the policy closes them.

    Demo leader targets press only ~5-15 ticks past the follower contact point,
    far below what the policy reproduces reliably, so raw policy outputs drift
    a few ticks open and drop the parcel. While latched, the sent target is
    clamped to a firm squeeze; the latch releases only after several
    consecutive clear-open commands so the basket drop still works.
    """

    def __init__(self, *, close_below: float, squeeze: float, open_above: float, open_steps: int):
        self.close_below = float(close_below)
        self.squeeze = float(squeeze)
        self.open_above = float(open_above)
        self.open_steps = max(1, int(open_steps))
        self._latched: dict[str, bool] = dict.fromkeys(GRIPPER_MOTORS, False)
        self._open_counts: dict[str, int] = dict.fromkeys(GRIPPER_MOTORS, 0)

    def apply(self, action: dict[str, float], *, step: int) -> dict[str, float]:
        adjusted = dict(action)
        for motor in GRIPPER_MOTORS:
            if motor not in adjusted:
                continue
            target = float(adjusted[motor])
            if not self._latched[motor]:
                if target < self.close_below:
                    self._latched[motor] = True
                    self._open_counts[motor] = 0
                    adjusted[motor] = min(target, self.squeeze)
                    print(
                        f"gripper latch engaged {motor}: target={target:.0f} -> {adjusted[motor]:.0f} "
                        f"(step={step + 1})",
                        flush=True,
                    )
                continue
            if target > self.open_above:
                self._open_counts[motor] += 1
                if self._open_counts[motor] >= self.open_steps:
                    self._latched[motor] = False
                    self._open_counts[motor] = 0
                    print(f"gripper latch released {motor}: target={target:.0f} (step={step + 1})", flush=True)
                    continue
            else:
                self._open_counts[motor] = 0
            adjusted[motor] = min(target, self.squeeze)
        return adjusted

    def latched_squeeze_targets(self) -> dict[str, float]:
        return {motor: self.squeeze for motor, latched in self._latched.items() if latched}

    def summary(self) -> str:
        flags = ",".join(
            f"{'L' if motor.startswith('left') else 'R'}:{'on' if self._latched[motor] else 'off'}"
            for motor in GRIPPER_MOTORS
        )
        return f"latch=({flags})"


class StaleCameraGuard:
    """Detect a frozen camera stream (e.g. cam bridge JPEG decode stall).

    The stream pumps keep serving the last received frame forever, so a dead
    camera silently feeds the policy a frozen image while proprio keeps
    moving. Live sensors never produce byte-identical consecutive frames, so
    repeated identical frames mean the stream stopped updating.
    """

    def __init__(
        self,
        *,
        warn_steps: int,
        abort_steps: int,
        names: tuple[str, ...] = ("head", "left_wrist", "right_wrist"),
    ):
        self.warn_steps = max(0, int(warn_steps))
        self.abort_steps = max(0, int(abort_steps))
        self.names = names
        self._last: dict[str, np.ndarray] = {}
        self._repeats: dict[str, int] = dict.fromkeys(names, 0)

    def check(self, raw_obs: dict[str, object], *, step: int) -> None:
        for name in self.names:
            frame = raw_obs.get(name)
            if not isinstance(frame, np.ndarray):
                continue
            prev = self._last.get(name)
            self._last[name] = frame
            if prev is None or prev.shape != frame.shape or not np.array_equal(prev, frame):
                self._repeats[name] = 0
                continue
            self._repeats[name] += 1
            repeats = self._repeats[name]
            if self.abort_steps and repeats >= self.abort_steps:
                raise RuntimeError(
                    f"{name} camera stream appears frozen: identical frame for {repeats} consecutive "
                    "policy steps; check the cam bridge / camera pipeline before rerunning"
                )
            if self.warn_steps and repeats % self.warn_steps == 0:
                print(
                    f"WARNING: {name} camera frame unchanged for {repeats} consecutive steps (step={step + 1})",
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded ACT live runner for Indory XLerobot.")
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="hanbin5/indory_xlerobot_pick_delivery")
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=10.0,
        help=(
            "Per-command motion cap in XLerobot leader units. With the default degrees mode, "
            "10.0 means roughly 113.75 Feetech ticks for arm/head joints and 10%% of gripper range."
        ),
    )
    parser.add_argument(
        "--head-pan-sign",
        type=float,
        default=1.0,
        help="Sign applied when converting absolute head_motor_1 tick targets to relative head pan commands.",
    )
    parser.add_argument(
        "--head-tilt-sign",
        type=float,
        default=-1.0,
        help="Sign applied when converting absolute head_motor_2 tick targets to relative head tilt commands.",
    )
    parser.add_argument("--command-lease-ms", type=int, default=300)
    parser.add_argument("--connect-timeout-s", type=int, default=5)
    parser.add_argument("--camera-transport", choices=("zmq", "rtp_udp", "cam_bridge"), default="zmq")
    parser.add_argument("--cam-bridge-base-url", default="ws://127.0.0.1:8870")
    parser.add_argument("--cam-bridge-resize-mode", choices=("center_crop", "stretch"), default="center_crop")
    parser.add_argument("--rtp-udp-bind-ip", default="0.0.0.0")
    parser.add_argument("--rtp-udp-payload-type", type=int, default=96)
    parser.add_argument("--rtp-udp-front-port", type=int, default=5600)
    parser.add_argument("--rtp-udp-wrist-left-port", type=int, default=5602)
    parser.add_argument("--rtp-udp-wrist-right-port", type=int, default=5604)
    parser.add_argument("--rtp-udp-depth-port", type=int, default=5610)
    parser.add_argument("--rtp-udp-ffmpeg-path", default=None)
    parser.add_argument("--print-head-rgb", action="store_true", help="Print received head RGB frame summaries.")
    parser.add_argument(
        "--print-head-rgb-every",
        type=int,
        default=20,
        help="Print one head RGB summary every N policy steps when --print-head-rgb is set.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-id", default="act_live")
    parser.add_argument("--source-role", default="policy_eval")
    parser.add_argument("--send", action="store_true", help="Actually send actions to the robot.")
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument(
        "--no-arm-recenter",
        action="store_true",
        help="Skip the initial arm recenter move before policy inference.",
    )
    parser.add_argument(
        "--arm-recenter-s",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for the initial arm recenter target before policy inference.",
    )
    parser.add_argument(
        "--arm-recenter-max-relative-target",
        type=float,
        default=5.0,
        help=(
            "Per-step cap used only during the initial arm recenter. "
            "With default degrees mode, 5.0 is roughly 56.9 Feetech ticks per 10 Hz step."
        ),
    )
    parser.add_argument(
        "--arm-recenter-targets",
        default=None,
        help="Comma-separated motor=value overrides for the default Indory arm recenter pose.",
    )
    parser.add_argument(
        "--no-arm-recenter-gripper-cycle",
        action="store_true",
        help="Do not open closed grippers before recentering arms and closing them afterward.",
    )
    parser.add_argument(
        "--arm-recenter-gripper-s",
        type=float,
        default=10.0,
        help="Maximum seconds to wait for each gripper phase during the initial arm recenter.",
    )
    parser.add_argument(
        "--arm-recenter-gripper-max-relative-target",
        type=float,
        default=10.0,
        help=(
            "Per-step cap used only while opening/closing grippers during the initial recenter. "
            "The arm/head recenter phase still uses --arm-recenter-max-relative-target."
        ),
    )
    parser.add_argument(
        "--arm-recenter-tolerance-ticks",
        type=float,
        default=25.0,
        help="Max raw tick error allowed before initial recenter is considered reached.",
    )
    parser.add_argument(
        "--arm-recenter-settle-steps",
        type=int,
        default=3,
        help="Consecutive in-tolerance observation cycles required before policy inference starts.",
    )
    parser.add_argument("--arm-recenter-gripper-open-ratio", type=float, default=0.35)
    parser.add_argument("--arm-recenter-gripper-closed-ratio", type=float, default=0.25)
    parser.add_argument(
        "--no-require-camera-frames",
        action="store_true",
        help="Do not wait for non-empty head/left_wrist/right_wrist frames before running.",
    )
    parser.add_argument(
        "--allow-base-action",
        action="store_true",
        help="Forward policy base velocity outputs. Default keeps x/y/theta velocity at zero.",
    )
    parser.add_argument(
        "--success-detector",
        default="none",
        help=(
            "none, manual-file, parcel-grasp-yolo, sky-blue-parcel-yolo, sky-blue-parcel-color, "
            "or a Python hook formatted as module:attribute."
        ),
    )
    parser.add_argument("--success-detector-kwargs", default=None, help="JSON object passed to a Python hook.")
    parser.add_argument("--success-file", type=Path, default=None, help="File watched by manual-file detector.")
    parser.add_argument(
        "--success-head-targets",
        default=None,
        help=(
            "Head target as 'head_motor_1,head_motor_2' or 'head_motor_1=...,head_motor_2=...'. "
            "Used as the policy override target only with --lock-head-for-success."
        ),
    )
    parser.add_argument(
        "--require-success-head-targets",
        action="store_true",
        help="Require observed head motors to match --success-head-targets before detector success is accepted.",
    )
    parser.add_argument(
        "--success-head-ranges",
        default=None,
        help=(
            "Only accept detector success while observed head motors are inside ranges, "
            "formatted as 'head_motor_1=min:max,head_motor_2=min:max'."
        ),
    )
    parser.add_argument(
        "--lock-head-for-success",
        action="store_true",
        help="Override policy head outputs with detector inspection head targets during the policy run.",
    )
    parser.add_argument(
        "--no-lock-head-for-success",
        dest="lock_head_for_success",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--success-min-steps",
        type=int,
        default=0,
        help="Ignore detector success until at least this many policy loop steps have run.",
    )
    parser.add_argument(
        "--success-min-elapsed-s",
        type=float,
        default=0.0,
        help="Ignore detector success until this many seconds have elapsed after inference starts.",
    )
    parser.add_argument("--no-stop-on-success", action="store_true")
    parser.add_argument("--no-hold-on-success", action="store_true")
    parser.add_argument(
        "--no-success-recovery",
        action="store_true",
        help="Skip release-gripper and return-home sequence after success.",
    )
    parser.add_argument("--release-s", type=float, default=0.7)
    parser.add_argument("--return-s", type=float, default=2.0)
    parser.add_argument(
        "--release-gripper-target",
        type=float,
        default=None,
        help="Raw Feetech tick target used to open both grippers. Default uses calibration range_max.",
    )
    parser.add_argument(
        "--gripper-latch",
        action="store_true",
        help="Clamp gripper targets to a firm squeeze once the policy closes, until a sustained open command.",
    )
    parser.add_argument(
        "--gripper-latch-close-below",
        type=float,
        default=2300.0,
        help=(
            "Raw tick threshold; policy gripper targets below this engage the latch. "
            "Demo hold targets reach ~2220 on the right arm, so keep this above that "
            "but below --gripper-latch-open-above."
        ),
    )
    parser.add_argument(
        "--gripper-latch-squeeze",
        type=float,
        default=2025.0,
        help="Raw tick target sent while latched. Lower presses harder; dataset leader minimum is ~2023.",
    )
    parser.add_argument(
        "--gripper-latch-open-above",
        type=float,
        default=2350.0,
        help="Raw tick threshold; policy gripper targets above this count toward releasing the latch.",
    )
    parser.add_argument(
        "--gripper-latch-open-steps",
        type=int,
        default=2,
        help="Consecutive steps above --gripper-latch-open-above required to release the latch.",
    )
    parser.add_argument(
        "--stale-camera-warn-steps",
        type=int,
        default=3,
        help="Warn when a camera frame is byte-identical for this many consecutive policy steps. 0 disables.",
    )
    parser.add_argument(
        "--stale-camera-abort-steps",
        type=int,
        default=20,
        help="Abort the run when a camera frame is byte-identical for this many consecutive policy steps. 0 disables.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--debug-web-view",
        action="store_true",
        help="Serve a local browser view with the active CLI config, head RGB overlay, and success metadata.",
    )
    parser.add_argument("--debug-web-host", default="0.0.0.0")
    parser.add_argument("--debug-web-port", type=int, default=8890)
    parser.add_argument("--debug-web-open", action="store_true", help="Open the debug web view in a browser.")
    parser.add_argument(
        "--debug-web-keepalive-s",
        type=float,
        default=30.0,
        help="Keep the final debug web frame available for this many seconds after the runner exits.",
    )
    return parser.parse_args()


def parse_head_targets(raw: str | None) -> dict[str, float] | None:
    if raw is None or raw.strip() == "":
        return None
    raw = raw.strip()
    if "=" not in raw:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
        if len(values) != 2:
            raise ValueError("--success-head-targets without names must be 'head_motor_1,head_motor_2'")
        return {"head_motor_1": values[0], "head_motor_2": values[1]}
    targets: dict[str, float] = {}
    for part in raw.split(","):
        name, sep, value = part.partition("=")
        if not sep:
            raise ValueError("--success-head-targets entries must be name=value")
        name = name.strip()
        if name not in HEAD_MOTORS:
            raise ValueError(f"unsupported head target {name!r}; expected one of {HEAD_MOTORS}")
        targets[name] = float(value)
    return targets


def parse_head_ranges(raw: str | None) -> dict[str, tuple[float, float]] | None:
    if raw is None or raw.strip() == "":
        return None
    ranges: dict[str, tuple[float, float]] = {}
    for part in raw.split(","):
        name, sep, value = part.partition("=")
        if not sep:
            raise ValueError("--success-head-ranges entries must be name=min:max")
        name = name.strip()
        if name not in HEAD_MOTORS:
            raise ValueError(f"unsupported head range {name!r}; expected one of {HEAD_MOTORS}")
        low_raw, sep, high_raw = value.strip().partition(":")
        if not sep:
            raise ValueError("--success-head-ranges values must be min:max")
        low, high = float(low_raw), float(high_raw)
        ranges[name] = (min(low, high), max(low, high))
    return ranges or None


def detector_kwargs(
    args: argparse.Namespace,
    *,
    success_head_targets: dict[str, float] | None,
    success_head_ranges: dict[str, tuple[float, float]] | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = parse_detector_kwargs(args.success_detector_kwargs)
    if args.require_success_head_targets and success_head_targets is not None and "head_targets" not in kwargs:
        kwargs["head_targets"] = success_head_targets
    if success_head_ranges is not None and "head_ranges" not in kwargs:
        kwargs["head_ranges"] = success_head_ranges
    return kwargs


def success_gate_reason(args: argparse.Namespace, *, step: int, elapsed_s: float) -> str | None:
    current_step = step + 1
    if args.success_min_steps > 0 and current_step < args.success_min_steps:
        return f"success_min_steps {current_step}/{args.success_min_steps}"
    if args.success_min_elapsed_s > 0 and elapsed_s < args.success_min_elapsed_s:
        return f"success_min_elapsed_s {elapsed_s:.2f}/{args.success_min_elapsed_s:.2f}"
    return None


def detector_inspection_head_targets(detector: object) -> dict[str, float] | None:
    method = getattr(detector, "inspection_head_targets", None)
    if not callable(method):
        return None
    targets = method()
    if targets is None:
        return None
    return {name: float(value) for name, value in targets.items() if name in HEAD_MOTORS}


def observation_frame(raw_obs: dict[str, object]) -> dict[str, np.ndarray]:
    state = np.asarray([float(raw_obs.get(name, 0.0) or 0.0) for name in STATE_NAMES], dtype=np.float32)
    return {
        "observation.state": state,
        "observation.images.head": np.asarray(raw_obs["head"], dtype=np.uint8),
        "observation.images.left_wrist": np.asarray(raw_obs["left_wrist"], dtype=np.uint8),
        "observation.images.right_wrist": np.asarray(raw_obs["right_wrist"], dtype=np.uint8),
    }


def cameras_ready(raw_obs: dict[str, object]) -> bool:
    for name in ("head", "left_wrist", "right_wrist"):
        frame = raw_obs.get(name)
        if not isinstance(frame, np.ndarray) or frame.shape != (480, 640, 3):
            return False
        if not np.any(frame):
            return False
    return True


def print_head_rgb(raw_obs: dict[str, object], enabled: bool, *, label: str) -> None:
    if not enabled:
        return
    frame = raw_obs.get("head")
    if not isinstance(frame, np.ndarray):
        print(f"head rgb {label}: missing frame", flush=True)
        return
    center_y = frame.shape[0] // 2 if frame.ndim >= 2 else 0
    center_x = frame.shape[1] // 2 if frame.ndim >= 2 else 0
    center_rgb = frame[center_y, center_x].tolist() if frame.ndim == 3 and frame.shape[2] == 3 else None
    print(
        f"head rgb {label}: shape={tuple(frame.shape)} dtype={frame.dtype} "
        f"min={int(frame.min())} max={int(frame.max())} mean={float(frame.mean()):.2f} "
        f"center_rgb={center_rgb}",
        flush=True,
    )


def wait_for_camera_frames(robot: XLerobotClient, *, timeout_s: float, require: bool) -> dict[str, object]:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_obs: dict[str, object] | None = None
    while True:
        last_obs = robot.get_observation()
        if cameras_ready(last_obs):
            return last_obs
        if time.perf_counter() >= deadline:
            if require:
                missing = [
                    name
                    for name in ("head", "left_wrist", "right_wrist")
                    if not isinstance(last_obs.get(name), np.ndarray) or not np.any(last_obs.get(name))
                ]
                raise RuntimeError(f"Camera warm-up timed out; missing non-empty frames for {missing}")
            return last_obs
        time.sleep(0.05)


def capped_policy_action(
    robot: XLerobotClient,
    action: dict[str, float],
    *,
    allow_base_action: bool,
    head_override: dict[str, float] | None = None,
) -> dict[str, float]:
    return robot.capped_canonical_action(
        action,
        allow_base_action=allow_base_action,
        head_override=head_override,
    )


def current_hold_action(
    robot: XLerobotClient,
    *,
    head_override: dict[str, float] | None = None,
    gripper_targets: dict[str, float] | None = None,
) -> dict[str, float]:
    return robot.current_hold_action(head_override=head_override, gripper_targets=gripper_targets)


def capped_target_action(
    robot: XLerobotClient,
    target_action: dict[str, float],
    *,
    max_relative_target: float | dict[str, float] | None = None,
) -> dict[str, float]:
    return robot.capped_target_action(target_action, max_relative_target=max_relative_target)


def gripper_release_targets(robot: XLerobotClient, explicit_target: float | None) -> dict[str, float]:
    if explicit_target is not None:
        return {"left_arm_gripper": float(explicit_target), "right_arm_gripper": float(explicit_target)}
    targets: dict[str, float] = {}
    for motor in ("left_arm_gripper", "right_arm_gripper"):
        calibration = robot.follower_calibration.get(motor, {})
        if "range_max" not in calibration:
            raise RuntimeError(
                f"Cannot infer release target for {motor}; pass --release-gripper-target explicitly."
            )
        targets[motor] = float(calibration["range_max"])
    return targets


def send_target_for(
    robot: XLerobotClient,
    target_action: dict[str, float],
    *,
    duration_s: float,
    fps: float,
    label: str,
    max_relative_target: float | dict[str, float] | None = None,
) -> None:
    n_steps = max(1, int(round(max(0.0, duration_s) * fps)))
    for step in range(n_steps):
        step_t = time.perf_counter()
        robot.get_observation()
        sent = capped_target_action(robot, target_action, max_relative_target=max_relative_target)
        send_canonical_action(robot, sent)
        if step == 0 or step == n_steps - 1:
            print(f"target motion {label}: step={step + 1}/{n_steps}", flush=True)
        precise_sleep(max(0.0, 1.0 / fps - (time.perf_counter() - step_t)))


def send_target_until_reached(
    robot: XLerobotClient,
    target_action: dict[str, float],
    *,
    timeout_s: float,
    fps: float,
    label: str,
    max_relative_target: float | dict[str, float] | None,
    tolerance_ticks: float,
    settle_steps: int,
    motors: tuple[str, ...],
) -> None:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    stable_steps = 0
    step = 0
    last_max_error = 0.0
    last_mean_error = 0.0
    print_interval = max(1, int(round(fps)))

    while True:
        step_t = time.perf_counter()
        robot.get_observation()
        current_action = current_hold_action(robot)
        last_max_error, last_mean_error = target_position_error(current_action, target_action, motors)
        if targets_within_tolerance(
            current_action,
            target_action,
            motors,
            tolerance_ticks=tolerance_ticks,
        ):
            stable_steps += 1
        else:
            stable_steps = 0

        sent = capped_target_action(robot, target_action, max_relative_target=max_relative_target)
        send_canonical_action(robot, sent)
        step += 1

        if step == 1 or step % print_interval == 0:
            print(
                f"target motion {label}: step={step} max_error={last_max_error:.1f} "
                f"mean_error={last_mean_error:.1f} stable={stable_steps}/{settle_steps}",
                flush=True,
            )

        if stable_steps >= settle_steps:
            print(
                f"target motion {label}: reached max_error={last_max_error:.1f} "
                f"mean_error={last_mean_error:.1f} after {step} steps",
                flush=True,
            )
            return

        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"target motion {label} did not reach target within {timeout_s:.1f}s; "
                f"max_error={last_max_error:.1f}, mean_error={last_mean_error:.1f}, "
                f"tolerance={abs(float(tolerance_ticks)):.1f}"
            )

        precise_sleep(max(0.0, 1.0 / fps - (time.perf_counter() - step_t)))


def run_initial_arm_recenter(
    robot: XLerobotClient,
    *,
    current_action: dict[str, float],
    recenter_action: dict[str, float],
    args: argparse.Namespace,
) -> None:
    if args.no_arm_recenter_gripper_cycle:
        send_target_until_reached(
            robot,
            recenter_action,
            timeout_s=args.arm_recenter_s,
            fps=args.fps,
            label="arm recenter",
            max_relative_target=args.arm_recenter_max_relative_target,
            tolerance_ticks=args.arm_recenter_tolerance_ticks,
            settle_steps=args.arm_recenter_settle_steps,
            motors=RECENTER_SETTLE_MOTORS,
        )
        return

    open_targets = moderate_gripper_open_targets(
        robot.follower_calibration,
        recenter_action,
        open_ratio=args.arm_recenter_gripper_open_ratio,
    )
    closed = closed_grippers(
        current_action,
        recenter_action,
        open_targets,
        closed_ratio=args.arm_recenter_gripper_closed_ratio,
    )
    if not closed:
        send_target_until_reached(
            robot,
            recenter_action,
            timeout_s=args.arm_recenter_s,
            fps=args.fps,
            label="arm recenter",
            max_relative_target=args.arm_recenter_max_relative_target,
            tolerance_ticks=args.arm_recenter_tolerance_ticks,
            settle_steps=args.arm_recenter_settle_steps,
            motors=RECENTER_SETTLE_MOTORS,
        )
        return

    print(f"arm recenter opening closed grippers first: {closed}", flush=True)
    open_action = dict(current_action)
    open_action.update(open_targets)
    send_target_until_reached(
        robot,
        open_action,
        timeout_s=args.arm_recenter_gripper_s,
        fps=args.fps,
        label="arm recenter open grippers",
        max_relative_target=args.arm_recenter_gripper_max_relative_target,
        tolerance_ticks=args.arm_recenter_tolerance_ticks,
        settle_steps=args.arm_recenter_settle_steps,
        motors=RECENTER_GRIPPER_SETTLE_MOTORS,
    )

    recenter_open_action = dict(recenter_action)
    recenter_open_action.update(open_targets)
    send_target_until_reached(
        robot,
        recenter_open_action,
        timeout_s=args.arm_recenter_s,
        fps=args.fps,
        label="arm recenter with grippers open",
        max_relative_target=args.arm_recenter_max_relative_target,
        tolerance_ticks=args.arm_recenter_tolerance_ticks,
        settle_steps=args.arm_recenter_settle_steps,
        motors=RECENTER_SETTLE_MOTORS,
    )

    send_target_until_reached(
        robot,
        recenter_action,
        timeout_s=args.arm_recenter_gripper_s,
        fps=args.fps,
        label="arm recenter close grippers",
        max_relative_target=args.arm_recenter_gripper_max_relative_target,
        tolerance_ticks=args.arm_recenter_tolerance_ticks,
        settle_steps=args.arm_recenter_settle_steps,
        motors=RECENTER_SETTLE_MOTORS,
    )


def run_success_recovery(
    robot: XLerobotClient,
    *,
    home_action: dict[str, float],
    head_override: dict[str, float] | None,
    args: argparse.Namespace,
) -> None:
    release_targets = gripper_release_targets(robot, args.release_gripper_target)
    release_action = current_hold_action(robot, head_override=head_override, gripper_targets=release_targets)
    send_target_for(robot, release_action, duration_s=args.release_s, fps=args.fps, label="release grippers")

    home_action = dict(home_action)
    for motor, value in release_targets.items():
        home_action[motor] = value
    send_target_for(robot, home_action, duration_s=args.return_s, fps=args.fps, label="return home")


def send_canonical_action(robot: XLerobotClient, action: dict[str, float]) -> None:
    robot.send_canonical_action(action, include_zero_base=True)


def joint_delta(raw_obs: dict[str, object], action: dict[str, float]) -> tuple[float, float]:
    values = [abs(float(action[name]) - float(raw_obs.get(name, 0.0) or 0.0)) for name in JOINT_NAMES]
    return max(values), sum(values) / max(1, len(values))


def print_action_table(raw_obs: dict[str, object], action: dict[str, float], *, prefix: str) -> None:
    print(prefix, flush=True)
    for name in STATE_NAMES:
        current = float(raw_obs.get(name, 0.0) or 0.0)
        target = float(action.get(name, 0.0) or 0.0)
        print(f"  {name:24s} current={current:9.3f} target={target:9.3f} delta={target-current:9.3f}", flush=True)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.duration_s < 0:
        raise ValueError("--duration-s must be non-negative")
    if args.print_head_rgb_every < 1:
        raise ValueError("--print-head-rgb-every must be at least 1")
    if args.success_min_steps < 0:
        raise ValueError("--success-min-steps must be non-negative")
    if args.success_min_elapsed_s < 0:
        raise ValueError("--success-min-elapsed-s must be non-negative")
    if args.arm_recenter_s < 0 or args.arm_recenter_gripper_s < 0 or args.release_s < 0 or args.return_s < 0:
        raise ValueError("--arm-recenter-s, --arm-recenter-gripper-s, --release-s, and --return-s must be non-negative")
    if args.arm_recenter_max_relative_target < 0:
        raise ValueError("--arm-recenter-max-relative-target must be non-negative")
    if args.arm_recenter_gripper_max_relative_target < 0:
        raise ValueError("--arm-recenter-gripper-max-relative-target must be non-negative")
    if args.arm_recenter_tolerance_ticks < 0:
        raise ValueError("--arm-recenter-tolerance-ticks must be non-negative")
    if args.arm_recenter_settle_steps < 1:
        raise ValueError("--arm-recenter-settle-steps must be at least 1")
    if not 0.0 <= args.arm_recenter_gripper_open_ratio <= 1.0:
        raise ValueError("--arm-recenter-gripper-open-ratio must be between 0 and 1")
    if not 0.0 <= args.arm_recenter_gripper_closed_ratio <= 1.0:
        raise ValueError("--arm-recenter-gripper-closed-ratio must be between 0 and 1")
    arm_recenter_targets = parse_arm_recenter_targets(args.arm_recenter_targets)

    gripper_latch: GripperLatch | None = None
    if args.gripper_latch:
        if args.gripper_latch_open_above <= args.gripper_latch_close_below:
            raise ValueError("--gripper-latch-open-above must be greater than --gripper-latch-close-below")
        if args.gripper_latch_squeeze > args.gripper_latch_close_below:
            raise ValueError("--gripper-latch-squeeze must not exceed --gripper-latch-close-below")
        if args.gripper_latch_open_steps < 1:
            raise ValueError("--gripper-latch-open-steps must be at least 1")
        gripper_latch = GripperLatch(
            close_below=args.gripper_latch_close_below,
            squeeze=args.gripper_latch_squeeze,
            open_above=args.gripper_latch_open_above,
            open_steps=args.gripper_latch_open_steps,
        )

    stale_guard = StaleCameraGuard(
        warn_steps=args.stale_camera_warn_steps,
        abort_steps=args.stale_camera_abort_steps,
    )

    success_head_targets = parse_head_targets(args.success_head_targets)
    success_head_ranges = parse_head_ranges(args.success_head_ranges)
    success_detector_kwargs = detector_kwargs(
        args,
        success_head_targets=success_head_targets,
        success_head_ranges=success_head_ranges,
    )
    detector = make_success_detector(
        args.success_detector,
        kwargs=success_detector_kwargs,
        manual_file=args.success_file,
    )
    detector.reset()
    inspection_head_targets = success_head_targets or detector_inspection_head_targets(detector)
    policy_head_override = inspection_head_targets if args.lock_head_for_success else None
    debug_view: LiveDebugWebView | None = None
    if args.debug_web_view:
        debug_cli = cli_debug_summary(args)
        debug_cli["effective_success_detector_kwargs"] = success_detector_kwargs
        debug_cli["inspection_head_targets"] = inspection_head_targets
        debug_cli["policy_head_override"] = policy_head_override
        debug_view = LiveDebugWebView(
            host=args.debug_web_host,
            port=args.debug_web_port,
            open_browser=args.debug_web_open,
            cli_args=debug_cli,
            detector=detector,
        )
        debug_view.start()

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = args.checkpoint
    cfg.device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
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
        id="indory_xlerobot_act_live",
        robot_id=args.robot_id,
        source_id=args.source_id,
        source_role=args.source_role,
        max_relative_target=args.max_relative_target,
        head_pan_sign=args.head_pan_sign,
        head_tilt_sign=args.head_tilt_sign,
        command_lease_ms=args.command_lease_ms,
        connect_timeout_s=args.connect_timeout_s,
        camera_transport=args.camera_transport,
        cam_bridge_base_url=args.cam_bridge_base_url,
        cam_bridge_resize_mode=args.cam_bridge_resize_mode,
        rtp_udp_bind_ip=args.rtp_udp_bind_ip,
        rtp_udp_payload_type=args.rtp_udp_payload_type,
        rtp_udp_front_port=args.rtp_udp_front_port,
        rtp_udp_wrist_left_port=args.rtp_udp_wrist_left_port,
        rtp_udp_wrist_right_port=args.rtp_udp_wrist_right_port,
        rtp_udp_depth_port=args.rtp_udp_depth_port,
        rtp_udp_ffmpeg_path=args.rtp_udp_ffmpeg_path,
    )
    robot = XLerobotClient(robot_cfg)

    n_steps = 1 if args.duration_s == 0 else max(1, int(round(args.duration_s * args.fps)))
    print(
        f"Connecting to {args.remote_ip}; send={args.send}; fps={args.fps}; duration_s={args.duration_s}; "
        f"steps={n_steps}; max_relative_target={args.max_relative_target}; "
        f"base={'policy' if args.allow_base_action else 'zero'}; success_detector={args.success_detector}; "
        f"inspection_head={inspection_head_targets}; lock_head_for_success={args.lock_head_for_success}; "
        f"require_success_head_targets={args.require_success_head_targets}; "
        f"success_head_ranges={success_head_ranges}; "
        f"gripper_latch={'on' if gripper_latch is not None else 'off'}",
        flush=True,
    )
    robot.connect()
    stop_reason = "duration complete"
    try:
        raw_obs = wait_for_camera_frames(
            robot,
            timeout_s=args.warmup_s,
            require=not args.no_require_camera_frames,
        )
        print("camera warm-up ready", flush=True)
        print_head_rgb(raw_obs, args.print_head_rgb, label="warmup")
        if debug_view is not None:
            debug_view.update(raw_obs, step=0, status="camera warm-up")
        current_start_action = current_hold_action(robot, head_override=policy_head_override)
        home_arm_action = current_start_action
        if args.no_arm_recenter:
            print("arm recenter skipped; current arm motor values are return-home pose", flush=True)
        else:
            home_arm_action = apply_arm_recenter_targets(current_start_action, arm_recenter_targets)
            print("arm recenter target prepared as return-home pose", flush=True)
            if args.send:
                run_initial_arm_recenter(
                    robot,
                    current_action=current_start_action,
                    recenter_action=home_arm_action,
                    args=args,
                )
                raw_obs = robot.get_observation()
                print_head_rgb(raw_obs, args.print_head_rgb, label="after_recenter")
                if debug_view is not None:
                    debug_view.update(raw_obs, step=0, status="after recenter")
            else:
                print("arm recenter not sent because --send is false", flush=True)

        start_run = time.perf_counter()
        for step in range(n_steps):
            step_t = time.perf_counter()
            elapsed_s = step_t - start_run
            if step > 0:
                raw_obs = robot.get_observation()
            if step == 0 or (step + 1) % args.print_head_rgb_every == 0:
                print_head_rgb(raw_obs, args.print_head_rgb, label=f"step_{step + 1:06d}")
            stale_guard.check(raw_obs, step=step)

            detection = detector.detect(raw_obs, step=step, elapsed_s=elapsed_s, task=args.task)
            gate_reason = success_gate_reason(args, step=step, elapsed_s=elapsed_s) if detection.success else None
            debug_status = "running"
            if detection.success and gate_reason is not None:
                debug_status = "success gated"
            elif detection.success:
                debug_status = "success"
            if debug_view is not None:
                debug_view.update(
                    raw_obs,
                    detection=detection,
                    step=step + 1,
                    elapsed_s=elapsed_s,
                    gate_reason=gate_reason,
                    status=debug_status,
                    extra={"gripper_latch": gripper_latch.summary() if gripper_latch is not None else None},
                )
            if detection.success:
                should_log_ignored = step == 0 or (
                    args.print_every > 0 and (step + 1) % args.print_every == 0
                )
                if gate_reason is not None:
                    if should_log_ignored:
                        print(
                            f"success detector fired at step={step + 1} "
                            f"(ignored: {gate_reason}): {detection.reason} {detection.metadata or ''}",
                            flush=True,
                        )
                elif args.no_stop_on_success:
                    print(
                        f"success detector fired at step={step + 1} (ignored: --no-stop-on-success): "
                        f"{detection.reason} {detection.metadata or ''}",
                        flush=True,
                    )
                else:
                    stop_reason = detection.reason or "success detector"
                    print(
                        f"success detected before step={step + 1}: {stop_reason} "
                        f"{detection.metadata or ''}",
                        flush=True,
                    )
                    if args.send and not args.no_success_recovery:
                        run_success_recovery(
                            robot,
                            home_action=home_arm_action,
                            head_override=policy_head_override,
                            args=args,
                        )
                    elif args.send and not args.no_hold_on_success:
                        hold_grippers = (
                            gripper_latch.latched_squeeze_targets() if gripper_latch is not None else None
                        )
                        send_canonical_action(robot, current_hold_action(robot, gripper_targets=hold_grippers))
                        print(f"sent current-pose hold command (gripper squeeze={hold_grippers})", flush=True)
                    break

            obs = prepare_observation_for_inference(
                observation_frame(raw_obs),
                torch.device(cfg.device),
                task=args.task,
                robot_type=robot.robot_type,
            )
            with torch.inference_mode():
                action = postprocessor(policy.select_action(preprocessor(obs)))
            robot_action = make_robot_action(action, ds_meta.features)
            if gripper_latch is not None:
                robot_action = gripper_latch.apply(robot_action, step=step)
            sent = capped_policy_action(
                robot,
                robot_action,
                allow_base_action=args.allow_base_action,
                head_override=policy_head_override,
            )
            if args.send:
                send_canonical_action(robot, sent)
            if debug_view is not None:
                debug_view.update(
                    raw_obs,
                    detection=detection,
                    step=step + 1,
                    elapsed_s=elapsed_s,
                    gate_reason=gate_reason,
                    sent_action=sent,
                    status=debug_status,
                    extra={"gripper_latch": gripper_latch.summary() if gripper_latch is not None else None},
                )

            if step == 0 or (args.print_every > 0 and (step + 1) % args.print_every == 0) or step == n_steps - 1:
                max_delta, mean_delta = joint_delta(raw_obs, sent)
                latch_info = f"{gripper_latch.summary()} " if gripper_latch is not None else ""
                print(
                    f"step={step+1:03d}/{n_steps} elapsed={time.perf_counter()-start_run:5.2f}s "
                    f"max_joint_delta={max_delta:7.2f} mean_joint_delta={mean_delta:7.2f} "
                    f"base=({sent['x.vel']:.3f},{sent['y.vel']:.3f},{sent['theta.vel']:.3f}) "
                    f"{latch_info}send={args.send}",
                    flush=True,
                )
                if step == 0:
                    print_action_table(raw_obs, sent, prefix="step=1 capped action")

            precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - step_t)))

        print(f"done: {stop_reason}", flush=True)
        if debug_view is not None:
            debug_view.set_status(f"done: {stop_reason}")
    finally:
        try:
            robot.disconnect()
        finally:
            if debug_view is not None:
                debug_view.set_status(f"finished: {stop_reason}")
                debug_view.keepalive(args.debug_web_keepalive_s)
                debug_view.close()


if __name__ == "__main__":
    main()
