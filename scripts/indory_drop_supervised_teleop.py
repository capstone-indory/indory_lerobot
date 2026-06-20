#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zmq

from lerobot.indory.arm_recenter import target_position_error, targets_within_tolerance
from lerobot.indory.live_debug_view import LiveDebugWebView, cli_debug_summary
from lerobot.indory.success_detector import make_success_detector, parse_detector_kwargs
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.utils.robot_utils import precise_sleep

HEAD_MOTORS = ("head_motor_1", "head_motor_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Super-side DROP teleop supervisor: gate Mac leader actions through a success detector."
    )
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--leader-bind-host", default="0.0.0.0")  # nosec B104 - accepts Mac leader stream on LAN.
    parser.add_argument("--leader-bind-port", type=int, default=8892)
    parser.add_argument("--leader-action-timeout-s", type=float, default=0.75)
    parser.add_argument(
        "--send", action="store_true", help="Forward gated leader actions to the adapter command socket."
    )
    parser.add_argument("--no-hold-on-success", action="store_true")
    parser.add_argument("--command-lease-ms", type=int, default=300)
    parser.add_argument("--max-relative-target", type=float, default=10.0)
    parser.add_argument("--camera-transport", choices=("zmq", "rtp_udp", "cam_bridge"), default="cam_bridge")
    parser.add_argument("--cam-bridge-base-url", default="ws://127.0.0.1:8870")
    parser.add_argument("--cam-bridge-resize-mode", choices=("center_crop", "stretch"), default="center_crop")
    parser.add_argument("--rtp-udp-bind-ip", default="0.0.0.0")  # nosec B104 - robot LAN UDP receiver.
    parser.add_argument("--rtp-udp-payload-type", type=int, default=96)
    parser.add_argument("--rtp-udp-front-port", type=int, default=5600)
    parser.add_argument("--rtp-udp-wrist-left-port", type=int, default=5602)
    parser.add_argument("--rtp-udp-wrist-right-port", type=int, default=5604)
    parser.add_argument("--rtp-udp-depth-port", type=int, default=5610)
    parser.add_argument("--rtp-udp-ffmpeg-path", default=None)
    parser.add_argument("--source-id", default="super_drop_supervisor")
    parser.add_argument("--source-role", default="drop_supervisor")
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--success-detector", default="sky-blue-parcel-color")
    parser.add_argument("--success-detector-kwargs", default=None)
    parser.add_argument("--success-file", type=Path, default=None)
    parser.add_argument("--success-head-targets", default="head_motor_1=2070,head_motor_2=2700")
    parser.add_argument("--require-success-head-targets", action="store_true")
    parser.add_argument("--success-head-ranges", default=None)
    parser.add_argument("--head-align-timeout-s", type=float, default=10.0)
    parser.add_argument("--head-align-tolerance-ticks", type=float, default=25.0)
    parser.add_argument("--head-align-settle-steps", type=int, default=3)
    parser.add_argument("--success-min-steps", type=int, default=0)
    parser.add_argument("--success-min-elapsed-s", type=float, default=0.0)
    parser.add_argument("--task", default="drop parcel")
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--debug-web-view", action="store_true")
    parser.add_argument("--debug-web-host", default="0.0.0.0")  # nosec B104 - optional debug UI binding.
    parser.add_argument("--debug-web-port", type=int, default=8890)
    parser.add_argument("--debug-web-open", action="store_true")
    parser.add_argument("--debug-web-keepalive-s", type=float, default=5.0)
    return parser.parse_args()


def parse_head_targets(raw: str | None) -> dict[str, float] | None:
    if raw is None or raw.strip() == "":
        return None
    if "=" not in raw:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
        if len(values) != 2:
            raise ValueError("--success-head-targets without names must be 'head_motor_1,head_motor_2'")
        return {"head_motor_1": values[0], "head_motor_2": values[1]}
    out: dict[str, float] = {}
    for item in raw.split(","):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError("--success-head-targets entries must be name=value")
        name = name.strip()
        if name not in HEAD_MOTORS:
            raise ValueError(f"unsupported head target {name!r}; expected one of {HEAD_MOTORS}")
        out[name] = float(value)
    return out or None


def parse_head_ranges(raw: str | None) -> dict[str, tuple[float, float]] | None:
    if raw is None or raw.strip() == "":
        return None
    out: dict[str, tuple[float, float]] = {}
    for item in raw.split(","):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError("--success-head-ranges entries must be name=min:max")
        name = name.strip()
        if name not in HEAD_MOTORS:
            raise ValueError(f"unsupported head range {name!r}; expected one of {HEAD_MOTORS}")
        low_text, range_sep, high_text = value.partition(":")
        if not range_sep:
            raise ValueError("--success-head-ranges values must be min:max")
        low, high = float(low_text), float(high_text)
        out[name] = (min(low, high), max(low, high))
    return out or None


def detector_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = parse_detector_kwargs(args.success_detector_kwargs)
    head_targets = parse_head_targets(args.success_head_targets)
    head_ranges = parse_head_ranges(args.success_head_ranges)
    if args.require_success_head_targets and head_targets is not None and "head_targets" not in kwargs:
        kwargs["head_targets"] = head_targets
    if head_ranges is not None and "head_ranges" not in kwargs:
        kwargs["head_ranges"] = head_ranges
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
    if not isinstance(targets, dict):
        return None
    return {name: float(value) for name, value in targets.items() if name in HEAD_MOTORS}


def has_head_frame(raw_obs: dict[str, Any]) -> bool:
    frame = raw_obs.get("head")
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return False
    return bool(np.any(frame))


def wait_for_head_frame(robot: XLerobotClient, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_obs: dict[str, Any] | None = None
    while time.perf_counter() <= deadline:
        last_obs = robot.get_observation()
        if has_head_frame(last_obs):
            return last_obs
        time.sleep(0.05)
    if last_obs is not None:
        return last_obs
    return robot.get_observation()


def wait_for_proprio_joint_positions(robot: XLerobotClient, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_obs: dict[str, Any] | None = None
    while True:
        last_obs = robot.get_observation()
        current = robot.current_canonical_ticks()
        if current is not None:
            return last_obs
        if time.perf_counter() >= deadline:
            latest_topics = ", ".join(sorted(robot.latest_topics.keys())) or "none"
            raise RuntimeError(
                f"Proprio warm-up timed out after {timeout_s:.1f}s; expected "
                f"proprio.{robot.robot_id} joint_pos with at least 14 values; latest_topics={latest_topics}"
            )
        time.sleep(0.05)


def send_canonical_action(robot: XLerobotClient, action: dict[str, float]) -> None:
    robot.send_canonical_action(action, include_zero_base=True)


def current_hold_action(
    robot: XLerobotClient,
    *,
    head_override: dict[str, float] | None = None,
) -> dict[str, float]:
    return robot.current_hold_action(head_override=head_override)


def capped_target_action(
    robot: XLerobotClient,
    target_action: dict[str, float],
    *,
    max_relative_target: float | dict[str, float] | None = None,
) -> dict[str, float]:
    return robot.capped_target_action(target_action, max_relative_target=max_relative_target)


def align_head_to_targets(
    robot: XLerobotClient,
    head_targets: dict[str, float] | None,
    *,
    args: argparse.Namespace,
) -> None:
    if not head_targets:
        return
    if not args.send:
        print(f"DROP head alignment skipped because --send is false: targets={head_targets}", flush=True)
        return

    deadline = time.perf_counter() + max(0.0, args.head_align_timeout_s)
    stable_steps = 0
    step = 0
    last_max_error = 0.0
    last_mean_error = 0.0
    print_interval = max(1, int(round(args.fps)))
    print(f"DROP head alignment target: {head_targets}", flush=True)

    while True:
        step_t = time.perf_counter()
        robot.get_observation()
        current_action = current_hold_action(robot)
        target_action = current_hold_action(robot, head_override=head_targets)
        last_max_error, last_mean_error = target_position_error(current_action, target_action, HEAD_MOTORS)
        if targets_within_tolerance(
            current_action,
            target_action,
            HEAD_MOTORS,
            tolerance_ticks=args.head_align_tolerance_ticks,
        ):
            stable_steps += 1
        else:
            stable_steps = 0

        sent = capped_target_action(robot, target_action, max_relative_target=args.max_relative_target)
        send_canonical_action(robot, sent)
        step += 1

        if step == 1 or step % print_interval == 0:
            print(
                f"DROP head alignment: step={step} max_error={last_max_error:.1f} "
                f"mean_error={last_mean_error:.1f} stable={stable_steps}/{args.head_align_settle_steps}",
                flush=True,
            )

        if stable_steps >= args.head_align_settle_steps:
            print(
                f"DROP head alignment reached: max_error={last_max_error:.1f} "
                f"mean_error={last_mean_error:.1f} after {step} steps",
                flush=True,
            )
            return

        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"DROP head alignment did not reach target within {args.head_align_timeout_s:.1f}s; "
                f"max_error={last_max_error:.1f}, mean_error={last_mean_error:.1f}, "
                f"tolerance={abs(float(args.head_align_tolerance_ticks)):.1f}"
            )

        precise_sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - step_t)))


def apply_head_override(
    action: dict[str, float],
    head_targets: dict[str, float] | None,
) -> dict[str, float]:
    if not head_targets:
        return action
    adjusted = dict(action)
    adjusted.update(head_targets)
    return adjusted


def build_head_locked_teleop_action(
    robot: XLerobotClient,
    action: dict[str, float],
    head_targets: dict[str, float] | None,
    *,
    max_relative_target: float | dict[str, float] | None,
) -> dict[str, float]:
    action_with_head = apply_head_override(action, head_targets)
    result = robot.command_builder.build(
        action_with_head,
        seq=0,
        source_id=robot.source_id,
        source_role=robot.source_role,
        lease_ms=robot.command_lease_ms,
        robot_id=robot.robot_id,
    )

    canonical = result.payload.get("arm_joint_pos_target")
    if canonical is None:
        canonical = robot.current_canonical_ticks()
    if canonical is None:
        raise RuntimeError("Cannot forward teleop action before receiving proprio joint positions.")

    base_cmd = result.payload.get("base_cmd_vel")
    target_action = robot.action_from_canonical_ticks(canonical, base_cmd=base_cmd)
    if head_targets:
        target_action.update(head_targets)
    return capped_target_action(robot, target_action, max_relative_target=max_relative_target)


def recv_latest_action(socket: zmq.Socket) -> tuple[dict[str, Any] | None, int]:
    latest: dict[str, Any] | None = None
    count = 0
    while True:
        try:
            raw = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        count += 1
        message = msgpack.unpackb(raw, raw=False)
        if not isinstance(message, dict):
            continue
        action = message.get("action")
        if isinstance(action, dict):
            message["_received_perf_s"] = time.perf_counter()
            latest = message
    return latest, count


def leader_action(message: dict[str, Any] | None) -> dict[str, float] | None:
    if not message:
        return None
    action = message.get("action")
    if not isinstance(action, dict):
        return None
    out: dict[str, float] = {}
    for key, value in action.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def leader_age_s(message: dict[str, Any] | None) -> float | None:
    if not message:
        return None
    received = message.get("_received_perf_s")
    if not isinstance(received, float):
        return None
    return time.perf_counter() - received


def hold_current(robot: XLerobotClient) -> None:
    robot.send_canonical_action(robot.current_hold_action(), include_zero_base=True)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.leader_action_timeout_s < 0:
        raise ValueError("--leader-action-timeout-s must be non-negative")
    if args.head_align_timeout_s < 0:
        raise ValueError("--head-align-timeout-s must be non-negative")
    if args.head_align_settle_steps < 1:
        raise ValueError("--head-align-settle-steps must be at least 1")

    kwargs = detector_kwargs(args)
    detector = make_success_detector(args.success_detector, kwargs=kwargs, manual_file=args.success_file)
    inspection_head_targets = parse_head_targets(
        args.success_head_targets
    ) or detector_inspection_head_targets(detector)
    debug_view = None
    if args.debug_web_view:
        debug_cli = cli_debug_summary(args)
        debug_cli["effective_success_detector_kwargs"] = kwargs
        debug_cli["inspection_head_targets"] = inspection_head_targets
        debug_view = LiveDebugWebView(
            host=args.debug_web_host,
            port=args.debug_web_port,
            open_browser=args.debug_web_open,
            cli_args=debug_cli,
            detector=detector,
        )
        debug_view.start()

    context = zmq.Context()
    leader_socket = context.socket(zmq.PULL)
    leader_socket.setsockopt(zmq.LINGER, 0)
    leader_socket.setsockopt(zmq.RCVHWM, 2)
    leader_endpoint = f"tcp://{args.leader_bind_host}:{args.leader_bind_port}"
    leader_socket.bind(leader_endpoint)

    robot_cfg = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        id="indory_xlerobot_drop_supervisor",
        robot_id=args.robot_id,
        source_id=args.source_id,
        source_role=args.source_role,
        command_lease_ms=args.command_lease_ms,
        max_relative_target=args.max_relative_target,
        leader_action_units="degrees",
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

    n_steps = max(1, int(round(args.duration_s * args.fps)))
    sleep_s = 1.0 / args.fps
    latest_leader: dict[str, Any] | None = None
    forwarded = 0
    received = 0
    stop_reason = "duration complete"
    exit_code = 3

    print(
        f"DROP supervisor binding {leader_endpoint}; remote_ip={args.remote_ip}; send={args.send}; "
        f"fps={args.fps}; duration_s={args.duration_s}; success_detector={args.success_detector}; kwargs={kwargs}",
        flush=True,
    )

    try:
        robot.connect()
        wait_for_proprio_joint_positions(robot, timeout_s=args.warmup_s)
        align_head_to_targets(robot, inspection_head_targets, args=args)
        raw_obs = wait_for_head_frame(robot, timeout_s=args.warmup_s)
        print("drop supervisor camera warm-up ready", flush=True)
        if debug_view is not None:
            debug_view.update(raw_obs, step=0, status="camera warm-up")

        start_run = time.perf_counter()
        for step in range(n_steps):
            step_t = time.perf_counter()
            elapsed_s = step_t - start_run
            raw_obs = robot.get_observation()
            newest, count = recv_latest_action(leader_socket)
            received += count
            if newest is not None:
                latest_leader = newest

            detection = detector.detect(raw_obs, step=step, elapsed_s=elapsed_s, task=args.task)
            gate_reason = (
                success_gate_reason(args, step=step, elapsed_s=elapsed_s) if detection.success else None
            )
            if detection.success and gate_reason is not None:
                status = f"success gated: {gate_reason}"
                detection = type(detection)(
                    False, reason=gate_reason, score=detection.score, metadata=detection.metadata
                )
            elif detection.success:
                status = "success"
            else:
                status = detection.reason or "running"

            action_age = leader_age_s(latest_leader)
            action = leader_action(latest_leader)
            action_fresh = action is not None and (
                args.leader_action_timeout_s == 0
                or (action_age is not None and action_age <= args.leader_action_timeout_s)
            )

            if debug_view is not None:
                debug_view.update(
                    raw_obs,
                    step=step + 1,
                    detection=detection,
                    status=f"{status}; leader_age={action_age if action_age is not None else 'none'}",
                )

            if detection.success:
                stop_reason = detection.reason or "success detector"
                exit_code = 0
                print(
                    f"success detected before step={step + 1}: {stop_reason}; "
                    f"received={received}; forwarded={forwarded}; metadata={detection.metadata}",
                    flush=True,
                )
                if args.send and not args.no_hold_on_success:
                    hold_current(robot)
                    print("sent current-pose hold command after DROP success", flush=True)
                break

            if args.send and action_fresh and action is not None:
                action_to_send = build_head_locked_teleop_action(
                    robot,
                    action,
                    inspection_head_targets,
                    max_relative_target=args.max_relative_target,
                )
                send_canonical_action(robot, action_to_send)
                forwarded += 1

            if args.print_every > 0 and (step == 0 or (step + 1) % args.print_every == 0):
                seq = latest_leader.get("seq") if latest_leader else None
                print(
                    f"step={step + 1}/{n_steps} status={status!r} leader_seq={seq} "
                    f"leader_age={action_age if action_age is not None else 'none'} "
                    f"fresh={action_fresh} received={received} forwarded={forwarded}",
                    flush=True,
                )

            time.sleep(max(0.0, sleep_s - (time.perf_counter() - step_t)))
        else:
            print(
                f"done: {stop_reason}; received={received}; forwarded={forwarded}; no success detector confirmation",
                flush=True,
            )
    finally:
        if debug_view is not None:
            debug_view.keepalive(args.debug_web_keepalive_s)
            debug_view.close()
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            leader_socket.close(0)
            context.term()

    print(f"done: {stop_reason}; received={received}; forwarded={forwarded}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
