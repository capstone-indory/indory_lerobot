#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.indory.live_debug_view import LiveDebugWebView, cli_debug_summary
from lerobot.indory.success_detector import make_success_detector, parse_detector_kwargs
from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient

HEAD_MOTORS = ("head_motor_1", "head_motor_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch XLerobot observations until a success detector fires."
    )
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
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
    parser.add_argument("--source-id", default="drop_success_watch")
    parser.add_argument("--source-role", default="success_watch")
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--success-detector", default="sky-blue-parcel-color")
    parser.add_argument("--success-detector-kwargs", default=None)
    parser.add_argument("--success-file", type=Path, default=None)
    parser.add_argument("--success-head-targets", default=None)
    parser.add_argument("--require-success-head-targets", action="store_true")
    parser.add_argument("--success-head-ranges", default=None)
    parser.add_argument("--success-min-steps", type=int, default=0)
    parser.add_argument("--success-min-elapsed-s", type=float, default=0.0)
    parser.add_argument("--task", default="drop parcel")
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--debug-web-view", action="store_true")
    parser.add_argument("--debug-web-host", default="0.0.0.0")  # nosec B104 - optional debug UI binding.
    parser.add_argument("--debug-web-port", type=int, default=8890)
    parser.add_argument("--debug-web-open", action="store_true")
    parser.add_argument(
        "--debug-web-keepalive-s",
        type=float,
        default=5.0,
        help="Seconds to keep the debug web view alive after the watcher exits.",
    )
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


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.success_min_steps < 0:
        raise ValueError("--success-min-steps must be non-negative")
    if args.success_min_elapsed_s < 0:
        raise ValueError("--success-min-elapsed-s must be non-negative")

    kwargs = detector_kwargs(args)
    detector = make_success_detector(args.success_detector, kwargs=kwargs, manual_file=args.success_file)
    detector.reset()

    debug_view: LiveDebugWebView | None = None
    if args.debug_web_view:
        debug_cli = cli_debug_summary(args)
        debug_cli["effective_success_detector_kwargs"] = kwargs
        debug_view = LiveDebugWebView(
            host=args.debug_web_host,
            port=args.debug_web_port,
            open_browser=args.debug_web_open,
            cli_args=debug_cli,
            detector=detector,
        )
        debug_view.start()

    robot_cfg = XLerobotClientConfig(
        remote_ip=args.remote_ip,
        id="indory_xlerobot_success_watch",
        robot_id=args.robot_id,
        source_id=args.source_id,
        source_role=args.source_role,
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
    print(
        f"Connecting to {args.remote_ip}; fps={args.fps}; duration_s={args.duration_s}; "
        f"steps={n_steps}; success_detector={args.success_detector}; kwargs={kwargs}",
        flush=True,
    )

    stop_reason = "duration complete"
    exit_code = 3
    try:
        robot.connect()
        raw_obs = wait_for_head_frame(robot, timeout_s=args.warmup_s)
        print("success watch camera warm-up ready", flush=True)
        if debug_view is not None:
            debug_view.update(raw_obs, step=0, status="camera warm-up")

        start_run = time.perf_counter()
        for step in range(n_steps):
            step_t = time.perf_counter()
            elapsed_s = step_t - start_run
            if step > 0:
                raw_obs = robot.get_observation()
            detection = detector.detect(raw_obs, step=step, elapsed_s=elapsed_s, task=args.task)
            gate_reason = (
                success_gate_reason(args, step=step, elapsed_s=elapsed_s) if detection.success else None
            )
            status = "success gated" if gate_reason else "success" if detection.success else "running"
            if debug_view is not None:
                debug_view.update(
                    raw_obs,
                    detection=detection,
                    step=step + 1,
                    elapsed_s=elapsed_s,
                    gate_reason=gate_reason,
                    status=status,
                )
            if args.print_every > 0 and (step == 0 or (step + 1) % args.print_every == 0):
                print(
                    f"watch step={step + 1}/{n_steps} status={status} "
                    f"reason={detection.reason} metadata={detection.metadata or {}}",
                    flush=True,
                )

            if detection.success and gate_reason is None:
                stop_reason = detection.reason or "success detector"
                print(
                    f"success detected before step={step + 1}: {stop_reason} {detection.metadata or {}}",
                    flush=True,
                )
                exit_code = 0
                break

            elapsed_step = time.perf_counter() - step_t
            time.sleep(max(0.0, sleep_s - elapsed_step))
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            if debug_view is not None:
                debug_view.set_status("done", extra={"stop_reason": stop_reason, "exit_code": exit_code})
                if args.debug_web_keepalive_s > 0:
                    time.sleep(args.debug_web_keepalive_s)
                debug_view.close()

    print(f"done: {stop_reason}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
