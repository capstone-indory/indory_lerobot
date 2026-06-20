#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import msgpack
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.teleoperators.bi_so_leader.bi_so_leader import BiSOLeader
from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig, SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader


SCHEMA = "indory_leader_action.v1"


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish local SO leader actions to a supervised DROP runner.")
    parser.add_argument("--server-url", default=env("INDORY_LEADER_SERVER_URL", "tcp://super:8892"))
    parser.add_argument("--fps", type=float, default=float(env("INDORY_FPS", "15") or 15))
    parser.add_argument("--source-id", default=env("INDORY_ZMQ_SOURCE_ID", "mac_xlerobot_drop_leader"))
    parser.add_argument("--source-role", default=env("INDORY_ZMQ_SOURCE_ROLE", "drop_leader"))
    parser.add_argument("--teleop-id", default=env("INDORY_TELEOP_ID", None))
    parser.add_argument("--teleop-arm-mode", default=env("INDORY_TELEOP_ARM_MODE", "auto"))
    parser.add_argument("--left-leader-port", default=env("INDORY_LEFT_LEADER_PORT", ""))
    parser.add_argument("--right-leader-port", default=env("INDORY_RIGHT_LEADER_PORT", ""))
    parser.add_argument("--single-leader-type", default=env("INDORY_SINGLE_LEADER_TYPE", "so101_leader"))
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--print-every", type=int, default=int(env("INDORY_PRINT_EVERY", "15") or 15))
    return parser.parse_args()


def leader_mode(args: argparse.Namespace) -> str:
    mode = str(args.teleop_arm_mode or "auto").strip().lower()
    if mode == "auto":
        if args.left_leader_port and args.right_leader_port:
            return "bimanual"
        if args.right_leader_port:
            return "right_only"
        raise ValueError("Set --right-leader-port, or set both --left-leader-port and --right-leader-port.")
    if mode in {"right", "right_only"}:
        if not args.right_leader_port:
            raise ValueError("--right-leader-port is required for right-only leader publishing.")
        return "right_only"
    if mode in {"bimanual", "bi"}:
        if not args.left_leader_port or not args.right_leader_port:
            raise ValueError("--left-leader-port and --right-leader-port are required for bimanual leader publishing.")
        return "bimanual"
    raise ValueError(f"Unsupported --teleop-arm-mode={args.teleop_arm_mode!r}; use auto, right_only, or bimanual.")


def make_teleop(args: argparse.Namespace, mode: str) -> SOLeader | BiSOLeader:
    if mode == "right_only":
        if args.single_leader_type not in {"so100_leader", "so101_leader"}:
            raise ValueError("--single-leader-type must be so100_leader or so101_leader")
        return SOLeader(
            SOLeaderTeleopConfig(
                id=args.teleop_id or "xlerobot_bi_so101_leader_right",
                port=args.right_leader_port,
                use_degrees=True,
            )
        )

    return BiSOLeader(
        BiSOLeaderConfig(
            id=args.teleop_id or "xlerobot_bi_so101_leader",
            left_arm_config=SOLeaderConfig(port=args.left_leader_port, use_degrees=True),
            right_arm_config=SOLeaderConfig(port=args.right_leader_port, use_degrees=True),
        )
    )


def sanitize_action(action: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in action.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[str(key)] = number
    return out


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    mode = leader_mode(args)
    teleop = make_teleop(args, mode)

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.connect(args.server_url)

    print(
        f"publishing leader actions to {args.server_url}; mode={mode}; fps={args.fps}; "
        f"source_id={args.source_id}",
        flush=True,
    )

    seq = 0
    sleep_s = 1.0 / args.fps
    teleop.connect(calibrate=not args.no_calibrate)
    try:
        while True:
            step_t = time.perf_counter()
            action = sanitize_action(teleop.get_action())
            message = {
                "schema": SCHEMA,
                "seq": seq,
                "stamp_ns": time.time_ns(),
                "source_id": args.source_id,
                "source_role": args.source_role,
                "teleop_arm_mode": mode,
                "action": action,
            }
            socket.send(msgpack.packb(message, use_bin_type=True), flags=zmq.NOBLOCK)
            seq += 1
            if args.print_every > 0 and (seq == 1 or seq % args.print_every == 0):
                print(f"leader action seq={seq} keys={len(action)}", flush=True)
            time.sleep(max(0.0, sleep_s - (time.perf_counter() - step_t)))
    except KeyboardInterrupt:
        print("leader publisher interrupted", flush=True)
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
