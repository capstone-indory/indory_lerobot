#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

import msgpack
import zmq

STATE_TOPICS = ("proprio.{robot_id}", "joint_states.{robot_id}", "odom.{robot_id}")
CAMERA_TOPICS = ("rgb.front.{robot_id}", "rgb.wrist_left.{robot_id}", "rgb.wrist_right.{robot_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-command Indory ZMQ preflight probe.")
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--rpc-port", type=int, default=8857)
    parser.add_argument("--state-port", type=int, default=8855)
    parser.add_argument("--camera-port", type=int, default=8866)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--skip-cameras", action="store_true")
    return parser.parse_args()


def rpc(ctx: zmq.Context, remote_ip: str, port: int, op: str, timeout_s: float) -> dict[str, Any]:
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    try:
        sock.connect(f"tcp://{remote_ip}:{port}")
        sock.send(msgpack.packb({"op": op}, use_bin_type=True))
        reply = msgpack.unpackb(sock.recv(), raw=False)
        return reply if isinstance(reply, dict) else {"ok": False, "error": "bad_rpc_reply"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        sock.close(0)


def collect_sub_messages(
    ctx: zmq.Context,
    remote_ip: str,
    port: int,
    topics: tuple[str, ...],
    timeout_s: float,
) -> dict[str, dict[str, Any]]:
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, 64)
    for topic in topics:
        sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    sock.connect(f"tcp://{remote_ip}:{port}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    deadline = time.monotonic() + max(0.0, timeout_s)
    seen: dict[str, dict[str, Any]] = {}
    try:
        while time.monotonic() < deadline and len(seen) < len(topics):
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            if sock not in dict(poller.poll(min(50, remaining_ms))):
                continue
            while True:
                try:
                    topic_raw, payload_raw = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                topic = topic_raw.decode("utf-8", errors="replace")
                try:
                    payload = msgpack.unpackb(payload_raw, raw=False)
                except Exception as exc:
                    payload = {"_decode_error": str(exc)}
                if isinstance(payload, dict):
                    seen[topic] = payload
    finally:
        sock.close(0)
    return seen


def summarize_rpc(reply: dict[str, Any]) -> dict[str, Any]:
    summary = {"ok": bool(reply.get("ok"))}
    health = reply.get("health")
    if isinstance(health, dict):
        summary["ok"] = bool(health.get("ok", summary["ok"]))
        for key in (
            "source",
            "estop",
            "motor_connected",
            "base_ready",
            "accepted_commands",
            "dropped_commands",
            "joint_state_age_ms",
            "odom_age_ms",
            "scan_age_ms",
            "last_command_source",
            "topics",
        ):
            if key in health:
                summary[key] = health[key]
    if "error" in reply:
        summary["error"] = reply["error"]
    if isinstance(reply.get("calibration"), dict):
        joints = reply["calibration"].get("joints")
        summary["calibration_joints"] = len(joints) if isinstance(joints, dict) else 0
    for key in ("topics", "active_sources", "owner", "lease"):
        if key in reply:
            summary[key] = reply[key]
    return summary


def tcp_connectable(remote_ip: str, port: int, timeout_s: float) -> dict[str, Any]:
    sock = socket.socket()
    sock.settimeout(timeout_s)
    try:
        sock.connect((remote_ip, port))
        return {"open": True}
    except Exception as exc:
        return {"open": False, "error": str(exc)}
    finally:
        sock.close()


def summarize_state(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"keys": sorted(payload.keys())}
    joint_pos = payload.get("joint_pos")
    if isinstance(joint_pos, list):
        summary["joint_pos_len"] = len(joint_pos)
        summary["joint_pos_first"] = joint_pos[:3]
    base_vel = payload.get("base_joint_vel")
    if isinstance(base_vel, list):
        summary["base_joint_vel"] = base_vel[:3]
    for key in ("stamp_ns", "seq", "frame"):
        if key in payload:
            summary[key] = payload[key]
    return summary


def summarize_camera(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "stamp_ns",
        "chunk_seq",
        "encoding",
        "width",
        "height",
        "format",
        "frame_id",
        "topic",
    ):
        if key in payload:
            summary[key] = payload[key]
    for key in ("data", "jpeg", "payload"):
        value = payload.get(key)
        if isinstance(value, bytes | bytearray):
            summary[f"{key}_bytes"] = len(value)
    summary["keys"] = sorted(payload.keys())
    return summary


def main() -> None:
    args = parse_args()
    state_topics = tuple(topic.format(robot_id=args.robot_id) for topic in STATE_TOPICS)
    camera_topics = tuple(topic.format(robot_id=args.robot_id) for topic in CAMERA_TOPICS)

    ctx = zmq.Context()
    try:
        health = rpc(ctx, args.remote_ip, args.rpc_port, "health", args.timeout_s)
        calibration = rpc(ctx, args.remote_ip, args.rpc_port, "calibration", args.timeout_s)
        topic_list = rpc(ctx, args.remote_ip, args.rpc_port, "topic_list", args.timeout_s)
        command_status = rpc(ctx, args.remote_ip, args.rpc_port, "command_status", args.timeout_s)
        state_messages = collect_sub_messages(
            ctx,
            args.remote_ip,
            args.state_port,
            state_topics,
            args.timeout_s,
        )
        camera_messages = (
            {}
            if args.skip_cameras
            else collect_sub_messages(ctx, args.remote_ip, args.camera_port, camera_topics, args.timeout_s)
        )
    finally:
        ctx.term()

    report = {
        "remote_ip": args.remote_ip,
        "ports": {
            "state": tcp_connectable(args.remote_ip, args.state_port, min(args.timeout_s, 2.0)),
            "command": tcp_connectable(args.remote_ip, 8856, min(args.timeout_s, 2.0)),
            "rpc": tcp_connectable(args.remote_ip, args.rpc_port, min(args.timeout_s, 2.0)),
            "camera": tcp_connectable(args.remote_ip, args.camera_port, min(args.timeout_s, 2.0)),
        },
        "rpc": {
            "health": summarize_rpc(health),
            "calibration": summarize_rpc(calibration),
            "topic_list": summarize_rpc(topic_list),
            "command_status": summarize_rpc(command_status),
        },
        "state": {topic: summarize_state(payload) for topic, payload in state_messages.items()},
        "cameras": {topic: summarize_camera(payload) for topic, payload in camera_messages.items()},
        "missing_state_topics": sorted(set(state_topics) - set(state_messages)),
        "missing_camera_topics": [] if args.skip_cameras else sorted(set(camera_topics) - set(camera_messages)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not health.get("ok"):
        raise SystemExit(1)
    if f"proprio.{args.robot_id}" not in state_messages:
        raise SystemExit(2)
    if not args.skip_cameras and camera_topics and not camera_messages:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
