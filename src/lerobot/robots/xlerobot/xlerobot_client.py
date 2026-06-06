# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import io
import json
import logging
import math
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

try:
    import av
except Exception:  # pragma: no cover - optional runtime dependency
    av = None

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_xlerobot import XLerobotClientConfig


SCHEMA_VERSION = "xlerobot_v1.1"
LEFT_MOTORS = (
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
    "left_arm_gripper",
)
RIGHT_MOTORS = (
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
)
HEAD_MOTORS = ("head_motor_1", "head_motor_2")
SHORT_TO_SUFFIX = {
    "shoulder_pan": "shoulder_pan",
    "shoulder_lift": "shoulder_lift",
    "elbow_flex": "elbow_flex",
    "wrist_flex": "wrist_flex",
    "wrist_roll": "wrist_roll",
    "gripper": "gripper",
}
LEGACY_CAMERA_TOPIC_TO_NAME = {
    "/xlerobot/head/rgb/image_raw": "head",
    "/xlerobot/wrist_left/rgb/image_raw": "left_wrist",
    "/xlerobot/wrist_right/rgb/image_raw": "right_wrist",
}


class XLerobotClient(Robot):
    """LeRobot Robot wrapper for the indory_zmq fast ZMQ server."""

    config_class = XLerobotClientConfig
    name = "xlerobot_client"

    def __init__(self, config: XLerobotClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type
        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations
        self.port_zmq_rpc = config.port_zmq_rpc
        self.port_zmq_cameras = config.port_zmq_cameras
        self.robot_id = int(config.robot_id)
        self.source_id = config.source_id
        self.teleop_keys = config.teleop_keys
        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s
        self.command_lease_ms = int(config.command_lease_ms)

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_state_socket = None
        self.zmq_camera_socket = None
        self._is_connected = False
        self._seq = 0
        self._warned_encodings: set[str] = set()
        self._h264_init_by_topic: dict[str, bytes] = {}
        self.last_frames: dict[str, np.ndarray] = {}
        self.last_remote_state: dict[str, Any] = {}
        self.latest_topics: dict[str, dict[str, Any]] = {}
        self.camera_topic_to_name = {
            f"rgb.front.{self.robot_id}": "head",
            f"rgb.wrist_left.{self.robot_id}": "left_wrist",
            f"rgb.wrist_right.{self.robot_id}": "right_wrist",
            **LEGACY_CAMERA_TOPIC_TO_NAME,
        }
        self.follower_calibration = self._load_follower_calibration(config.follower_calibration_path)
        self.logs = {}

        self.speed_levels = [
            {"xy": 0.10, "theta": math.radians(30)},
            {"xy": 0.20, "theta": math.radians(60)},
            {"xy": 0.30, "theta": math.radians(90)},
        ]
        self.speed_index = 0

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (*LEFT_MOTORS, *RIGHT_MOTORS, *HEAD_MOTORS, "x.vel", "y.vel", "theta.vel"),
            float,
        )

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def cameras(self) -> dict[str, Any]:
        return self.config.cameras

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError("XLerobotClient is already connected.")
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self._make_push_socket(self.port_zmq_cmd)
        self.zmq_state_socket = self._make_sub_socket(
            self.port_zmq_observations,
            [f"proprio.{self.robot_id}", f"joint_states.{self.robot_id}", f"odom.{self.robot_id}"],
        )
        camera_topics = [
            topic for topic, name in self.camera_topic_to_name.items() if name in self._cameras_ft
        ]
        self.zmq_camera_socket = self._make_sub_socket(self.port_zmq_cameras, camera_topics)
        health = self._rpc("health")
        if not health.get("ok"):
            self.disconnect_sockets()
            raise DeviceNotConnectedError(f"indory_zmq health check failed: {health}")
        self._merge_remote_calibration(self._rpc("calibration"))
        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("XLerobotClient is not connected.")
        self._poll_zmq()
        obs_dict = self._state_from_latest_topics()
        for cam_name, shape in self._cameras_ft.items():
            frame = self.last_frames.get(cam_name)
            obs_dict[cam_name] = frame if frame is not None else np.zeros(shape, dtype=np.uint8)
        self.last_remote_state = obs_dict
        return obs_dict

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, len(self.speed_levels) - 1)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed = self.speed_levels[self.speed_index]
        x_cmd = y_cmd = theta_cmd = 0.0
        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += speed["xy"]
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= speed["xy"]
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += speed["xy"]
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= speed["xy"]
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += speed["theta"]
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= speed["theta"]
        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected or self.zmq_cmd_socket is None:
            raise DeviceNotConnectedError("XLerobotClient is not connected.")
        action = dict(action)
        base_cmd = self._base_cmd_from_action(action)
        joint_targets = self._joint_targets_from_action(action)
        payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "source_id": self.source_id,
            "source_role": "teleop",
            "seq": self._seq,
            "stamp_ns": time.time_ns(),
            "lease_ms": self.command_lease_ms,
            "frame": "body",
        }
        if base_cmd is not None:
            payload["base_cmd_vel"] = base_cmd
        if joint_targets is not None:
            payload["joint_targets_sparse"] = joint_targets
        self._seq += 1
        if "base_cmd_vel" in payload or "joint_targets_sparse" in payload:
            self.zmq_cmd_socket.send(msgpack.packb(payload, use_bin_type=True), flags=zmq.NOBLOCK)
        return self._sent_action_vector(base_cmd, joint_targets)

    def disconnect(self):
        if not self._is_connected:
            raise DeviceNotConnectedError("XLerobotClient is not connected.")
        try:
            self.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            pass
        self.disconnect_sockets()
        self._is_connected = False

    def disconnect_sockets(self) -> None:
        for attr in ("zmq_camera_socket", "zmq_state_socket", "zmq_cmd_socket"):
            sock = getattr(self, attr)
            if sock is not None:
                sock.close(0)
                setattr(self, attr, None)
        if self.zmq_context is not None:
            self.zmq_context.term()
            self.zmq_context = None

    def _make_push_socket(self, port: int):
        sock = self.zmq_context.socket(zmq.PUSH)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 1)
        sock.setsockopt(zmq.SNDTIMEO, 0)
        try:
            sock.setsockopt(zmq.CONFLATE, 1)
        except zmq.ZMQError:
            pass
        sock.connect(f"tcp://{self.remote_ip}:{port}")
        return sock

    def _make_sub_socket(self, port: int, topics: list[str]):
        sock = self.zmq_context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 64)
        sock.setsockopt(zmq.RCVTIMEO, 0)
        for topic in topics:
            sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        sock.connect(f"tcp://{self.remote_ip}:{port}")
        return sock

    def _rpc(self, op: str) -> dict[str, Any]:
        sock = self.zmq_context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(self.connect_timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(self.connect_timeout_s * 1000))
        try:
            sock.connect(f"tcp://{self.remote_ip}:{self.port_zmq_rpc}")
            sock.send(msgpack.packb({"op": op}, use_bin_type=True))
            reply = msgpack.unpackb(sock.recv(), raw=False)
            return reply if isinstance(reply, dict) else {"ok": False, "error": "bad_rpc_reply"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            sock.close(0)

    def _poll_zmq(self) -> None:
        poller = zmq.Poller()
        poller.register(self.zmq_state_socket, zmq.POLLIN)
        poller.register(self.zmq_camera_socket, zmq.POLLIN)
        events = dict(poller.poll(self.polling_timeout_ms))
        if self.zmq_state_socket in events:
            self._drain_state_socket()
        if self.zmq_camera_socket in events:
            self._drain_camera_socket()

    def _drain_state_socket(self) -> None:
        while True:
            try:
                topic_raw, payload_raw = self.zmq_state_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            topic = topic_raw.decode("utf-8", errors="replace")
            payload = msgpack.unpackb(payload_raw, raw=False)
            if isinstance(payload, dict):
                self.latest_topics[topic] = payload

    def _drain_camera_socket(self) -> None:
        while True:
            try:
                topic_raw, payload_raw = self.zmq_camera_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            topic = topic_raw.decode("utf-8", errors="replace")
            payload = msgpack.unpackb(payload_raw, raw=False)
            if isinstance(payload, dict):
                self._update_camera_frame(topic, payload)

    def _update_camera_frame(self, topic: str, payload: dict[str, Any]) -> None:
        cam_name = self.camera_topic_to_name.get(topic)
        data = payload.get("data")
        encoding = str(payload.get("encoding") or "")
        if cam_name not in self._cameras_ft or not isinstance(data, (bytes, bytearray)):
            return
        if encoding == "jpeg":
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif encoding == "h264_fmp4":
            frame = self._decode_h264_fmp4(topic, payload, bytes(data))
        else:
            if encoding not in self._warned_encodings:
                logging.warning("Skipping %s camera payload with unsupported encoding=%s", topic, encoding)
                self._warned_encodings.add(encoding)
            return
        if frame is not None:
            self.last_frames[cam_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _decode_h264_fmp4(self, topic: str, payload: dict[str, Any], data: bytes) -> np.ndarray | None:
        if av is None:
            if "h264_fmp4" not in self._warned_encodings:
                logging.warning("Skipping h264_fmp4 camera payloads because PyAV is not installed.")
                self._warned_encodings.add("h264_fmp4")
            return None
        init = payload.get("init")
        if isinstance(init, (bytes, bytearray)) and init:
            self._h264_init_by_topic[topic] = bytes(init)
        init_bytes = self._h264_init_by_topic.get(topic, b"")
        if not init_bytes:
            return None
        try:
            with av.open(io.BytesIO(init_bytes + data), mode="r", format="mp4") as container:
                last_frame = None
                for frame in container.decode(video=0):
                    last_frame = frame
                if last_frame is None:
                    return None
                return last_frame.to_ndarray(format="bgr24")
        except Exception as exc:
            if topic not in self._warned_encodings:
                logging.warning("Failed to decode h264_fmp4 camera payload for %s: %s", topic, exc)
                self._warned_encodings.add(topic)
            return None

    def _state_from_latest_topics(self) -> dict[str, Any]:
        state = {key: 0.0 for key in self._state_order}
        proprio = self.latest_topics.get(f"proprio.{self.robot_id}", {})
        joint_pos = proprio.get("joint_pos") if isinstance(proprio, dict) else None
        if isinstance(joint_pos, list) and len(joint_pos) >= 14:
            for name, value in zip(RIGHT_MOTORS, joint_pos[:6], strict=False):
                state[name] = float(value)
            for name, value in zip(LEFT_MOTORS, joint_pos[6:12], strict=False):
                state[name] = float(value)
            state["head_motor_1"] = float(joint_pos[12])
            state["head_motor_2"] = float(joint_pos[13])
        base_vel = proprio.get("base_joint_vel") if isinstance(proprio, dict) else None
        if isinstance(base_vel, list) and len(base_vel) >= 3:
            state["x.vel"], state["y.vel"], state["theta.vel"] = map(float, base_vel[:3])
        state["observation.state"] = np.array([state[k] for k in self._state_order], dtype=np.float32)
        return state

    def _base_cmd_from_action(self, action: dict[str, Any]) -> list[float] | None:
        vals = [float(action.get(k, 0.0) or 0.0) for k in ("x.vel", "y.vel", "theta.vel")]
        return vals if any(abs(v) > 1e-9 for v in vals) else [0.0, 0.0, 0.0]

    def _joint_targets_from_action(self, action: dict[str, Any]) -> list[float | None] | None:
        targets: list[float | None] = [None] * 14
        self._fill_side_targets(action, targets, "left", LEFT_MOTORS, 0)
        self._fill_side_targets(action, targets, "right", RIGHT_MOTORS, 6)
        return targets if any(value is not None for value in targets) else None

    def _fill_side_targets(
        self, action: dict[str, Any], targets: list[float | None], side: str, motors: tuple[str, ...], offset: int
    ) -> None:
        for i, motor in enumerate(motors):
            suffix = motor.removeprefix(f"{side}_arm_")
            keys = (
                f"{motor}.pos",
                f"arm_{side}_{suffix}.pos",
                f"{side}_{suffix}.pos",
                f"arm_{suffix}.pos" if side == "right" else "",
                f"{suffix}.pos" if side == "right" else "",
            )
            for key in keys:
                if key and key in action:
                    try:
                        targets[offset + i] = self._to_raw_tick(motor, float(action[key]))
                    except ValueError as exc:
                        logging.warning("Dropping %s target from %s: %s", motor, key, exc)
                    break

    def _to_raw_tick(self, motor: str, value: float) -> float:
        if value < -100.0 or value > 100.0:
            return float(np.clip(value, 0.0, 4095.0))
        cal = self.follower_calibration.get(motor)
        if not cal:
            logging.warning("No follower calibration for %s; dropping normalized target", motor)
            raise ValueError(f"missing calibration for {motor}")
        min_v, max_v = float(cal["range_min"]), float(cal["range_max"])
        drive_mode = bool(cal.get("drive_mode", 0))
        if motor.endswith("gripper"):
            bounded = np.clip(100.0 - value if drive_mode else value, 0.0, 100.0)
            return float((bounded / 100.0) * (max_v - min_v) + min_v)
        bounded = np.clip(-value if drive_mode else value, -100.0, 100.0)
        return float(((bounded + 100.0) / 200.0) * (max_v - min_v) + min_v)

    def _sent_action_vector(
        self, base_cmd: list[float] | None, joint_targets: list[float | None] | None
    ) -> dict[str, Any]:
        values = {key: 0.0 for key in self._state_order}
        if joint_targets is not None:
            for name, value in zip(LEFT_MOTORS, joint_targets[:6], strict=False):
                if value is not None:
                    values[name] = float(value)
            for name, value in zip(RIGHT_MOTORS, joint_targets[6:12], strict=False):
                if value is not None:
                    values[name] = float(value)
        if base_cmd is not None:
            values["x.vel"], values["y.vel"], values["theta.vel"] = base_cmd
        values["action"] = np.array([values[k] for k in self._state_order], dtype=np.float32)
        return values

    def _load_follower_calibration(self, path: str | None) -> dict[str, dict[str, Any]]:
        candidates = []
        if path:
            candidates.append(Path(path).expanduser())
        candidates.append(HF_LEROBOT_CALIBRATION / ROBOTS / "xlerobot" / f"{self.id}.json")
        for candidate in candidates:
            if candidate.is_file():
                try:
                    data = json.loads(candidate.read_text())
                    return data if isinstance(data, dict) else {}
                except Exception as exc:
                    logging.warning("Failed to load follower calibration %s: %s", candidate, exc)
        return {}

    def _merge_remote_calibration(self, reply: dict[str, Any]) -> None:
        if not reply.get("ok"):
            if not self.follower_calibration:
                logging.warning("No local follower calibration and remote calibration RPC failed: %s", reply)
            return
        calibration = reply.get("calibration")
        joints = calibration.get("joints") if isinstance(calibration, dict) else None
        if not isinstance(joints, dict):
            return
        for name, value in joints.items():
            if isinstance(value, dict) and name not in self.follower_calibration:
                self.follower_calibration[str(name)] = dict(value)
