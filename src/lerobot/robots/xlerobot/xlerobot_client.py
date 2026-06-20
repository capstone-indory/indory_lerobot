# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import contextlib
import json
import logging
import math
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_xlerobot import IndoryFastZMQCameraConfig, XLerobotClientConfig
from .xlerobot_camera_materializer import materialize_camera_archive as materialize_camera_archive_file
from .xlerobot_camera_stream import CameraStreamPump
from .xlerobot_command_builder import XLerobotCommandBuilder
from .xlerobot_constants import HEAD_MOTORS, LEFT_MOTORS, LEGACY_CAMERA_TOPIC_TO_NAME, RIGHT_MOTORS
from .xlerobot_keyboard_control import (
    action_from_pressed_keys,
)
from .xlerobot_leader_kinematics import LeaderKinematicMapper
from .xlerobot_rgbd_decode import RgbdDepthDecoder


class XLerobotClient(Robot):
    """LeRobot Robot wrapper for the indory_server adapter ZMQ endpoint."""

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
        self.port_zmq_rgbd = config.port_zmq_rgbd
        self.robot_id = int(config.robot_id)
        self.source_id = config.source_id
        self.source_role = config.source_role
        self.teleop_keys = config.teleop_keys
        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s
        self.command_lease_ms = int(config.command_lease_ms)
        self.leader_action_units = config.leader_action_units

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_state_socket = None
        self.zmq_rgbd_socket = None
        self.camera_stream: CameraStreamPump | None = None
        self._is_connected = False
        self._seq = 0
        self._warned_encodings: set[str] = set()
        self.rgbd_decoder = RgbdDepthDecoder()
        self.last_depth_frames: dict[str, np.ndarray] = {}
        self.last_rgbd_metadata: dict[str, dict[str, Any]] = {}
        self.last_remote_state: dict[str, Any] = {}
        self.latest_topics: dict[str, dict[str, Any]] = {}
        self.camera_topic_to_name = self._camera_topic_mapping() or {
            f"rgb.front.{self.robot_id}": "head",
            f"rgb.wrist_left.{self.robot_id}": "left_wrist",
            f"rgb.wrist_right.{self.robot_id}": "right_wrist",
            **LEGACY_CAMERA_TOPIC_TO_NAME,
        }
        self.follower_calibration = self._load_follower_calibration(config.follower_calibration_path)
        self.leader_mapper = LeaderKinematicMapper(self.follower_calibration, self.leader_action_units)
        self.command_builder = XLerobotCommandBuilder(
            config,
            self.leader_mapper,
            self.follower_calibration,
            self.latest_topics,
            self._state_order,
        )
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
    def _depth_ft(self) -> dict[str, dict[str, Any]]:
        if not self.config.enable_rgbd:
            return {}
        return {
            str(self.config.rgbd_depth_feature): {
                "dtype": "uint16",
                "shape": (int(self.config.rgbd_depth_height), int(self.config.rgbd_depth_width)),
                "names": ["height", "width"],
            }
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple | dict]:
        return {**self._state_ft, **self._cameras_ft, **self._depth_ft}

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
        self.camera_stream = CameraStreamPump(
            remote_ip=self.remote_ip,
            port=self.port_zmq_cameras,
            topics=camera_topics,
            topic_to_name=self.camera_topic_to_name,
            camera_names=set(self._cameras_ft),
        )
        self.camera_stream.start()
        if self.config.enable_rgbd:
            self.zmq_rgbd_socket = self._make_sub_socket(self.port_zmq_rgbd, [self.config.rgbd_topic])
        health = self._rpc("health")
        if not health.get("ok"):
            self.disconnect_sockets()
            raise DeviceNotConnectedError(f"indory_server adapter health check failed: {health}")
        self._merge_remote_calibration(self._rpc("calibration"))
        self._is_connected = True
        self._warm_up_cameras()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def start_camera_archive(self, root: Path | str, episode_index: int) -> Path | None:
        if self.camera_stream is None:
            return None
        return self.camera_stream.start_archive(root, episode_index)

    def stop_camera_archive(self, *, keep: bool = True) -> Path | None:
        if self.camera_stream is None:
            return None
        return self.camera_stream.stop_archive(keep=keep)

    def materialize_camera_archive(self, dataset: Any, archive_path: Path | str | None) -> dict[str, int]:
        return materialize_camera_archive_file(dataset, archive_path)

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("XLerobotClient is not connected.")
        self._poll_zmq()
        obs_dict = self._state_from_latest_topics()
        latest_frames = self.camera_stream.frames() if self.camera_stream is not None else {}
        for cam_name, shape in self._cameras_ft.items():
            frame = latest_frames.get(cam_name)
            obs_dict[cam_name] = frame if frame is not None else np.zeros(shape, dtype=np.uint8)
        for name, ft in self._depth_ft.items():
            depth = self.last_depth_frames.get(name)
            obs_dict[name] = depth if depth is not None else np.zeros(ft["shape"], dtype=np.uint16)
        self.last_remote_state = obs_dict
        return obs_dict

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        action, self.speed_index = action_from_pressed_keys(
            pressed_keys,
            self.teleop_keys,
            self.speed_levels,
            self.speed_index,
            self.config.head_step_rad,
            self.config.head_pan_sign,
            self.config.head_tilt_sign,
        )
        return action

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected or self.zmq_cmd_socket is None:
            raise DeviceNotConnectedError("XLerobotClient is not connected.")
        result = self.command_builder.build(
            action,
            seq=self._seq,
            source_id=self.source_id,
            source_role=self.source_role,
            lease_ms=self.command_lease_ms,
            robot_id=self.robot_id,
        )
        self._seq += 1
        if self.command_builder.is_material_command(result.payload):
            self.zmq_cmd_socket.send(msgpack.packb(result.payload, use_bin_type=True), flags=zmq.NOBLOCK)
        return result.sent_action

    def disconnect(self):
        if not self._is_connected:
            raise DeviceNotConnectedError("XLerobotClient is not connected.")
        with contextlib.suppress(Exception):
            self.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        self.disconnect_sockets()
        self._is_connected = False

    def disconnect_sockets(self) -> None:
        if self.camera_stream is not None:
            self.camera_stream.stop()
            self.camera_stream = None
        for attr in ("zmq_rgbd_socket", "zmq_state_socket", "zmq_cmd_socket"):
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
        with contextlib.suppress(zmq.ZMQError):
            sock.setsockopt(zmq.CONFLATE, 1)
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
        if self.zmq_rgbd_socket is not None:
            poller.register(self.zmq_rgbd_socket, zmq.POLLIN)
        events = dict(poller.poll(self.polling_timeout_ms))
        if self.zmq_state_socket in events:
            self._drain_state_socket()
        if self.zmq_rgbd_socket is not None and self.zmq_rgbd_socket in events:
            self._drain_rgbd_socket()

    def _warm_up_cameras(self) -> None:
        if not self._cameras_ft:
            return
        if self.camera_stream is None:
            return
        missing = self.camera_stream.warm_up(min(3.0, max(0.5, float(self.connect_timeout_s))))
        if missing:
            logging.warning("Camera warm-up incomplete; missing initial frames for %s", ", ".join(missing))

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

    def _drain_rgbd_socket(self) -> None:
        latest_payload: tuple[str, dict[str, Any]] | None = None
        while True:
            try:
                topic_raw, payload_raw = self.zmq_rgbd_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            topic = topic_raw.decode("utf-8", errors="replace")
            payload = msgpack.unpackb(payload_raw, raw=False)
            if isinstance(payload, dict):
                latest_payload = (topic, payload)
        if latest_payload is not None:
            self._update_rgbd_frame(*latest_payload)

    def _update_rgbd_frame(self, topic: str, payload: dict[str, Any]) -> None:
        if topic != self.config.rgbd_topic:
            return
        feature_name = str(self.config.rgbd_depth_feature)
        depth = self.rgbd_decoder.decode_depth(payload, self._warned_encodings)
        if depth is None:
            return
        target_shape = self._depth_ft.get(feature_name, {}).get("shape")
        if target_shape and depth.shape != tuple(target_shape):
            depth = cv2.resize(
                depth,
                (int(target_shape[1]), int(target_shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
        self.last_depth_frames[feature_name] = np.ascontiguousarray(depth.astype(np.uint16, copy=False))
        self.last_rgbd_metadata[feature_name] = {
            "stamp_ns": payload.get("stamp_ns"),
            "depth_format": payload.get("depth_format"),
            "depth_units": payload.get("depth_units"),
            "aligned_depth_to_color": payload.get("aligned_depth_to_color"),
        }

    def _camera_topic_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for name, cfg in self.config.cameras.items():
            if isinstance(cfg, IndoryFastZMQCameraConfig):
                mapping[str(cfg.topic)] = str(name)
        if mapping:
            mapping.update(LEGACY_CAMERA_TOPIC_TO_NAME)
        return mapping

    def _state_from_latest_topics(self) -> dict[str, Any]:
        state = dict.fromkeys(self._state_order, 0.0)
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
            if isinstance(value, dict):
                self.follower_calibration[str(name)] = dict(value)
