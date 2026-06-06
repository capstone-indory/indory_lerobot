#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import os
import time

import cv2
import zmq

from .xlerobot import XLerobot
from .config_xlerobot import XLerobotConfig, XLerobotHostConfig


WEB_LINEAR_SPEED_MPS = float(os.getenv("XLEROBOT_WEB_LINEAR_SPEED_MPS", "0.2"))
WEB_STRAFE_SPEED_MPS = float(os.getenv("XLEROBOT_WEB_STRAFE_SPEED_MPS", "0.2"))
WEB_YAW_SPEED_DEGPS = float(os.getenv("XLEROBOT_WEB_YAW_SPEED_DEGPS", "60.0"))

LEFT_ARM_JOINTS = (
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
    "left_arm_gripper",
)
RIGHT_ARM_JOINTS = (
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
)


class XLerobotHost:
    def __init__(self, config: XLerobotHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz
        self.web_protocol_active = False

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _response(response_type: str, data: dict | None = None) -> dict:
    return {
        "type": "response",
        "response": response_type,
        "data": data or {},
        "timestamp": time.time(),
    }


def _send_payload(host: XLerobotHost, payload: dict) -> None:
    try:
        host.zmq_observation_socket.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
    except zmq.Again:
        logging.info("Dropping observation, no client connected")


def _is_web_control_command(message: dict) -> bool:
    return message.get("type") == "command" or "command" in message


def _clamp_speed(speed: object) -> float:
    try:
        return max(0.0, min(1.0, float(speed)))
    except (TypeError, ValueError):
        return 0.0


def _web_move_to_action(direction: str, speed: object) -> dict[str, float]:
    clamped_speed = _clamp_speed(speed)
    action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    if direction == "forward":
        action["x.vel"] = WEB_LINEAR_SPEED_MPS * clamped_speed
    elif direction == "backward":
        action["x.vel"] = -WEB_LINEAR_SPEED_MPS * clamped_speed
    elif direction == "left":
        action["y.vel"] = WEB_STRAFE_SPEED_MPS * clamped_speed
    elif direction == "right":
        action["y.vel"] = -WEB_STRAFE_SPEED_MPS * clamped_speed
    elif direction == "rotate_left":
        action["theta.vel"] = WEB_YAW_SPEED_DEGPS * clamped_speed
    elif direction == "rotate_right":
        action["theta.vel"] = -WEB_YAW_SPEED_DEGPS * clamped_speed
    elif direction in ("stop", None):
        pass
    else:
        logging.warning("Unknown web_control move direction: %s", direction)

    return action


def _state_response_from_observation(observation: dict) -> dict:
    def joint_values(names: tuple[str, ...]) -> list[float]:
        return [float(observation.get(f"{name}.pos", 0.0)) for name in names]

    x_vel = float(observation.get("x.vel", 0.0))
    y_vel = float(observation.get("y.vel", 0.0))
    theta_vel = float(observation.get("theta.vel", 0.0))
    state = {
        "status": "connected",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "arm_joints": {
            "left": joint_values(LEFT_ARM_JOINTS),
            "right": joint_values(RIGHT_ARM_JOINTS),
        },
        "base_joints": [x_vel, y_vel, theta_vel],
        "velocity": {
            "linear": {"x": x_vel, "y": y_vel, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": theta_vel},
        },
    }
    return _response("state", state)


def _video_response_from_observation(observation: dict, camera_shapes: dict[str, tuple[int, int]]) -> dict | None:
    for camera_id, (width, height) in camera_shapes.items():
        frame = observation.get(camera_id)
        if not frame:
            continue
        return _response(
            "video",
            {
                "frame": frame,
                "width": width,
                "height": height,
                "quality": 90,
                "camera_id": camera_id,
                "format": "jpeg",
            },
        )
    return None


def _encode_camera_observations(robot: XLerobot, observation: dict) -> dict[str, tuple[int, int]]:
    camera_shapes = {}
    for cam_key, _ in robot.cameras.items():
        frame = observation[cam_key]
        height, width = frame.shape[:2]
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        observation[cam_key] = base64.b64encode(buffer).decode("utf-8") if ret else ""
        camera_shapes[cam_key] = (width, height)
    return camera_shapes


def _handle_web_control_command(host: XLerobotHost, robot: XLerobot, message: dict) -> bool:
    command = message.get("command")
    data = message.get("data") or {}

    if command == "ping":
        _send_payload(host, _response("pong", {"timestamp": time.time()}))
        return False

    if command == "move":
        action = _web_move_to_action(data.get("direction", "stop"), data.get("speed", 1.0))
        robot.send_action(action)
        return True

    if command == "stop":
        robot.stop_base()
        return True

    if command == "reset":
        robot.stop_base()
        _send_payload(host, _response("success", {"message": "Base stopped"}))
        return True

    if command == "get_state":
        observation = robot.get_observation()
        _encode_camera_observations(robot, observation)
        _send_payload(host, _state_response_from_observation(observation))
        return False

    logging.warning("Unsupported web_control command: %s", command)
    _send_payload(host, _response("error", {"message": f"Unsupported command: {command}"}))
    return False


def main():
    logging.info("Configuring Xlerobot")
    robot_config = XLerobotConfig(id="my_xlerobot_pc")
    robot = XLerobot(robot_config)
    right_arm_only = os.getenv("XLEROBOT_RIGHT_ARM_ONLY", "0").lower() in ("1", "true", "yes", "on")
    if right_arm_only:
        logging.warning("XLEROBOT_RIGHT_ARM_ONLY=1: bus2 will use right arm motors only; base 7/8/9 are disabled")
        robot.bus2.motors = {name: motor for name, motor in robot.bus2.motors.items() if name.startswith("right_arm")}
        robot.bus2.ids = [motor.id for motor in robot.bus2.motors.values()]
        robot.bus2.models = [motor.model for motor in robot.bus2.motors.values()]
        robot.bus2._id_to_name_dict = {motor.id: name for name, motor in robot.bus2.motors.items()}
        robot.bus2._id_to_model_dict = {motor.id: motor.model for motor in robot.bus2.motors.values()}
        robot.base_motors = []

    logging.info("Connecting Xlerobot")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = XLerobotHostConfig()
    host = XLerobotHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            sent_web_response = False
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                if _is_web_control_command(data):
                    host.web_protocol_active = True
                    movement_command = _handle_web_control_command(host, robot, data)
                    sent_web_response = not movement_command
                    if movement_command:
                        last_cmd_time = time.time()
                        watchdog_active = False
                else:
                    _action_sent = robot.send_action(data)
                    last_cmd_time = time.time()
                    watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            last_observation = robot.get_observation()
            camera_shapes = _encode_camera_observations(robot, last_observation)

            # Send the observation to the remote agent
            if host.web_protocol_active:
                if not sent_web_response:
                    _send_payload(host, _state_response_from_observation(last_observation))
                video_response = _video_response_from_observation(last_observation, camera_shapes)
                if video_response is not None:
                    _send_payload(host, video_response)
            else:
                _send_payload(host, last_observation)

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
