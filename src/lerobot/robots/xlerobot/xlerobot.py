#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_xlerobot import XLerobotConfig
from .xlerobot_bus_setup import (
    calibrate_robot,
    configure_robot,
    make_motor_buses,
    motor_groups,
    restore_or_calibrate,
    setup_robot_motors,
)
from .xlerobot_constants import HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS
from .xlerobot_local_motion import (
    body_to_wheel_raw,
    keyboard_to_base_action,
    send_robot_action,
    wheel_raw_to_body,
)

logger = logging.getLogger(__name__)


class XLerobot(Robot):
    """
    Local hardware wrapper for XLeRobot.

    The class owns lifecycle and observation assembly. Bus setup, calibration,
    base kinematics, and action writes live in helper modules so this file stays
    readable and close to the LeRobot Robot API surface.
    """

    config_class = XLerobotConfig
    name = "xlerobot"

    def __init__(self, config: XLerobotConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        self.logs = {}
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        self.speed_index = 0
        self.bus1, self.bus2 = make_motor_buses(self.config, self.calibration)
        (
            self.left_arm_motors,
            self.right_arm_motors,
            self.head_motors,
            self.base_motors,
        ) = motor_groups(self.bus1, self.bus2)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                *(f"{motor}.pos" for motor in LEFT_MOTORS),
                *(f"{motor}.pos" for motor in RIGHT_MOTORS),
                *(f"{motor}.pos" for motor in HEAD_MOTORS),
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus1.is_connected and self.bus2.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return self.bus1.is_calibrated and self.bus2.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bus1.connect()
        self.bus2.connect()
        restore_or_calibrate(self, calibrate)
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        calibrate_robot(self)

    def configure(self):
        configure_robot(self)

    def setup_motors(self) -> None:
        setup_robot_motors(self)

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        return int(np.clip(int(round(degps * 4096.0 / 360.0)), -0x8000, 0x7FFF))

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        return float(raw_speed) / (4096.0 / 360.0)

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict[str, int]:
        return body_to_wheel_raw(x, y, theta, wheel_radius, base_radius, max_raw)

    def _wheel_raw_to_body(
        self,
        left_wheel_speed: int,
        back_wheel_speed: int,
        right_wheel_speed: int,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        return wheel_raw_to_body(
            left_wheel_speed,
            back_wheel_speed,
            right_wheel_speed,
            wheel_radius,
            base_radius,
        )

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        return keyboard_to_base_action(self, pressed_keys)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        start = time.perf_counter()
        left_arm_pos = self.bus1.sync_read("Present_Position", self.left_arm_motors)
        right_arm_pos = self.bus2.sync_read("Present_Position", self.right_arm_motors, num_retry=3)
        head_pos = self.bus1.sync_read("Present_Position", self.head_motors)
        if self.base_motors:
            base_wheel_vel = self.bus2.sync_read("Present_Velocity", self.base_motors)
            base_vel = self._wheel_raw_to_body(
                base_wheel_vel["base_left_wheel"],
                base_wheel_vel["base_back_wheel"],
                base_wheel_vel["base_right_wheel"],
            )
        else:
            base_vel = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        logger.debug("%s read state: %.1fms", self, (time.perf_counter() - start) * 1e3)
        return {
            **{f"{k}.pos": v for k, v in left_arm_pos.items()},
            **{f"{k}.pos": v for k, v in right_arm_pos.items()},
            **{f"{k}.pos": v for k, v in head_pos.items()},
            **base_vel,
            **self.get_camera_observation(),
        }

    def get_camera_observation(self):
        obs_dict = {}
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            logger.debug("%s read %s: %.1fms", self, cam_key, (time.perf_counter() - start) * 1e3)
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return send_robot_action(self, action)

    def stop_base(self):
        if not self.base_motors:
            logger.info("Base motors skipped: no base motors configured")
            return
        self.bus2.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.stop_base()
        self.bus1.disconnect(self.config.disable_torque_on_disconnect)
        self.bus2.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)
