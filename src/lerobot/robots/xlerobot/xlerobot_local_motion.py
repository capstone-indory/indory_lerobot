from __future__ import annotations

import logging
from itertools import chain
from typing import Any

import numpy as np

from lerobot.motors import MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.errors import DeviceNotConnectedError

from ..utils import ensure_safe_goal_position

logger = logging.getLogger(__name__)
CALIBRATION_CLAMP_EXEMPT_MOTOR_PARTS = ("wrist_flex", "wrist_roll")


def keyboard_to_base_action(robot: Any, pressed_keys: np.ndarray) -> dict[str, float]:
    if robot.teleop_keys["speed_up"] in pressed_keys:
        robot.speed_index = min(robot.speed_index + 1, len(robot.speed_levels) - 1)
    if robot.teleop_keys["speed_down"] in pressed_keys:
        robot.speed_index = max(robot.speed_index - 1, 0)
    speed = robot.speed_levels[robot.speed_index]
    x_cmd = y_cmd = theta_cmd = 0.0
    if robot.teleop_keys["forward"] in pressed_keys:
        x_cmd += speed["xy"]
    if robot.teleop_keys["backward"] in pressed_keys:
        x_cmd -= speed["xy"]
    if robot.teleop_keys["left"] in pressed_keys:
        y_cmd += speed["xy"]
    if robot.teleop_keys["right"] in pressed_keys:
        y_cmd -= speed["xy"]
    if robot.teleop_keys["rotate_left"] in pressed_keys:
        theta_cmd += speed["theta"]
    if robot.teleop_keys["rotate_right"] in pressed_keys:
        theta_cmd -= speed["theta"]
    return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}


def send_robot_action(robot: Any, action: dict[str, Any]) -> dict[str, Any]:
    if not robot.is_connected:
        raise DeviceNotConnectedError(f"{robot} is not connected.")
    left_arm_pos = _select(action, "left_arm_", ".pos")
    right_arm_pos = _select(action, "right_arm_", ".pos")
    head_pos = _select(action, "head_", ".pos")
    base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
    base_wheel_goal_vel = (
        body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )
        if robot.base_motors
        else {}
    )
    if robot.config.max_relative_target is not None:
        left_arm_pos, right_arm_pos, head_pos = _safe_position_targets(
            robot, left_arm_pos, right_arm_pos, head_pos
        )
    left_raw, left_clamped = _prepare_position_targets(robot.bus1, left_arm_pos)
    right_raw, right_clamped = _prepare_position_targets(robot.bus2, right_arm_pos)
    head_raw, head_clamped = _prepare_position_targets(robot.bus1, head_pos)
    left_roll_norm, left_roll_raw = _pop_periodic_raw_goal(robot.bus1, left_raw, "left_arm_wrist_roll")
    right_roll_norm, right_roll_raw = _pop_periodic_raw_goal(robot.bus2, right_raw, "right_arm_wrist_roll")
    _record_clamp_logs(robot, left_clamped, right_clamped, head_clamped)
    _write_goal_positions(robot.bus1, left_raw, "left_arm_wrist_roll", left_roll_raw)
    _write_goal_positions(robot.bus2, right_raw, "right_arm_wrist_roll", right_roll_raw)
    if head_raw:
        robot.bus1.sync_write("Goal_Position", head_raw)
    if base_wheel_goal_vel:
        robot.bus2.sync_write("Goal_Velocity", base_wheel_goal_vel)
    return {
        **{f"{motor}.pos": value for motor, value in left_raw.items()},
        **({"left_arm_wrist_roll.pos": left_roll_norm} if left_roll_raw is not None else {}),
        **{f"{motor}.pos": value for motor, value in right_raw.items()},
        **({"right_arm_wrist_roll.pos": right_roll_norm} if right_roll_raw is not None else {}),
        **{f"{motor}.pos": value for motor, value in head_raw.items()},
        **base_goal_vel,
    }


def body_to_wheel_raw(
    x: float,
    y: float,
    theta: float,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
    max_raw: int = 3000,
) -> dict[str, int]:
    velocity_vector = np.array([x, y, theta * (np.pi / 180.0)])
    angles = np.radians(np.array([240, 0, 120]) - 90)
    wheel_linear_speeds = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles]).dot(
        velocity_vector
    )
    wheel_degps = (wheel_linear_speeds / wheel_radius) * (180.0 / np.pi)
    raw_floats = [abs(degps) * 4096.0 / 360.0 for degps in wheel_degps]
    if max(raw_floats) > max_raw:
        wheel_degps = wheel_degps * (max_raw / max(raw_floats))
    wheel_raw = [_degps_to_raw(deg) for deg in wheel_degps]
    return {
        "base_left_wheel": wheel_raw[0],
        "base_back_wheel": wheel_raw[1],
        "base_right_wheel": wheel_raw[2],
    }


def wheel_raw_to_body(
    left_wheel_speed: int,
    back_wheel_speed: int,
    right_wheel_speed: int,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
) -> dict[str, float]:
    wheel_degps = np.array([
        _raw_to_degps(left_wheel_speed),
        _raw_to_degps(back_wheel_speed),
        _raw_to_degps(right_wheel_speed),
    ])
    wheel_linear_speeds = wheel_degps * (np.pi / 180.0) * wheel_radius
    angles = np.radians(np.array([240, 0, 120]) - 90)
    matrix = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
    x, y, theta_rad = np.linalg.inv(matrix).dot(wheel_linear_speeds)
    return {"x.vel": x, "y.vel": y, "theta.vel": theta_rad * (180.0 / np.pi)}


def _select(action: dict[str, Any], prefix: str, suffix: str) -> dict[str, Any]:
    return {k: v for k, v in action.items() if k.startswith(prefix) and k.endswith(suffix)}


def _safe_position_targets(
    robot: Any,
    left_arm_pos: dict[str, Any],
    right_arm_pos: dict[str, Any],
    head_pos: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    present_pos = {
        **robot.bus1.sync_read("Present_Position", robot.left_arm_motors),
        **robot.bus2.sync_read("Present_Position", robot.right_arm_motors),
        **robot.bus1.sync_read("Present_Position", robot.head_motors),
    }
    goal_present_pos = {
        key: (g_pos, present_pos[key.replace(".pos", "")])
        for key, g_pos in chain(left_arm_pos.items(), right_arm_pos.items(), head_pos.items())
    }
    safe_goal_pos = ensure_safe_goal_position(goal_present_pos, robot.config.max_relative_target)
    return (
        {k: v for k, v in safe_goal_pos.items() if k in left_arm_pos},
        {k: v for k, v in safe_goal_pos.items() if k in right_arm_pos},
        {k: v for k, v in safe_goal_pos.items() if k in head_pos},
    )


def _prepare_position_targets(
    bus: FeetechMotorsBus, action_targets: dict[str, Any]
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    raw = {k.replace(".pos", ""): v for k, v in action_targets.items()}
    return _clamp_position_targets_to_calibration(bus, raw)


def _write_goal_positions(
    bus: FeetechMotorsBus, targets: dict[str, float], periodic_motor: str, periodic_raw: int | None
) -> None:
    if targets:
        bus.sync_write("Goal_Position", targets)
    if periodic_raw is not None:
        bus.sync_write("Goal_Position", {periodic_motor: periodic_raw}, normalize=False)


def _record_clamp_logs(
    robot: Any,
    left_clamped: dict[str, tuple[float, float]],
    right_clamped: dict[str, tuple[float, float]],
    head_clamped: dict[str, tuple[float, float]],
) -> None:
    if left_clamped or right_clamped or head_clamped:
        robot.logs["clamped_position_targets"] = {**left_clamped, **right_clamped, **head_clamped}
    else:
        robot.logs.pop("clamped_position_targets", None)


def _clamp_position_targets_to_calibration(
    bus: FeetechMotorsBus, targets: dict[str, Any]
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    clamped: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for motor, value in targets.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            logger.warning("Dropping non-numeric position target for '%s': %r", motor, value)
            continue
        if np.isnan(numeric_value):
            logger.warning("Dropping NaN position target for '%s'.", motor)
            continue
        if any(part in motor for part in CALIBRATION_CLAMP_EXEMPT_MOTOR_PARTS):
            clamped[motor] = numeric_value
            continue
        low, high = _position_bounds_from_calibration(bus, motor)
        clamped_value = float(np.clip(numeric_value, low, high))
        clamped[motor] = clamped_value
        if clamped_value != numeric_value:
            ranges[motor] = (low, high)
    return clamped, ranges


def _position_bounds_from_calibration(bus: FeetechMotorsBus, motor: str) -> tuple[float, float]:
    norm_mode = bus.motors[motor].norm_mode
    if norm_mode is MotorNormMode.RANGE_M100_100:
        return -100.0, 100.0
    if norm_mode is MotorNormMode.RANGE_0_100:
        return 0.0, 100.0
    if norm_mode is MotorNormMode.DEGREES:
        calibration = bus.calibration[motor]
        mid = (calibration.range_min + calibration.range_max) / 2
        max_res = bus.model_resolution_table[bus._id_to_model(bus.motors[motor].id)] - 1
        return (
            (calibration.range_min - mid) * 360 / max_res,
            (calibration.range_max - mid) * 360 / max_res,
        )
    raise NotImplementedError(f"Unsupported motor norm mode for '{motor}': {norm_mode}")


def _pop_periodic_raw_goal(
    bus: FeetechMotorsBus, targets: dict[str, float], motor: str
) -> tuple[float | None, int | None]:
    norm_goal = targets.pop(motor, None)
    if norm_goal is None:
        return None, None
    motor_id = bus.motors[motor].id
    period = bus.model_resolution_table[bus.motors[motor].model]
    try:
        present_raw = bus.read("Present_Position", motor, normalize=False, num_retry=2)
    except Exception as exc:
        logger.warning("Skipping '%s' command; failed to read current raw position: %s", motor, exc)
        return float(norm_goal), None
    nominal_goal_raw = bus._unnormalize({motor_id: float(norm_goal)})[motor_id]
    raw_goal = _nearest_periodic_raw_goal(int(present_raw), int(nominal_goal_raw), period)
    return float(norm_goal), raw_goal


def _nearest_periodic_raw_goal(present: int, goal: int, period: int) -> int:
    if period <= 0:
        raise ValueError("period must be positive")
    offset = round((present - goal) / period)
    return min((goal + (offset + delta) * period for delta in (-1, 0, 1)), key=lambda c: abs(c - present))


def _degps_to_raw(degps: float) -> int:
    speed_int = int(round(degps * 4096.0 / 360.0))
    return int(np.clip(speed_int, -0x8000, 0x7FFF))


def _raw_to_degps(raw_speed: int) -> float:
    return float(raw_speed) / (4096.0 / 360.0)
