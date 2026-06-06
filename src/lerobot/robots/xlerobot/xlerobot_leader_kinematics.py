from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np


FEETECH_TICKS_PER_REV = 4095.0
CALIBRATION_CLAMP_EXEMPT_MOTOR_PARTS = ("wrist_flex", "wrist_roll")
JOINT_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class LeaderKinematicMapper:
    """Convert calibrated leader-arm actions into follower Feetech tick targets."""

    def __init__(self, follower_calibration: dict[str, dict[str, Any]], action_units: str):
        self.follower_calibration = follower_calibration
        self.action_units = action_units

    def targets_for_side(
        self,
        action: dict[str, Any],
        side: str,
        motors: tuple[str, ...],
    ) -> list[float | None]:
        values = self._leader_values(action, side, motors)
        if not values:
            return [None] * len(motors)
        if self.action_units == "normalized":
            return [
                self._normalized_to_raw(motor, values[suffix]) if suffix in values else None
                for motor, suffix in zip(motors, JOINT_SUFFIXES, strict=True)
            ]
        if self.action_units != "degrees":
            raise ValueError(f"unsupported leader_action_units={self.action_units!r}")
        degree_targets = dict(values)
        return [
            self._value_to_raw(motor, suffix, degree_targets[suffix]) if suffix in degree_targets else None
            for motor, suffix in zip(motors, JOINT_SUFFIXES, strict=True)
        ]

    def _leader_values(
        self,
        action: dict[str, Any],
        side: str,
        motors: tuple[str, ...],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for motor, suffix in zip(motors, JOINT_SUFFIXES, strict=True):
            for key in self._action_keys(side, motor, suffix):
                if key in action:
                    try:
                        out[suffix] = float(action[key])
                    except (TypeError, ValueError) as exc:
                        logging.warning("Dropping %s target from %s: %s", motor, key, exc)
                    break
        return out

    @staticmethod
    def _action_keys(side: str, motor: str, suffix: str) -> tuple[str, ...]:
        right_legacy = (f"arm_{suffix}.pos", f"{suffix}.pos") if side == "right" else ()
        return (
            f"{motor}.pos",
            f"arm_{side}_{suffix}.pos",
            f"{side}_{suffix}.pos",
            *right_legacy,
        )

    def _value_to_raw(self, motor: str, suffix: str, value: float) -> float:
        if suffix == "gripper":
            return self._normalized_to_raw(motor, value)
        return self._degree_to_raw(motor, value)

    def _degree_to_raw(self, motor: str, degrees: float) -> float:
        cal = self.follower_calibration.get(motor)
        if not cal:
            logging.warning("No follower calibration for %s; dropping degree target", motor)
            raise ValueError(f"missing calibration for {motor}")
        min_v, max_v = self._range(cal)
        center = (min_v + max_v) * 0.5
        raw = center + (float(degrees) * FEETECH_TICKS_PER_REV / 360.0)
        if _calibration_clamp_exempt(motor):
            return float(np.clip(raw, 0.0, FEETECH_TICKS_PER_REV))
        return float(np.clip(raw, min_v, max_v))

    def _normalized_to_raw(self, motor: str, value: float) -> float:
        if value < -100.0 or value > 100.0:
            return float(np.clip(value, 0.0, FEETECH_TICKS_PER_REV))
        cal = self.follower_calibration.get(motor)
        if not cal:
            logging.warning("No follower calibration for %s; dropping normalized target", motor)
            raise ValueError(f"missing calibration for {motor}")
        min_v, max_v = self._range(cal)
        drive_mode = bool(cal.get("drive_mode", 0))
        if motor.endswith("gripper"):
            bounded = np.clip(100.0 - value if drive_mode else value, 0.0, 100.0)
            return float((bounded / 100.0) * (max_v - min_v) + min_v)
        bounded = np.clip(-value if drive_mode else value, -100.0, 100.0)
        return float(((bounded + 100.0) / 200.0) * (max_v - min_v) + min_v)

    @staticmethod
    def _range(cal: dict[str, Any]) -> tuple[float, float]:
        min_v, max_v = float(cal["range_min"]), float(cal["range_max"])
        if not math.isfinite(min_v) or not math.isfinite(max_v) or max_v <= min_v:
            raise ValueError("invalid calibration range")
        return min_v, max_v


def cap_raw_targets_to_current(
    targets: list[float],
    current: list[float],
    motors: tuple[str, ...],
    calibration: dict[str, dict[str, Any]],
    max_relative_target: float | dict[str, float] | None,
    action_units: str,
) -> list[float]:
    if max_relative_target is None:
        return targets
    capped = list(targets)
    for index, (target, present, motor) in enumerate(zip(targets, current, motors, strict=False)):
        limit = _raw_delta_limit(motor, calibration, max_relative_target, action_units)
        if limit is None:
            continue
        capped[index] = float(np.clip(float(target), float(present) - limit, float(present) + limit))
    return capped


def _raw_delta_limit(
    motor: str,
    calibration: dict[str, dict[str, Any]],
    max_relative_target: float | dict[str, float],
    action_units: str,
) -> float | None:
    if isinstance(max_relative_target, dict):
        raw_cap = max_relative_target.get(motor, max_relative_target.get(f"{motor}.pos"))
        if raw_cap is None:
            return None
    else:
        raw_cap = max_relative_target
    cap = abs(float(raw_cap))
    if not math.isfinite(cap):
        return None
    if motor.endswith("gripper"):
        cal = calibration.get(motor)
        if not cal:
            return None
        min_v, max_v = LeaderKinematicMapper._range(cal)
        return (max_v - min_v) * cap / 100.0
    if action_units == "degrees":
        return cap * FEETECH_TICKS_PER_REV / 360.0
    cal = calibration.get(motor)
    if not cal:
        return None
    min_v, max_v = LeaderKinematicMapper._range(cal)
    return (max_v - min_v) * cap / 200.0


def _calibration_clamp_exempt(motor: str) -> bool:
    return any(part in motor for part in CALIBRATION_CLAMP_EXEMPT_MOTOR_PARTS)
