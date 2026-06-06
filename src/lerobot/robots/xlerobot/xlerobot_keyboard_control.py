from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

FEETECH_TICK_MAX = 4095.0


def action_from_pressed_keys(
    pressed_keys: Any,
    teleop_keys: dict[str, str],
    speed_levels: list[dict[str, float]],
    speed_index: int,
    head_step_rad: float,
    head_pan_sign: float,
    head_tilt_sign: float,
) -> tuple[dict[str, Any], int]:
    keys = _pressed_set(pressed_keys)
    if _key(teleop_keys, "speed_up") in keys:
        speed_index = min(speed_index + 1, len(speed_levels) - 1)
    if _key(teleop_keys, "speed_down") in keys:
        speed_index = max(speed_index - 1, 0)
    speed = speed_levels[speed_index]
    action: dict[str, Any] = {
        "x.vel": _axis(keys, teleop_keys, "forward", "backward") * speed["xy"],
        "y.vel": _axis(keys, teleop_keys, "left", "right") * speed["xy"],
        "theta.vel": _axis(keys, teleop_keys, "rotate_left", "rotate_right") * speed["theta"],
    }
    pan = _axis(keys, teleop_keys, "head_right", "head_left") * head_step_rad * head_pan_sign
    tilt = _axis(keys, teleop_keys, "head_up", "head_down") * head_step_rad * head_tilt_sign
    if abs(pan) > 1e-9:
        action["head.pan"] = pan
    if abs(tilt) > 1e-9:
        action["head.tilt"] = tilt
    if _key(teleop_keys, "head_recenter") in keys:
        action["head.recenter"] = True
    if _key(teleop_keys, "arm_recenter") in keys:
        action["arm.recenter"] = True
    return action, speed_index


def head_relative_target_from_action(
    action: dict[str, Any],
    latest_topics: dict[str, dict[str, Any]],
    robot_id: int,
) -> dict[str, float] | None:
    if action.get("head.recenter"):
        target = _head_recenter_delta(latest_topics, robot_id)
        if target is None:
            logging.warning("Cannot recenter head before receiving proprio.%s joint_pos_urdf_rad.", robot_id)
        return target
    pan = _finite_float(action.get("head.pan", 0.0))
    tilt = _finite_float(action.get("head.tilt", 0.0))
    if abs(pan) <= 1e-9 and abs(tilt) <= 1e-9:
        return None
    return {"head_pan": pan, "head_tilt": tilt}


def arm_recenter_offsets(
    action: dict[str, Any],
    current_canonical: list[float] | None,
    mapper: Any,
    left_motors: tuple[str, ...],
    right_motors: tuple[str, ...],
) -> dict[str, float]:
    if current_canonical is None:
        logging.warning("Cannot recenter arms before receiving proprio joint positions.")
        return {}
    offsets: dict[str, float] = {}
    for side, motors, start in (("right", right_motors, 0), ("left", left_motors, 6)):
        try:
            targets = mapper.targets_for_side(action, side, motors)
        except ValueError as exc:
            logging.warning("Cannot recenter %s arm: %s", side, exc)
            continue
        for index, target in enumerate(targets):
            if target is not None:
                offsets[motors[index]] = float(current_canonical[start + index]) - float(target)
    return offsets


def bounded_raw_target(motor: str, value: float, calibration: dict[str, Any] | None) -> float:
    lower, upper = 0.0, FEETECH_TICK_MAX
    if calibration and not any(part in motor for part in ("wrist_flex", "wrist_roll")):
        lower = _finite_float(calibration.get("range_min"), lower)
        upper = _finite_float(calibration.get("range_max"), upper)
    if upper <= lower:
        lower, upper = 0.0, FEETECH_TICK_MAX
    return float(np.clip(float(value), lower, upper))


def _pressed_set(pressed_keys: Any) -> set[str]:
    items = pressed_keys.keys() if isinstance(pressed_keys, dict) else pressed_keys
    try:
        return {str(key).lower() for key in items if isinstance(key, str)}
    except TypeError:
        return set()


def _axis(keys: set[str], teleop_keys: dict[str, str], positive: str, negative: str) -> float:
    return float(_key(teleop_keys, positive) in keys) - float(_key(teleop_keys, negative) in keys)


def _key(teleop_keys: dict[str, str], name: str) -> str:
    return str(teleop_keys.get(name, "")).lower()


def _head_recenter_delta(
    latest_topics: dict[str, dict[str, Any]],
    robot_id: int,
) -> dict[str, float] | None:
    proprio = latest_topics.get(f"proprio.{robot_id}", {})
    names = proprio.get("joint_names_urdf") if isinstance(proprio, dict) else None
    values = proprio.get("joint_pos_urdf_rad") if isinstance(proprio, dict) else None
    if not isinstance(names, list) or not isinstance(values, list):
        return None
    try:
        pan = _finite_float(values[names.index("head_pan_joint")])
        tilt = _finite_float(values[names.index("head_tilt_joint")])
    except (ValueError, IndexError):
        return None
    if not math.isfinite(pan) or not math.isfinite(tilt):
        return None
    return {"head_pan": -pan, "head_tilt": -tilt}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
