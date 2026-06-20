from __future__ import annotations

from typing import Any

from lerobot.robots.xlerobot.xlerobot_constants import LEFT_MOTORS, RIGHT_MOTORS


ARM_RECENTER_MOTORS = (*LEFT_MOTORS, *RIGHT_MOTORS)
GRIPPER_MOTORS = ("left_arm_gripper", "right_arm_gripper")

DEFAULT_ARM_RECENTER_TICKS: dict[str, float] = {
    "left_arm_shoulder_pan": 2056.0,
    "left_arm_shoulder_lift": 949.0,
    "left_arm_elbow_flex": 3072.0,
    "left_arm_wrist_flex": 823.0,
    "left_arm_wrist_roll": 2080.0,
    "left_arm_gripper": 2095.0,
    "right_arm_shoulder_pan": 1981.0,
    "right_arm_shoulder_lift": 800.0,
    "right_arm_elbow_flex": 3131.0,
    "right_arm_wrist_flex": 747.0,
    "right_arm_wrist_roll": 2074.0,
    "right_arm_gripper": 2031.0,
}


def parse_arm_recenter_targets(raw: str | None) -> dict[str, float]:
    if raw is None or raw.strip() == "":
        return dict(DEFAULT_ARM_RECENTER_TICKS)

    targets = dict(DEFAULT_ARM_RECENTER_TICKS)
    for part in raw.split(","):
        name, sep, value = part.partition("=")
        if not sep:
            raise ValueError("--arm-recenter-targets entries must be formatted as motor=value")
        name = name.strip()
        if name not in ARM_RECENTER_MOTORS:
            raise ValueError(f"unsupported arm recenter motor {name!r}; expected one of {ARM_RECENTER_MOTORS}")
        targets[name] = float(value)
    return targets


def apply_arm_recenter_targets(base_action: dict[str, Any], targets: dict[str, float]) -> dict[str, float]:
    action = {name: float(value) for name, value in base_action.items()}
    missing = [name for name in ARM_RECENTER_MOTORS if name not in targets]
    if missing:
        raise ValueError(f"arm recenter targets missing motors: {missing}")
    for name in ARM_RECENTER_MOTORS:
        action[name] = float(targets[name])
    return action


def recovery_home_actions(
    home_action: dict[str, Any],
    release_targets: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    closed_home = {name: float(value) for name, value in home_action.items()}
    open_home = dict(closed_home)
    missing = [motor for motor in GRIPPER_MOTORS if motor not in release_targets]
    if missing:
        raise ValueError(f"release targets missing gripper motors: {missing}")
    for motor in GRIPPER_MOTORS:
        open_home[motor] = float(release_targets[motor])
    return open_home, closed_home


def moderate_gripper_open_targets(
    calibration: dict[str, dict[str, Any]],
    close_targets: dict[str, float],
    *,
    open_ratio: float,
    fallback_delta_ticks: float = 450.0,
) -> dict[str, float]:
    ratio = max(0.0, min(1.0, float(open_ratio)))
    targets: dict[str, float] = {}
    for motor in GRIPPER_MOTORS:
        close_target = float(close_targets[motor])
        cal = calibration.get(motor, {})
        try:
            range_max = float(cal["range_max"])
        except (KeyError, TypeError, ValueError):
            range_max = close_target + abs(float(fallback_delta_ticks))
        targets[motor] = close_target + (range_max - close_target) * ratio
    return targets


def closed_grippers(
    current_action: dict[str, Any],
    close_targets: dict[str, float],
    open_targets: dict[str, float],
    *,
    closed_ratio: float,
) -> list[str]:
    ratio = max(0.0, min(1.0, float(closed_ratio)))
    closed: list[str] = []
    for motor in GRIPPER_MOTORS:
        current = float(current_action[motor])
        close_target = float(close_targets[motor])
        open_target = float(open_targets[motor])
        threshold = close_target + (open_target - close_target) * ratio
        if open_target >= close_target:
            is_closed = current <= threshold
        else:
            is_closed = current >= threshold
        if is_closed:
            closed.append(motor)
    return closed


def target_position_error(
    current_action: dict[str, Any],
    target_action: dict[str, Any],
    motors: tuple[str, ...],
) -> tuple[float, float]:
    errors = [abs(float(current_action[motor]) - float(target_action[motor])) for motor in motors]
    if not errors:
        return 0.0, 0.0
    return max(errors), sum(errors) / len(errors)


def targets_within_tolerance(
    current_action: dict[str, Any],
    target_action: dict[str, Any],
    motors: tuple[str, ...],
    *,
    tolerance_ticks: float,
) -> bool:
    max_error, _ = target_position_error(current_action, target_action, motors)
    return max_error <= abs(float(tolerance_ticks))
