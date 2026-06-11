from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from .xlerobot_constants import CANONICAL_MOTORS, HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS, SCHEMA_VERSION


BASE_VELOCITY_KEYS = ("x.vel", "y.vel", "theta.vel")
STATE_NAMES = (*LEFT_MOTORS, *RIGHT_MOTORS, *HEAD_MOTORS, *BASE_VELOCITY_KEYS)


def base_cmd_from_action(action: dict[str, Any], *, allow_base_action: bool) -> list[float]:
    if not allow_base_action:
        return [0.0, 0.0, 0.0]
    return [float(action.get(key, 0.0) or 0.0) for key in BASE_VELOCITY_KEYS]


def canonical_ticks_from_action(
    action: dict[str, Any],
    *,
    current: Sequence[float] | None = None,
    head_override: dict[str, float] | None = None,
    require_all: bool = False,
) -> list[float]:
    if current is None:
        if not require_all:
            raise ValueError("current canonical ticks are required when require_all=False")
        values: list[float] = []
        for motor in CANONICAL_MOTORS:
            if motor not in action:
                raise KeyError(f"missing canonical motor target {motor!r}")
            values.append(float(action[motor]))
        return values

    if len(current) < len(CANONICAL_MOTORS):
        raise ValueError(f"current canonical ticks must contain {len(CANONICAL_MOTORS)} values")
    values = [float(value) for value in current[: len(CANONICAL_MOTORS)]]
    for index, motor in enumerate(CANONICAL_MOTORS):
        if head_override is not None and motor in head_override:
            values[index] = float(head_override[motor])
        elif motor in action:
            values[index] = float(action[motor])
    return values


def canonical_ticks_to_action(
    canonical_ticks: Sequence[float],
    *,
    base_cmd: Sequence[float] | None = None,
) -> dict[str, float]:
    if len(canonical_ticks) < len(CANONICAL_MOTORS):
        raise ValueError(f"canonical ticks must contain {len(CANONICAL_MOTORS)} values")
    action = {name: 0.0 for name in STATE_NAMES}
    for motor, value in zip(CANONICAL_MOTORS, canonical_ticks, strict=False):
        action[motor] = float(value)
    base = list(base_cmd or (0.0, 0.0, 0.0))
    for key, value in zip(BASE_VELOCITY_KEYS, base, strict=False):
        action[key] = float(value)
    return action


def base_payload(
    *,
    seq: int,
    source_id: str,
    source_role: str,
    lease_ms: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "source_id": source_id,
        "source_role": source_role,
        "seq": int(seq),
        "stamp_ns": time.time_ns(),
        "lease_ms": int(lease_ms),
        "frame": "body",
    }


def canonical_payload(
    *,
    seq: int,
    source_id: str,
    source_role: str,
    lease_ms: int,
    canonical_ticks: Sequence[float],
    base_cmd: Sequence[float] | None = None,
    include_zero_base: bool = False,
) -> dict[str, Any]:
    canonical = [float(value) for value in canonical_ticks[: len(CANONICAL_MOTORS)]]
    if len(canonical) != len(CANONICAL_MOTORS):
        raise ValueError(f"canonical ticks must contain {len(CANONICAL_MOTORS)} values")
    payload = base_payload(seq=seq, source_id=source_id, source_role=source_role, lease_ms=lease_ms)
    payload["arm_joint_pos_target"] = canonical
    payload["arm_joint_pos_target_units"] = "feetech_ticks"
    base = [float(value) for value in (base_cmd or (0.0, 0.0, 0.0))]
    if include_zero_base or any(abs(value) > 1e-9 for value in base):
        payload["base_cmd_vel"] = base[:3]
    return payload
