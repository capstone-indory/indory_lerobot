from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .xlerobot_constants import CANONICAL_MOTORS, HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS, SCHEMA_VERSION
from .xlerobot_keyboard_control import (
    arm_recenter_offsets,
    bounded_raw_target,
    head_relative_target_from_action,
)
from .xlerobot_leader_kinematics import FEETECH_TICKS_PER_REV, LeaderKinematicMapper, cap_raw_targets_to_current


HEAD_TICK_TO_RAD = 2 * math.pi / FEETECH_TICKS_PER_REV


@dataclass
class CommandBuildResult:
    payload: dict[str, Any]
    sent_action: dict[str, Any]


class XLerobotCommandBuilder:
    def __init__(
        self,
        config: Any,
        leader_mapper: LeaderKinematicMapper,
        follower_calibration: dict[str, dict[str, Any]],
        latest_topics: dict[str, dict[str, Any]],
        state_order: tuple[str, ...],
    ) -> None:
        self.config = config
        self.leader_mapper = leader_mapper
        self.follower_calibration = follower_calibration
        self.latest_topics = latest_topics
        self.state_order = state_order
        self.arm_target_offsets = dict.fromkeys((*LEFT_MOTORS, *RIGHT_MOTORS), 0.0)

    def build(
        self,
        action: dict[str, Any],
        *,
        seq: int,
        source_id: str,
        source_role: str,
        lease_ms: int,
        robot_id: int,
    ) -> CommandBuildResult:
        action = dict(action)
        if action.get("arm.recenter"):
            self.arm_target_offsets.update(
                arm_recenter_offsets(
                    action,
                    self.current_canonical_ticks(robot_id),
                    self.leader_mapper,
                    LEFT_MOTORS,
                    RIGHT_MOTORS,
                )
            )
        base_cmd = self._base_cmd_from_action(action)
        joint_targets = self._joint_targets_from_action(action)
        head_target = head_relative_target_from_action(action, self.latest_topics, robot_id)
        head_targets = self._head_raw_targets_from_relative(head_target, robot_id)
        if head_target is None:
            head_target, head_targets = self._head_relative_target_from_absolute_action(action, robot_id)
        if head_targets is None:
            head_targets = self._current_head_targets(robot_id)
        payload = self._base_payload(seq, source_id, source_role, lease_ms)
        if base_cmd is not None:
            payload["base_cmd_vel"] = base_cmd
        if joint_targets is not None:
            self._add_joint_targets(payload, joint_targets, robot_id)
        if head_target is not None:
            payload["head_joint_relative_target"] = head_target
        return CommandBuildResult(payload, self._sent_action_vector(base_cmd, joint_targets, head_targets))

    @staticmethod
    def is_material_command(payload: dict[str, Any]) -> bool:
        return any(
            key in payload
            for key in (
                "base_cmd_vel",
                "joint_targets_sparse",
                "arm_joint_pos_target",
                "head_joint_relative_target",
            )
        )

    @staticmethod
    def _base_payload(seq: int, source_id: str, source_role: str, lease_ms: int) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "source_id": source_id,
            "source_role": source_role,
            "seq": seq,
            "stamp_ns": time.time_ns(),
            "lease_ms": lease_ms,
            "frame": "body",
        }

    @staticmethod
    def _base_cmd_from_action(action: dict[str, Any]) -> list[float] | None:
        if not any(k in action for k in ("x.vel", "y.vel", "theta.vel")):
            return None
        return [float(action.get(k, 0.0) or 0.0) for k in ("x.vel", "y.vel", "theta.vel")]

    def _joint_targets_from_action(self, action: dict[str, Any]) -> list[float | None] | None:
        targets: list[float | None] = [None] * 14
        self._fill_side_targets(action, targets, "left", LEFT_MOTORS, 0)
        self._fill_side_targets(action, targets, "right", RIGHT_MOTORS, 6)
        return targets if any(value is not None for value in targets) else None

    def _add_joint_targets(self, payload: dict[str, Any], targets: list[float | None], robot_id: int) -> None:
        canonical_targets = self._canonical_targets_from_ros_sparse(targets, robot_id)
        if canonical_targets is None:
            payload["joint_targets_sparse"] = targets
            return
        payload["arm_joint_pos_target"] = canonical_targets
        payload["arm_joint_pos_target_units"] = "feetech_ticks"

    def _canonical_targets_from_ros_sparse(
        self, targets: list[float | None], robot_id: int
    ) -> list[float] | None:
        current = self.current_canonical_ticks(robot_id)
        if current is None:
            return None
        canonical = list(current)
        for idx, value in enumerate(targets[6:12]):
            if value is not None:
                canonical[idx] = float(value)
        for idx, value in enumerate(targets[:6]):
            if value is not None:
                canonical[6 + idx] = float(value)
        for idx, value in enumerate(targets[12:14], start=12):
            if value is not None:
                canonical[idx] = float(value)
        return cap_raw_targets_to_current(
            canonical,
            current,
            CANONICAL_MOTORS,
            self.follower_calibration,
            self.config.max_relative_target,
            self.config.leader_action_units,
        )

    def current_canonical_ticks(self, robot_id: int) -> list[float] | None:
        proprio = self.latest_topics.get(f"proprio.{robot_id}", {})
        joint_pos = proprio.get("joint_pos") if isinstance(proprio, dict) else None
        if not isinstance(joint_pos, list) or len(joint_pos) < 14:
            return None
        try:
            values = [float(value) for value in joint_pos[:14]]
        except (TypeError, ValueError):
            return None
        return values if all(math.isfinite(value) for value in values) else None

    def _fill_side_targets(
        self,
        action: dict[str, Any],
        targets: list[float | None],
        side: str,
        motors: tuple[str, ...],
        offset: int,
    ) -> None:
        try:
            side_targets = self.leader_mapper.targets_for_side(action, side, motors)
        except ValueError as exc:
            logging.warning("Dropping %s arm targets: %s", side, exc)
            return
        for i, value in enumerate(side_targets):
            if value is not None:
                value += self.arm_target_offsets.get(motors[i], 0.0)
                targets[offset + i] = bounded_raw_target(
                    motors[i], value, self.follower_calibration.get(motors[i])
                )

    def _head_relative_target_from_absolute_action(
        self,
        action: dict[str, Any],
        robot_id: int,
    ) -> tuple[dict[str, float] | None, list[float] | None]:
        current = self.current_canonical_ticks(robot_id)
        if current is None:
            if any(self._head_action_value(action, motor) is not None for motor in HEAD_MOTORS):
                logging.warning("Dropping head targets before receiving proprio.%s joint positions.", robot_id)
            return None, None

        current_head = [float(current[12]), float(current[13])]
        target_head = list(current_head)
        saw_target = False
        saw_nonzero_target = False
        for index, motor in enumerate(HEAD_MOTORS):
            value = self._head_action_value(action, motor)
            if value is None:
                continue
            saw_target = True
            saw_nonzero_target = saw_nonzero_target or abs(value) > 1e-6
            target_head[index] = bounded_raw_target(motor, value, self.follower_calibration.get(motor))

        # A legacy unpatched dataset has zero-filled head actions. Treat an all-zero
        # policy head output as "no head command" instead of driving toward raw tick 0.
        if not saw_target or not saw_nonzero_target:
            return None, None

        target_head = self._cap_head_targets(target_head, current_head)
        relative = self._head_relative_target_from_raw(current_head, target_head)
        return relative, target_head

    @staticmethod
    def _head_action_value(action: dict[str, Any], motor: str) -> float | None:
        for key in (motor, f"{motor}.pos"):
            if key not in action:
                continue
            try:
                value = float(action[key])
            except (TypeError, ValueError):
                logging.warning("Dropping head target from %s: %r", key, action[key])
                return None
            return value if math.isfinite(value) else None
        return None

    def _head_raw_targets_from_relative(
        self,
        relative: dict[str, float] | None,
        robot_id: int,
    ) -> list[float] | None:
        if relative is None:
            return None
        current = self.current_canonical_ticks(robot_id)
        if current is None:
            return None
        current_head = [float(current[12]), float(current[13])]
        target_head = [
            current_head[0]
            + float(relative.get("head_pan", 0.0)) / (HEAD_TICK_TO_RAD * self._head_sign("pan")),
            current_head[1]
            + float(relative.get("head_tilt", 0.0)) / (HEAD_TICK_TO_RAD * self._head_sign("tilt")),
        ]
        return self._cap_head_targets(target_head, current_head)

    def _current_head_targets(self, robot_id: int) -> list[float] | None:
        current = self.current_canonical_ticks(robot_id)
        if current is None:
            return None
        return [float(current[12]), float(current[13])]

    def _head_relative_target_from_raw(
        self,
        current_head: list[float],
        target_head: list[float],
    ) -> dict[str, float] | None:
        pan = (target_head[0] - current_head[0]) * HEAD_TICK_TO_RAD * self._head_sign("pan")
        tilt = (target_head[1] - current_head[1]) * HEAD_TICK_TO_RAD * self._head_sign("tilt")
        if abs(pan) <= 1e-9 and abs(tilt) <= 1e-9:
            return None
        return {"head_pan": pan, "head_tilt": tilt}

    def _cap_head_targets(self, target_head: list[float], current_head: list[float]) -> list[float]:
        capped = list(target_head)
        for index, motor in enumerate(HEAD_MOTORS):
            limit = self._head_delta_limit_ticks(motor)
            if limit is not None:
                capped[index] = float(
                    np.clip(capped[index], current_head[index] - limit, current_head[index] + limit)
                )
            capped[index] = bounded_raw_target(motor, capped[index], self.follower_calibration.get(motor))
        return capped

    def _head_delta_limit_ticks(self, motor: str) -> float | None:
        max_relative_target = self.config.max_relative_target
        if max_relative_target is None:
            return None
        if isinstance(max_relative_target, dict):
            raw_cap = max_relative_target.get(motor, max_relative_target.get(f"{motor}.pos"))
            if raw_cap is None:
                return None
        else:
            raw_cap = max_relative_target
        cap_degrees = abs(float(raw_cap))
        if not math.isfinite(cap_degrees):
            return None
        return cap_degrees * FEETECH_TICKS_PER_REV / 360.0

    def _head_sign(self, axis: str) -> float:
        attr = "head_pan_sign" if axis == "pan" else "head_tilt_sign"
        try:
            sign = float(getattr(self.config, attr))
        except (TypeError, ValueError):
            sign = 1.0
        return sign if math.isfinite(sign) and abs(sign) > 1e-9 else 1.0

    def _sent_action_vector(
        self,
        base_cmd: list[float] | None,
        joint_targets: list[float | None] | None,
        head_targets: list[float] | None,
    ) -> dict[str, Any]:
        values = {key: 0.0 for key in self.state_order}
        if joint_targets is not None:
            for name, value in zip(LEFT_MOTORS, joint_targets[:6], strict=False):
                if value is not None:
                    values[name] = float(value)
            for name, value in zip(RIGHT_MOTORS, joint_targets[6:12], strict=False):
                if value is not None:
                    values[name] = float(value)
        if head_targets is not None:
            for name, value in zip(HEAD_MOTORS, head_targets, strict=False):
                values[name] = float(value)
        if base_cmd is not None:
            values["x.vel"], values["y.vel"], values["theta.vel"] = base_cmd
        values["action"] = np.array([values[k] for k in self.state_order], dtype=np.float32)
        return values
