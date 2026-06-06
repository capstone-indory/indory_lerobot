from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .xlerobot_constants import CANONICAL_MOTORS, LEFT_MOTORS, RIGHT_MOTORS, SCHEMA_VERSION
from .xlerobot_keyboard_control import (
    arm_recenter_offsets,
    bounded_raw_target,
    head_relative_target_from_action,
)
from .xlerobot_leader_kinematics import LeaderKinematicMapper, cap_raw_targets_to_current


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
        payload = self._base_payload(seq, source_id, source_role, lease_ms)
        if base_cmd is not None:
            payload["base_cmd_vel"] = base_cmd
        if joint_targets is not None:
            self._add_joint_targets(payload, joint_targets, robot_id)
        if head_target is not None:
            payload["head_joint_relative_target"] = head_target
        return CommandBuildResult(payload, self._sent_action_vector(base_cmd, joint_targets))

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

    def _sent_action_vector(
        self,
        base_cmd: list[float] | None,
        joint_targets: list[float | None] | None,
    ) -> dict[str, Any]:
        values = {key: 0.0 for key in self.state_order}
        if joint_targets is not None:
            for name, value in zip(LEFT_MOTORS, joint_targets[:6], strict=False):
                if value is not None:
                    values[name] = float(value)
            for name, value in zip(RIGHT_MOTORS, joint_targets[6:12], strict=False):
                if value is not None:
                    values[name] = float(value)
        if base_cmd is not None:
            values["x.vel"], values["y.vel"], values["theta.vel"] = base_cmd
        values["action"] = np.array([values[k] for k in self.state_order], dtype=np.float32)
        return values
