from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from lerobot.robots.xlerobot.xlerobot_command_builder import HEAD_TICK_TO_RAD, XLerobotCommandBuilder
from lerobot.robots.xlerobot.xlerobot_constants import HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS


class NoopLeaderMapper:
    def targets_for_side(self, action, side, motors):
        return [None] * len(motors)


def make_builder(*, max_relative_target=10.0, head_pan_sign=1.0, head_tilt_sign=1.0):
    config = SimpleNamespace(
        max_relative_target=max_relative_target,
        leader_action_units="degrees",
        head_pan_sign=head_pan_sign,
        head_tilt_sign=head_tilt_sign,
    )
    latest_topics = {
        "proprio.0": {
            "joint_pos": [
                1000.0,
                1001.0,
                1002.0,
                1003.0,
                1004.0,
                1005.0,
                2000.0,
                2001.0,
                2002.0,
                2003.0,
                2004.0,
                2005.0,
                2100.0,
                2600.0,
            ]
        }
    }
    state_order = (*LEFT_MOTORS, *RIGHT_MOTORS, *HEAD_MOTORS, "x.vel", "y.vel", "theta.vel")
    return XLerobotCommandBuilder(config, NoopLeaderMapper(), {}, latest_topics, state_order)


def test_policy_head_motor_targets_become_relative_head_command():
    builder = make_builder()

    result = builder.build(
        {"head_motor_1": 2120.0, "head_motor_2": 2570.0},
        seq=0,
        source_id="test",
        source_role="policy",
        lease_ms=300,
        robot_id=0,
    )

    assert result.payload["head_joint_relative_target"] == pytest.approx(
        {
            "head_pan": 20.0 * HEAD_TICK_TO_RAD,
            "head_tilt": -30.0 * HEAD_TICK_TO_RAD,
        }
    )
    assert result.sent_action["head_motor_1"] == pytest.approx(2120.0)
    assert result.sent_action["head_motor_2"] == pytest.approx(2570.0)


def test_base_commands_are_sent_and_recorded():
    builder = make_builder()

    result = builder.build(
        {"x.vel": 0.1, "y.vel": -0.2, "theta.vel": 0.3},
        seq=0,
        source_id="test",
        source_role="teleop",
        lease_ms=300,
        robot_id=0,
    )

    assert result.payload["base_cmd_vel"] == pytest.approx([0.1, -0.2, 0.3])
    assert result.sent_action["x.vel"] == pytest.approx(0.1)
    assert result.sent_action["y.vel"] == pytest.approx(-0.2)
    assert result.sent_action["theta.vel"] == pytest.approx(0.3)


def test_zero_filled_policy_head_targets_are_ignored():
    builder = make_builder()

    result = builder.build(
        {"head_motor_1": 0.0, "head_motor_2": 0.0},
        seq=0,
        source_id="test",
        source_role="policy",
        lease_ms=300,
        robot_id=0,
    )

    assert "head_joint_relative_target" not in result.payload
    assert result.sent_action["head_motor_1"] == pytest.approx(2100.0)
    assert result.sent_action["head_motor_2"] == pytest.approx(2600.0)


def test_policy_head_motor_targets_are_capped_by_relative_limit():
    builder = make_builder(max_relative_target=1.0)

    result = builder.build(
        {"head_motor_1": 3000.0, "head_motor_2": 1000.0},
        seq=0,
        source_id="test",
        source_role="policy",
        lease_ms=300,
        robot_id=0,
    )

    assert result.payload["head_joint_relative_target"] == pytest.approx(
        {
            "head_pan": math.radians(1.0),
            "head_tilt": -math.radians(1.0),
        }
    )


def test_explicit_relative_head_command_takes_precedence():
    builder = make_builder()

    result = builder.build(
        {"head.pan": 0.05, "head.tilt": -0.02, "head_motor_1": 3000.0, "head_motor_2": 1000.0},
        seq=0,
        source_id="test",
        source_role="teleop",
        lease_ms=300,
        robot_id=0,
    )

    assert result.payload["head_joint_relative_target"] == pytest.approx(
        {"head_pan": 0.05, "head_tilt": -0.02}
    )
    assert result.sent_action["head_motor_1"] == pytest.approx(2100.0 + 0.05 / HEAD_TICK_TO_RAD)
    assert result.sent_action["head_motor_2"] == pytest.approx(2600.0 - 0.02 / HEAD_TICK_TO_RAD)
