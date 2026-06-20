from __future__ import annotations

import pytest

from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.robots.xlerobot.xlerobot_constants import CANONICAL_MOTORS, LEFT_MOTORS, RIGHT_MOTORS
from scripts.indory_drop_supervised_teleop import build_head_locked_teleop_action


def test_head_locked_teleop_action_preserves_current_left_arm_and_embeds_head_targets():
    robot = XLerobotClient(
        XLerobotClientConfig(
            remote_ip="127.0.0.1",
            id="test_drop_supervisor",
            max_relative_target=None,
            leader_action_units="degrees",
        )
    )
    current = [
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
    robot.latest_topics["proprio.0"] = {"joint_pos": current}
    robot.follower_calibration.update(
        {
            motor: {"range_min": 0.0, "range_max": 4095.0, "drive_mode": 0}
            for motor in CANONICAL_MOTORS
        }
    )

    action = build_head_locked_teleop_action(
        robot,
        {"shoulder_pan.pos": 10.0},
        {"head_motor_1": 2070.0, "head_motor_2": 2700.0},
        max_relative_target=None,
    )

    assert action[RIGHT_MOTORS[0]] == pytest.approx(2047.5 + 10.0 * 4095.0 / 360.0)
    assert action[LEFT_MOTORS[0]] == pytest.approx(current[6])
    assert action["head_motor_1"] == pytest.approx(2070.0)
    assert action["head_motor_2"] == pytest.approx(2700.0)
