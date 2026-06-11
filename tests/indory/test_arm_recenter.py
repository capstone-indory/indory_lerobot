from __future__ import annotations

import pytest

from lerobot.indory.arm_recenter import (
    DEFAULT_ARM_RECENTER_TICKS,
    apply_arm_recenter_targets,
    closed_grippers,
    moderate_gripper_open_targets,
    parse_arm_recenter_targets,
    target_position_error,
    targets_within_tolerance,
)


def test_parse_default_arm_recenter_targets():
    targets = parse_arm_recenter_targets(None)

    assert targets == DEFAULT_ARM_RECENTER_TICKS
    assert targets["left_arm_shoulder_pan"] == 2056.0
    assert targets["right_arm_gripper"] == 2031.0


def test_parse_arm_recenter_target_overrides():
    targets = parse_arm_recenter_targets("left_arm_gripper=2200,right_arm_gripper=2100")

    assert targets["left_arm_gripper"] == 2200.0
    assert targets["right_arm_gripper"] == 2100.0
    assert targets["left_arm_shoulder_lift"] == DEFAULT_ARM_RECENTER_TICKS["left_arm_shoulder_lift"]


def test_parse_arm_recenter_rejects_unknown_motor():
    with pytest.raises(ValueError, match="unsupported arm recenter motor"):
        parse_arm_recenter_targets("head_motor_1=2070")


def test_apply_arm_recenter_targets_preserves_head_and_base():
    base_action = {
        "left_arm_shoulder_pan": 1.0,
        "left_arm_shoulder_lift": 2.0,
        "left_arm_elbow_flex": 3.0,
        "left_arm_wrist_flex": 4.0,
        "left_arm_wrist_roll": 5.0,
        "left_arm_gripper": 6.0,
        "right_arm_shoulder_pan": 7.0,
        "right_arm_shoulder_lift": 8.0,
        "right_arm_elbow_flex": 9.0,
        "right_arm_wrist_flex": 10.0,
        "right_arm_wrist_roll": 11.0,
        "right_arm_gripper": 12.0,
        "head_motor_1": 2070.0,
        "head_motor_2": 2700.0,
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }

    action = apply_arm_recenter_targets(base_action, DEFAULT_ARM_RECENTER_TICKS)

    assert action["left_arm_shoulder_pan"] == 2056.0
    assert action["right_arm_shoulder_pan"] == 1981.0
    assert action["head_motor_1"] == 2070.0
    assert action["head_motor_2"] == 2700.0
    assert action["x.vel"] == 0.0


def test_moderate_gripper_open_targets_use_calibration_range_max():
    targets = moderate_gripper_open_targets(
        {
            "left_arm_gripper": {"range_max": 3558},
            "right_arm_gripper": {"range_max": 3516},
        },
        DEFAULT_ARM_RECENTER_TICKS,
        open_ratio=0.35,
    )

    assert targets["left_arm_gripper"] == pytest.approx(2095.0 + (3558.0 - 2095.0) * 0.35)
    assert targets["right_arm_gripper"] == pytest.approx(2031.0 + (3516.0 - 2031.0) * 0.35)


def test_closed_grippers_detects_grippers_near_close_target():
    open_targets = {
        "left_arm_gripper": 2600.0,
        "right_arm_gripper": 2550.0,
    }
    current = {
        **DEFAULT_ARM_RECENTER_TICKS,
        "left_arm_gripper": 2100.0,
        "right_arm_gripper": 2500.0,
    }

    assert closed_grippers(
        current,
        DEFAULT_ARM_RECENTER_TICKS,
        open_targets,
        closed_ratio=0.25,
    ) == ["left_arm_gripper"]


def test_target_position_error_and_tolerance():
    current = {
        **DEFAULT_ARM_RECENTER_TICKS,
        "left_arm_shoulder_pan": 2040.0,
        "right_arm_wrist_roll": 2084.0,
    }

    max_error, mean_error = target_position_error(
        current,
        DEFAULT_ARM_RECENTER_TICKS,
        ("left_arm_shoulder_pan", "right_arm_wrist_roll"),
    )

    assert max_error == pytest.approx(16.0)
    assert mean_error == pytest.approx(13.0)
    assert targets_within_tolerance(
        current,
        DEFAULT_ARM_RECENTER_TICKS,
        ("left_arm_shoulder_pan", "right_arm_wrist_roll"),
        tolerance_ticks=16.0,
    )
    assert not targets_within_tolerance(
        current,
        DEFAULT_ARM_RECENTER_TICKS,
        ("left_arm_shoulder_pan", "right_arm_wrist_roll"),
        tolerance_ticks=15.9,
    )
