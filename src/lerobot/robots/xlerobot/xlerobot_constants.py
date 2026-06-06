from __future__ import annotations

LEFT_MOTORS = (
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
    "left_arm_gripper",
)
RIGHT_MOTORS = (
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
)
HEAD_MOTORS = ("head_motor_1", "head_motor_2")
BASE_MOTORS = ("base_left_wheel", "base_back_wheel", "base_right_wheel")
CANONICAL_MOTORS = RIGHT_MOTORS + LEFT_MOTORS + HEAD_MOTORS
LEGACY_CAMERA_TOPIC_TO_NAME = {
    "/xlerobot/head/rgb/image_raw": "head",
    "/xlerobot/wrist_left/rgb/image_raw": "left_wrist",
    "/xlerobot/wrist_right/rgb/image_raw": "right_wrist",
}
SCHEMA_VERSION = "xlerobot_v1.1"
