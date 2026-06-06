from __future__ import annotations

import logging
from itertools import chain
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from .xlerobot_constants import BASE_MOTORS, HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS

logger = logging.getLogger(__name__)


def make_motor_buses(config: Any, calibration: dict[str, Any]) -> tuple[FeetechMotorsBus, FeetechMotorsBus]:
    norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
    bus1 = FeetechMotorsBus(
        port=config.port1,
        motors={
            "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            "head_motor_1": Motor(7, "sts3215", norm_mode_body),
            "head_motor_2": Motor(8, "sts3215", norm_mode_body),
        },
        calibration=_calibration_subset(calibration, (*LEFT_MOTORS, *HEAD_MOTORS), "left_arm_shoulder_pan"),
    )
    bus2 = FeetechMotorsBus(
        port=config.port2,
        motors={
            "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "right_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "right_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
            "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
            "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
        },
        calibration=_calibration_subset(calibration, (*RIGHT_MOTORS, *BASE_MOTORS), "right_arm_shoulder_pan"),
    )
    return bus1, bus2


def motor_groups(bus1: FeetechMotorsBus, bus2: FeetechMotorsBus) -> tuple[list[str], list[str], list[str], list[str]]:
    return (
        [motor for motor in bus1.motors if motor.startswith("left_arm")],
        [motor for motor in bus2.motors if motor.startswith("right_arm")],
        [motor for motor in bus1.motors if motor.startswith("head")],
        [motor for motor in bus2.motors if motor.startswith("base")],
    )


def restore_or_calibrate(robot: Any, calibrate: bool) -> None:
    if robot.calibration_fpath.is_file():
        logger.info("Calibration file found at %s", robot.calibration_fpath)
        user_input = input(
            "Press ENTER to restore calibration from file, or type 'c' and press ENTER to run manual calibration: "
        )
        if user_input.strip().lower() == "c":
            logger.info("User chose manual calibration...")
            if calibrate:
                calibrate_robot(robot)
            return
        _restore_calibration(robot, calibrate)
    elif calibrate:
        logger.info("No calibration file found, proceeding with manual calibration...")
        calibrate_robot(robot)


def calibrate_robot(robot: Any) -> None:
    logger.info("\nRunning calibration of %s", robot)
    left_motors = robot.left_arm_motors + robot.head_motors
    robot.bus1.disable_torque()
    for name in left_motors:
        robot.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
    input("Move left arm and head motors to the middle of their range of motion and press ENTER....")
    homing_offsets = robot.bus1.set_half_turn_homings(left_motors)
    homing_offsets.update(dict.fromkeys(robot.right_arm_motors + robot.base_motors, 0))
    print(
        "Move all left arm and head joints sequentially through their entire ranges of motion.\n"
        "Recording positions. Press ENTER to stop..."
    )
    range_mins, range_maxes = robot.bus1.record_ranges_of_motion(left_motors)
    calibration_left = _calibration_from_ranges(robot.bus1, homing_offsets, range_mins, range_maxes)
    robot.bus1.write_calibration(calibration_left)

    right_motors = robot.right_arm_motors + robot.base_motors
    robot.bus2.disable_torque(robot.right_arm_motors)
    for name in robot.right_arm_motors:
        robot.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
    input("Move right arm motors to the middle of their range of motion and press ENTER....")
    homing_offsets = robot.bus2.set_half_turn_homings(robot.right_arm_motors)
    homing_offsets.update(dict.fromkeys(robot.base_motors, 0))
    full_turn_motors = [motor for motor in right_motors if "wheel" in motor]
    range_mins, range_maxes = robot.bus2.record_ranges_of_motion(
        [motor for motor in right_motors if motor not in full_turn_motors]
    )
    for name in full_turn_motors:
        range_mins[name] = 0
        range_maxes[name] = 4095
    calibration_right = _calibration_from_ranges(robot.bus2, homing_offsets, range_mins, range_maxes)
    robot.bus2.write_calibration(calibration_right)
    robot.calibration = {**calibration_left, **calibration_right}
    robot._save_calibration()
    print("Calibration saved to", robot.calibration_fpath)


def configure_robot(robot: Any) -> None:
    robot.bus1.disable_torque(num_retry=3)
    robot.bus1.configure_motors(maximum_acceleration=80, acceleration=80)
    robot.bus2.disable_torque(num_retry=3)
    robot.bus2.configure_motors(maximum_acceleration=80, acceleration=80)
    for bus, motors in (
        (robot.bus1, robot.left_arm_motors),
        (robot.bus1, robot.head_motors),
        (robot.bus2, robot.right_arm_motors),
    ):
        for name in motors:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", name, 16)
            bus.write("I_Coefficient", name, 0)
            bus.write("D_Coefficient", name, 43)
    for name in robot.base_motors:
        robot.bus2.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
    robot.bus1.enable_torque(num_retry=3)
    robot.bus2.enable_torque(num_retry=3)


def setup_robot_motors(robot: Any) -> None:
    for motor in chain(reversed(robot.left_arm_motors), reversed(robot.head_motors)):
        input(f"Connect the controller board to the '{motor}' motor only and press enter.")
        robot.bus1.setup_motor(motor)
        print(f"'{motor}' motor id set to {robot.bus1.motors[motor].id}")
    for motor in chain(reversed(robot.right_arm_motors), reversed(robot.base_motors)):
        input(f"Connect the controller board to the '{motor}' motor only and press enter.")
        robot.bus2.setup_motor(motor)
        print(f"'{motor}' motor id set to {robot.bus2.motors[motor].id}")


def _calibration_subset(calibration: dict[str, Any], motors: tuple[str, ...], sentinel: str) -> dict[str, Any]:
    if calibration.get(sentinel) is None:
        return calibration
    return {motor: calibration.get(motor) for motor in motors}


def _restore_calibration(robot: Any, calibrate: bool) -> None:
    logger.info("Attempting to restore calibration from file...")
    try:
        robot.bus1.calibration = {k: v for k, v in robot.calibration.items() if k in robot.bus1.motors}
        robot.bus2.calibration = {k: v for k, v in robot.calibration.items() if k in robot.bus2.motors}
        robot.bus1.write_calibration(robot.bus1.calibration)
        robot.bus2.write_calibration(robot.bus2.calibration)
        logger.info("Calibration restored successfully from file!")
    except Exception as exc:
        logger.warning("Failed to restore calibration from file: %s", exc)
        if calibrate:
            calibrate_robot(robot)


def _calibration_from_ranges(
    bus: FeetechMotorsBus,
    homing_offsets: dict[str, int],
    range_mins: dict[str, int],
    range_maxes: dict[str, int],
) -> dict[str, MotorCalibration]:
    return {
        name: MotorCalibration(
            id=motor.id,
            drive_mode=0,
            homing_offset=homing_offsets[name],
            range_min=range_mins[name],
            range_max=range_maxes[name],
        )
        for name, motor in bus.motors.items()
    }
