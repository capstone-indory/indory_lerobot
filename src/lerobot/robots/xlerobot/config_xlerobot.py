# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@CameraConfig.register_subclass("indory_fast_zmq")
@dataclass
class IndoryFastZMQCameraConfig(CameraConfig):
    """Feature metadata for RGB frames decoded by XLerobotClient itself.

    This is intentionally not `ZMQCameraConfig`: the Indoory optimized camera
    socket uses msgpack multipart payloads, not the generic JSON/base64 camera
    protocol.
    """

    topic: str = "rgb.front.0"
    color_mode: ColorMode = ColorMode.RGB

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        if not str(self.topic).strip():
            raise ValueError("`topic` cannot be empty.")


def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    return {
        "head": RealSenseCameraConfig(
            serial_number_or_name=os.getenv("XLEROBOT_REALSENSE_SERIAL", "944122072978"),
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=False,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=os.getenv(
                "XLEROBOT_LEFT_WRIST_CAMERA",
                "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1.3:1.0-video-index0",
            ),
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc="MJPG",
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=os.getenv(
                "XLEROBOT_RIGHT_WRIST_CAMERA",
                "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1.4:1.0-video-index0",
            ),
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.ROTATE_180,
            warmup_s=2,
            fourcc="MJPG",
        ),
    }


def xlerobot_client_cameras_config() -> dict[str, CameraConfig]:
    """Remote camera feature definitions for the indory_server adapter path.

    XLerobotClient reads images from the optimized adapter camera socket
    directly, so these configs provide LeRobot feature metadata only.
    """
    return {
        "head": IndoryFastZMQCameraConfig(
            topic="rgb.front.0",
            fps=15,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        "left_wrist": IndoryFastZMQCameraConfig(
            topic="rgb.wrist_left.0",
            fps=15,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        "right_wrist": IndoryFastZMQCameraConfig(
            topic="rgb.wrist_right.0",
            fps=15,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
    }


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):
    port1: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D044741-if00"  # left arm + head
    port2: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032190-if00"  # right arm + mobile base
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the per-command positional jump for safety.
    # With RANGE_M100_100 normalization, 5.0 is a conservative soft-start step.
    max_relative_target: float | dict[str, float] | None = 5.0

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "q",
            "rotate_right": "e",
            # Speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )


@RobotConfig.register_subclass("xlerobot_client")
@dataclass
class XLerobotClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 8856
    port_zmq_observations: int = 8855
    port_zmq_rpc: int = 8857
    port_zmq_cameras: int = 8866
    port_zmq_rgbd: int = 8867
    camera_transport: str = "zmq"
    cam_bridge_base_url: str = "ws://127.0.0.1:8870"
    cam_bridge_resize_mode: str = "center_crop"
    rtp_udp_bind_ip: str = "0.0.0.0"  # nosec B104 - receives robot camera UDP on the robot LAN.
    rtp_udp_payload_type: int = 96
    rtp_udp_front_port: int = 5600
    rtp_udp_wrist_left_port: int = 5602
    rtp_udp_wrist_right_port: int = 5604
    rtp_udp_depth_port: int = 5610
    rtp_udp_ffmpeg_path: str | None = None
    robot_id: int = 0
    source_id: str = "mac_xlerobot_client"
    source_role: str = "teleop"
    command_lease_ms: int = 300
    follower_calibration_path: str | None = None
    leader_action_units: str = "degrees"
    max_relative_target: float | dict[str, float] | None = 10.0
    head_step_rad: float = 0.05
    head_pan_sign: float = 1.0
    head_tilt_sign: float = 1.0
    enable_rgbd: bool = False
    rgbd_topic: str = "/xlerobot/head/rgbd"
    rgbd_depth_feature: str = "depth.head"
    rgbd_depth_width: int = 640
    rgbd_depth_height: int = 480

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "q",
            "rotate_right": "e",
            # Head camera
            "head_up": "i",
            "head_down": "k",
            "head_left": "j",
            "head_right": "l",
            "head_recenter": "h",
            "arm_recenter": "r",
            # Speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_client_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.camera_transport = str(self.camera_transport or "zmq").strip().lower().replace("-", "_")
        if self.camera_transport not in {"zmq", "rtp_udp", "cam_bridge"}:
            raise ValueError("camera_transport must be 'zmq', 'rtp_udp', or 'cam_bridge'")
        self.cam_bridge_resize_mode = (
            str(self.cam_bridge_resize_mode or "center_crop").strip().lower().replace("-", "_")
        )
        if self.cam_bridge_resize_mode not in {"center_crop", "stretch"}:
            raise ValueError("cam_bridge_resize_mode must be 'center_crop' or 'stretch'")
