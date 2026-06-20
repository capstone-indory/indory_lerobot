from __future__ import annotations

from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from lerobot.robots.xlerobot.xlerobot_constants import (
    CANONICAL_MOTORS,
    HEAD_MOTORS,
    LEFT_MOTORS,
    RIGHT_MOTORS,
)
from scripts.gr00t_n17.indory_n17_robot_smoke import (
    BASE_KEYS,
    LEFT_ARM_MOTORS,
    LEFT_GRIPPER_MOTORS,
    RIGHT_ARM_MOTORS,
    RIGHT_GRIPPER_MOTORS,
    STATE_NAMES,
    capped_raw_policy_action,
    close_robot,
    ensure_tcp_connectable,
    gr00t_observation,
    robot_action_from_chunk,
    send_capped_raw_policy_action,
    warn_zero_images,
)


def raw_observation() -> dict[str, object]:
    obs = {name: float(i) for i, name in enumerate(STATE_NAMES)}
    obs["head"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["left_wrist"] = np.ones((480, 640, 3), dtype=np.uint8)
    obs["right_wrist"] = np.full((480, 640, 3), 2, dtype=np.uint8)
    return obs


def action_chunk() -> dict[str, np.ndarray]:
    return {
        "left_arm": np.asarray([[[10.0, 11.0, 12.0, 13.0, 14.0]]], dtype=np.float32),
        "left_gripper": np.asarray([[[15.0]]], dtype=np.float32),
        "right_arm": np.asarray([[[20.0, 21.0, 22.0, 23.0, 24.0]]], dtype=np.float32),
        "right_gripper": np.asarray([[[25.0]]], dtype=np.float32),
        "head": np.asarray([[[30.0, 31.0]]], dtype=np.float32),
        "base_velocity": np.asarray([[[0.1, -0.2, 0.3]]], dtype=np.float32),
    }


def fake_robot(*, current: list[float] | None = None, max_relative_target=None):
    current = current or [1000.0 + i for i in range(14)]

    class CommandBuilder:
        def current_canonical_ticks(self, robot_id):
            assert robot_id == 0
            return current

        @staticmethod
        def _base_payload(seq, source_id, source_role, lease_ms):
            return {
                "schema": "xlerobot_v1.1",
                "seq": seq,
                "source_id": source_id,
                "source_role": source_role,
                "lease_ms": lease_ms,
            }

    class Socket:
        def __init__(self):
            self.messages = []

        def send(self, payload, flags=0):
            self.messages.append((payload, flags))

    return SimpleNamespace(
        robot_id=0,
        follower_calibration={},
        config=SimpleNamespace(max_relative_target=max_relative_target, leader_action_units="degrees"),
        command_builder=CommandBuilder(),
        _seq=7,
        source_id="test_policy",
        source_role="policy_smoke",
        command_lease_ms=200,
        zmq_cmd_socket=Socket(),
    )


def test_gr00t_observation_uses_expected_modalities():
    obs = gr00t_observation(raw_observation(), "pick task", "annotation.human.task_description")

    assert set(obs) == {"video", "state", "language"}
    assert obs["video"]["head"].shape == (1, 1, 480, 640, 3)
    assert obs["video"]["head"].dtype == np.uint8
    assert obs["state"]["left_arm"].shape == (1, 1, 5)
    assert obs["state"]["left_gripper"].shape == (1, 1, 1)
    assert obs["state"]["right_arm"].shape == (1, 1, 5)
    assert obs["state"]["right_gripper"].shape == (1, 1, 1)
    assert obs["state"]["head"].shape == (1, 1, 2)
    assert obs["state"]["base_velocity"].shape == (1, 1, 3)
    assert obs["language"]["annotation.human.task_description"] == [["pick task"]]


def test_warn_zero_images_reports_empty_camera_frames():
    obs = raw_observation()
    assert warn_zero_images(obs) == ["head"]


def test_ensure_tcp_connectable_reports_camera_port_failure(monkeypatch):
    def fail_create_connection(address, timeout):
        raise ConnectionRefusedError("refused")

    import scripts.gr00t_n17.indory_n17_robot_smoke as smoke

    monkeypatch.setattr(smoke.socket, "create_connection", fail_create_connection)

    with pytest.raises(RuntimeError, match=r"Camera port 1\.2\.3\.4:8866 is not reachable"):
        ensure_tcp_connectable("1.2.3.4", 8866, 0.1, label="Camera")


def test_robot_action_from_chunk_blocks_base_by_default():
    action = robot_action_from_chunk(action_chunk())

    assert [action[name] for name in LEFT_ARM_MOTORS] == pytest.approx([10, 11, 12, 13, 14])
    assert [action[name] for name in LEFT_GRIPPER_MOTORS] == pytest.approx([15])
    assert [action[name] for name in RIGHT_ARM_MOTORS] == pytest.approx([20, 21, 22, 23, 24])
    assert [action[name] for name in RIGHT_GRIPPER_MOTORS] == pytest.approx([25])
    assert [action[name] for name in HEAD_MOTORS] == pytest.approx([30, 31])
    assert [action[name] for name in BASE_KEYS] == pytest.approx([0, 0, 0])


def test_robot_action_from_chunk_allows_base_only_when_explicit():
    action = robot_action_from_chunk(action_chunk(), allow_base_motion=True)

    assert [action[name] for name in BASE_KEYS] == pytest.approx([0.1, -0.2, 0.3])


def test_capped_raw_policy_action_maps_canonical_order_and_blocks_base():
    robot = fake_robot(max_relative_target=None)
    action = robot_action_from_chunk(action_chunk(), allow_base_motion=True)

    sent = capped_raw_policy_action(robot, action)

    assert [sent[name] for name in RIGHT_MOTORS] == pytest.approx([20, 21, 22, 23, 24, 25])
    assert [sent[name] for name in LEFT_MOTORS] == pytest.approx([10, 11, 12, 13, 14, 15])
    assert [sent[name] for name in HEAD_MOTORS] == pytest.approx([30, 31])
    assert [sent[name] for name in BASE_KEYS] == pytest.approx([0, 0, 0])


def test_send_payload_omits_base_velocity_without_explicit_allowance():
    robot = fake_robot(max_relative_target=None)
    action = robot_action_from_chunk(action_chunk(), allow_base_motion=True)

    sent = send_capped_raw_policy_action(robot, action)

    payload = msgpack.unpackb(robot.zmq_cmd_socket.messages[0][0], raw=False)
    assert "base_cmd_vel" not in payload
    assert payload["arm_joint_pos_target"] == pytest.approx([sent[motor] for motor in CANONICAL_MOTORS])
    assert payload["arm_joint_pos_target_units"] == "feetech_ticks"
    assert robot._seq == 8


def test_send_payload_includes_base_velocity_only_when_allowed():
    robot = fake_robot(max_relative_target=None)
    action = robot_action_from_chunk(action_chunk(), allow_base_motion=True)

    send_capped_raw_policy_action(robot, action, allow_base_motion=True)

    payload = msgpack.unpackb(robot.zmq_cmd_socket.messages[0][0], raw=False)
    assert payload["base_cmd_vel"] == pytest.approx([0.1, -0.2, 0.3])


def test_close_robot_no_send_closes_sockets_without_stop_command():
    calls = []
    robot = SimpleNamespace(
        is_connected=True,
        _is_connected=True,
        disconnect=lambda: calls.append("disconnect"),
        disconnect_sockets=lambda: calls.append("disconnect_sockets"),
    )

    close_robot(robot, send_stop=False)

    assert calls == ["disconnect_sockets"]
    assert robot._is_connected is False


def test_close_robot_send_uses_robot_disconnect_for_stop_command():
    calls = []
    robot = SimpleNamespace(
        is_connected=True,
        _is_connected=True,
        disconnect=lambda: calls.append("disconnect"),
        disconnect_sockets=lambda: calls.append("disconnect_sockets"),
    )

    close_robot(robot, send_stop=True)

    assert calls == ["disconnect"]
