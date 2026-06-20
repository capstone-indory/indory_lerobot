from __future__ import annotations

import msgpack
import pytest

from lerobot.robots.xlerobot.config_xlerobot import XLerobotClientConfig
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient
from lerobot.robots.xlerobot.xlerobot_constants import CANONICAL_MOTORS, HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS
from lerobot.robots.xlerobot.xlerobot_protocol import (
    base_cmd_from_action,
    canonical_payload,
    canonical_ticks_from_action,
    canonical_ticks_to_action,
)


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, int]] = []

    def send(self, data: bytes, flags: int = 0) -> None:
        self.sent.append((data, flags))


def canonical_action(values: list[float] | None = None) -> dict[str, float]:
    return canonical_ticks_to_action(values or [float(i) for i in range(len(CANONICAL_MOTORS))])


def test_canonical_ticks_to_action_uses_pi_wire_order():
    action = canonical_ticks_to_action([float(i) for i in range(14)], base_cmd=[0.1, -0.2, 0.3])

    assert action[RIGHT_MOTORS[0]] == 0.0
    assert action[RIGHT_MOTORS[-1]] == 5.0
    assert action[LEFT_MOTORS[0]] == 6.0
    assert action[LEFT_MOTORS[-1]] == 11.0
    assert action[HEAD_MOTORS[0]] == 12.0
    assert action[HEAD_MOTORS[1]] == 13.0
    assert action["x.vel"] == pytest.approx(0.1)
    assert action["y.vel"] == pytest.approx(-0.2)
    assert action["theta.vel"] == pytest.approx(0.3)


def test_canonical_ticks_from_action_fills_current_and_applies_head_override():
    current = [float(i) for i in range(14)]
    action = {
        RIGHT_MOTORS[0]: 100.0,
        LEFT_MOTORS[0]: 200.0,
        HEAD_MOTORS[0]: 300.0,
    }

    canonical = canonical_ticks_from_action(
        action,
        current=current,
        head_override={HEAD_MOTORS[0]: 400.0},
    )

    assert canonical[0] == 100.0
    assert canonical[6] == 200.0
    assert canonical[12] == 400.0
    assert canonical[13] == 13.0


def test_canonical_payload_omits_zero_base_unless_requested():
    payload = canonical_payload(
        seq=7,
        source_id="test",
        source_role="policy",
        lease_ms=300,
        canonical_ticks=[float(i) for i in range(14)],
    )

    assert payload["schema"] == "xlerobot_v1.1"
    assert payload["seq"] == 7
    assert payload["arm_joint_pos_target"] == [float(i) for i in range(14)]
    assert "base_cmd_vel" not in payload

    payload = canonical_payload(
        seq=8,
        source_id="test",
        source_role="policy",
        lease_ms=300,
        canonical_ticks=[float(i) for i in range(14)],
        include_zero_base=True,
    )
    assert payload["base_cmd_vel"] == [0.0, 0.0, 0.0]


def test_base_cmd_from_action_can_lock_base():
    assert base_cmd_from_action({"x.vel": 0.2, "theta.vel": -0.1}, allow_base_action=True) == [
        0.2,
        0.0,
        -0.1,
    ]
    assert base_cmd_from_action({"x.vel": 0.2, "theta.vel": -0.1}, allow_base_action=False) == [
        0.0,
        0.0,
        0.0,
    ]


def test_xlerobot_client_send_canonical_action_owns_seq_and_payload():
    robot = XLerobotClient(
        XLerobotClientConfig(
            remote_ip="127.0.0.1",
            id="test_client",
            source_id="unit_test",
            source_role="policy_eval",
        )
    )
    fake_socket = FakeSocket()
    robot._is_connected = True
    robot.zmq_cmd_socket = fake_socket
    robot._seq = 12

    sent = robot.send_canonical_action(canonical_action(), include_zero_base=True)

    assert sent[RIGHT_MOTORS[0]] == 0.0
    assert sent[LEFT_MOTORS[0]] == 6.0
    assert robot._seq == 13
    assert len(fake_socket.sent) == 1
    payload = msgpack.unpackb(fake_socket.sent[0][0], raw=False)
    assert payload["seq"] == 12
    assert payload["source_id"] == "unit_test"
    assert payload["source_role"] == "policy_eval"
    assert payload["arm_joint_pos_target"] == [float(i) for i in range(14)]
    assert payload["base_cmd_vel"] == [0.0, 0.0, 0.0]


def test_xlerobot_client_capped_canonical_action_uses_public_current_state():
    robot = XLerobotClient(
        XLerobotClientConfig(
            remote_ip="127.0.0.1",
            id="test_client",
            max_relative_target=None,
        )
    )
    robot.latest_topics["proprio.0"] = {"joint_pos": [1000.0 + i for i in range(14)]}
    target = {motor: 2000.0 + i for i, motor in enumerate(CANONICAL_MOTORS)}
    target["x.vel"] = 0.5

    action = robot.capped_canonical_action(
        target,
        allow_base_action=False,
        head_override={HEAD_MOTORS[0]: 3333.0},
    )

    assert action[RIGHT_MOTORS[0]] == 2000.0
    assert action[LEFT_MOTORS[0]] == 2006.0
    assert action[HEAD_MOTORS[0]] == 3333.0
    assert action["x.vel"] == 0.0
