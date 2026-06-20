from __future__ import annotations

import numpy as np
import pytest

from lerobot.robots.xlerobot.xlerobot_cam_bridge_stream import resize_frame_for_lerobot


def test_resize_frame_for_lerobot_center_crops_wide_frame_without_distortion():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(8, dtype=np.uint8)

    out = resize_frame_for_lerobot(frame, width=4, height=4, mode="center_crop")

    assert out.shape == (4, 4, 3)
    assert out[:, 0, 0].tolist() == [2, 2, 2, 2]
    assert out[:, -1, 0].tolist() == [5, 5, 5, 5]


def test_resize_frame_for_lerobot_center_crops_tall_frame_without_distortion():
    frame = np.zeros((8, 4, 3), dtype=np.uint8)
    frame[:, :, 1] = np.arange(8, dtype=np.uint8)[:, None]

    out = resize_frame_for_lerobot(frame, width=4, height=4, mode="center_crop")

    assert out.shape == (4, 4, 3)
    assert out[0, :, 1].tolist() == [2, 2, 2, 2]
    assert out[-1, :, 1].tolist() == [5, 5, 5, 5]


def test_resize_frame_for_lerobot_stretch_keeps_full_frame():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(8, dtype=np.uint8)

    out = resize_frame_for_lerobot(frame, width=4, height=4, mode="stretch")

    assert out.shape == (4, 4, 3)
    assert out[:, 0, 0].tolist() == [0, 0, 0, 0]
    assert out[:, -1, 0].tolist() == [6, 6, 6, 6]


def test_resize_frame_for_lerobot_rejects_unknown_mode():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="resize mode"):
        resize_frame_for_lerobot(frame, width=4, height=4, mode="letterbox")
