from __future__ import annotations

import json
import urllib.request

import numpy as np

from lerobot.indory.live_debug_view import LiveDebugWebView, detector_debug_summary
from lerobot.indory.success_detector import SkyBlueParcelColorSegmentDetector


def test_detector_debug_summary_includes_color_segment_thresholds():
    detector = SkyBlueParcelColorSegmentDetector(
        bottom_y_ratio=0.42,
        position_metric="centroid",
        required_consecutive=15,
    )

    summary = detector_debug_summary(detector)

    assert summary["type"] == "SkyBlueParcelColorSegmentDetector"
    assert summary["bottom_y_ratio"] == 0.42
    assert summary["success_region"] == "lower"
    assert summary["position_metric"] == "centroid"
    assert summary["required_consecutive"] == 15


def test_live_debug_web_view_serves_snapshot_and_annotated_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[330:450, 260:520] = np.array([90, 185, 230], dtype=np.uint8)
    detector = SkyBlueParcelColorSegmentDetector(bottom_y_ratio=0.65, min_area_ratio=0.01)
    detection = detector.detect({"head": frame}, step=0, elapsed_s=0.0, task="debug")
    view = LiveDebugWebView(
        host="127.0.0.1",
        port=0,
        open_browser=False,
        cli_args={"task": "debug"},
        detector=detector,
    )

    view.start()
    try:
        view.update({"head": frame, "head_motor_1": 2070, "head_motor_2": 2700}, detection=detection, step=1)

        snapshot = json.loads(urllib.request.urlopen(view.url + "snapshot.json", timeout=2).read())
        frame_jpeg = urllib.request.urlopen(view.url + "frame.jpg", timeout=2).read()
    finally:
        view.close()

    assert snapshot["detection"]["success"] is True
    assert snapshot["head"] == {"head_motor_1": 2070, "head_motor_2": 2700}
    assert frame_jpeg[:2] == b"\xff\xd8"
