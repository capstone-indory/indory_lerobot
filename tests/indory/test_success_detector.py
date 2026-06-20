from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lerobot.indory.parcel_grasp_detector import DEFAULT_MODEL_PATH
from lerobot.indory.success_detector import (
    DetectionBox,
    SkyBlueParcelColorSegmentDetector,
    SkyBlueParcelYoloDetector,
    boxes_from_ultralytics,
    make_success_detector,
    parse_detector_kwargs,
)


def test_manual_file_success_detector(tmp_path):
    flag = tmp_path / "success.flag"
    detector = make_success_detector("manual-file", manual_file=flag)

    assert not detector.detect({}, step=0, elapsed_s=0.0, task="pick").success

    flag.write_text("success")
    detection = detector.detect({}, step=1, elapsed_s=0.1, task="pick")

    assert detection.success
    assert str(flag) in detection.reason


def test_python_success_detector_hook(tmp_path, monkeypatch):
    module = tmp_path / "custom_detector.py"
    module.write_text(
        """
class Detector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def detect(self, raw_observation, *, step, elapsed_s, task):
        return {
            "success": raw_observation["score"] >= self.threshold,
            "reason": task,
            "step": step,
        }
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    detector = make_success_detector(
        "custom_detector:Detector",
        kwargs=parse_detector_kwargs('{"threshold": 0.8}'),
    )
    detector.reset()
    detection = detector.detect({"score": 0.9}, step=3, elapsed_s=1.0, task="basket")

    assert detection.success
    assert detection.reason == "basket"
    assert detection.metadata["step"] == 3


def test_sky_blue_parcel_yolo_detector_requires_head_pose_and_lower_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = SkyBlueParcelYoloDetector(
        head_targets={"head_motor_1": 2100.0, "head_motor_2": 2600.0},
        head_tolerance_ticks=5.0,
        bottom_y_ratio=0.6,
        predictor=lambda _frame: [
            DetectionBox((200.0, 340.0, 360.0, 460.0), confidence=0.9, class_name="parcel")
        ],
    )

    not_ready = detector.detect(
        {"head": frame, "head_motor_1": 2000.0, "head_motor_2": 2600.0},
        step=0,
        elapsed_s=0.0,
        task="pick",
    )
    assert not not_ready.success
    assert "head" in not_ready.reason

    ready = detector.detect(
        {"head": frame, "head_motor_1": 2102.0, "head_motor_2": 2598.0},
        step=1,
        elapsed_s=0.1,
        task="pick",
    )
    assert ready.success
    assert ready.metadata["bbox_xyxy"] == (200.0, 340.0, 360.0, 460.0)


def test_sky_blue_parcel_yolo_detector_accepts_head_ranges():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = SkyBlueParcelYoloDetector(
        head_ranges={"head_motor_1": [2000.0, 2200.0], "head_motor_2": [2500.0, 2800.0]},
        bottom_y_ratio=0.6,
        predictor=lambda _frame: [
            DetectionBox((200.0, 340.0, 360.0, 460.0), confidence=0.9, class_name="parcel")
        ],
    )

    outside = detector.detect(
        {"head": frame, "head_motor_1": 2300.0, "head_motor_2": 2600.0},
        step=0,
        elapsed_s=0.0,
        task="pick",
    )
    assert not outside.success

    inside = detector.detect(
        {"head": frame, "head_motor_1": 2100.0, "head_motor_2": 2700.0},
        step=1,
        elapsed_s=0.1,
        task="pick",
    )
    assert inside.success


def test_sky_blue_parcel_yolo_detector_can_require_upper_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = SkyBlueParcelYoloDetector(
        bottom_y_ratio=0.4,
        success_region="upper",
        predictor=lambda _frame: [
            DetectionBox((200.0, 80.0, 360.0, 180.0), confidence=0.9, class_name="parcel")
        ],
    )

    detection = detector.detect({"head": frame}, step=1, elapsed_s=0.1, task="drop")

    assert detection.success
    assert detection.metadata["success_region"] == "upper"
    assert detection.metadata["metric_y_ratio"] <= 0.4


def test_sky_blue_parcel_color_segment_detector_requires_lower_segment():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sky_blue = np.array([90, 185, 230], dtype=np.uint8)
    detector = SkyBlueParcelColorSegmentDetector(bottom_y_ratio=0.65, min_area_ratio=0.01)

    frame[60:180, 260:520] = sky_blue
    top = detector.detect({"head": frame}, step=0, elapsed_s=0.0, task="pick")
    assert not top.success
    assert top.metadata["centroid_y_ratio"] < 0.65

    frame[:] = 0
    frame[330:455, 230:520] = sky_blue
    bottom = detector.detect({"head": frame}, step=1, elapsed_s=0.1, task="pick")
    assert bottom.success
    assert bottom.metadata["centroid_y_ratio"] >= 0.65


def test_sky_blue_parcel_color_segment_detector_can_require_upper_segment():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sky_blue = np.array([90, 185, 230], dtype=np.uint8)
    detector = SkyBlueParcelColorSegmentDetector(
        bottom_y_ratio=0.45,
        success_region="upper",
        min_area_ratio=0.01,
        position_metric="centroid",
    )

    frame[330:455, 230:520] = sky_blue
    bottom = detector.detect({"head": frame}, step=0, elapsed_s=0.0, task="drop")
    assert not bottom.success
    assert bottom.metadata["metric_y_ratio"] > 0.45

    frame[:] = 0
    frame[60:180, 260:520] = sky_blue
    top = detector.detect({"head": frame}, step=1, elapsed_s=0.1, task="drop")
    assert top.success
    assert top.metadata["success_region"] == "upper"
    assert top.metadata["metric_y_ratio"] <= 0.45


def test_sky_blue_parcel_color_segment_detector_rejects_oversized_segment():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = np.array([90, 185, 230], dtype=np.uint8)
    detector = SkyBlueParcelColorSegmentDetector(max_area_ratio=0.45)

    detection = detector.detect({"head": frame}, step=0, elapsed_s=0.0, task="pick")

    assert not detection.success
    assert detection.reason == "no sky-blue color segment"


def test_make_success_detector_builds_sky_blue_parcel_yolo_detector():
    detector = make_success_detector(
        "sky-blue-parcel-yolo",
        kwargs={"head_targets": [2100.0, 2600.0], "bottom_y_ratio": 0.7},
    )

    assert isinstance(detector, SkyBlueParcelYoloDetector)
    assert detector.model_path == DEFAULT_MODEL_PATH
    assert detector.target_class_names == {"parcel"}
    assert detector.inspection_head_targets() == {"head_motor_1": 2100.0, "head_motor_2": 2600.0}


def test_make_success_detector_builds_sky_blue_parcel_color_segment_detector():
    detector = make_success_detector(
        "sky-blue-parcel-color",
        kwargs={"hsv_lower": [82, 70, 80], "bottom_y_ratio": 0.7},
    )

    assert isinstance(detector, SkyBlueParcelColorSegmentDetector)


def test_make_success_detector_builds_default_parcel_grasp_yolo_detector():
    detector = make_success_detector("parcel-grasp-yolo")

    assert isinstance(detector, SkyBlueParcelYoloDetector)
    assert detector.model_path == DEFAULT_MODEL_PATH
    assert detector.target_class_names == {"parcel"}


def test_boxes_from_ultralytics_reads_obb_polygons():
    result = SimpleNamespace(
        names={0: "parcel"},
        boxes=None,
        obb=SimpleNamespace(
            xyxyxyxy=np.array([[[10.0, 20.0], [40.0, 15.0], [50.0, 60.0], [12.0, 65.0]]]),
            conf=np.array([0.87]),
            cls=np.array([0]),
        ),
    )

    boxes = boxes_from_ultralytics([result])

    assert boxes == [DetectionBox((10.0, 15.0, 50.0, 65.0), 0.87, "parcel")]
