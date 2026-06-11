from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from lerobot.indory.parcel_grasp_detector import DEFAULT_MODEL_PATH as DEFAULT_PARCEL_GRASP_MODEL_PATH


@dataclass
class SuccessDetection:
    success: bool
    reason: str = ""
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionBox:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str = ""


@dataclass
class ColorSegment:
    xyxy: tuple[float, float, float, float]
    area_ratio: float
    centroid: tuple[float, float]
    centroid_y_ratio: float
    bbox_center_y_ratio: float
    bbox_bottom_y_ratio: float


class SuccessDetector(Protocol):
    def reset(self) -> None: ...

    def inspection_head_targets(self) -> dict[str, float] | None: ...

    def detect(
        self,
        raw_observation: dict[str, Any],
        *,
        step: int,
        elapsed_s: float,
        task: str,
    ) -> SuccessDetection: ...


class NullSuccessDetector:
    def reset(self) -> None:
        pass

    def inspection_head_targets(self) -> dict[str, float] | None:
        return None

    def detect(
        self,
        raw_observation: dict[str, Any],
        *,
        step: int,
        elapsed_s: float,
        task: str,
    ) -> SuccessDetection:
        return SuccessDetection(False)


class ManualFileSuccessDetector:
    """Stops when a file contains a truthy success value.

    This is useful while bootstrapping a vision detector: another process or a
    human-in-the-loop tool can write "success" to the file, and the live runner
    will stop on the next observation cycle.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def reset(self) -> None:
        pass

    def inspection_head_targets(self) -> dict[str, float] | None:
        return None

    def detect(
        self,
        raw_observation: dict[str, Any],
        *,
        step: int,
        elapsed_s: float,
        task: str,
    ) -> SuccessDetection:
        if not self.path.exists():
            return SuccessDetection(False)
        text = self.path.read_text(errors="ignore").strip().lower()
        if text in {"", "1", "true", "yes", "success", "done", "stop"}:
            return SuccessDetection(True, reason=f"manual file {self.path}")
        return SuccessDetection(False, reason=f"manual file value {text!r}")


class PythonCallableSuccessDetector:
    def __init__(self, spec: str, kwargs: dict[str, Any] | None = None):
        module_name, sep, attr_name = spec.partition(":")
        if not sep or not module_name or not attr_name:
            raise ValueError("--success-detector python hook must be formatted as module:attribute")
        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)
        kwargs = kwargs or {}
        instance = target(**kwargs) if isinstance(kwargs, dict) else target(kwargs)
        self._target = instance

    def reset(self) -> None:
        if hasattr(self._target, "reset"):
            self._target.reset()

    def inspection_head_targets(self) -> dict[str, float] | None:
        if hasattr(self._target, "inspection_head_targets"):
            targets = self._target.inspection_head_targets()
            return normalize_head_targets(targets)
        return None

    def detect(
        self,
        raw_observation: dict[str, Any],
        *,
        step: int,
        elapsed_s: float,
        task: str,
    ) -> SuccessDetection:
        if hasattr(self._target, "detect"):
            result = self._target.detect(raw_observation, step=step, elapsed_s=elapsed_s, task=task)
        else:
            result = self._target(raw_observation, step=step, elapsed_s=elapsed_s, task=task)
        return normalize_detection(result)


class SkyBlueParcelYoloDetector:
    """Detect task success from the fixed-pose head RGB view.

    The YOLO model is intentionally loaded lazily so the runner can be wired now
    and a trained parcel detector can be dropped in later with only a model path.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        target_class_names: str | list[str] | tuple[str, ...] | None = None,
        confidence_threshold: float = 0.25,
        bottom_y_ratio: float = 0.65,
        min_box_area_ratio: float = 0.0,
        required_consecutive: int = 1,
        head_targets: dict[str, float] | list[float] | tuple[float, float] | None = None,
        head_ranges: dict[str, Any] | list[Any] | tuple[Any, ...] | None = None,
        head_tolerance_ticks: float = 40.0,
        frame_key: str = "head",
        predictor: Callable[[np.ndarray], Any] | None = None,
    ):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.target_class_names = normalize_class_names(target_class_names)
        self.confidence_threshold = float(confidence_threshold)
        self.bottom_y_ratio = float(bottom_y_ratio)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.required_consecutive = max(1, int(required_consecutive))
        self.head_targets = normalize_head_targets(head_targets)
        self.head_ranges = normalize_head_ranges(head_ranges)
        self.head_tolerance_ticks = abs(float(head_tolerance_ticks))
        self.frame_key = frame_key
        self._predictor = predictor
        self._model: Any | None = None
        self._consecutive = 0

    def reset(self) -> None:
        self._consecutive = 0

    def inspection_head_targets(self) -> dict[str, float] | None:
        return dict(self.head_targets) if self.head_targets else None

    def detect(
        self,
        raw_observation: dict[str, Any],
        *,
        step: int,
        elapsed_s: float,
        task: str,
    ) -> SuccessDetection:
        del elapsed_s, task
        if not self._head_ready(raw_observation):
            self._consecutive = 0
            return SuccessDetection(False, reason="head not at success inspection target")

        frame = raw_observation.get(self.frame_key)
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            self._consecutive = 0
            return SuccessDetection(False, reason=f"missing RGB frame {self.frame_key!r}")

        best = self._best_success_box(frame)
        if best is None:
            self._consecutive = 0
            return SuccessDetection(False, reason="no sky-blue parcel box in success zone")

        self._consecutive += 1
        metadata = {
            "bbox_xyxy": best.xyxy,
            "confidence": best.confidence,
            "class_name": best.class_name,
            "consecutive": self._consecutive,
            "step": step,
        }
        if self._consecutive < self.required_consecutive:
            return SuccessDetection(False, reason="success box needs consecutive confirmation", metadata=metadata)
        return SuccessDetection(True, reason="sky-blue parcel box is in lower head RGB region", metadata=metadata)

    def _head_ready(self, raw_observation: dict[str, Any]) -> bool:
        return head_pose_ready(
            raw_observation,
            head_targets=self.head_targets,
            head_ranges=self.head_ranges,
            head_tolerance_ticks=self.head_tolerance_ticks,
        )

    def _best_success_box(self, frame: np.ndarray) -> DetectionBox | None:
        height, width = frame.shape[:2]
        min_area = float(height * width) * max(0.0, self.min_box_area_ratio)
        candidates: list[DetectionBox] = []
        for box in self._detect_boxes(frame):
            if box.confidence < self.confidence_threshold:
                continue
            if self.target_class_names and box.class_name.lower() not in self.target_class_names:
                continue
            x1, y1, x2, y2 = box.xyxy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < min_area:
                continue
            center_y_ratio = ((y1 + y2) * 0.5) / max(1.0, float(height))
            if center_y_ratio >= self.bottom_y_ratio:
                candidates.append(box)
        return max(candidates, key=lambda b: b.confidence, default=None)

    def _detect_boxes(self, frame: np.ndarray) -> list[DetectionBox]:
        if self._predictor is not None:
            return normalize_boxes(self._predictor(frame))
        if self.model_path is None:
            return []
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is required for sky-blue-parcel-yolo success detection"
                ) from exc
            self._model = YOLO(str(self.model_path))
        return boxes_from_ultralytics(self._model.predict(frame, verbose=False, conf=self.confidence_threshold))


class SkyBlueParcelColorSegmentDetector:
    """Detect task success from a sky-blue HSV segment in the fixed head RGB view."""

    def __init__(
        self,
        *,
        hsv_lower: list[int] | tuple[int, int, int] = (82, 70, 80),
        hsv_upper: list[int] | tuple[int, int, int] = (125, 255, 255),
        bottom_y_ratio: float = 0.65,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.45,
        required_consecutive: int = 1,
        position_metric: str = "bbox_bottom",
        median_blur: int = 7,
        close_kernel: int = 9,
        open_kernel: int = 5,
        head_targets: dict[str, float] | list[float] | tuple[float, float] | None = None,
        head_ranges: dict[str, Any] | list[Any] | tuple[Any, ...] | None = None,
        head_tolerance_ticks: float = 40.0,
        frame_key: str = "head",
    ):
        self.hsv_lower = normalize_hsv_triplet(hsv_lower, name="hsv_lower")
        self.hsv_upper = normalize_hsv_triplet(hsv_upper, name="hsv_upper")
        self.bottom_y_ratio = float(bottom_y_ratio)
        self.min_area_ratio = max(0.0, float(min_area_ratio))
        self.max_area_ratio = max(self.min_area_ratio, float(max_area_ratio))
        self.required_consecutive = max(1, int(required_consecutive))
        self.position_metric = str(position_metric or "centroid").strip().lower()
        if self.position_metric not in {"centroid", "bbox_center", "bbox_bottom"}:
            raise ValueError("position_metric must be 'centroid', 'bbox_center', or 'bbox_bottom'")
        self.median_blur = max(1, int(median_blur))
        if self.median_blur % 2 == 0:
            self.median_blur += 1
        self.close_kernel = max(0, int(close_kernel))
        self.open_kernel = max(0, int(open_kernel))
        self.head_targets = normalize_head_targets(head_targets)
        self.head_ranges = normalize_head_ranges(head_ranges)
        self.head_tolerance_ticks = abs(float(head_tolerance_ticks))
        self.frame_key = frame_key
        self._consecutive = 0

    def reset(self) -> None:
        self._consecutive = 0

    def inspection_head_targets(self) -> dict[str, float] | None:
        return dict(self.head_targets) if self.head_targets else None

    def detect(
        self,
        raw_observation: dict[str, Any],
        *,
        step: int,
        elapsed_s: float,
        task: str,
    ) -> SuccessDetection:
        del elapsed_s, task
        if not head_pose_ready(
            raw_observation,
            head_targets=self.head_targets,
            head_ranges=self.head_ranges,
            head_tolerance_ticks=self.head_tolerance_ticks,
        ):
            self._consecutive = 0
            return SuccessDetection(False, reason="head not at success inspection target")

        frame = raw_observation.get(self.frame_key)
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            self._consecutive = 0
            return SuccessDetection(False, reason=f"missing RGB frame {self.frame_key!r}")

        segment = self._best_segment(frame)
        if segment is None:
            self._consecutive = 0
            return SuccessDetection(False, reason="no sky-blue color segment")

        metric_y = self._metric_y(segment)
        metadata = {
            "bbox_xyxy": segment.xyxy,
            "area_ratio": segment.area_ratio,
            "centroid": segment.centroid,
            "centroid_y_ratio": segment.centroid_y_ratio,
            "bbox_center_y_ratio": segment.bbox_center_y_ratio,
            "bbox_bottom_y_ratio": segment.bbox_bottom_y_ratio,
            "position_metric": self.position_metric,
            "metric_y_ratio": metric_y,
            "consecutive": self._consecutive,
            "step": step,
        }
        if metric_y < self.bottom_y_ratio:
            self._consecutive = 0
            return SuccessDetection(False, reason="sky-blue color segment is not in lower region", metadata=metadata)

        self._consecutive += 1
        metadata["consecutive"] = self._consecutive
        if self._consecutive < self.required_consecutive:
            return SuccessDetection(False, reason="success color segment needs consecutive confirmation", metadata=metadata)
        return SuccessDetection(True, reason="sky-blue color segment is in lower head RGB region", metadata=metadata)

    def _metric_y(self, segment: ColorSegment) -> float:
        if self.position_metric == "bbox_center":
            return segment.bbox_center_y_ratio
        if self.position_metric == "bbox_bottom":
            return segment.bbox_bottom_y_ratio
        return segment.centroid_y_ratio

    def _best_segment(self, frame: np.ndarray) -> ColorSegment | None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for sky-blue color segmentation") from exc

        rgb = np.asarray(frame, dtype=np.uint8)
        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.asarray(self.hsv_lower, dtype=np.uint8), np.asarray(self.hsv_upper, dtype=np.uint8))
        if self.median_blur > 1:
            mask = cv2.medianBlur(mask, self.median_blur)
        if self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if self.open_kernel > 1:
            kernel = np.ones((self.open_kernel, self.open_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = max(1.0, float(width * height))
        candidates: list[ColorSegment] = []
        for contour in contours:
            area_ratio = float(cv2.contourArea(contour)) / image_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx = float(x + box_width * 0.5)
                cy = float(y + box_height * 0.5)
            candidates.append(
                ColorSegment(
                    xyxy=(float(x), float(y), float(x + box_width), float(y + box_height)),
                    area_ratio=area_ratio,
                    centroid=(cx, cy),
                    centroid_y_ratio=cy / max(1.0, float(height)),
                    bbox_center_y_ratio=(y + box_height * 0.5) / max(1.0, float(height)),
                    bbox_bottom_y_ratio=(y + box_height) / max(1.0, float(height)),
                )
            )
        return max(candidates, key=lambda segment: segment.area_ratio, default=None)


def normalize_detection(result: Any) -> SuccessDetection:
    if isinstance(result, SuccessDetection):
        return result
    if isinstance(result, bool):
        return SuccessDetection(result)
    if isinstance(result, dict):
        return SuccessDetection(
            bool(result.get("success", False)),
            reason=str(result.get("reason", "")),
            score=result.get("score"),
            metadata={k: v for k, v in result.items() if k not in {"success", "reason", "score"}},
        )
    raise TypeError(
        "Success detector must return SuccessDetection, bool, or dict with a 'success' key; "
        f"got {type(result).__name__}"
    )


def normalize_head_targets(targets: Any) -> dict[str, float] | None:
    if targets is None or targets == "":
        return None
    if isinstance(targets, dict):
        out = {
            str(name): float(value)
            for name, value in targets.items()
            if str(name) in {"head_motor_1", "head_motor_2"}
        }
    elif isinstance(targets, (list, tuple)) and len(targets) == 2:
        out = {"head_motor_1": float(targets[0]), "head_motor_2": float(targets[1])}
    else:
        raise ValueError("head_targets must be {'head_motor_1': ..., 'head_motor_2': ...} or [pan, tilt]")
    return out or None


def head_pose_ready(
    raw_observation: dict[str, Any],
    *,
    head_targets: dict[str, float] | None,
    head_ranges: dict[str, tuple[float, float]] | None,
    head_tolerance_ticks: float,
) -> bool:
    if not head_targets and not head_ranges:
        return True
    for name, target in (head_targets or {}).items():
        try:
            current = float(raw_observation[name])
        except (KeyError, TypeError, ValueError):
            return False
        if abs(current - target) > head_tolerance_ticks:
            return False
    for name, (low, high) in (head_ranges or {}).items():
        try:
            current = float(raw_observation[name])
        except (KeyError, TypeError, ValueError):
            return False
        if current < low or current > high:
            return False
    return True


def normalize_head_ranges(ranges: Any) -> dict[str, tuple[float, float]] | None:
    if ranges is None or ranges == "":
        return None
    if isinstance(ranges, dict):
        out: dict[str, tuple[float, float]] = {}
        for name, raw_range in ranges.items():
            name = str(name)
            if name not in {"head_motor_1", "head_motor_2"}:
                continue
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                raise ValueError("head_ranges entries must be [min, max]")
            low, high = float(raw_range[0]), float(raw_range[1])
            out[name] = (min(low, high), max(low, high))
    elif isinstance(ranges, (list, tuple)) and len(ranges) == 2:
        out = {
            "head_motor_1": _normalize_head_range_pair(ranges[0]),
            "head_motor_2": _normalize_head_range_pair(ranges[1]),
        }
    else:
        raise ValueError(
            "head_ranges must be {'head_motor_1': [min, max], 'head_motor_2': [min, max]} "
            "or [[pan_min, pan_max], [tilt_min, tilt_max]]"
        )
    return out or None


def _normalize_head_range_pair(raw_range: Any) -> tuple[float, float]:
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        raise ValueError("head_ranges list entries must be [min, max]")
    low, high = float(raw_range[0]), float(raw_range[1])
    return min(low, high), max(low, high)


def normalize_class_names(names: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if names is None or names == "":
        return set()
    if isinstance(names, str):
        values = names.split(",")
    else:
        values = names
    return {str(name).strip().lower() for name in values if str(name).strip()}


def normalize_hsv_triplet(values: Any, *, name: str) -> tuple[int, int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must be [h, s, v]")
    h, s, v = (int(round(float(value))) for value in values)
    return max(0, min(179, h)), max(0, min(255, s)), max(0, min(255, v))


def normalize_boxes(raw: Any) -> list[DetectionBox]:
    if raw is None:
        return []
    boxes: list[DetectionBox] = []
    for item in raw:
        if isinstance(item, DetectionBox):
            boxes.append(item)
            continue
        if isinstance(item, dict):
            xyxy = item.get("xyxy") or item.get("bbox") or item.get("box")
            if xyxy is None or len(xyxy) != 4:
                continue
            boxes.append(
                DetectionBox(
                    xyxy=tuple(float(value) for value in xyxy),
                    confidence=float(item.get("confidence", item.get("conf", 1.0))),
                    class_name=str(item.get("class_name", item.get("label", ""))),
                )
            )
    return boxes


def boxes_from_ultralytics(results: Any) -> list[DetectionBox]:
    if not results:
        return []
    result = results[0]
    raw_obb = getattr(result, "obb", None)
    if raw_obb is not None:
        obb_boxes = boxes_from_ultralytics_obb(result, raw_obb)
        if obb_boxes:
            return obb_boxes
    raw_boxes = getattr(result, "boxes", None)
    if raw_boxes is None:
        return []
    xyxy = tensor_to_numpy(getattr(raw_boxes, "xyxy", []))
    confs = tensor_to_numpy(getattr(raw_boxes, "conf", []))
    classes = tensor_to_numpy(getattr(raw_boxes, "cls", []))
    names = getattr(result, "names", {}) or {}
    boxes: list[DetectionBox] = []
    for index, coords in enumerate(xyxy):
        class_index = int(classes[index]) if index < len(classes) else -1
        class_name = str(names.get(class_index, class_index if class_index >= 0 else ""))
        confidence = float(confs[index]) if index < len(confs) else 1.0
        boxes.append(DetectionBox(tuple(float(value) for value in coords[:4]), confidence, class_name))
    return boxes


def boxes_from_ultralytics_obb(result: Any, raw_obb: Any) -> list[DetectionBox]:
    names = getattr(result, "names", {}) or {}
    confs = tensor_to_numpy(getattr(raw_obb, "conf", []))
    classes = tensor_to_numpy(getattr(raw_obb, "cls", []))
    polygons = _optional_numpy_attr(raw_obb, "xyxyxyxy")
    boxes: list[DetectionBox] = []
    if polygons.size:
        polygons = np.asarray(polygons, dtype=np.float32).reshape((-1, 8))
        for index, coords in enumerate(polygons):
            boxes.append(
                DetectionBox(
                    _axis_aligned_xyxy_from_polygon(coords),
                    _indexed_confidence(confs, index),
                    _indexed_class_name(classes, names, index),
                )
            )
        return boxes

    xywhr = _optional_numpy_attr(raw_obb, "xywhr")
    if not xywhr.size:
        return []
    xywhr = np.asarray(xywhr, dtype=np.float32).reshape((-1, 5))
    for index, row in enumerate(xywhr):
        cx, cy, width, height, _rotation = (float(value) for value in row[:5])
        boxes.append(
            DetectionBox(
                (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5),
                _indexed_confidence(confs, index),
                _indexed_class_name(classes, names, index),
            )
        )
    return boxes


def _axis_aligned_xyxy_from_polygon(coords: np.ndarray) -> tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _optional_numpy_attr(obj: Any, name: str) -> np.ndarray:
    try:
        value = getattr(obj, name)
    except Exception:
        return np.asarray([])
    if value is None:
        return np.asarray([])
    return tensor_to_numpy(value)


def _indexed_confidence(confs: np.ndarray, index: int) -> float:
    return float(confs[index]) if index < len(confs) else 1.0


def _indexed_class_name(classes: np.ndarray, names: dict[Any, Any], index: int) -> str:
    class_index = int(classes[index]) if index < len(classes) else -1
    return str(names.get(class_index, class_index if class_index >= 0 else ""))


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def parse_detector_kwargs(raw: str | None) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--success-detector-kwargs must decode to a JSON object")
    return parsed


def default_parcel_yolo_kwargs(kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    detector_kwargs = dict(kwargs or {})
    detector_kwargs.setdefault("model_path", DEFAULT_PARCEL_GRASP_MODEL_PATH)
    detector_kwargs.setdefault("target_class_names", "parcel")
    return detector_kwargs


def make_success_detector(
    spec: str | None,
    *,
    kwargs: dict[str, Any] | None = None,
    manual_file: str | Path | None = None,
) -> SuccessDetector:
    if spec is None or spec == "" or spec == "none":
        return NullSuccessDetector()
    if spec == "manual-file":
        if manual_file is None:
            raise ValueError("--success-file is required with --success-detector=manual-file")
        return ManualFileSuccessDetector(manual_file)
    if spec in {"parcel-grasp-yolo", "parcel-obb-yolo", "indory-parcel-grasp-yolo"}:
        return SkyBlueParcelYoloDetector(**default_parcel_yolo_kwargs(kwargs))
    if spec in {"sky-blue-parcel-yolo", "yolo-sky-blue-parcel"}:
        return SkyBlueParcelYoloDetector(**default_parcel_yolo_kwargs(kwargs))
    if spec in {"sky-blue-parcel-color", "sky-blue-parcel-seg", "sky-blue-color-seg"}:
        return SkyBlueParcelColorSegmentDetector(**(kwargs or {}))
    return PythonCallableSuccessDetector(spec, kwargs)
