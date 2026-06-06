from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np

from lerobot.datasets.image_writer import write_image
from lerobot.utils.constants import OBS_PREFIX

from .xlerobot_camera_decode import H264Fmp4Decoder


def materialize_camera_archive(dataset: Any, archive_path: Path | str | None) -> dict[str, int]:
    if archive_path is None or dataset.episode_buffer is None:
        return {}
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        return {}
    if hasattr(dataset, "_wait_image_writer"):
        dataset._wait_image_writer()

    decoded, start_ns = _decode_archive(archive_path)
    if not decoded:
        return {}
    timestamps = [float(t) for t in dataset.episode_buffer.get("timestamp", [])]
    if not timestamps:
        return {}
    target_ns = [int(start_ns + timestamp * 1_000_000_000) for timestamp in timestamps]

    counts: dict[str, int] = {}
    for feature_key in _camera_feature_keys(dataset):
        cam_name = _camera_name_from_feature_key(feature_key)
        frames = decoded.get(cam_name)
        paths = dataset.episode_buffer.get(feature_key)
        if not frames or not paths:
            continue
        frame_times = [stamp_ns for stamp_ns, _frame in frames]
        count = 0
        for index, path in enumerate(paths):
            if index >= len(target_ns):
                break
            selected = _nearest_frame_index(frame_times, target_ns[index])
            if selected is None:
                continue
            rgb_frame = frames[selected][1]
            write_image(rgb_frame, Path(path), compress_level=1)
            count += 1
        counts[feature_key] = count
    if counts:
        logging.info("Materialized camera archive %s into dataset frames: %s", archive_path, counts)
    return counts


def _decode_archive(archive_path: Path) -> tuple[dict[str, list[tuple[int, np.ndarray]]], int]:
    decoder = H264Fmp4Decoder()
    warned: set[str] = set()
    topic_to_name: dict[str, str] = {}
    decoded: dict[str, list[tuple[int, np.ndarray]]] = {}
    start_ns: int | None = None
    first_recv_ns: int | None = None

    with archive_path.open("rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for record in unpacker:
            if not isinstance(record, dict):
                continue
            if record.get("kind") == "metadata":
                start_ns = _optional_int(record.get("start_ns"), start_ns)
                raw_mapping = record.get("topic_to_name")
                if isinstance(raw_mapping, dict):
                    topic_to_name = {str(k): str(v) for k, v in raw_mapping.items()}
                continue
            topic = str(record.get("topic") or "")
            payload = record.get("payload")
            if not topic or not isinstance(payload, dict):
                continue
            recv_ns = _optional_int(record.get("recv_ns"), None)
            if recv_ns is None:
                continue
            if first_recv_ns is None:
                first_recv_ns = recv_ns
            frame_bgr = decoder.decode_rgb_payload(topic, payload, warned)
            if frame_bgr is None:
                continue
            cam_name = topic_to_name.get(topic) or _fallback_camera_name(topic)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decoded.setdefault(cam_name, []).append((recv_ns, frame_rgb))
    return decoded, start_ns or first_recv_ns or 0


def _camera_feature_keys(dataset: Any) -> list[str]:
    return [
        key
        for key, feature in dataset.features.items()
        if feature.get("dtype") in ("image", "video") and key in dataset.episode_buffer
    ]


def _camera_name_from_feature_key(feature_key: str) -> str:
    key = feature_key
    if key.startswith(f"{OBS_PREFIX}images."):
        return key.removeprefix(f"{OBS_PREFIX}images.")
    if key.startswith(OBS_PREFIX):
        key = key.removeprefix(OBS_PREFIX)
    return key.rsplit(".", 1)[-1]


def _nearest_frame_index(frame_times: list[int], target_ns: int) -> int | None:
    if not frame_times:
        return None
    pos = bisect.bisect_left(frame_times, target_ns)
    if pos <= 0:
        return 0
    if pos >= len(frame_times):
        return len(frame_times) - 1
    before = pos - 1
    return before if target_ns - frame_times[before] <= frame_times[pos] - target_ns else pos


def _fallback_camera_name(topic: str) -> str:
    if "wrist_left" in topic:
        return "left_wrist"
    if "wrist_right" in topic:
        return "right_wrist"
    if "front" in topic or "head" in topic:
        return "head"
    return topic.rsplit(".", 1)[0].replace("rgb.", "")


def _optional_int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
