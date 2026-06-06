from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional runtime dependency
    zstd = None


class RgbdDepthDecoder:
    def __init__(self) -> None:
        self._zstd_decompressor = zstd.ZstdDecompressor() if zstd is not None else None

    def decode_depth(
        self,
        payload: dict[str, Any],
        warned_encodings: set[str],
    ) -> np.ndarray | None:
        data = payload.get("depth_data")
        if not isinstance(data, (bytes, bytearray)):
            return None
        depth_format = str(payload.get("depth_format") or "").lower()
        width = int(payload.get("depth_width") or 0)
        height = int(payload.get("depth_height") or 0)
        if width <= 0 or height <= 0:
            return None
        raw: bytes | None
        if depth_format == "zstd;16uc1":
            raw = self._decode_zstd(payload, bytes(data), width, height, warned_encodings)
        elif depth_format == "png;16uc1":
            return self._decode_png(data)
        elif depth_format in {"raw16uc1-le", "raw16;16uc1", "raw16"}:
            raw = bytes(data)
        else:
            key = f"rgbd_depth:{depth_format}"
            if key not in warned_encodings:
                logging.warning("Skipping RGB-D depth payload with unsupported depth_format=%s", depth_format)
                warned_encodings.add(key)
            return None
        return _raw_u16_depth(raw, width, height)

    def _decode_zstd(
        self,
        payload: dict[str, Any],
        data: bytes,
        width: int,
        height: int,
        warned_encodings: set[str],
    ) -> bytes | None:
        if self._zstd_decompressor is None:
            if "rgbd_zstd" not in warned_encodings:
                logging.warning("Skipping RGB-D depth payloads because zstandard is not installed.")
                warned_encodings.add("rgbd_zstd")
            return None
        try:
            max_len = int(payload.get("depth_uncompressed_len") or width * height * 2)
            return self._zstd_decompressor.decompress(data, max_output_size=max_len)
        except Exception as exc:
            if "rgbd_zstd_decode" not in warned_encodings:
                logging.warning("Failed to decode RGB-D zstd depth payload: %s", exc)
                warned_encodings.add("rgbd_zstd_decode")
            return None

    @staticmethod
    def _decode_png(data: bytes | bytearray) -> np.ndarray | None:
        depth = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return np.ascontiguousarray(depth.astype(np.uint16, copy=False))


def _raw_u16_depth(raw: bytes | None, width: int, height: int) -> np.ndarray | None:
    expected_len = width * height * 2
    if raw is None or len(raw) < expected_len:
        return None
    depth = np.frombuffer(raw[:expected_len], dtype="<u2").reshape((height, width))
    return np.ascontiguousarray(depth)
