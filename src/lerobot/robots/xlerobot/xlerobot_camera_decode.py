from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

try:
    import av
    from av.codec.hwaccel import HWAccel, hwdevices_available

    av.logging.set_level(av.logging.ERROR)
except Exception:  # pragma: no cover - optional runtime dependency
    av = None
    HWAccel = None
    hwdevices_available = None


class H264Fmp4Decoder:
    def __init__(self) -> None:
        self._init_by_topic: dict[str, bytes] = {}
        self._decoders_by_topic: dict[str, Any] = {}
        self._started_by_topic: set[str] = set()
        self._nal_length_size_by_topic: dict[str, int] = {}
        self._annexb_params_by_topic: dict[str, bytes] = {}

    def reset(self) -> None:
        self._decoders_by_topic.clear()
        self._started_by_topic.clear()

    def decode_rgb_payload(
        self,
        topic: str,
        payload: dict[str, Any],
        warned_encodings: set[str],
    ) -> np.ndarray | None:
        data = payload.get("data")
        encoding = str(payload.get("encoding") or "")
        if not isinstance(data, (bytes, bytearray)):
            return None
        if encoding == "jpeg":
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if encoding == "h264_fmp4":
            return self._decode_h264_fmp4(topic, payload, bytes(data), warned_encodings)
        if encoding not in warned_encodings:
            logging.warning("Skipping %s camera payload with unsupported encoding=%s", topic, encoding)
            warned_encodings.add(encoding)
        return None

    def needs_keyframe(self, topic: str) -> bool:
        return topic not in self._started_by_topic

    def payload_has_idr(
        self,
        topic: str,
        payload: dict[str, Any],
        warned_encodings: set[str],
    ) -> bool:
        data = payload.get("data")
        if av is None or not isinstance(data, (bytes, bytearray)):
            return False
        if str(payload.get("encoding") or "") != "h264_fmp4":
            return False
        self._remember_init(topic, payload, warned_encodings)
        if topic not in self._init_by_topic:
            return False
        annexb = _fmp4_fragment_to_annexb(
            bytes(data),
            self._nal_length_size_by_topic.get(topic, 4),
            b"",
        )
        return _annexb_has_idr(annexb) if annexb is not None else False

    def _decode_h264_fmp4(
        self,
        topic: str,
        payload: dict[str, Any],
        data: bytes,
        warned_encodings: set[str],
    ) -> np.ndarray | None:
        if av is None:
            if "h264_fmp4" not in warned_encodings:
                logging.warning("Skipping h264_fmp4 camera payloads because PyAV is not installed.")
                warned_encodings.add("h264_fmp4")
            return None
        self._remember_init(topic, payload, warned_encodings)
        if topic not in self._init_by_topic:
            return None
        annexb = _fmp4_fragment_to_annexb(
            data,
            self._nal_length_size_by_topic.get(topic, 4),
            self._annexb_params_by_topic.get(topic, b""),
        )
        if annexb is None:
            key = f"{topic}:h264_fragment"
            if key not in warned_encodings:
                logging.warning("Skipping malformed h264_fmp4 fragment for %s.", topic)
                warned_encodings.add(key)
            return None
        if topic not in self._started_by_topic and not _annexb_has_idr(annexb):
            return None
        try:
            decoder = self._decoders_by_topic.get(topic)
            if decoder is None:
                decoder = _create_h264_decoder()
                self._decoders_by_topic[topic] = decoder
            frame = _decode_latest_frame(decoder, annexb)
            if frame is not None:
                self._started_by_topic.add(topic)
            return frame
        except Exception as exc:
            self._decoders_by_topic.pop(topic, None)
            if topic not in warned_encodings:
                logging.warning("Failed to decode h264_fmp4 camera payload for %s: %s", topic, exc)
                warned_encodings.add(topic)
            return None

    def _remember_init(self, topic: str, payload: dict[str, Any], warned_encodings: set[str]) -> None:
        init = payload.get("init")
        if isinstance(init, (bytes, bytearray)) and init:
            self._configure_topic(topic, bytes(init), warned_encodings)

    def _configure_topic(self, topic: str, init_bytes: bytes, warned_encodings: set[str]) -> None:
        if self._init_by_topic.get(topic) == init_bytes:
            return
        parsed = _parse_avcc(init_bytes)
        if parsed is None:
            key = f"{topic}:h264_init"
            if key not in warned_encodings:
                logging.warning("Skipping h264_fmp4 init segment without avcC for %s.", topic)
                warned_encodings.add(key)
            return
        length_size, params = parsed
        self._init_by_topic[topic] = init_bytes
        self._nal_length_size_by_topic[topic] = length_size
        self._annexb_params_by_topic[topic] = params
        self._decoders_by_topic.pop(topic, None)
        self._started_by_topic.discard(topic)


def _decode_latest_frame(decoder: Any, annexb: bytes) -> np.ndarray | None:
    frames = decoder.decode(av.Packet(annexb))
    return frames[-1].to_ndarray(format="bgr24") if frames else None


def _create_h264_decoder():
    if HWAccel is not None and hwdevices_available is not None:
        try:
            if "videotoolbox" in hwdevices_available():
                hwaccel = HWAccel("videotoolbox", allow_software_fallback=True)
                return av.CodecContext.create("h264", "r", hwaccel=hwaccel)
        except Exception as exc:
            logging.debug("Could not enable VideoToolbox H.264 decode: %s", exc)
    return av.CodecContext.create("h264", "r")


def _annexb_has_idr(data: bytes) -> bool:
    pos = 0
    while True:
        start = data.find(b"\x00\x00\x00\x01", pos)
        if start < 0:
            return False
        nal_start = start + 4
        if nal_start < len(data) and data[nal_start] & 0x1F == 5:
            return True
        pos = nal_start


def _iter_top_level_mp4_payloads(data: bytes, box_name: bytes):
    offset = 0
    length = len(data)
    while offset + 8 <= length:
        size = int.from_bytes(data[offset : offset + 4], "big")
        name = data[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if offset + 16 > length:
                return
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = length - offset
        if size < header_size or offset + size > length:
            return
        if name == box_name:
            yield data[offset + header_size : offset + size]
        offset += size


def _parse_avcc(init: bytes) -> tuple[int, bytes] | None:
    search_from = 0
    while True:
        marker = init.find(b"avcC", search_from)
        if marker < 4:
            return None
        box_start = marker - 4
        size = int.from_bytes(init[box_start:marker], "big")
        box_end = box_start + size
        search_from = marker + 4
        if size < 12 or box_end > len(init):
            continue
        parsed = _parse_avcc_payload(init[marker + 4 : box_end])
        if parsed is not None:
            return parsed


def _parse_avcc_payload(payload: bytes) -> tuple[int, bytes] | None:
    if len(payload) < 7 or payload[0] != 1:
        return None
    length_size = (payload[4] & 0x03) + 1
    pos = 6
    nals: list[bytes] = []
    for _ in range(payload[5] & 0x1F):
        if pos + 2 > len(payload):
            return None
        nal_len = int.from_bytes(payload[pos : pos + 2], "big")
        pos += 2
        if nal_len <= 0 or pos + nal_len > len(payload):
            return None
        nals.append(payload[pos : pos + nal_len])
        pos += nal_len
    if pos >= len(payload):
        return None
    pps_count = payload[pos]
    pos += 1
    for _ in range(pps_count):
        if pos + 2 > len(payload):
            return None
        nal_len = int.from_bytes(payload[pos : pos + 2], "big")
        pos += 2
        if nal_len <= 0 or pos + nal_len > len(payload):
            return None
        nals.append(payload[pos : pos + nal_len])
        pos += nal_len
    return length_size, b"".join(b"\x00\x00\x00\x01" + nal for nal in nals)


def _fmp4_fragment_to_annexb(data: bytes, length_size: int, prefix: bytes = b"") -> bytes | None:
    if length_size not in (1, 2, 3, 4):
        length_size = 4
    annexb = bytearray(prefix)
    wrote_media = False
    for mdat in _iter_top_level_mp4_payloads(data, b"mdat"):
        pos = 0
        while pos + length_size <= len(mdat):
            nal_len = int.from_bytes(mdat[pos : pos + length_size], "big")
            pos += length_size
            if nal_len <= 0 or pos + nal_len > len(mdat):
                return None
            annexb.extend(b"\x00\x00\x00\x01")
            annexb.extend(mdat[pos : pos + nal_len])
            pos += nal_len
            wrote_media = True
        if pos != len(mdat):
            return None
    return bytes(annexb) if wrote_media else None
