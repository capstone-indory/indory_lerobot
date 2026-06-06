from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

from .xlerobot_camera_decode import H264Fmp4Decoder


class CameraStreamPump:
    def __init__(
        self,
        *,
        remote_ip: str,
        port: int,
        topics: list[str],
        topic_to_name: dict[str, str],
        camera_names: set[str],
    ) -> None:
        self.remote_ip = remote_ip
        self.port = int(port)
        self.topics = list(topics)
        self.topic_to_name = dict(topic_to_name)
        self.camera_names = set(camera_names)
        self.decoder = H264Fmp4Decoder()
        self.warned_encodings: set[str] = set()
        self.last_frames: dict[str, np.ndarray] = {}
        self.last_payload_meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: zmq.Context | None = None
        self._archive_file = None
        self._archive_path: Path | None = None
        self._archive_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="xlerobot-camera-pump", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.stop_archive(keep=True)
        self.decoder.reset()

    def frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self.last_frames)

    def warm_up(self, timeout_s: float) -> list[str]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                missing = sorted(name for name in self.camera_names if name not in self.last_frames)
            if not missing:
                return []
            time.sleep(0.005)
        with self._lock:
            return sorted(name for name in self.camera_names if name not in self.last_frames)

    def start_archive(self, root: Path | str, episode_index: int) -> Path:
        archive_dir = Path(root) / "camera_archives" / f"episode-{int(episode_index):06d}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / "fragments.msgpack"
        with self._archive_lock:
            self._close_archive_locked()
            self._archive_path = archive_path
            self._archive_file = archive_path.open("wb")
            self._write_archive_record_locked(
                {
                    "kind": "metadata",
                    "schema": "indory_camera_archive_v1",
                    "episode_index": int(episode_index),
                    "start_ns": time.time_ns(),
                    "topics": self.topics,
                    "topic_to_name": self.topic_to_name,
                }
            )
        return archive_path

    def stop_archive(self, *, keep: bool) -> Path | None:
        with self._archive_lock:
            archive_path = self._archive_path
            self._close_archive_locked()
        if archive_path is not None and not keep:
            shutil.rmtree(archive_path.parent, ignore_errors=True)
        return archive_path

    def _close_archive_locked(self) -> None:
        if self._archive_file is not None:
            self._archive_file.close()
            self._archive_file = None
        self._archive_path = None

    def _run(self) -> None:
        self._context = zmq.Context()
        sock = self._context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 512)
        sock.setsockopt(zmq.RCVTIMEO, 50)
        for topic in self.topics:
            sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        sock.connect(f"tcp://{self.remote_ip}:{self.port}")
        try:
            while not self._stop_event.is_set():
                try:
                    topic_raw, payload_raw = sock.recv_multipart()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    if not self._stop_event.is_set():
                        logging.exception("Camera stream socket failed.")
                    break
                topic = topic_raw.decode("utf-8", errors="replace")
                try:
                    payload = msgpack.unpackb(payload_raw, raw=False)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    self._archive_payload(topic, payload)
                    self._update_preview_frame(topic, payload)
        finally:
            sock.close(0)
            if self._context is not None:
                self._context.term()
                self._context = None

    def _archive_payload(self, topic: str, payload: dict[str, Any]) -> None:
        record = {
            "kind": "payload",
            "recv_ns": time.time_ns(),
            "topic": topic,
            "payload": payload,
        }
        with self._archive_lock:
            self._write_archive_record_locked(record)

    def _write_archive_record_locked(self, record: dict[str, Any]) -> None:
        if self._archive_file is not None:
            self._archive_file.write(msgpack.packb(record, use_bin_type=True))

    def _update_preview_frame(self, topic: str, payload: dict[str, Any]) -> None:
        cam_name = self.topic_to_name.get(topic)
        if cam_name not in self.camera_names:
            return
        frame = self.decoder.decode_rgb_payload(topic, payload, self.warned_encodings)
        if frame is None:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            self.last_frames[cam_name] = rgb_frame
            self.last_payload_meta[cam_name] = {
                "topic": topic,
                "stamp_ns": payload.get("stamp_ns"),
                "chunk_seq": payload.get("chunk_seq"),
                "encoding": payload.get("encoding"),
            }
