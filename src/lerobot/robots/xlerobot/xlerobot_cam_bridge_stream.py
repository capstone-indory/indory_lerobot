from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CamBridgeCameraSpec:
    name: str
    bridge_name: str
    width: int
    height: int
    fps: float


def resize_frame_for_lerobot(
    frame_rgb: np.ndarray, *, width: int, height: int, mode: str = "center_crop"
) -> np.ndarray:
    frame = np.asarray(frame_rgb, dtype=np.uint8)
    target_width = int(width)
    target_height = int(height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"target size must be positive, got {target_width}x{target_height}")
    if frame.ndim != 3:
        raise ValueError(f"camera frame must be HxWxC, got shape {frame.shape}")
    if frame.shape[:2] == (target_height, target_width):
        return np.ascontiguousarray(frame)

    mode = str(mode or "center_crop").strip().lower().replace("-", "_")
    if mode == "stretch":
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)
    if mode != "center_crop":
        raise ValueError("cam bridge resize mode must be 'center_crop' or 'stretch'")

    source_height, source_width = frame.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = source_width / source_height
    if source_aspect > target_aspect:
        crop_width = max(1, min(source_width, int(round(source_height * target_aspect))))
        x0 = max(0, (source_width - crop_width) // 2)
        frame = frame[:, x0 : x0 + crop_width]
    elif source_aspect < target_aspect:
        crop_height = max(1, min(source_height, int(round(source_width / target_aspect))))
        y0 = max(0, (source_height - crop_height) // 2)
        frame = frame[y0 : y0 + crop_height, :]

    if frame.shape[:2] != (target_height, target_width):
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


class CamBridgeCameraStreamPump:
    """Receive JPEG frames from indory_pi_cam_bridge LeRobot camera endpoints."""

    def __init__(
        self, *, specs: list[CamBridgeCameraSpec], base_url: str, resize_mode: str = "center_crop"
    ) -> None:
        self.specs = list(specs)
        self.base_url = str(base_url or "ws://127.0.0.1:8870").rstrip("/")
        self.resize_mode = str(resize_mode or "center_crop").strip().lower().replace("-", "_")
        if self.resize_mode not in {"center_crop", "stretch"}:
            raise ValueError("resize_mode must be 'center_crop' or 'stretch'")
        self.last_frames: dict[str, np.ndarray] = {}
        self.last_payload_meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._workers: list[_CamBridgeCameraWorker] = []

    def start(self) -> None:
        if any(worker.is_alive() for worker in self._workers):
            return
        self._stop_event.clear()
        self._workers = [
            _CamBridgeCameraWorker(
                spec=spec,
                url=f"{self.base_url}/ws/lerobot/camera/{spec.bridge_name}",
                stop_event=self._stop_event,
                update_frame=self._update_frame,
            )
            for spec in self.specs
        ]
        for worker in self._workers:
            worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout=2.0)
        self._workers = []

    def frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self.last_frames)

    def warm_up(self, timeout_s: float) -> list[str]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        expected = {spec.name for spec in self.specs}
        while time.monotonic() < deadline:
            with self._lock:
                missing = sorted(expected.difference(self.last_frames))
            if not missing:
                return []
            time.sleep(0.005)
        with self._lock:
            return sorted(expected.difference(self.last_frames))

    def start_archive(self, root: Any, episode_index: int) -> None:
        return None

    def stop_archive(self, *, keep: bool = True) -> None:
        return None

    def _update_frame(self, spec: CamBridgeCameraSpec, frame_rgb: np.ndarray) -> None:
        source_shape = tuple(np.asarray(frame_rgb).shape)
        frame = resize_frame_for_lerobot(
            frame_rgb,
            width=int(spec.width),
            height=int(spec.height),
            mode=self.resize_mode,
        )
        with self._lock:
            self.last_frames[spec.name] = frame
            self.last_payload_meta[spec.name] = {
                "transport": "cam_bridge_jpeg_ws",
                "bridge_name": spec.bridge_name,
                "source_shape": source_shape,
                "resize_mode": self.resize_mode,
                "stamp_ns": time.time_ns(),
            }


class _CamBridgeCameraWorker(threading.Thread):
    def __init__(
        self,
        *,
        spec: CamBridgeCameraSpec,
        url: str,
        stop_event: threading.Event,
        update_frame: Any,
    ) -> None:
        super().__init__(name=f"xlerobot-cam-bridge-{spec.name}", daemon=True)
        self.spec = spec
        self.url = url
        self.stop_event = stop_event
        self.update_frame = update_frame
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_error_log = 0.0

    def stop(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(lambda: None)

    def run(self) -> None:
        try:
            asyncio.run(self._run_forever())
        except Exception as exc:
            self._log_error(f"{self.spec.name} cam bridge receiver stopped: {exc}")

    async def _run_forever(self) -> None:
        self._loop = asyncio.get_running_loop()
        while not self.stop_event.is_set():
            try:
                await self._run_once()
            except Exception as exc:
                self._log_error(f"{self.spec.name} cam bridge receiver failed: {exc}")
            if not self.stop_event.is_set():
                await asyncio.sleep(0.2)

    async def _run_once(self) -> None:
        try:
            import websockets
        except Exception as exc:
            raise RuntimeError(
                "camera_transport=cam_bridge requires websockets in this Python environment"
            ) from exc

        try:
            async with websockets.connect(self.url, max_size=16 * 1024 * 1024) as ws:
                while not self.stop_event.is_set():
                    try:
                        payload = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except TimeoutError:
                        continue
                    if isinstance(payload, str) or not payload:
                        continue
                    encoded = np.frombuffer(payload, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    self.update_frame(self.spec, frame_rgb)
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosed:
            pass

    def _log_error(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= 2.0:
            logging.warning(msg)
            self._last_error_log = now
