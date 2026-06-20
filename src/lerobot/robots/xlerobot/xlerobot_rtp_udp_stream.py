from __future__ import annotations

import logging
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional depth dependency
    zstd = None


@dataclass(frozen=True)
class RtpUdpCameraSpec:
    name: str
    port: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class Idd2DepthUdpSpec:
    name: str
    port: int
    width: int
    height: int


class RtpUdpCameraStreamPump:
    """Receive Indoory ROS-profile H.264 RTP/UDP streams as LeRobot RGB frames."""

    def __init__(
        self,
        *,
        specs: list[RtpUdpCameraSpec],
        bind_ip: str,
        payload_type: int,
        ffmpeg_path: str | None,
        depth_spec: Idd2DepthUdpSpec | None = None,
    ) -> None:
        self.specs = list(specs)
        self.bind_ip = str(bind_ip or "0.0.0.0")
        self.payload_type = int(payload_type)
        self.ffmpeg_path = ffmpeg_path
        self.depth_spec = depth_spec
        self.last_frames: dict[str, np.ndarray] = {}
        self.last_depth_frames: dict[str, np.ndarray] = {}
        self.last_payload_meta: dict[str, dict[str, Any]] = {}
        self.last_depth_meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._workers: list[_RtpUdpCameraWorker] = []
        self._depth_worker: _Idd2DepthUdpWorker | None = None

    def start(self) -> None:
        if any(worker.is_alive() for worker in self._workers):
            return
        ffmpeg = self._resolve_ffmpeg()
        self._stop_event.clear()
        self._workers = [
            _RtpUdpCameraWorker(
                spec=spec,
                bind_ip=self.bind_ip,
                payload_type=self.payload_type,
                ffmpeg=ffmpeg,
                stop_event=self._stop_event,
                update_frame=self._update_frame,
            )
            for spec in self.specs
        ]
        if self.depth_spec is not None:
            self._depth_worker = _Idd2DepthUdpWorker(
                spec=self.depth_spec,
                bind_ip=self.bind_ip,
                stop_event=self._stop_event,
                update_depth=self._update_depth,
            )
        for worker in self._workers:
            worker.start()
        if self._depth_worker is not None:
            self._depth_worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        for worker in self._workers:
            worker.stop()
        if self._depth_worker is not None:
            self._depth_worker.stop()
        for worker in self._workers:
            worker.join(timeout=2.0)
        if self._depth_worker is not None:
            self._depth_worker.join(timeout=2.0)
            self._depth_worker = None
        self._workers = []

    def frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self.last_frames)

    def depth_frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self.last_depth_frames)

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

    def start_archive(self, root: Path | str, episode_index: int) -> None:
        return None

    def stop_archive(self, *, keep: bool = True) -> None:
        return None

    def _update_frame(self, spec: RtpUdpCameraSpec, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self.last_frames[spec.name] = frame_rgb
            self.last_payload_meta[spec.name] = {
                "transport": "rtp_udp",
                "port": int(spec.port),
                "stamp_ns": time.time_ns(),
            }

    def _update_depth(self, spec: Idd2DepthUdpSpec, depth: np.ndarray, meta: dict[str, Any]) -> None:
        with self._lock:
            self.last_depth_frames[spec.name] = depth
            self.last_depth_meta[spec.name] = meta

    def _resolve_ffmpeg(self) -> str:
        if self.ffmpeg_path:
            return str(self.ffmpeg_path)
        ffmpeg = os.environ.get("INDOORY_FFMPEG") or shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError(
                "ffmpeg is required for camera_transport=rtp_udp; set rtp_udp_ffmpeg_path or INDOORY_FFMPEG"
            )
        return ffmpeg


class _DepthChunks:
    def __init__(self, *, chunk_count: int, compressed_len: int, first_seen: float, meta: dict[str, Any]) -> None:
        self.chunk_count = int(chunk_count)
        self.compressed_len = int(compressed_len)
        self.first_seen = float(first_seen)
        self.meta = dict(meta)
        self.chunks: list[bytes | None] = [None] * int(chunk_count)

    def add(self, index: int, chunk: bytes) -> None:
        if 0 <= index < self.chunk_count:
            self.chunks[index] = bytes(chunk)

    def complete(self) -> bool:
        return all(chunk is not None for chunk in self.chunks)

    def join(self) -> bytes:
        data = b"".join(chunk or b"" for chunk in self.chunks)
        return data[: self.compressed_len]


class _Idd2DepthUdpWorker(threading.Thread):
    def __init__(
        self,
        *,
        spec: Idd2DepthUdpSpec,
        bind_ip: str,
        stop_event: threading.Event,
        update_depth: Any,
    ) -> None:
        super().__init__(name=f"xlerobot-idd2-depth-{spec.name}", daemon=True)
        self.spec = spec
        self.bind_ip = str(bind_ip or "0.0.0.0")
        self.stop_event = stop_event
        self.update_depth = update_depth
        self._sock: socket.socket | None = None
        self._frames: dict[int, _DepthChunks] = {}
        self._warned_zstd = False
        self._last_error_log = 0.0

    def stop(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    def run(self) -> None:
        if zstd is None:
            self._log_error("zstandard is required to decode IDD2 depth UDP; depth frames will be unavailable")
            return
        decompressor = zstd.ZstdDecompressor()
        while not self.stop_event.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.bind_ip, int(self.spec.port)))
                sock.settimeout(0.2)
                self._sock = sock
                while not self.stop_event.is_set():
                    try:
                        packet, _addr = sock.recvfrom(65535)
                    except socket.timeout:
                        self._drop_stale_frames()
                        continue
                    except OSError:
                        break
                    depth = self._handle_packet(packet, decompressor)
                    if depth is not None:
                        frame, meta = depth
                        self.update_depth(self.spec, frame, meta)
            except Exception as exc:
                self._log_error(f"{self.spec.name} IDD2 depth receiver failed: {exc}")
            finally:
                self.stop()
            if not self.stop_event.is_set():
                self.stop_event.wait(0.2)

    def _handle_packet(self, packet: bytes, decompressor: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
        if len(packet) < 80 or packet[:4] != b"IDD2":
            return None
        header_size = struct.calcsize(">4sHHIHHHHIIIQ")
        if len(packet) < header_size:
            return None
        (
            _magic,
            version,
            header_len,
            frame_seq,
            chunk_index,
            chunk_count,
            width,
            height,
            quant_um,
            raw_len,
            compressed_len,
            stamp_ns,
        ) = struct.unpack(">4sHHIHHHHIIIQ", packet[:header_size])
        if version != 2 or header_len < 80 or chunk_count <= 0:
            return None
        if len(packet) < header_len:
            return None
        payload = packet[header_len:]
        meta = {
            "transport": "idd2_depth_udp",
            "port": int(self.spec.port),
            "frame_seq": int(frame_seq),
            "stamp_ns": int(stamp_ns),
            "depth_width": int(width),
            "depth_height": int(height),
            "depth_format": "idd2;zstd;quantized16uc1",
            "depth_quant_um": int(quant_um),
            "depth_uncompressed_len": int(raw_len),
            "depth_compressed_len": int(compressed_len),
        }
        frame = self._frames.get(frame_seq)
        if frame is None or frame.chunk_count != int(chunk_count):
            frame = _DepthChunks(
                chunk_count=int(chunk_count),
                compressed_len=int(compressed_len),
                first_seen=time.monotonic(),
                meta=meta,
            )
            self._frames[frame_seq] = frame
        frame.add(int(chunk_index), payload)
        self._drop_stale_frames()
        if not frame.complete():
            return None
        self._frames.pop(frame_seq, None)
        compressed = frame.join()
        try:
            raw = decompressor.decompress(compressed, max_output_size=int(raw_len))
        except Exception as exc:
            self._log_error(f"failed to decode IDD2 zstd depth frame: {exc}")
            return None
        expected = int(width) * int(height) * 2
        if len(raw) < expected:
            return None
        quantized = np.frombuffer(raw[:expected], dtype="<u2").reshape((int(height), int(width)))
        scale = max(1, int(quant_um)) / 1000.0
        depth_mm = np.rint(quantized.astype(np.float32) * scale)
        depth_u16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        return np.ascontiguousarray(depth_u16), frame.meta

    def _drop_stale_frames(self) -> None:
        now = time.monotonic()
        stale = [seq for seq, frame in self._frames.items() if now - frame.first_seen > 1.0]
        for seq in stale:
            self._frames.pop(seq, None)

    def _log_error(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= 2.0:
            logging.warning(msg)
            self._last_error_log = now


class _RtpUdpCameraWorker(threading.Thread):
    def __init__(
        self,
        *,
        spec: RtpUdpCameraSpec,
        bind_ip: str,
        payload_type: int,
        ffmpeg: str,
        stop_event: threading.Event,
        update_frame: Any,
    ) -> None:
        super().__init__(name=f"xlerobot-rtp-udp-{spec.name}", daemon=True)
        self.spec = spec
        self.bind_ip = bind_ip
        self.payload_type = int(payload_type)
        self.ffmpeg = ffmpeg
        self.stop_event = stop_event
        self.update_frame = update_frame
        self._proc: subprocess.Popen | None = None
        self._sdp_path: Path | None = None
        self._last_error_log = 0.0

    def stop(self) -> None:
        self._stop_process()

    def run(self) -> None:
        frame_len = int(self.spec.width) * int(self.spec.height) * 3
        while not self.stop_event.is_set():
            try:
                proc = self._start_process()
                assert proc.stdout is not None
                while not self.stop_event.is_set():
                    raw = self._read_exact(proc.stdout, frame_len)
                    if raw is None:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (int(self.spec.height), int(self.spec.width), 3)
                    )
                    self.update_frame(self.spec, np.ascontiguousarray(frame))
            except Exception as exc:
                self._log_error(f"{self.spec.name} RTP/UDP receiver failed: {exc}")
            finally:
                self._stop_process()
                self._cleanup_sdp()
            if not self.stop_event.is_set():
                self.stop_event.wait(0.2)
        self._cleanup_sdp()

    def _start_process(self) -> subprocess.Popen:
        self._stop_process()
        self._sdp_path = self._write_sdp()
        vf = (
            f"scale={int(self.spec.width)}:{int(self.spec.height)}:"
            "force_original_aspect_ratio=increase,"
            f"crop={int(self.spec.width)}:{int(self.spec.height)},format=bgr24"
        )
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file,udp,rtp",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-i",
            str(self._sdp_path),
            "-an",
            "-vf",
            vf,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        return self._proc

    def _write_sdp(self) -> Path:
        text = "\n".join(
            [
                "v=0",
                "o=- 0 0 IN IP4 127.0.0.1",
                f"s=Indoory {self.spec.name}",
                f"c=IN IP4 {self.bind_ip}",
                "t=0 0",
                f"m=video {int(self.spec.port)} RTP/AVP {self.payload_type}",
                f"a=rtpmap:{self.payload_type} H264/90000",
                f"a=fmtp:{self.payload_type} packetization-mode=1",
                "a=recvonly",
                "",
            ]
        )
        fd, path = tempfile.mkstemp(prefix=f"indory_{self.spec.name}_", suffix=".sdp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        return Path(path)

    def _read_exact(self, stream: Any, length: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = int(length)
        while remaining > 0 and not self.stop_event.is_set():
            chunk = stream.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining:
            return None
        return b"".join(chunks)

    def _stop_process(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        for stream in (proc.stdout, proc.stderr, proc.stdin):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            deadline = time.monotonic() + 1.0
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            if proc.poll() is None:
                proc.kill()

    def _cleanup_sdp(self) -> None:
        path = self._sdp_path
        self._sdp_path = None
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _log_error(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= 2.0:
            logging.warning(msg)
            self._last_error_log = now
