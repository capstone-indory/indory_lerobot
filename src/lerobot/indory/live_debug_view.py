from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from lerobot.indory.success_detector import SuccessDetection


DETECTOR_DEBUG_ATTRS = (
    "bottom_y_ratio",
    "success_region",
    "position_metric",
    "required_consecutive",
    "hsv_lower",
    "hsv_upper",
    "min_area_ratio",
    "max_area_ratio",
    "confidence_threshold",
    "min_box_area_ratio",
    "head_targets",
    "head_ranges",
    "head_tolerance_ticks",
    "frame_key",
)


def detector_debug_summary(detector: object) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(detector).__name__}
    for attr in DETECTOR_DEBUG_ATTRS:
        if hasattr(detector, attr):
            summary[attr] = _jsonable(getattr(detector, attr))
    return summary


def cli_debug_summary(args: argparse.Namespace) -> dict[str, Any]:
    hidden = {"debug_web_open"}
    return {key: _jsonable(value) for key, value in vars(args).items() if key not in hidden}


class LiveDebugWebView:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        open_browser: bool,
        cli_args: dict[str, Any],
        detector: object,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.open_browser = bool(open_browser)
        self.cli_args = _jsonable(cli_args)
        self.detector = detector_debug_summary(detector)
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._state: dict[str, Any] = {
            "status": "starting",
            "updated_at": None,
            "step": None,
            "elapsed_s": 0.0,
            "gate_reason": None,
            "head": {},
            "detection": None,
            "sent_action": None,
            "extra": {},
        }
        self._frame: np.ndarray | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        handler = self._handler_class()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="indory-debug-web", daemon=True)
        self._thread.start()
        print(f"debug web view: {self.url}", flush=True)
        if self.open_browser:
            try:
                webbrowser.open(self.url)
            except Exception as exc:
                print(f"debug web view: browser open failed: {exc}", flush=True)

    def close(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_status(self, status: str, *, extra: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._state["status"] = str(status)
            self._state["updated_at"] = time.time()
            if extra:
                merged = dict(self._state.get("extra") or {})
                merged.update(_jsonable(extra))
                self._state["extra"] = merged
            self._seq += 1

    def keepalive(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if seconds <= 0.0:
            return
        print(f"debug web view: keeping final frame for {seconds:.1f}s at {self.url}", flush=True)
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.perf_counter())))

    def update(
        self,
        raw_observation: dict[str, Any] | None,
        *,
        detection: SuccessDetection | None = None,
        step: int | None = None,
        elapsed_s: float | None = None,
        gate_reason: str | None = None,
        sent_action: dict[str, Any] | None = None,
        status: str = "running",
        extra: dict[str, Any] | None = None,
    ) -> None:
        frame = None
        head: dict[str, Any] = {}
        if raw_observation:
            candidate = raw_observation.get("head")
            if isinstance(candidate, np.ndarray) and candidate.ndim == 3:
                frame = np.asarray(candidate, dtype=np.uint8).copy()
            for name in ("head_motor_1", "head_motor_2"):
                if name in raw_observation:
                    head[name] = _jsonable(raw_observation[name])

        with self._lock:
            if frame is not None:
                self._frame = frame
            self._state.update(
                {
                    "status": status,
                    "updated_at": time.time(),
                    "step": step,
                    "elapsed_s": float(elapsed_s or 0.0),
                    "gate_reason": gate_reason,
                    "head": head,
                    "detection": detection_to_dict(detection),
                    "sent_action": _compact_action(sent_action),
                    "extra": _jsonable(extra or {}),
                }
            )
            self._seq += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seq": self._seq,
                "cli_args": self.cli_args,
                "detector": self.detector,
                **_jsonable(self._state),
                "has_frame": self._frame is not None,
            }

    def frame_jpeg(self) -> bytes | None:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            state = _jsonable(self._state)
            detector = dict(self.detector)
        if frame is None:
            return None
        return annotated_jpeg(frame, state=state, detector=detector)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/":
                    self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/snapshot.json":
                    body = json.dumps(owner.snapshot(), ensure_ascii=False).encode("utf-8")
                    self._send_bytes(body, "application/json; charset=utf-8")
                    return
                if path == "/frame.jpg":
                    body = owner.frame_jpeg()
                    if body is None:
                        self.send_error(404, "no frame yet")
                        return
                    self._send_bytes(body, "image/jpeg")
                    return
                self.send_error(404)

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_bytes(self, body: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        return Handler


def detection_to_dict(detection: SuccessDetection | None) -> dict[str, Any] | None:
    if detection is None:
        return None
    return {
        "success": bool(detection.success),
        "reason": str(detection.reason or ""),
        "score": _jsonable(detection.score),
        "metadata": _jsonable(detection.metadata or {}),
    }


def annotated_jpeg(frame_rgb: np.ndarray, *, state: dict[str, Any], detector: dict[str, Any]) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for debug web image rendering") from exc

    frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    detection = state.get("detection") or {}
    metadata = detection.get("metadata") or {}
    threshold = _as_float(detector.get("bottom_y_ratio"))
    success_region = str(metadata.get("success_region") or detector.get("success_region") or "lower")
    metric_name = str(metadata.get("position_metric") or detector.get("position_metric") or "metric")
    metric_y = _as_float(metadata.get("metric_y_ratio"))
    success = bool(detection.get("success"))
    gate_reason = state.get("gate_reason")
    spatial_pass = success or (
        threshold is not None
        and metric_y is not None
        and (metric_y <= threshold if success_region == "upper" else metric_y >= threshold)
    )
    color = (0, 220, 0) if spatial_pass and not gate_reason else (0, 165, 255) if spatial_pass else (0, 0, 255)

    if threshold is not None:
        y = int(round(threshold * height))
        y = max(0, min(height - 1, y))
        cv2.line(image, (0, y), (width - 1, y), (0, 255, 255), 2)
        _put_text(
            image,
            f"{metric_name} {success_region} threshold={threshold:.3f} y={y}",
            10,
            max(24, y - 8),
            (0, 255, 255),
        )

    bbox = _xyxy(metadata.get("bbox_xyxy"))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    centroid = _xy(metadata.get("centroid"))
    if centroid is not None:
        cx, cy = centroid
        cv2.circle(image, (cx, cy), 6, color, -1)
        cv2.line(image, (0, cy), (width - 1, cy), color, 1)

    label_parts = []
    if metric_y is not None:
        label_parts.append(f"{metric_name}={metric_y:.3f}")
    for key in ("centroid_y_ratio", "bbox_bottom_y_ratio", "area_ratio"):
        value = _as_float(metadata.get(key))
        if value is not None:
            label_parts.append(f"{key.replace('_y_ratio', '')}={value:.3f}")
    consecutive = metadata.get("consecutive")
    if consecutive is not None:
        label_parts.append(f"consec={consecutive}/{detector.get('required_consecutive', '?')}")
    label = " ".join(label_parts)
    if label:
        x = bbox[0] if bbox else 10
        y = max(24, (bbox[1] - 10) if bbox else 24)
        _put_text(image, label, x, y, color)

    status = str(state.get("status") or "")
    step = state.get("step")
    elapsed = _as_float(state.get("elapsed_s"))
    top = f"{status} step={step} elapsed={elapsed:.2f}s" if elapsed is not None else status
    if gate_reason:
        top += f" gate={gate_reason}"
    _put_text(image, top, 10, 24, (255, 255, 255))

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("failed to encode debug frame")
    return encoded.tobytes()


def _put_text(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    import cv2

    x = max(0, int(x))
    y = max(18, int(y))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def _compact_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not action:
        return None
    keys = (
        "x.vel",
        "y.vel",
        "theta.vel",
        "head_motor_1",
        "head_motor_2",
        "left_arm_gripper",
        "right_arm_gripper",
    )
    return {key: _jsonable(action[key]) for key in keys if key in action}


def _xyxy(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(part))) for part in value]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def _xy(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Indory VLA Debug</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101418;
      color: #e8eef2;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 390px;
      gap: 0;
      background: #101418;
    }
    main {
      display: flex;
      flex-direction: column;
      min-width: 0;
      border-right: 1px solid #2a3238;
    }
    header {
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      border-bottom: 1px solid #2a3238;
      background: #151b20;
    }
    h1 {
      margin: 0;
      font-size: 15px;
      font-weight: 650;
      letter-spacing: 0;
    }
    #status {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #a7b5be;
    }
    .stage {
      flex: 1;
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 12px;
      background: #0b0f12;
    }
    img {
      width: min(100%, calc((100vh - 72px) * 1.333));
      max-height: calc(100vh - 72px);
      object-fit: contain;
      background: #050607;
      border: 1px solid #2a3238;
    }
    aside {
      overflow: auto;
      padding: 14px;
      background: #151b20;
    }
    section {
      margin: 0 0 14px;
      padding-bottom: 14px;
      border-bottom: 1px solid #2a3238;
    }
    h2 {
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 700;
      color: #9fb0bb;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    dl {
      display: grid;
      grid-template-columns: 138px minmax(0, 1fr);
      gap: 5px 8px;
      margin: 0;
      font-size: 12px;
    }
    dt {
      color: #91a0aa;
    }
    dd {
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
      color: #e8eef2;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 11px;
      line-height: 1.35;
      color: #d3dde4;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    @media (max-width: 980px) {
      body {
        grid-template-columns: 1fr;
      }
      main {
        border-right: 0;
        border-bottom: 1px solid #2a3238;
      }
      img {
        width: 100%;
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Indory VLA Debug</h1>
      <div id="status">waiting</div>
    </header>
    <div class="stage">
      <img id="frame" alt="head camera debug frame">
    </div>
  </main>
  <aside>
    <section>
      <h2>Run</h2>
      <dl id="run"></dl>
    </section>
    <section>
      <h2>Detection</h2>
      <dl id="detection"></dl>
    </section>
    <section>
      <h2>Detector</h2>
      <pre id="detector"></pre>
    </section>
    <section>
      <h2>CLI</h2>
      <pre id="cli"></pre>
    </section>
  </aside>
  <script>
    let lastSeq = -1;
    const statusEl = document.getElementById("status");
    const frameEl = document.getElementById("frame");
    const runEl = document.getElementById("run");
    const detectionEl = document.getElementById("detection");
    const detectorEl = document.getElementById("detector");
    const cliEl = document.getElementById("cli");

    function row(key, value) {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value === undefined || value === null ? "" : String(value);
      return [dt, dd];
    }

    function fillDl(el, obj) {
      el.replaceChildren();
      Object.entries(obj || {}).forEach(([key, value]) => {
        const [dt, dd] = row(key, typeof value === "object" && value !== null ? JSON.stringify(value) : value);
        el.append(dt, dd);
      });
    }

    async function refresh() {
      try {
        const res = await fetch("/snapshot.json", { cache: "no-store" });
        const data = await res.json();
        if (data.seq !== lastSeq) {
          lastSeq = data.seq;
          statusEl.textContent = `${data.status || ""} step=${data.step ?? ""} seq=${data.seq}`;
          if (data.has_frame) frameEl.src = `/frame.jpg?seq=${data.seq}`;
          const det = data.detection || {};
          fillDl(runEl, {
            status: data.status,
            step: data.step,
            elapsed_s: Number(data.elapsed_s || 0).toFixed(2),
            gate_reason: data.gate_reason,
            head: data.head,
            sent_action: data.sent_action,
            extra: data.extra
          });
          fillDl(detectionEl, {
            success: det.success,
            reason: det.reason,
            score: det.score,
            metadata: det.metadata
          });
          detectorEl.textContent = JSON.stringify(data.detector || {}, null, 2);
          cliEl.textContent = JSON.stringify(data.cli_args || {}, null, 2);
        }
      } catch (err) {
        statusEl.textContent = `offline ${err}`;
      }
    }

    refresh();
    setInterval(refresh, 250);
  </script>
</body>
</html>
"""
