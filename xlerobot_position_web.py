#!/usr/bin/env python3
"""Tiny read-only web view for XLerobot joint positions.

Run from /home/pi/lerobot:
    python xlerobot_position_web.py --host 0.0.0.0 --port 8090

This script reads Present_Position only. It does not send Goal_Position or
velocity commands.
"""

from __future__ import annotations

import argparse
import html
import json
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

RIGHT_ARM = (
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
)
LEFT_ARM = (
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
    "left_arm_gripper",
)
HEAD = ("head_motor_1", "head_motor_2")
BASE = ("base_left_wheel", "base_back_wheel", "base_right_wheel")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class XLerobotPositionReader:
    def __init__(self, right_only: bool = False, raw_only: bool = False):
        from lerobot.robots.xlerobot import XLerobot, XLerobotConfig

        self.robot = XLerobot(XLerobotConfig())
        self.right_only = right_only
        self.raw_only = raw_only
        self.lock = threading.Lock()
        self.connected: dict[str, bool] = {"bus1": False, "bus2": False}
        self.connect_errors: dict[str, str] = {}
        self._connect()

    def _cache_calibration(self, bus: Any) -> int:
        calibration = {name: cal for name, cal in self.robot.calibration.items() if name in bus.motors}
        if calibration:
            bus.calibration = calibration
        return len(calibration)

    def _connect_bus(self, label: str, bus: Any) -> None:
        try:
            bus.connect()
            loaded = self._cache_calibration(bus)
            self.connected[label] = True
            self.connect_errors.pop(label, None)
            print(f"[{label}] connected on {bus.port}; cached_calibration={loaded}", flush=True)
        except Exception as exc:  # keep the web server alive and show the error
            self.connected[label] = False
            self.connect_errors[label] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"[{label}] connect failed: {self.connect_errors[label]}", flush=True)

    def _connect(self) -> None:
        if not self.right_only:
            self._connect_bus("bus1", self.robot.bus1)
        self._connect_bus("bus2", self.robot.bus2)

    def close(self) -> None:
        for label, bus in (("bus1", self.robot.bus1), ("bus2", self.robot.bus2)):
            if self.connected.get(label):
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    pass

    def _read_positions(self, bus: Any, motors: tuple[str, ...]) -> dict[str, Any]:
        present: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        if not self.raw_only:
            present = bus.sync_read("Present_Position", list(motors), num_retry=1)
        raw = bus.sync_read("Present_Position", list(motors), normalize=False, num_retry=1)
        return {"position": present, "raw_position": raw}

    def read(self) -> dict[str, Any]:
        with self.lock:
            data: dict[str, Any] = {
                "ok": True,
                "timestamp": time.time(),
                "connected": dict(self.connected),
                "connect_errors": dict(self.connect_errors),
                "ports": {
                    "bus1": self.robot.bus1.port,
                    "bus2": self.robot.bus2.port,
                },
                "groups": {},
                "read_errors": {},
            }

            if self.connected.get("bus1"):
                try:
                    data["groups"]["left_arm"] = self._read_positions(self.robot.bus1, LEFT_ARM)
                    data["groups"]["head"] = self._read_positions(self.robot.bus1, HEAD)
                except Exception as exc:
                    data["ok"] = False
                    data["read_errors"]["bus1"] = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()

            if self.connected.get("bus2"):
                try:
                    data["groups"]["right_arm"] = self._read_positions(self.robot.bus2, RIGHT_ARM)
                    data["groups"]["base"] = self._read_positions(self.robot.bus2, BASE)
                except Exception as exc:
                    data["ok"] = False
                    data["read_errors"]["bus2"] = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()

            if not any(self.connected.values()):
                data["ok"] = False
            return data


def render_page(payload: dict[str, Any], refresh_ms: int) -> bytes:
    body = json.dumps(payload, indent=2, default=_json_default)
    ok = "OK" if payload.get("ok") else "ERROR"
    status_class = "ok" if payload.get("ok") else "bad"
    escaped = html.escape(body)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{max(refresh_ms, 250) / 1000:.2f}">
  <title>XLerobot Positions</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101412;
      --panel: #171e1a;
      --ink: #ecf3ed;
      --muted: #9eb0a4;
      --line: #2a342d;
      --ok: #65d58a;
      --bad: #ff7a70;
    }}
    body {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 24px;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .pill {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      border-radius: 6px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .pill.ok {{ color: var(--ok); }}
    .pill.bad {{ color: var(--bad); }}
    pre {{
      margin: 0;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: auto;
      line-height: 1.45;
      font-size: 13px;
    }}
    a {{ color: var(--ok); }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>XLerobot Positions</h1>
      <span class="pill {status_class}">{ok}</span>
    </header>
    <pre>{escaped}</pre>
  </main>
</body>
</html>
"""
    return page.encode("utf-8")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_handler(reader: XLerobotPositionReader, refresh_ms: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} {fmt % args}", flush=True)

        def do_GET(self) -> None:
            if self.path.startswith("/api/positions"):
                payload = reader.read()
                content = json.dumps(payload, default=_json_default).encode("utf-8")
                self.send_response(200 if payload.get("ok") else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            payload = reader.read()
            content = render_page(payload, refresh_ms)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only XLerobot joint position web view")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--refresh-ms", type=int, default=500)
    parser.add_argument("--right-only", action="store_true", help="Connect only bus2: right arm + base")
    parser.add_argument("--raw-only", action="store_true", help="Skip normalized reads and show raw ticks only")
    args = parser.parse_args()

    reader = XLerobotPositionReader(right_only=args.right_only, raw_only=args.raw_only)
    server = ReusableThreadingHTTPServer((args.host, args.port), make_handler(reader, args.refresh_ms))

    def stop(_signum: int, _frame: Any) -> None:
        print("\nStopping xlerobot position web...", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"XLerobot position web: http://{args.host}:{args.port}", flush=True)
    print("JSON endpoint: /api/positions", flush=True)
    try:
        server.serve_forever()
    finally:
        reader.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
