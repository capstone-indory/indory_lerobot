#!/usr/bin/env python

import argparse
import time

from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from lerobot.robots.xlerobot.xlerobot import XLerobot


WRIST_ROLLS = {
    "left": ("bus1", "left_arm_wrist_roll"),
    "right": ("bus2", "right_arm_wrist_roll"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print XLeRobot wrist roll motor positions while you move them by hand."
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Which wrist roll motor to monitor.",
    )
    parser.add_argument("--hz", type=float, default=10.0, help="Print rate in Hz.")
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="Do not disable wrist roll torque before reading.",
    )
    parser.add_argument("--port1", default=None, help="Override bus1 port.")
    parser.add_argument("--port2", default=None, help="Override bus2 port.")
    return parser.parse_args()


def selected_wrist_rolls(side: str) -> dict[str, tuple[str, str]]:
    if side == "both":
        return WRIST_ROLLS
    return {side: WRIST_ROLLS[side]}


def main() -> None:
    args = parse_args()
    cfg = XLerobotConfig()
    if args.port1:
        cfg.port1 = args.port1
    if args.port2:
        cfg.port2 = args.port2

    robot = XLerobot(cfg)
    targets = selected_wrist_rolls(args.side)
    buses_to_connect = sorted({bus_name for bus_name, _ in targets.values()})

    connected_buses = []
    try:
        for bus_name in buses_to_connect:
            bus = getattr(robot, bus_name)
            bus.connect(handshake=True)
            connected_buses.append(bus)

        if not args.keep_torque:
            for side, (bus_name, motor) in targets.items():
                bus = getattr(robot, bus_name)
                bus.disable_torque(motor, num_retry=5)
                print(f"[{side}] disabled torque on {motor}")

        print()
        print("Move the wrist roll by hand. Press Ctrl+C to stop.")
        print("Columns: raw_tick is the motor encoder value; norm is LeRobot normalized/calibrated value.")
        print("Observed min/max update as you move, so use those values at the mechanical stops.")
        print()

        period_s = 1.0 / max(args.hz, 0.1)
        start_s = time.monotonic()
        last_values: dict[str, float] = {}
        raw_min: dict[str, float] = {}
        raw_max: dict[str, float] = {}
        norm_min: dict[str, float] = {}
        norm_max: dict[str, float] = {}

        while True:
            elapsed_s = time.monotonic() - start_s
            parts = [f"t={elapsed_s:7.2f}s"]

            for side, (bus_name, motor) in targets.items():
                bus = getattr(robot, bus_name)
                try:
                    raw_tick = bus.read("Present_Position", motor, normalize=False, num_retry=3)
                    norm_pos = bus.read("Present_Position", motor, normalize=True, num_retry=3)
                    delta = ""
                    if side in last_values:
                        delta = f" d_raw={raw_tick - last_values[side]:+7.1f}"
                    last_values[side] = raw_tick
                    raw_min[side] = min(raw_min.get(side, raw_tick), raw_tick)
                    raw_max[side] = max(raw_max.get(side, raw_tick), raw_tick)
                    norm_min[side] = min(norm_min.get(side, norm_pos), norm_pos)
                    norm_max[side] = max(norm_max.get(side, norm_pos), norm_pos)
                    parts.append(
                        f"{side}: raw_tick={raw_tick:8.1f} norm={norm_pos:8.3f}{delta} "
                        f"raw_range=[{raw_min[side]:.1f}, {raw_max[side]:.1f}] "
                        f"norm_range=[{norm_min[side]:.3f}, {norm_max[side]:.3f}]"
                    )
                except Exception as exc:
                    parts.append(f"{side}: ERROR {exc}")

            print(" | ".join(parts), flush=True)
            time.sleep(period_s)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for bus in connected_buses:
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
