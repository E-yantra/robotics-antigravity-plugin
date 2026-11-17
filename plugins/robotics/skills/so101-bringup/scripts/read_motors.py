# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
"""Read-only SO-101 Feetech STS3215 motor bus inspection (never turns torque on)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_common import ALL_MOTORS, BODY_JOINTS  # noqa: E402

SERIAL_PORT_PATTERNS = ("/dev/tty.usbmodem*", "/dev/ttyACM*")


def autodetect_serial_port() -> str | None:
  """Returns the only USB serial candidate, or None if zero or several exist."""
  candidates = sorted({p for pat in SERIAL_PORT_PATTERNS for p in glob.glob(pat)})
  return candidates[0] if len(candidates) == 1 else None


def build_bus(port: str):
  from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

  motors = {
      "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
      "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
      "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
      "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
      "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
      "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
  }
  return SerialMotorsBus(port=port, motors=motors)


def read_motors_readonly(port: str, output_path: Path | None = None) -> dict:
  """Read Present_Position and Torque_Enable for all 6 STS3215 servos without enabling torque."""
  bus = build_bus(port)
  bus.connect(handshake=True)
  try:
    raw_ticks = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    torque_enable = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    raw_ticks = {k: int(v) for k, v in raw_ticks.items()}
    torque_enable = {k: int(v) for k, v in torque_enable.items()}
  finally:
    bus.disconnect(disable_torque=False)

  print(f"Port: {port}")
  print(f"Raw ticks (0-4095): {raw_ticks}")
  print(f"Torque_Enable state: {torque_enable}")

  if any(v != 0 for v in torque_enable.values()):
    print("WARNING: One or more motors reported Torque_Enable != 0!")
  else:
    print("All 6 motors answered; Torque_Enable is 0 (OFF) on all 6 motors.")

  snapshot = {
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "port": port,
      "body_joints": BODY_JOINTS,
      "raw_ticks": raw_ticks,
      "torque_enable": torque_enable,
  }
  if output_path is not None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved snapshot -> {output_path}")
  return snapshot


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Read-only SO-101 motor bus check (Torque OFF)."
  )
  parser.add_argument(
      "--config",
      type=Path,
      default=Path("config/robot_config.json"),
      help="Path to robot JSON config.",
  )
  parser.add_argument(
      "--port",
      type=str,
      default=None,
      help=(
          "Override serial port (e.g. /dev/tty.usbmodem... or /dev/ttyACM0), or"
          " 'auto' to use the single detected USB serial device."
      ),
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("data/stage1/motors_readonly.json"),
      help="Where to save the read-only motor snapshot.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print planned read-only bus query without opening the serial port.",
  )
  args = parser.parse_args()

  if args.port == "auto":
    detected = autodetect_serial_port()
    if detected is None:
      sys.exit(
          "Could not auto-detect a single USB serial port "
          f"({', '.join(SERIAL_PORT_PATTERNS)}). Run lerobot-find-port instead."
      )
    port = detected
    print(f"Auto-detected serial port: {port}")
  elif args.port:
    port = str(args.port)
  else:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    port = str(cfg.get("port") or cfg.get("serial_port"))

  if args.dry_run:
    print(
        f"[DRY-RUN] Would open SerialMotorsBus on {port} "
        "(handshake only, Torque_Enable untouched), read raw Present_Position (0-4095) "
        f"and Torque_Enable for {ALL_MOTORS}, and save snapshot to {args.output}."
    )
    return

  read_motors_readonly(port=port, output_path=args.output)


if __name__ == "__main__":
  main()
