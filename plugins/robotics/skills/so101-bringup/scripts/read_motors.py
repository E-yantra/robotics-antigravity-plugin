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
import dataclasses
from datetime import datetime, timezone
import glob
import importlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common  # noqa: E402  # pylint: disable=g-import-not-at-top

ALL_MOTORS = so101_common.ALL_MOTORS
BODY_JOINTS = so101_common.BODY_JOINTS
ticks_to_norm = so101_common.ticks_to_norm

SERIAL_PORT_PATTERNS = ("/dev/tty.usbmodem*", "/dev/ttyACM*")


def autodetect_serial_port() -> str | None:
  """Returns the only USB serial candidate, or None if zero or several exist."""
  candidates = sorted({p for pat in SERIAL_PORT_PATTERNS for p in glob.glob(pat)})
  return candidates[0] if len(candidates) == 1 else None


def build_robot(port: str, robot_id: str = "my_follower") -> Any:
  so_mod = importlib.import_module("lerobot.robots.so_follower")
  follower_cls = getattr(so_mod, "SO101Follower")
  config_cls = getattr(so_mod, "SO101FollowerConfig")
  return follower_cls(config_cls(id=robot_id, port=port))


def is_valid_eeprom_calibration(
    cal: dict[str, Any], raw_ticks: dict[str, int] | None = None
) -> tuple[bool, dict[str, float]]:
  """Return whether EEPROM holds calibrated bounds rather than factory [0, 4095]."""
  if any(m not in cal for m in ALL_MOTORS):
    return False, {}
  spans = {
      m: round((cal[m].range_max - cal[m].range_min) * 360.0 / 4095.0, 1)
      for m in ALL_MOTORS
  }
  for m in ALL_MOTORS:
    r_min, r_max = int(cal[m].range_min), int(cal[m].range_max)
    if not 0 <= r_min < r_max <= 4095:
      return False, spans
    if m != "wrist_roll" and (r_min, r_max) == (0, 4095):
      return False, spans
    if spans[m] < (20.0 if m == "gripper" else 45.0):
      return False, spans
    if raw_ticks and m != "wrist_roll" and m in raw_ticks:
      if raw_ticks[m] < r_min - 57 or raw_ticks[m] > r_max + 57:
        return False, spans
  return True, spans


def read_motors_readonly(
    port: str,
    output_path: Path | None = None,
    robot_id: str = "my_follower",
) -> dict[str, Any]:
  """Read positions, Torque_Enable, and EEPROM calibration without enabling torque."""
  robot = build_robot(port=port, robot_id=robot_id)
  bus = robot.bus
  bus.connect(handshake=True)
  try:
    raw_ticks = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    torque_enable = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    eeprom_cal = bus.read_calibration()
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

  calib_valid, joint_spans_deg = is_valid_eeprom_calibration(
      eeprom_cal, raw_ticks
  )
  if calib_valid:
    if robot.calibration != eeprom_cal or not robot.calibration_fpath.is_file():
      robot.calibration_fpath.parent.mkdir(parents=True, exist_ok=True)
      payload = {k: dataclasses.asdict(v) for k, v in eeprom_cal.items()}
      robot.calibration_fpath.write_text(
          json.dumps(payload, indent=4) + "\n", encoding="utf-8"
      )
      print(f"Saved motor EEPROM calibration -> {robot.calibration_fpath}")
    else:
      print(f"Verified motor EEPROM calibration -> {robot.calibration_fpath}")
  else:
    print(
        "No valid calibration in motor EEPROM; run lerobot-calibrate in"
        " Stage 2."
    )

  snapshot: dict[str, Any] = {
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "port": port,
      "robot_id": robot_id,
      "body_joints": BODY_JOINTS,
      "raw_ticks": raw_ticks,
      "torque_enable": torque_enable,
      "calibration_valid": calib_valid,
      "calibration_file": (
          str(robot.calibration_fpath) if calib_valid else None
      ),
      "joint_spans_deg": joint_spans_deg,
      "rest_pose": (
          ticks_to_norm(raw_ticks, eeprom_cal) if calib_valid else None
      ),
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
      description="Read-only SO-101 motor bus & EEPROM check (Torque OFF)."
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
      "--robot-id",
      type=str,
      default=None,
      help="Robot ID for LeRobot calibration cache (default: my_follower).",
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

  cfg: dict[str, Any] = (
      json.loads(args.config.read_text(encoding="utf-8"))
      if args.config.is_file()
      else {}
  )
  robot_id = str(args.robot_id or cfg.get("robot_id") or "my_follower")

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
    port = str(cfg.get("port") or cfg.get("serial_port"))

  if args.dry_run:
    print(
        f"[DRY-RUN] Would open Feetech bus on {port} (Torque_Enable untouched),"
        f" read Present_Position, Torque_Enable, and EEPROM calibration for"
        f" {ALL_MOTORS}, and save snapshot to {args.output}."
    )
    return

  read_motors_readonly(port=port, output_path=args.output, robot_id=robot_id)


if __name__ == "__main__":
  main()
