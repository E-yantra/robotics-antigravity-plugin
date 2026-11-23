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
"""Shared hardware, safety (R1-R16), USB bandwidth, and camera helpers for SO-101 skills."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import select
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
  import cv2
except ImportError:
  cv2 = None  # type: ignore[assignment]

BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
ALL_MOTORS = BODY_JOINTS + ["gripper"]

# R5 hardware safety constants
R5_DEFAULT_MAX_STEP_DEG = 20.0
R5_HARD_MAX_STEP_DEG = 45.0
R5_DEFAULT_MAX_SPEED_DEG_PER_SEC = 30.0
R5_MAX_SPEED_DEG_PER_SEC = 90.0
R5_JOINT_MARGIN_DEG = 5.0
R5_HARD_FLOOR_M = -0.05
R6_MAX_FRAME_AGE_SEC = 0.3


def require_cv2() -> Any:
  if cv2 is None:
    raise RuntimeError(
        "opencv-python is required for camera/image operations. "
        "Install with: pip install opencv-python"
    )
  return cv2


def get_camera_index(cam_cfg: dict[str, Any]) -> int:
  """Return OpenCV camera index supporting both 'opencv_index' and 'index' keys."""
  if "opencv_index" in cam_cfg:
    return int(cam_cfg["opencv_index"])
  if "index" in cam_cfg:
    return int(cam_cfg["index"])
  raise KeyError("Camera config must specify 'opencv_index' or 'index'.")


def get_configured_cameras(robot_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
  """Return dictionary of active configured cameras (at least 1 required by R6)."""
  cams = robot_cfg.get("cameras", {})
  if not isinstance(cams, dict) or not cams:
    raise ValueError(
        "[R6 FAIL] At least one camera must be configured in robot_config['cameras']."
    )
  active: dict[str, dict[str, Any]] = {}
  for role, cfg in cams.items():
    if (
        isinstance(cfg, dict)
        and ("opencv_index" in cfg or "index" in cfg)
        and cfg.get("enabled", True)
    ):
      active[role] = dict(cfg)
  if not active:
    raise ValueError("[R6 FAIL] No enabled cameras found in robot_config['cameras'].")
  return active


def get_fixed_camera_role_and_config(
    robot_cfg: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
  """Return (role, cfg) for the stationary fixed/overhead camera if configured, else None."""
  cams = get_configured_cameras(robot_cfg)
  for role in ("fixed", "overhead"):
    if role in cams:
      return role, cams[role]
  for role, cfg in cams.items():
    if cfg.get("is_fixed", False):
      return role, cfg
  return None


def compute_lighting_stats(frame: np.ndarray) -> dict[str, Any]:
  """Compute image brightness and glare diagnostics (0-255) for Stage 1 & Stage 4."""
  arr = np.asarray(frame, dtype=np.float32)
  if arr.ndim == 3:
    gray = arr.mean(axis=2)
  else:
    gray = arr
  mean_brightness = float(np.mean(gray))
  std_brightness = float(np.std(gray))
  dark_ratio = float(np.mean(gray < 25.0))
  glare_ratio = float(np.mean(gray > 245.0))
  if mean_brightness < 40.0:
    status = "TOO_DARK"
  elif mean_brightness > 235.0 or glare_ratio > 0.20:
    status = "TOO_BRIGHT_OR_GLARE"
  else:
    status = "OK"
  return {
      "mean_brightness": round(mean_brightness, 2),
      "std_brightness": round(std_brightness, 2),
      "dark_pixel_ratio": round(dark_ratio, 4),
      "glare_pixel_ratio": round(glare_ratio, 4),
      "status": status,
  }


def estimate_usb_bandwidth_mbps(
    width: int, height: int, fps: float, fourcc: str = "MJPG"
) -> dict[str, float]:
  """Estimate raw YUYV (16 bpp) vs compressed MJPG (~2.2 bpp) USB bandwidth in Mbps."""
  pixels_per_sec = float(width) * float(height) * float(fps)
  yuyv_mbps = (pixels_per_sec * 16.0) / 1e6
  mjpg_mbps = (pixels_per_sec * 2.2) / 1e6
  active_mbps = mjpg_mbps if str(fourcc).upper() == "MJPG" else yuyv_mbps
  usb2_usable_mbps = 320.0
  return {
      "yuyv_uncompressed_mbps": round(yuyv_mbps, 2),
      "mjpg_estimated_mbps": round(mjpg_mbps, 2),
      "active_estimated_mbps": round(active_mbps, 2),
      "usb2_theoretical_mbps": 480.0,
      "usb2_usable_mbps": usb2_usable_mbps,
      "usb2_utilization_pct": round((active_mbps / usb2_usable_mbps) * 100.0, 1),
  }


def inspect_usb_topology() -> dict[str, Any]:
  """Inspect OS USB controller tree and link speeds (system_profiler on macOS, lsusb -t on Linux)."""
  sys_name = platform.system()
  cmd: list[str] | None = None
  if sys_name == "Darwin" and shutil.which("system_profiler"):
    cmd = ["system_profiler", "SPUSBDataType"]
  elif shutil.which("lsusb"):
    cmd = ["lsusb", "-t"]

  if cmd is None:
    return {
        "os": sys_name,
        "command": None,
        "raw_output": "",
        "detected_speeds": [],
    }
  try:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=8.0, check=False
    )
    raw = proc.stdout or ""
  except Exception as exc:
    raw = f"error running {' '.join(cmd)}: {exc}"

  speeds: list[str] = []
  for token in ("480 Mb/sec", "480M", "5 Gb/sec", "5000M", "10 Gb/sec", "12 Mb/sec"):
    if token in raw:
      speeds.append(token)
  return {
      "os": sys_name,
      "command": " ".join(cmd),
      "raw_output": raw[:4000],
      "detected_speeds": speeds,
  }


def load_calibrations(calib_path: Path) -> dict[str, Any]:
  """Load LeRobot/Feetech servo calibration dictionary from JSON."""
  raw = json.loads(calib_path.read_text(encoding="utf-8"))
  try:
    from lerobot.motors import MotorCalibration
  except ImportError:
    MotorCalibration = None

  out: dict[str, Any] = {}
  for name, d in raw.items():
    if MotorCalibration is not None:
      out[name] = MotorCalibration(
          id=int(d["id"]),
          drive_mode=int(d["drive_mode"]),
          homing_offset=int(d["homing_offset"]),
          range_min=int(d["range_min"]),
          range_max=int(d["range_max"]),
      )
    else:
      out[name] = {
          "id": int(d["id"]),
          "drive_mode": int(d["drive_mode"]),
          "homing_offset": int(d["homing_offset"]),
          "range_min": int(d["range_min"]),
          "range_max": int(d["range_max"]),
      }
  return out


def _cal_field(cal: Any, field: str) -> int:
  return int(cal[field] if isinstance(cal, dict) else getattr(cal, field))


def ticks_to_norm(
    raw_ticks: dict[str, int], calibrations: dict[str, Any]
) -> dict[str, float]:
  """Convert raw STS3215 ticks (0-4095) to normalized degrees / 0-100 gripper."""
  norm: dict[str, float] = {}
  for name in BODY_JOINTS:
    c = calibrations[name]
    r_min = _cal_field(c, "range_min")
    r_max = _cal_field(c, "range_max")
    mid = (r_min + r_max) / 2.0
    deg = (float(raw_ticks[name]) - mid) * 360.0 / 4095.0
    norm[name] = round(deg, 3)

  cg = calibrations["gripper"]
  g_min = _cal_field(cg, "range_min")
  g_max = _cal_field(cg, "range_max")
  g_val = ((float(raw_ticks["gripper"]) - g_min) / max(g_max - g_min, 1)) * 100.0
  g_val = max(0.0, min(100.0, g_val))
  norm["gripper"] = round(g_val, 2)
  return norm


def joint_calibrated_bounds_deg(
    calibrations: dict[str, Any],
) -> dict[str, tuple[float, float]]:
  """Return calibrated [min, max] span in degrees for each body joint and [0, 100] for gripper."""
  bounds: dict[str, tuple[float, float]] = {}
  for name in BODY_JOINTS:
    c = calibrations[name]
    r_min = _cal_field(c, "range_min")
    r_max = _cal_field(c, "range_max")
    mid = (r_min + r_max) / 2.0
    min_deg = (r_min - mid) * 360.0 / 4095.0
    max_deg = (r_max - mid) * 360.0 / 4095.0
    bounds[name] = (round(min_deg, 2), round(max_deg, 2))
  bounds["gripper"] = (0.0, 100.0)
  return bounds


def open_configured_camera(cam_cfg: dict[str, Any]) -> Any:
  """Open an OpenCV VideoCapture with explicit FOURCC, resolution, FPS, and buffer flush."""
  cv = require_cv2()
  idx = get_camera_index(cam_cfg)
  cap = cv.VideoCapture(idx)
  if not cap.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open camera at OpenCV index {idx}")
  if cam_cfg.get("fourcc"):
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*str(cam_cfg["fourcc"])))
  cap.set(cv.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))
  cap.set(cv.CAP_PROP_FPS, float(cam_cfg["fps"]))
  cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
  for _ in range(5):
    cap.read()
  return cap


def annotate_and_save_frame(
    frame: np.ndarray,
    out_path: Path,
    cam_label: str,
    phase_label: str,
    pose: dict[str, float],
    tip_xyz: list[float] | None,
    stage_tag: str = "SO-101",
) -> None:
  """Overlay timestamp, joint angles, and FK tip coordinates onto a frame and save."""
  cv = require_cv2()
  img = frame.copy()
  ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
  pan_txt = (
      ", ".join(f"{k}:{pose[k]:+.1f}" for k in BODY_JOINTS)
      + f", grp:{pose['gripper']:.1f}"
  )
  lines = [
      f"{stage_tag} | {cam_label.upper()} | {phase_label} | {ts}",
      f"Joints: {pan_txt}",
  ]
  if tip_xyz is not None:
    lines.append(
        f"FK Tip (gripper_frame_link): x={tip_xyz[0]:+.4f}m "
        f"y={tip_xyz[1]:+.4f}m z={tip_xyz[2]:+.4f}m"
    )

  overlay = img.copy()
  cv.rectangle(
      overlay, (0, 0), (img.shape[1], 24 * len(lines) + 12), (10, 10, 10), -1
  )
  cv.addWeighted(overlay, 0.65, img, 0.35, 0.0, img)

  for idx, line in enumerate(lines):
    y = 22 + idx * 24
    cv.putText(
        img,
        line,
        (10, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55 if img.shape[1] >= 1000 else 0.42,
        (80, 255, 160),
        1,
        cv.LINE_AA,
    )

  out_path.parent.mkdir(parents=True, exist_ok=True)
  cv.imwrite(str(out_path), img)


class AbortWatcher:
  """Non-blocking watcher for terminal 'q' + Enter and SIGINT (Ctrl-C) abort requests."""

  def __init__(self) -> None:
    self.abort_reason: str | None = None
    self._prev_sigint = signal.getsignal(signal.SIGINT)

  def _sigint_handler(self, signum: int, frame: Any) -> None:
    del signum, frame
    self.abort_reason = "SIGINT (Ctrl-C)"

  def __enter__(self) -> AbortWatcher:
    signal.signal(signal.SIGINT, self._sigint_handler)
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    del exc_type, exc_val, exc_tb
    signal.signal(signal.SIGINT, self._prev_sigint)

  def check_abort(self) -> str | None:
    if self.abort_reason is not None:
      return self.abort_reason
    try:
      rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
      if rlist:
        line = sys.stdin.readline()
        if not line or line.strip().lower().startswith("q"):
          self.abort_reason = (
              f"stdin '{line.strip()}' + Enter" if line.strip() else "stdin EOF"
          )
          return self.abort_reason
    except Exception:
      pass
    return self.abort_reason


def run_preflight(
    robot_cfg: dict[str, Any], check_cfg: dict[str, Any]
) -> tuple[Any, Any]:
  """Pre-flight check (R6): verify all 6 STS3215 servos answer (Torque OFF) and all configured cameras (1 or more) deliver fresh frames <= 0.3 s."""
  from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

  pf_cfg = check_cfg.get("preflight", {})
  max_age = float(pf_cfg.get("max_frame_age_sec", R6_MAX_FRAME_AGE_SEC))
  num_checks = int(pf_cfg.get("num_checks", 3))
  interval = float(pf_cfg.get("interval_sec", 0.1))

  print("\n--- Running Pre-Flight Checks (Before Torque ON) ---")
  port = str(robot_cfg.get("port") or robot_cfg.get("serial_port"))
  calib_path = Path(
      check_cfg.get("calibration_file") or robot_cfg.get("calibration_file")
  ).expanduser()
  calibs = load_calibrations(calib_path)
  bus = SerialMotorsBus(
      port=port,
      motors={
          "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
          "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
          "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
          "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
          "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
          "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
      },
      calibration=calibs,
  )
  bus.connect(handshake=True)
  try:
    ticks = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    if not bus.is_calibrated:
      raise RuntimeError(
          f"[PRE-FLIGHT FAIL] Calibration in '{calib_path}' does not match motor EEPROM!"
      )
    print(
        "[Motor Check PASS] All 6 servos answered & EEPROM calibration verified: "
        f"{ {k: int(v) for k, v in ticks.items()} }"
    )
  finally:
    bus.disconnect(disable_torque=False)

  cams = get_configured_cameras(robot_cfg)
  opened_caps: dict[str, Any] = {}
  try:
    for role, cam_cfg in cams.items():
      cfg_copy = dict(cam_cfg)
      cfg_copy.setdefault("fourcc", "MJPG")
      opened_caps[role] = open_configured_camera(cfg_copy)

    for check_idx in range(1, num_checks + 1):
      summaries: list[str] = []
      for role, cap in opened_caps.items():
        cam_cfg = cams[role]
        t_req = time.monotonic()
        ok, frame = cap.read()
        age = time.monotonic() - t_req
        if not ok or frame is None or age > max_age:
          raise RuntimeError(
              f"[PRE-FLIGHT FAIL] Stale/unreadable '{role}' frame (ok={ok}, age={age:.3f}s > {max_age:.2f}s)"
          )
        if (frame.shape[1], frame.shape[0]) != (
            int(cam_cfg["width"]),
            int(cam_cfg["height"]),
        ):
          raise RuntimeError(
              f"[PRE-FLIGHT FAIL] Unexpected '{role}' resolution {frame.shape}"
          )
        summaries.append(
            f"{role}={frame.shape[1]}x{frame.shape[0]} (age={age:.3f}s)"
        )
      print(
          f"[Camera Check {check_idx}/{num_checks} PASS] "
          + ", ".join(summaries)
          + f" <= {max_age:.2f}s"
      )
      if check_idx < num_checks:
        time.sleep(interval)
  except Exception:
    for cap in opened_caps.values():
      cap.release()
    raise

  cap_wr = opened_caps.get("wrist")
  cap_ov = opened_caps.get("overhead") or opened_caps.get("fixed")
  if cap_wr is None and opened_caps:
    first_role = next(iter(opened_caps))
    cap_wr = opened_caps[first_role]
  return cap_wr, cap_ov


def execute_r3_retreat(
    robot: Any,
    travelled_waypoints: list[dict[str, float]],
    rest_pose: dict[str, float],
    control_hz: float,
    log_file: Any,
) -> bool:
  """Smoothly unwind backwards along the travelled path back to rest_pose."""
  print("=== Executing Smooth Retreat Along Travelled Path Back to Rest Pose ===")
  dt = 1.0 / control_hz
  reversed_full = list(reversed(travelled_waypoints))
  retreat_path: list[dict[str, float]] = []
  if reversed_full:
    last_kept = reversed_full[0]
    for wp in reversed_full[1:]:
      max_delta = max(abs(wp[j] - last_kept[j]) for j in ALL_MOTORS)
      if max_delta >= 0.25:
        retreat_path.append(wp)
        last_kept = wp
  if not retreat_path or retreat_path[-1] != rest_pose:
    retreat_path.append(dict(rest_pose))

  for step_idx, target_wp in enumerate(retreat_path, start=1):
    action = {f"{j}.pos": float(target_wp[j]) for j in ALL_MOTORS}
    sent = robot.send_action(action)
    if log_file is not None:
      log_file.write(
          json.dumps({
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "event": "retreat_step",
              "step": step_idx,
              "total_steps": len(retreat_path),
              "sent": {
                  k.removesuffix(".pos"): round(float(v), 3)
                  for k, v in sent.items()
              },
          })
          + "\n"
      )
      log_file.flush()
    time.sleep(dt)

  time.sleep(0.25)
  obs = robot.get_observation()
  max_err = max(
      abs(float(obs[f"{j}.pos"]) - float(rest_pose[j])) for j in BODY_JOINTS
  )
  print(f"Retreat completed. Max body joint error vs rest pose = {max_err:.2f} deg.")
  return max_err <= 5.0


def safe_disconnect_robot(
    robot: Any,
    rest_pose: dict[str, float],
    rest_tol_deg: float,
) -> bool:
  """Never disable torque away from rest; verify via motor bus without camera dependency."""
  if robot is None or not getattr(robot, "is_connected", False):
    return True

  at_rest = False
  max_err = float("inf")
  try:
    bus_pos = robot.bus.sync_read("Present_Position", num_retry=2)
    if rest_pose:
      max_err = max(
          abs(float(bus_pos[j]) - float(rest_pose[j])) for j in BODY_JOINTS
      )
      at_rest = max_err <= rest_tol_deg
  except Exception as err:
    print(f"[SAFE EXIT WARNING] Failed to read motor bus position: {err}")
    at_rest = False

  if at_rest:
    print(
        f"[Safe Exit] Arm verified at rest pose via motor bus "
        f"(max error = {max_err:.2f} deg <= {rest_tol_deg} deg). Disabling torque."
    )
    robot.config.disable_torque_on_disconnect = True
    robot.disconnect()
    return True

  print("\n" + "!" * 76)
  print(
      f"[CRITICAL WARNING] ARM IS AWAY FROM REST (max_err={max_err:.2f} deg) OR BUS UNREADABLE!"
  )
  print("LEAVING MOTOR TORQUE **ON** SO THE LIMP ARM DOES NOT DROP ONTO THE TABLE.")
  print("HUMAN: PLEASE PHYSICALLY SUPPORT THE ARM BEFORE CUTTING POWER!")
  print("!" * 76 + "\n")
  robot.config.disable_torque_on_disconnect = False
  robot.disconnect()
  return False
