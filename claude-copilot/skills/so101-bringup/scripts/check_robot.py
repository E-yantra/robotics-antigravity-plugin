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
"""SO-101 bring-up, read-only bus check, cosine-eased limit checks (R5), q+Enter abort retreat (R3), and torque-safe exit (R4)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_motors import read_motors_readonly  # noqa: E402
from so101_common import (  # noqa: E402
    ALL_MOTORS,
    AbortWatcher,
    BODY_JOINTS,
    R5_DEFAULT_MAX_SPEED_DEG_PER_SEC,
    R5_DEFAULT_MAX_STEP_DEG,
    R5_HARD_MAX_STEP_DEG,
    R5_MAX_SPEED_DEG_PER_SEC,
    annotate_and_save_frame,
    execute_r3_retreat,
    joint_calibrated_bounds_deg,
    load_calibrations,
    run_preflight,
    safe_disconnect_robot,
    ticks_to_norm,
)


def _append_cosine_segment(
    waypoints: list[dict[str, float]],
    pose_a: dict[str, float],
    pose_b: dict[str, float],
    duration_sec: float,
    control_hz: float,
    max_step_deg: float = R5_DEFAULT_MAX_STEP_DEG,
    max_speed_deg_per_sec: float = R5_DEFAULT_MAX_SPEED_DEG_PER_SEC,
) -> None:
  """Append a cosine-eased segment from pose_a to pose_b automatically sized to respect R5 step & speed limits."""
  max_joint_delta = max(abs(float(pose_b[j]) - float(pose_a[j])) for j in ALL_MOTORS)
  min_dur_for_speed = (math.pi / 2.0) * max_joint_delta / max(max_speed_deg_per_sec * 0.95, 1.0)
  min_steps_for_step = int(math.ceil((math.pi / 2.0) * max_joint_delta / max(max_step_deg * 0.95, 0.5)))
  eff_dur = max(float(duration_sec), min_dur_for_speed)
  steps = max(2, int(round(eff_dur * control_hz)), min_steps_for_step)
  for k in range(1, steps + 1):
    s = k / steps
    ease = 0.5 * (1.0 - math.cos(math.pi * s))
    wp = {
        j: round(float(pose_a[j]) + (float(pose_b[j]) - float(pose_a[j])) * ease, 4)
        for j in ALL_MOTORS
    }
    waypoints.append(wp)


def generate_cosine_trajectory(
    start_pose: dict[str, float],
    mode: str,
    mode_cfg: dict[str, Any],
    control_hz: float,
) -> list[dict[str, float]]:
  """Build smooth cosine-eased waypoint trajectory starting and ending at start_pose."""
  if mode == "check":
    return [dict(start_pose)]

  if mode == "hello-world":
    seg_dur = float(mode_cfg.get("segment_duration_sec", 4.0))
    grip_dur = float(mode_cfg.get("gripper_duration_sec", 2.0))
    max_step = float(mode_cfg.get("max_step_deg", 8.0))
    max_speed = float(mode_cfg.get("max_speed_deg_per_sec", 20.0))
    open_val = float(mode_cfg.get("gripper_open", 75.0))
    close_val = float(mode_cfg.get("gripper_closed", 10.0))

    midpoint_pose = {j: 0.0 for j in BODY_JOINTS}
    midpoint_pose["gripper"] = float(start_pose["gripper"])
    midpoint_open = dict(midpoint_pose, gripper=open_val)
    midpoint_closed = dict(midpoint_pose, gripper=close_val)

    waypoints: list[dict[str, float]] = []
    _append_cosine_segment(
        waypoints, start_pose, midpoint_pose, seg_dur, control_hz, max_step, max_speed
    )
    _append_cosine_segment(
        waypoints, midpoint_pose, midpoint_open, grip_dur, control_hz, max_step, max_speed
    )
    _append_cosine_segment(
        waypoints, midpoint_open, midpoint_closed, grip_dur, control_hz, max_step, max_speed
    )
    _append_cosine_segment(
        waypoints, midpoint_closed, start_pose, seg_dur, control_hz, max_step, max_speed
    )
    return waypoints

  joint = str(mode_cfg["joint"])
  delta = float(mode_cfg["delta_deg"])
  seg_dur = float(mode_cfg["segment_duration_sec"])
  steps_per_seg = max(2, int(round(seg_dur * control_hz)))

  waypoints = []
  for k in range(1, steps_per_seg + 1):
    s = k / steps_per_seg
    ease = 0.5 * (1.0 - math.cos(math.pi * s))
    wp = dict(start_pose)
    wp[joint] = round(start_pose[joint] + delta * ease, 4)
    waypoints.append(wp)

  for k in range(1, steps_per_seg + 1):
    s = k / steps_per_seg
    ease = 0.5 * (1.0 - math.cos(math.pi * s))
    wp = dict(start_pose)
    wp[joint] = round(start_pose[joint] + delta * (1.0 - ease), 4)
    waypoints.append(wp)

  return waypoints


def validate_and_analyze_trajectory(
    start_pose: dict[str, float],
    waypoints: list[dict[str, float]],
    calibrations: dict[str, Any],
    kin: Any,
    limits_cfg: dict[str, Any],
) -> dict[str, Any]:
  """Validate R5 limits (step <= 45 deg ceiling, speed <= 90 deg/s ceiling, joint margin, Z floor) and compute FK.

  The joint margin keeps moved joints away from the calibrated stops, but never
  blocks the start pose (rest) itself: a joint resting inside the margin may
  move away from rest and return to it, just not go further toward the stop.
  """
  max_step_allowed = min(
      float(limits_cfg["max_step_deg"]), R5_HARD_MAX_STEP_DEG
  )
  max_speed_allowed = min(
      float(limits_cfg["max_speed_deg_per_sec"]), R5_MAX_SPEED_DEG_PER_SEC
  )
  margin_deg = float(limits_cfg["joint_margin_deg"])
  hard_z_floor = float(limits_cfg["hard_z_floor_m"])
  dt = 1.0 / float(limits_cfg["control_hz"])

  bounds = joint_calibrated_bounds_deg(calibrations)
  full_path = [start_pose] + waypoints

  max_step_seen = 0.0
  peak_speed_seen = 0.0
  moved_joints = {
      j for j in ALL_MOTORS if any(abs(wp[j] - start_pose[j]) > 1e-6 for wp in full_path)
  }

  for i in range(1, len(full_path)):
    prev = full_path[i - 1]
    curr = full_path[i]
    for j in ALL_MOTORS:
      step = abs(curr[j] - prev[j])
      max_step_seen = max(max_step_seen, step)
      speed = step / dt
      peak_speed_seen = max(peak_speed_seen, speed)
      if step > max_step_allowed + 1e-6:
        raise ValueError(
            f"R5 VIOLATION at waypoint {i}: joint {j} step {step:.3f} deg > {max_step_allowed} deg"
        )
      if speed > max_speed_allowed + 1e-6:
        raise ValueError(
            f"R5 VIOLATION at waypoint {i}: joint {j} speed {speed:.2f} deg/s > {max_speed_allowed} deg/s"
        )

  for i, wp in enumerate(full_path):
    for j in BODY_JOINTS:
      bmin, bmax = bounds[j]
      if j in moved_joints:
        lo = min(bmin + margin_deg, start_pose[j])
        hi = max(bmax - margin_deg, start_pose[j])
        if wp[j] < lo - 1e-6 or wp[j] > hi + 1e-6:
          raise ValueError(
              f"R5 VIOLATION at waypoint {i}: moved joint {j}={wp[j]:.2f} deg "
              f"outside calibrated safe range [{lo:.2f}, {hi:.2f}] deg"
          )
      else:
        if wp[j] < bmin or wp[j] > bmax:
          raise ValueError(
              f"R5 VIOLATION at waypoint {i}: holding joint {j}={wp[j]:.2f} deg "
              f"outside calibrated range [{bmin:.2f}, {bmax:.2f}] deg"
          )

  min_z = float("inf")
  tip_xyz_start = None
  tip_xyz_peak = None
  max_delta_idx = 0
  max_delta_val = -1.0

  for i, wp in enumerate(full_path):
    q_deg = np.array([wp[j] for j in ALL_MOTORS], dtype=float)
    t_ee = kin.forward_kinematics(q_deg)
    xyz = t_ee[:3, 3]
    z = float(xyz[2])
    min_z = min(min_z, z)
    if z < hard_z_floor:
      raise ValueError(
          f"R5 VIOLATION at waypoint {i}: FK tip height z={z:.4f} m < hard floor {hard_z_floor:.4f} m"
      )
    if i == 0:
      tip_xyz_start = [round(float(v), 4) for v in xyz]
    dev = sum(abs(wp[j] - start_pose[j]) for j in ALL_MOTORS)
    if dev > max_delta_val:
      max_delta_val = dev
      max_delta_idx = i
      tip_xyz_peak = [round(float(v), 4) for v in xyz]

  return {
      "num_waypoints": len(waypoints),
      "moved_joints": sorted(moved_joints),
      "max_step_deg": round(max_step_seen, 4),
      "peak_speed_deg_per_sec": round(peak_speed_seen, 3),
      "min_tip_height_m": round(min_z, 4),
      "tip_xyz_start_m": tip_xyz_start,
      "tip_xyz_peak_m": tip_xyz_peak,
      "peak_waypoint_index": max_delta_idx,
      "peak_pose": full_path[max_delta_idx],
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description="SO-101 bring-up, read-only bus check, safety checks, and staged motion verification."
  )
  parser.add_argument(
      "--read-only",
      action="store_true",
      help="Stage 1 read-only motor bus inspection (verifies Torque_Enable == 0 and reads 6 encoders).",
  )
  parser.add_argument(
      "--mode",
      choices=["check", "nudge", "abort-test", "hello-world"],
      default="check",
      help="Execution mode: check, nudge (+-5 deg pan), abort-test (q+Enter retreat), or hello-world (Stage 3).",
  )
  parser.add_argument(
      "--robot-config",
      type=Path,
      default=Path("config/robot_config.json"),
  )
  parser.add_argument(
      "--config",
      type=Path,
      default=Path("config/check_robot.json"),
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Compute waypoints, R5 limits, and FK tip heights without connecting or turning torque on.",
  )
  args = parser.parse_args()

  if args.read_only:
    cfg_path = args.robot_config if args.robot_config.exists() else args.config
    robot_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    port = str(robot_cfg.get("port") or robot_cfg.get("serial_port"))
    if args.dry_run:
      print(f"[DRY-RUN] --read-only: would read 6 servos on {port} with Torque OFF.")
      return
    read_motors_readonly(port=port, output_path=Path("data/stage1/motors_readonly.json"))
    return

  from lerobot.model.kinematics import RobotKinematics
  from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

  robot_cfg = json.loads(args.robot_config.read_text(encoding="utf-8"))
  raw_check = json.loads(args.config.read_text(encoding="utf-8"))
  check_cfg = raw_check.get("check_robot", raw_check)
  calibs = load_calibrations(Path(check_cfg["calibration_file"]))
  limits_cfg = check_cfg["limits"]
  mode_cfg = check_cfg.get("modes", {}).get(args.mode, {})

  kin = RobotKinematics(
      urdf_path=str(robot_cfg["urdf_path"]),
      target_frame_name=str(
          robot_cfg.get("target_frame") or robot_cfg.get("target_frame_name", "gripper_frame_link")
      ),
      joint_names=BODY_JOINTS,
  )

  if args.dry_run:
    if "rest_pose" in robot_cfg:
      rest_pose = dict(robot_cfg["rest_pose"])
    else:
      snap = json.loads(
          Path(check_cfg["readonly_snapshot"]).read_text(encoding="utf-8")
      )
      rest_ticks = {k: int(v) for k, v in snap["raw_ticks"].items()}
      rest_pose = ticks_to_norm(rest_ticks, calibs)
    waypoints = generate_cosine_trajectory(
        start_pose=rest_pose,
        mode=args.mode,
        mode_cfg=mode_cfg,
        control_hz=float(limits_cfg["control_hz"]),
    )
    analysis = validate_and_analyze_trajectory(
        start_pose=rest_pose,
        waypoints=waypoints,
        calibrations=calibs,
        kin=kin,
        limits_cfg=limits_cfg,
    )
    print(f"[DRY-RUN] Mode: {args.mode} — NO hardware touched, Torque OFF")
    print(f"Start / End Pose: {rest_pose}")
    print(f"Peak Pose:        {analysis['peak_pose']}")
    print(
        f"Max step: {analysis['max_step_deg']:.4f} deg | "
        f"Peak speed: {analysis['peak_speed_deg_per_sec']:.2f} deg/s | "
        f"Min tip Z: {analysis['min_tip_height_m']:.4f} m"
    )
    return

  out_dir = Path(check_cfg["output_dir"])
  out_dir.mkdir(parents=True, exist_ok=True)
  log_path = out_dir / f"{args.mode}_log.jsonl"
  summary_path = out_dir / f"{args.mode}_summary.json"

  cap_wr, cap_ov = run_preflight(robot_cfg, check_cfg)
  robot = None
  rest_pose: dict[str, float] = {}
  aborted = False
  abort_reason: str | None = None

  try:
    port = str(robot_cfg.get("port") or robot_cfg.get("serial_port"))
    so_cfg = SO101FollowerConfig(
        id=str(robot_cfg["robot_id"]),
        port=port,
        use_degrees=True,
        max_relative_target=float(limits_cfg["max_step_deg"]),
        disable_torque_on_disconnect=False,
    )
    robot = SO101Follower(so_cfg)
    if not robot.calibration:
      raise RuntimeError(
          f"Calibration file for '{robot_cfg['robot_id']}' not loaded. Refusing to move."
      )
    robot.connect(calibrate=False)
    obs = robot.get_observation()
    rest_pose = {
        **{j: round(float(obs[f"{j}.pos"]), 3) for j in BODY_JOINTS},
        "gripper": round(float(obs["gripper.pos"]), 2),
    }

    robot_cfg["rest_pose"] = rest_pose
    robot_cfg["rest_pose_measured_utc"] = datetime.now(timezone.utc).isoformat()
    args.robot_config.write_text(
        json.dumps(robot_cfg, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "rest_pose.json").write_text(
        json.dumps(rest_pose, indent=2) + "\n", encoding="utf-8"
    )

    waypoints = generate_cosine_trajectory(
        start_pose=rest_pose,
        mode=args.mode,
        mode_cfg=mode_cfg,
        control_hz=float(limits_cfg["control_hz"]),
    )
    analysis = validate_and_analyze_trajectory(
        start_pose=rest_pose,
        waypoints=waypoints,
        calibrations=calibs,
        kin=kin,
        limits_cfg=limits_cfg,
    )

    f_wr_pre = cap_wr.read()[1] if cap_wr is not None else None
    f_ov_pre = cap_ov.read()[1] if cap_ov is not None else None
    if f_ov_pre is not None:
      annotate_and_save_frame(
          f_ov_pre,
          out_dir / f"{args.mode}_pre_overhead.png",
          "overhead",
          "PRE-MOTION",
          rest_pose,
          analysis["tip_xyz_start_m"],
      )
    if f_wr_pre is not None:
      annotate_and_save_frame(
          f_wr_pre,
          out_dir / f"{args.mode}_pre_wrist.png",
          "wrist",
          "PRE-MOTION",
          rest_pose,
          analysis["tip_xyz_start_m"],
      )

    control_hz = float(limits_cfg["control_hz"])
    dt = 1.0 / control_hz
    travelled_waypoints: list[dict[str, float]] = [dict(rest_pose)]
    captured_peak = False

    with log_path.open("w", encoding="utf-8") as log_file, AbortWatcher() as watcher:
      log_file.write(
          json.dumps({
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "event": "motion_start",
              "mode": args.mode,
              "rest_pose": rest_pose,
              "analysis": analysis,
          })
          + "\n"
      )
      log_file.flush()

      if args.mode != "check":
        print(
            "\n[MOVING] Type 'q' + Enter (or Ctrl-C) at any time to abort & retreat!"
        )

      for step_idx, target_wp in enumerate(waypoints, start=1):
        reason = watcher.check_abort()
        if reason is not None:
          aborted = True
          abort_reason = reason
          print(f"\n[ABORT RECEIVED] ({reason}) Initiating retreat!")
          log_file.write(
              json.dumps({
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "event": "abort_triggered",
                  "reason": reason,
                  "at_step": step_idx,
                  "pose": travelled_waypoints[-1],
              })
              + "\n"
          )
          log_file.flush()
          execute_r3_retreat(
              robot, travelled_waypoints, rest_pose, control_hz, log_file
          )
          break

        if args.mode != "check":
          action = {f"{j}.pos": float(target_wp[j]) for j in ALL_MOTORS}
          sent = robot.send_action(action)
          travelled_waypoints.append(dict(target_wp))

          if not captured_peak and step_idx == analysis["peak_waypoint_index"]:
            captured_peak = True
            time.sleep(0.05)
            f_wr_pk = cap_wr.read()[1] if cap_wr is not None else None
            f_ov_pk = cap_ov.read()[1] if cap_ov is not None else None
            if f_ov_pk is not None:
              annotate_and_save_frame(
                  f_ov_pk,
                  out_dir / f"{args.mode}_peak_overhead.png",
                  "overhead",
                  "PEAK EXCURSION",
                  target_wp,
                  analysis["tip_xyz_peak_m"],
              )
            if f_wr_pk is not None:
              annotate_and_save_frame(
                  f_wr_pk,
                  out_dir / f"{args.mode}_peak_wrist.png",
                  "wrist",
                  "PEAK EXCURSION",
                  target_wp,
                  analysis["tip_xyz_peak_m"],
              )

          log_file.write(
              json.dumps({
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "event": "waypoint",
                  "step": step_idx,
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
      final_obs = robot.get_observation()
      final_pose = {
          **{j: round(float(final_obs[f"{j}.pos"]), 3) for j in BODY_JOINTS},
          "gripper": round(float(final_obs["gripper.pos"]), 2),
      }
      summary = {
          "timestamp_utc": datetime.now(timezone.utc).isoformat(),
          "mode": args.mode,
          "rest_pose": rest_pose,
          "final_pose": final_pose,
          "aborted": aborted,
          "abort_reason": abort_reason,
          "analysis": analysis,
      }
      summary_path.write_text(
          json.dumps(summary, indent=2) + "\n", encoding="utf-8"
      )
      print(f"\nFinal Joint Positions after {args.mode}: {final_pose}")

  finally:
    if cap_wr is not None:
      cap_wr.release()
    if cap_ov is not None and cap_ov is not cap_wr:
      cap_ov.release()
    safe_disconnect_robot(
        robot, rest_pose, float(limits_cfg["rest_tolerance_deg"])
    )


if __name__ == "__main__":
  main()
