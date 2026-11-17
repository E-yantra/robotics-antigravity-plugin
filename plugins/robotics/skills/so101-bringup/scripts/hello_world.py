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
"""Stage 3 'Hello World' motion & video milestone for SO-101 (works with minimal 1-camera or 2-camera setup).

Sequence:
  1. Run R6 pre-flight check (all 6 motors + all configured cameras <= 0.3 s frame gap).
  2. Start continuous timestamped MP4 video recording (R7) before leaving rest_pose.
  3. Move smoothly from rest_pose -> calibration midpoint (0 deg on the 5 body joints)
     within R5 limits (<= 45 deg/step, <= 90 deg/s, 5 deg margin, FK height floor).
  4. Open and close the gripper jaws once at the midpoint pose.
  5. Return smoothly to rest_pose, disable motor torque (R4), and stop video recording.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_robot import (  # noqa: E402
    generate_cosine_trajectory,
    validate_and_analyze_trajectory,
)
from record_overhead import burn_timestamp_overlay  # noqa: E402
from so101_common import (  # noqa: E402
    ALL_MOTORS,
    AbortWatcher,
    BODY_JOINTS,
    execute_r3_retreat,
    load_calibrations,
    require_cv2,
    run_preflight,
    safe_disconnect_robot,
)


def build_hello_world_trajectory(
    rest_pose: dict[str, float],
    control_hz: float = 20.0,
    max_step_deg: float = 8.0,
    max_speed_deg_per_sec: float = 20.0,
    gripper_open: float = 75.0,
    gripper_closed: float = 10.0,
) -> list[dict[str, float]]:
  """Build the Stage 3 Hello World trajectory (rest -> 0 deg midpoint -> open/close gripper -> rest)."""
  mode_cfg = {
      "segment_duration_sec": 4.0,
      "gripper_duration_sec": 2.0,
      "max_step_deg": max_step_deg,
      "max_speed_deg_per_sec": max_speed_deg_per_sec,
      "gripper_open": gripper_open,
      "gripper_closed": gripper_closed,
  }
  return generate_cosine_trajectory(
      start_pose=rest_pose,
      mode="hello-world",
      mode_cfg=mode_cfg,
      control_hz=control_hz,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Stage 3 SO-101 Hello World motion and timestamped video recording."
  )
  parser.add_argument(
      "--config",
      type=Path,
      default=Path("config/robot_config.json"),
  )
  parser.add_argument(
      "--output-mp4",
      type=Path,
      default=Path("logs/stage3_hello_world_wrist.mp4"),
  )
  parser.add_argument(
      "--output-json",
      type=Path,
      default=Path("logs/stage3_hello_world.json"),
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Validate Stage 3 Hello World trajectory and FK without touching hardware.",
  )
  args = parser.parse_args()

  from lerobot.model.kinematics import RobotKinematics
  from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

  robot_cfg = json.loads(args.config.read_text(encoding="utf-8"))
  calib_path = Path(robot_cfg["calibration_file"]).expanduser()
  calibs = load_calibrations(calib_path)
  safety_cfg = robot_cfg.get("safety", {})
  limits_cfg = {
      "max_step_deg": float(safety_cfg.get("max_step_deg", 20.0)),
      "max_speed_deg_per_sec": float(
          safety_cfg.get("max_speed_deg_per_sec", 30.0)
      ),
      "joint_margin_deg": float(safety_cfg.get("joint_margin_deg", 5.0)),
      "hard_z_floor_m": float(safety_cfg.get("z_floor_search_m", -0.05)),
      "control_hz": float(safety_cfg.get("control_hz", 20.0)),
      "rest_tolerance_deg": float(safety_cfg.get("rest_tolerance_deg", 5.0)),
  }
  rest_pose = dict(robot_cfg["rest_pose"])

  kin = RobotKinematics(
      urdf_path=str(robot_cfg["urdf_path"]),
      target_frame_name=str(
          robot_cfg.get("target_frame")
          or robot_cfg.get("target_frame_name", "gripper_frame_link")
      ),
      joint_names=BODY_JOINTS,
  )

  waypoints = build_hello_world_trajectory(
      rest_pose=rest_pose,
      control_hz=limits_cfg["control_hz"],
      max_step_deg=limits_cfg["max_step_deg"],
      max_speed_deg_per_sec=limits_cfg["max_speed_deg_per_sec"],
  )
  analysis = validate_and_analyze_trajectory(
      start_pose=rest_pose,
      waypoints=waypoints,
      calibrations=calibs,
      kin=kin,
      limits_cfg=limits_cfg,
  )

  if args.dry_run:
    print("[DRY-RUN] Stage 3 'Hello World' — NO hardware touched, Torque OFF")
    print(f"Rest Pose:     {rest_pose}")
    print(f"Midpoint Pose: { {j: 0.0 for j in BODY_JOINTS} } + Gripper Open/Close")
    print(
        f"Waypoints: {analysis['num_waypoints']} | "
        f"Max step: {analysis['max_step_deg']:.3f} deg | "
        f"Peak speed: {analysis['peak_speed_deg_per_sec']:.2f} deg/s | "
        f"Min tip Z: {analysis['min_tip_height_m']:.4f} m"
    )
    return

  cv = require_cv2()
  cap_wr, cap_ov = run_preflight(
      robot_cfg,
      {
          "calibration_file": str(calib_path),
          "preflight": {
              "max_frame_age_sec": float(
                  safety_cfg.get("max_frame_gap_sec", 0.3)
              ),
              "num_checks": 3,
              "interval_sec": 0.1,
          },
      },
  )
  rec_cap = cap_wr if cap_wr is not None else cap_ov
  if rec_cap is None:
    raise RuntimeError("No active camera available for Stage 3 video recording.")

  ok_init, init_frame = rec_cap.read()
  if not ok_init or init_frame is None:
    raise RuntimeError("Failed to read initial frame before Stage 3 motion.")
  h, w = init_frame.shape[:2]
  args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.parent.mkdir(parents=True, exist_ok=True)

  writer = cv.VideoWriter(
      str(args.output_mp4),
      cv.VideoWriter_fourcc(*"mp4v"),
      limits_cfg["control_hz"],
      (w, h),
  )

  robot = None
  aborted = False
  abort_reason: str | None = None
  frames_written = 0
  t_start = time.monotonic()
  total_est_sec = len(waypoints) / limits_cfg["control_hz"] + 1.0

  try:
    for _ in range(5):
      ok_f, frm = rec_cap.read()
      if ok_f and frm is not None:
        frames_written += 1
        writer.write(
            burn_timestamp_overlay(
                frm,
                "STAGE 3 HELLO WORLD (PRE-MOTION)",
                frames_written,
                time.monotonic() - t_start,
                total_est_sec,
                frames_written,
                0,
            )
        )

    port = str(robot_cfg.get("port") or robot_cfg.get("serial_port"))
    so_cfg = SO101FollowerConfig(
        id=str(robot_cfg["robot_id"]),
        port=port,
        use_degrees=True,
        max_relative_target=limits_cfg["max_step_deg"],
        disable_torque_on_disconnect=False,
    )
    robot = SO101Follower(so_cfg)
    robot.connect(calibrate=False)

    dt = 1.0 / limits_cfg["control_hz"]
    travelled: list[dict[str, float]] = [dict(rest_pose)]

    with AbortWatcher() as watcher:
      for step_idx, wp in enumerate(waypoints, start=1):
        reason = watcher.check_abort()
        if reason is not None:
          aborted = True
          abort_reason = reason
          execute_r3_retreat(
              robot, travelled, rest_pose, limits_cfg["control_hz"], None
          )
          break

        action = {f"{j}.pos": float(wp[j]) for j in ALL_MOTORS}
        robot.send_action(action)
        travelled.append(dict(wp))

        ok_f, frm = rec_cap.read()
        if ok_f and frm is not None:
          frames_written += 1
          writer.write(
              burn_timestamp_overlay(
                  frm,
                  "STAGE 3 HELLO WORLD",
                  frames_written,
                  time.monotonic() - t_start,
                  total_est_sec,
                  frames_written,
                  0,
              )
          )
        time.sleep(dt)

    safe_disconnect_robot(robot, rest_pose, limits_cfg["rest_tolerance_deg"])
    robot = None

    for _ in range(5):
      ok_f, frm = rec_cap.read()
      if ok_f and frm is not None:
        frames_written += 1
        writer.write(
            burn_timestamp_overlay(
                frm,
                "STAGE 3 HELLO WORLD (POST-MOTION TORQUE OFF)",
                frames_written,
                time.monotonic() - t_start,
                total_est_sec,
                frames_written,
                0,
            )
        )

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage 3 Hello World",
        "mp4_path": str(args.output_mp4),
        "frames_written": frames_written,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "analysis": analysis,
    }
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
  finally:
    writer.release()
    if cap_wr is not None:
      cap_wr.release()
    if cap_ov is not None and cap_ov is not cap_wr:
      cap_ov.release()
    if robot is not None:
      safe_disconnect_robot(robot, rest_pose, limits_cfg["rest_tolerance_deg"])


if __name__ == "__main__":
  main()
