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
"""Guarded vertical table surface touch-down probe (z_table) for SO-101 (Stage 5 Part B)."""

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
from so101_common import (  # noqa: E402
    ALL_MOTORS,
    AbortWatcher,
    BODY_JOINTS,
    annotate_and_save_frame,
    execute_r3_retreat,
    get_fixed_camera_role_and_config,
    joint_calibrated_bounds_deg,
    load_calibrations,
    run_preflight,
    safe_disconnect_robot,
    ticks_to_norm,
)


def solve_vertical_tip_ik(
    kin: Any,
    seed_deg: dict[str, float],
    tx: float,
    ty: float,
    tz: float,
    gripper_pos: float,
) -> dict[str, float]:
  """Solve 5-DOF IK with closed gripper pointing straight down (z_axis = [0, 0, -1])."""
  pan_rad = math.atan2(-ty, max(tx, 1e-4))
  t_des = np.eye(4)
  t_des[:3, 2] = [0.0, 0.0, -1.0]
  t_des[:3, 0] = [-math.cos(pan_rad), math.sin(pan_rad), 0.0]
  t_des[:3, 1] = [math.sin(pan_rad), math.cos(pan_rad), 0.0]
  t_des[:3, 3] = [tx, ty, tz]

  q_seed = np.array([seed_deg[j] for j in BODY_JOINTS], dtype=float)
  q_sol = kin.inverse_kinematics(
      q_seed, t_des, position_weight=1.0, orientation_weight=0.04, max_iters=35
  )
  t_fk = kin.forward_kinematics(q_sol)
  pos_err = float(np.linalg.norm(t_fk[:3, 3] - np.array([tx, ty, tz])))
  if pos_err > 0.0025:
    raise RuntimeError(
        f"IK failed for target ({tx:.4f}, {ty:.4f}, {tz:.4f}): residual {pos_err*1000:.2f} mm > 2.5 mm"
    )
  if float(t_fk[2, 2]) > -0.94:
    raise RuntimeError(
        f"IK orientation drifted too far from vertical at ({tx:.4f}, {ty:.4f}, {tz:.4f}): z_ax={t_fk[:3,2]}"
    )

  out = {
      BODY_JOINTS[i]: round(float(q_sol[i]), 3)
      for i in range(len(BODY_JOINTS))
  }
  out["gripper"] = round(float(gripper_pos), 2)
  return out


def interpolate_segment(
    p0: dict[str, float],
    p1: dict[str, float],
    max_speed_deg_s: float,
    control_hz: float,
    min_duration_sec: float = 0.25,
) -> list[dict[str, float]]:
  """Cosine-eased joint interpolation respecting angular velocity and step limits."""
  max_delta = max(abs(p1[j] - p0[j]) for j in ALL_MOTORS)
  dur = max(
      min_duration_sec,
      (math.pi * max_delta) / max(2.0 * max_speed_deg_s, 1e-3),
  )
  n_steps = max(2, int(math.ceil(dur * control_hz)))
  wps: list[dict[str, float]] = []
  for k in range(1, n_steps + 1):
    s = k / n_steps
    ease = 0.5 * (1.0 - math.cos(math.pi * s))
    wp = {j: round(p0[j] + (p1[j] - p0[j]) * ease, 4) for j in ALL_MOTORS}
    wps.append(wp)
  return wps


def build_touchdown_plan(
    rest_pose: dict[str, float],
    td_cfg: dict[str, Any],
    calibs: dict[str, Any],
    kin: Any,
) -> dict[str, Any]:
  """Plan vertical descent waypoints and enforce calibrated joint margins and hard Z floor."""
  tx, ty = float(td_cfg["patch_xy_m"][0]), float(td_cfg["patch_xy_m"][1])
  z_app = float(td_cfg["z_approach_start_m"])
  z_desc_start = float(td_cfg["z_descent_start_m"])
  z_floor = float(td_cfg["z_hard_floor_m"])
  dz = float(td_cfg["step_z_m"])
  grip_closed = float(td_cfg["closed_gripper_pos"])
  margin = float(td_cfg["limits"]["joint_margin_deg"])

  if z_floor < -0.05001:
    raise ValueError(
        f"Hard floor {z_floor} m violates -0.050 m table-touchdown safety floor!"
    )

  bounds = joint_calibrated_bounds_deg(calibs)
  mid_seed = {
      "shoulder_pan": -15.0,
      "shoulder_lift": -20.0,
      "elbow_flex": 42.0,
      "wrist_flex": 68.0,
      "wrist_roll": rest_pose["wrist_roll"],
      "gripper": grip_closed,
  }
  hover_pose = solve_vertical_tip_ik(kin, mid_seed, tx, ty, z_app, grip_closed)
  clear_above_rest = solve_vertical_tip_ik(
      kin,
      {
          "shoulder_pan": rest_pose["shoulder_pan"],
          **{k: mid_seed[k] for k in BODY_JOINTS if k != "shoulder_pan"},
          "gripper": grip_closed,
      },
      0.155,
      0.0,
      z_app,
      grip_closed,
  )

  for label_pose, pose_chk in [
      ("clear_above_rest", clear_above_rest),
      ("hover_above_patch", hover_pose),
  ]:
    for j in BODY_JOINTS:
      bmin, bmax = bounds[j]
      if pose_chk[j] < (bmin + margin) or pose_chk[j] > (bmax - margin):
        raise ValueError(
            f"R5 VIOLATION in {label_pose}: joint {j}={pose_chk[j]:.2f} outside safe [{bmin+margin:.1f}, {bmax-margin:.1f}]"
        )

  descent_poses: list[dict[str, Any]] = []
  seed = dict(hover_pose)
  z_curr = z_desc_start
  while z_curr >= z_floor - 1e-6:
    z_target = round(max(z_curr, z_floor), 4)
    pose = solve_vertical_tip_ik(kin, seed, tx, ty, z_target, grip_closed)
    for j in BODY_JOINTS:
      bmin, bmax = bounds[j]
      if pose[j] < (bmin + margin) or pose[j] > (bmax - margin):
        raise ValueError(
            f"R5 VIOLATION at z={z_target:.4f}m: joint {j}={pose[j]:.2f} outside safe [{bmin+margin:.1f}, {bmax-margin:.1f}]"
        )
    descent_poses.append({"z_cmd_m": z_target, "pose": pose})
    seed = dict(pose)
    z_curr -= dz

  return {
      "patch_xy_m": [tx, ty],
      "clear_above_rest_pose": clear_above_rest,
      "hover_above_patch_pose": hover_pose,
      "descent_probes": descent_poses,
      "min_possible_tip_z_m": descent_poses[-1]["z_cmd_m"],
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Guarded table height touch-down measurement for SO-101 (Stage 5 Part B)."
  )
  parser.add_argument(
      "--robot-config", type=Path, default=Path("config/robot_config.json")
  )
  parser.add_argument(
      "--check-config", type=Path, default=Path("config/check_robot.json")
  )
  parser.add_argument(
      "--config", type=Path, default=Path("config/table_touchdown.json")
  )
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  from lerobot.model.kinematics import RobotKinematics
  from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

  robot_cfg = json.loads(args.robot_config.read_text(encoding="utf-8"))
  if get_fixed_camera_role_and_config(robot_cfg) is None:
    raise RuntimeError(
        "[STAGE 5 STEP 0 STOP] No fixed/overhead camera is configured in robot_config['cameras']. "
        "Stage 5 requires a fixed camera viewing the arm and tabletop. Add a fixed camera or stop at Stage 3."
    )

  raw_check = json.loads(args.check_config.read_text(encoding="utf-8"))
  check_cfg = raw_check.get("check_robot", raw_check)
  raw_td = json.loads(args.config.read_text(encoding="utf-8"))
  td_cfg = raw_td.get("table_touchdown", raw_td)
  out_dir = Path(td_cfg["output_dir"])
  out_dir.mkdir(parents=True, exist_ok=True)

  calibs = load_calibrations(Path(check_cfg["calibration_file"]))
  kin = RobotKinematics(
      urdf_path=str(robot_cfg["urdf_path"]),
      target_frame_name=str(
          robot_cfg.get("target_frame")
          or robot_cfg.get("target_frame_name", "gripper_frame_link")
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
      rest_pose = ticks_to_norm(
          {k: int(v) for k, v in snap["raw_ticks"].items()}, calibs
      )
    plan = build_touchdown_plan(rest_pose, td_cfg, calibs, kin)
    print("[DRY-RUN] Guarded Table Touch-Down Plan — NO hardware touched, Torque OFF")
    print(f"Patch XY (m):            {plan['patch_xy_m']}")
    print(f"Descent steps:           {len(plan['descent_probes'])}")
    print(f"Lowest possible tip Z:   {plan['min_possible_tip_z_m']:+.4f} m")
    return

  cap_wr, cap_ov = run_preflight(robot_cfg, check_cfg)
  robot = None
  rest_pose: dict[str, float] = {}
  limits = td_cfg["limits"]
  control_hz = float(limits["control_hz"])
  dt = 1.0 / control_hz
  max_speed = float(limits["max_speed_deg_per_sec"])

  log_path = out_dir / "touchdown_log.jsonl"
  result_path = out_dir / "table_height.json"

  try:
    port = str(robot_cfg.get("port") or robot_cfg.get("serial_port"))
    so_cfg = SO101FollowerConfig(
        id=str(robot_cfg["robot_id"]),
        port=port,
        use_degrees=True,
        max_relative_target=float(limits["max_step_deg"]),
        disable_torque_on_disconnect=False,
    )
    robot = SO101Follower(so_cfg)
    if not robot.calibration:
      raise RuntimeError("Robot calibration missing. Refusing to move.")
    robot.connect(calibrate=False)

    obs0 = robot.get_observation()
    rest_pose = {
        **{j: round(float(obs0[f"{j}.pos"]), 3) for j in BODY_JOINTS},
        "gripper": round(float(obs0["gripper.pos"]), 2),
    }
    plan = build_touchdown_plan(rest_pose, td_cfg, calibs, kin)
    travelled: list[dict[str, float]] = [dict(rest_pose)]

    with log_path.open("w", encoding="utf-8") as jf, AbortWatcher() as watcher:

      def run_eased_move(
          p_start: dict[str, float], p_goal: dict[str, float], label: str
      ) -> bool:
        wps = interpolate_segment(p_start, p_goal, max_speed, control_hz)
        for wp in wps:
          reason = watcher.check_abort()
          if reason is not None:
            print(f"\n[ABORT RECEIVED] ({reason}) during {label}! Retreating...")
            jf.write(
                json.dumps({"event": "abort", "reason": reason, "stage": label})
                + "\n"
            )
            jf.flush()
            execute_r3_retreat(robot, travelled, rest_pose, control_hz, jf)
            return False
          robot.send_action({f"{j}.pos": float(wp[j]) for j in ALL_MOTORS})
          travelled.append(dict(wp))
          time.sleep(dt)
        return True

      if not run_eased_move(
          rest_pose, plan["clear_above_rest_pose"], "lift_from_rest"
      ):
        return
      if not run_eased_move(
          plan["clear_above_rest_pose"],
          plan["hover_above_patch_pose"],
          "pan_to_patch",
      ):
        return

      contact_cfg = td_cfg["contact_detection"]
      baseline_sags: list[float] = []
      baseline_loads: list[float] = []
      prev_z_meas: float | None = None
      stall_count = 0
      contact_detected = False
      contact_record: dict[str, Any] | None = None
      probes_log: list[dict[str, Any]] = []

      curr_pose = dict(plan["hover_above_patch_pose"])
      for idx, probe in enumerate(plan["descent_probes"], start=1):
        z_cmd = float(probe["z_cmd_m"])
        target_pose = dict(probe["pose"])
        if not run_eased_move(curr_pose, target_pose, f"descent_step_{idx}"):
          return
        curr_pose = target_pose

        time.sleep(0.18)
        obs = robot.get_observation()
        q_meas = {j: round(float(obs[f"{j}.pos"]), 3) for j in BODY_JOINTS}
        loads_raw = robot.bus.sync_read(
            "Present_Load", normalize=False, num_retry=2
        )
        abs_loads = {k: abs(int(v)) for k, v in loads_raw.items()}
        lift_elbow_load = float(
            abs_loads["shoulder_lift"]
            + abs_loads["elbow_flex"]
            + abs_loads["wrist_flex"]
        )

        t_meas = kin.forward_kinematics(
            np.array([q_meas[j] for j in BODY_JOINTS], dtype=float)
        )
        x_meas, y_meas, z_meas = (
            float(t_meas[0, 3]),
            float(t_meas[1, 3]),
            float(t_meas[2, 3]),
        )
        raw_sag = z_cmd - z_meas

        if idx <= 4:
          baseline_sags.append(raw_sag)
          baseline_loads.append(lift_elbow_load)
          sag_excess = 0.0
          load_excess = 0.0
        else:
          sag_excess = raw_sag - float(np.mean(baseline_sags))
          load_excess = lift_elbow_load - float(np.mean(baseline_loads))
          if (
              prev_z_meas is not None
              and (prev_z_meas - z_meas)
              < float(contact_cfg["stall_z_drop_threshold_m"])
          ):
            stall_count += 1
          else:
            stall_count = 0

        prev_z_meas = z_meas
        step_info = {
            "probe_step": idx,
            "z_cmd_m": round(z_cmd, 4),
            "z_meas_fk_m": round(z_meas, 4),
            "xy_meas_fk_m": [round(x_meas, 4), round(y_meas, 4)],
            "sag_excess_mm": round(sag_excess * 1000.0, 2),
            "load_excess": round(load_excess, 1),
            "stall_count": stall_count,
            "q_meas_deg": q_meas,
        }
        probes_log.append(step_info)
        jf.write(json.dumps({"event": "descent_probe", **step_info}) + "\n")
        jf.flush()

        if idx > 4 and (
            sag_excess >= float(contact_cfg["min_sag_increase_m"])
            or load_excess >= float(contact_cfg["min_load_increase"])
            or stall_count >= int(contact_cfg["min_stall_steps"])
        ):
          contact_detected = True
          contact_record = dict(step_info)
          print(
              f"\n[CONTACT DETECTED AT PROBE #{idx}] Measured table surface z_table = {z_meas:+.4f} m"
          )
          f_wr_td = cap_wr.read()[1] if cap_wr is not None else None
          f_ov_td = cap_ov.read()[1] if cap_ov is not None else None
          pose_with_grip = {
              **q_meas,
              "gripper": float(target_pose["gripper"]),
          }
          if f_ov_td is not None:
            annotate_and_save_frame(
                f_ov_td,
                out_dir / "touchdown_contact_overhead.png",
                "overhead",
                f"TABLE CONTACT z={z_meas:+.4f}m",
                pose_with_grip,
                [x_meas, y_meas, z_meas],
            )
          if f_wr_td is not None:
            annotate_and_save_frame(
                f_wr_td,
                out_dir / "touchdown_contact_wrist.png",
                "wrist",
                f"TABLE CONTACT z={z_meas:+.4f}m",
                pose_with_grip,
                [x_meas, y_meas, z_meas],
            )
          break

      execute_r3_retreat(robot, travelled, rest_pose, control_hz, jf)

      out_summary = {
          "timestamp_utc": datetime.now(timezone.utc).isoformat(),
          "contact_detected": contact_detected,
          "z_table_m": (
              round(float(contact_record["z_meas_fk_m"]), 4)
              if contact_record
              else None
          ),
          "patch_xy_m": plan["patch_xy_m"],
          "contact_probe": contact_record,
          "probes": probes_log,
      }
      result_path.write_text(
          json.dumps(out_summary, indent=2) + "\n", encoding="utf-8"
      )

      if contact_detected and contact_record is not None:
        z_tbl = round(float(contact_record["z_meas_fk_m"]), 4)
        robot_cfg["z_table_m"] = z_tbl
        robot_cfg["z_table_patch_xy_m"] = plan["patch_xy_m"]
        robot_cfg["z_table_measured_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        args.robot_config.write_text(
            json.dumps(robot_cfg, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved z_table_m = {z_tbl:+.4f} m -> {result_path}")
  finally:
    if cap_wr is not None:
      cap_wr.release()
    if cap_ov is not None and cap_ov is not cap_wr:
      cap_ov.release()
    safe_disconnect_robot(robot, rest_pose, float(limits["rest_tolerance_deg"]))


if __name__ == "__main__":
  main()
