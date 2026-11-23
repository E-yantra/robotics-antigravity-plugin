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
"""Multi-pose 3D hand-eye calibration sweep and perspective-regularized 3x4 projection fitter (Stage 5 Part C)."""

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
from detect_marker import detect_gripper_marker  # noqa: E402
from measure_table import interpolate_segment, solve_vertical_tip_ik  # noqa: E402
from so101_common import (  # noqa: E402
    ALL_MOTORS,
    AbortWatcher,
    BODY_JOINTS,
    execute_r3_retreat,
    get_fixed_camera_role_and_config,
    joint_calibrated_bounds_deg,
    load_calibrations,
    require_cv2,
    run_preflight,
    safe_disconnect_robot,
    ticks_to_norm,
)


def build_calibration_grid(
    rest_pose: dict[str, float],
    z_table_m: float,
    cal_cfg: dict[str, Any],
    calibs: dict[str, Any],
    kin: Any,
) -> list[dict[str, Any]]:
  """Build safe 2-height workspace calibration poses strictly above measured z_table_m."""
  bounds = joint_calibrated_bounds_deg(calibs)
  margin = float(cal_cfg["limits"]["joint_margin_deg"])
  grip_closed = float(cal_cfg["closed_gripper_pos"])

  poses: list[dict[str, Any]] = []
  seed = {
      "shoulder_pan": 0.0,
      "shoulder_lift": -35.0,
      "elbow_flex": 55.0,
      "wrist_flex": 70.0,
      "wrist_roll": rest_pose.get("wrist_roll", 0.0),
      "gripper": grip_closed,
  }

  for level_idx, dz_clear in enumerate(cal_cfg["z_clearances_above_table_m"]):
    z_target = round(z_table_m + float(dz_clear), 4)
    if z_target <= z_table_m + 0.030:
      raise ValueError(
          f"Calibration height {z_target:.4f}m too close to table {z_table_m:.4f}m!"
      )
    xy_list = (
        cal_cfg["grid_xy_m"]
        if level_idx % 2 == 0
        else list(reversed(cal_cfg["grid_xy_m"]))
    )
    for tx, ty in xy_list:
      pose = solve_vertical_tip_ik(
          kin, seed, float(tx), float(ty), z_target, grip_closed
      )
      for j in BODY_JOINTS:
        bmin, bmax = bounds[j]
        if pose[j] < (bmin + margin) or pose[j] > (bmax - margin):
          raise ValueError(
              f"R5 VIOLATION at ({tx:.3f},{ty:.3f},{z_target:.3f}): joint {j}={pose[j]:.2f} outside safe range"
          )
      poses.append({
          "pose_idx": len(poses) + 1,
          "target_tip_xyz_m": [
              round(float(tx), 4),
              round(float(ty), 4),
              z_target,
          ],
          "clearance_above_table_m": round(float(dz_clear), 4),
          "pose": pose,
      })
      seed = dict(pose)
  return poses


def pixel_to_world_at_z(
    proj_matrix: np.ndarray, u: float, v: float, z_plane_m: float
) -> tuple[float, float]:
  """Closed-form ray-plane intersection mapping overhead pixel (u, v) to robot (X, Y) at Z = z_plane_m."""
  m_mat = np.asarray(proj_matrix, dtype=float)
  a_mat = np.array(
      [
          [m_mat[0, 0] - u * m_mat[2, 0], m_mat[0, 1] - u * m_mat[2, 1]],
          [m_mat[1, 0] - v * m_mat[2, 0], m_mat[1, 1] - v * m_mat[2, 1]],
      ],
      dtype=float,
  )
  b_vec = np.array(
      [
          u * (m_mat[2, 2] * z_plane_m + m_mat[2, 3])
          - (m_mat[0, 2] * z_plane_m + m_mat[0, 3]),
          v * (m_mat[2, 2] * z_plane_m + m_mat[2, 3])
          - (m_mat[1, 2] * z_plane_m + m_mat[1, 3]),
      ],
      dtype=float,
  )
  xy = np.linalg.solve(a_mat, b_vec)
  return float(xy[0]), float(xy[1])


def fit_projective_handeye(
    samples: list[dict[str, Any]],
    mount_z_offset_m: float = 0.098,
    reg_lambda: float = 500.0,
) -> dict[str, Any]:
  """Fit perspective-regularized 3x4 projective camera matrix M mapping [X_tip, Y_tip, Z_tip, 1]^T -> [u, v, 1]^T."""
  if len(samples) < 8:
    raise RuntimeError(
        f"Need at least 8 valid calibration samples, got {len(samples)}"
    )

  pts = np.array([s["tip_xyz_m"] for s in samples], dtype=float)
  uvs = np.array([s["marker_uv_px"] for s in samples], dtype=float)

  rows = []
  rhs = []
  for (x_w, y_w, z_w), (u_px, v_px) in zip(pts, uvs):
    rows.append([
        x_w,
        y_w,
        z_w,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -u_px * x_w,
        -u_px * y_w,
        -u_px * z_w,
    ])
    rhs.append(u_px)
    rows.append([
        0.0,
        0.0,
        0.0,
        0.0,
        x_w,
        y_w,
        z_w,
        1.0,
        -v_px * x_w,
        -v_px * y_w,
        -v_px * z_w,
    ])
    rhs.append(v_px)

  for col in (8, 9, 10):
    reg = [0.0] * 11
    reg[col] = float(reg_lambda)
    rows.append(reg)
    rhs.append(0.0)

  theta_m, _, _, _ = np.linalg.lstsq(
      np.array(rows, dtype=float), np.array(rhs, dtype=float), rcond=None
  )
  m_final = [
      [
          float(theta_m[0]),
          float(theta_m[1]),
          float(theta_m[2]),
          float(theta_m[3]),
      ],
      [
          float(theta_m[4]),
          float(theta_m[5]),
          float(theta_m[6]),
          float(theta_m[7]),
      ],
      [float(theta_m[8]), float(theta_m[9]), float(theta_m[10]), 1.0],
  ]
  m_np = np.array(m_final, dtype=float)

  per_pose_errors = []
  for s in samples:
    xt, yt, zt = [float(v) for v in s["tip_xyz_m"]]
    proj = m_np @ np.array([xt, yt, zt, 1.0], dtype=float)
    u_hat, v_hat = float(proj[0] / proj[2]), float(proj[1] / proj[2])
    u_obs_k, v_obs_k = float(s["marker_uv_px"][0]), float(s["marker_uv_px"][1])
    px_err = math.hypot(u_hat - u_obs_k, v_hat - v_obs_k)

    xy_hat = pixel_to_world_at_z(m_np, u_obs_k, v_obs_k, zt)
    mm_err = math.hypot(xy_hat[0] - xt, xy_hat[1] - yt) * 1000.0
    per_pose_errors.append({
        "pose_idx": s["pose_idx"],
        "tip_xyz_m": [round(xt, 4), round(yt, 4), round(zt, 4)],
        "obs_uv_px": [round(u_obs_k, 2), round(v_obs_k, 2)],
        "pred_uv_px": [round(u_hat, 2), round(v_hat, 2)],
        "reproj_error_px": round(px_err, 2),
        "table_xy_error_mm": round(mm_err, 2),
    })

  rmse_px = float(
      np.sqrt(np.mean([e["reproj_error_px"] ** 2 for e in per_pose_errors]))
  )
  rmse_mm = float(
      np.sqrt(np.mean([e["table_xy_error_mm"] ** 2 for e in per_pose_errors]))
  )

  return {
      "projection_matrix_3x4": m_final,
      "mount_z_offset_m": mount_z_offset_m,
      "num_samples": len(samples),
      "rmse_reprojection_px": round(rmse_px, 2),
      "rmse_localization_mm": round(rmse_mm, 2),
      "per_pose_errors": per_pose_errors,
  }


def save_calibration_grid_image(
    base_frame: np.ndarray,
    fit: dict[str, Any],
    z_table_m: float,
    out_path: Path,
) -> None:
  """Render projected tabletop coordinate grid at Z = z_table_m onto base_frame."""
  cv = require_cv2()
  vis = base_frame.copy()
  m_mat = np.array(fit["projection_matrix_3x4"], dtype=float)

  for x_m in [0.12, 0.16, 0.20, 0.24]:
    pts = []
    for y_m in np.linspace(-0.12, 0.06, 30):
      proj = m_mat @ np.array([x_m, y_m, z_table_m, 1.0])
      pts.append([int(round(proj[0] / proj[2])), int(round(proj[1] / proj[2]))])
    cv.polylines(
        vis, [np.array(pts, dtype=np.int32)], False, (255, 210, 60), 1, cv.LINE_AA
    )
    cv.putText(
        vis,
        f"X={x_m*100:.0f}cm",
        (pts[-1][0] - 66, pts[-1][1] + 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 225, 100),
        1,
        cv.LINE_AA,
    )

  for y_m in [-0.10, -0.05, 0.00, 0.05]:
    pts = []
    for x_m in np.linspace(0.11, 0.25, 30):
      proj = m_mat @ np.array([x_m, y_m, z_table_m, 1.0])
      pts.append([int(round(proj[0] / proj[2])), int(round(proj[1] / proj[2]))])
    cv.polylines(
        vis, [np.array(pts, dtype=np.int32)], False, (255, 210, 60), 1, cv.LINE_AA
    )
    cv.putText(
        vis,
        f"Y={y_m*100:+.0f}cm",
        (pts[-1][0] - 24, pts[-1][1] - 8),
        cv.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 225, 100),
        1,
        cv.LINE_AA,
    )

  for err in fit["per_pose_errors"]:
    uo, vo = int(round(err["obs_uv_px"][0])), int(round(err["obs_uv_px"][1]))
    up, vp = int(round(err["pred_uv_px"][0])), int(round(err["pred_uv_px"][1]))
    cv.circle(vis, (uo, vo), 6, (80, 255, 120), 2)
    cv.drawMarker(vis, (up, vp), (0, 60, 255), cv.MARKER_CROSS, 10, 2)

  bar = vis.copy()
  cv.rectangle(bar, (0, 0), (vis.shape[1], 56), (10, 10, 10), -1)
  cv.addWeighted(bar, 0.72, vis, 0.28, 0.0, vis)
  cv.putText(
      vis,
      f"SO-101 3D Hand-Eye Calibration (N={fit['num_samples']} poses | "
      f"RMSE={fit['rmse_reprojection_px']:.2f}px / {fit['rmse_localization_mm']:.2f}mm)",
      (12, 22),
      cv.FONT_HERSHEY_SIMPLEX,
      0.54,
      (90, 255, 160),
      1,
      cv.LINE_AA,
  )
  cv.putText(
      vis,
      f"Cyan grid = tabletop plane z_table={z_table_m:+.4f}m | Green circles = observed | Red crosses = model",
      (12, 46),
      cv.FONT_HERSHEY_SIMPLEX,
      0.46,
      (240, 240, 240),
      1,
      cv.LINE_AA,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  cv.imwrite(str(out_path), vis)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Multi-pose 3D hand-eye calibration for SO-101 (Stage 5 Part C)."
  )
  parser.add_argument(
      "--robot-config", type=Path, default=Path("config/robot_config.json")
  )
  parser.add_argument(
      "--check-config", type=Path, default=Path("config/check_robot.json")
  )
  parser.add_argument(
      "--marker-config", type=Path, default=Path("config/marker_detect.json")
  )
  parser.add_argument(
      "--config", type=Path, default=Path("config/handeye_calib.json")
  )
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument(
      "--fit-from-log",
      action="store_true",
      help="Fit calibration model from sweep log + touchdown contact frame without moving motors.",
  )
  args = parser.parse_args()

  from lerobot.model.kinematics import RobotKinematics
  from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

  robot_cfg = json.loads(args.robot_config.read_text(encoding="utf-8"))
  if get_fixed_camera_role_and_config(robot_cfg) is None:
    raise RuntimeError(
        "[STAGE 5 STEP 0 STOP] No fixed/overhead camera is configured in robot_config['cameras']. "
        "Stage 5 Hand-Eye Calibration requires a fixed camera viewing the workspace."
    )

  raw_check = json.loads(args.check_config.read_text(encoding="utf-8"))
  check_cfg = raw_check.get("check_robot", raw_check)
  raw_marker = json.loads(args.marker_config.read_text(encoding="utf-8"))
  marker_cfg = raw_marker.get("marker_detect", raw_marker)
  raw_cal = json.loads(args.config.read_text(encoding="utf-8"))
  cal_cfg = raw_cal.get("handeye_calib", raw_cal)
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
    z_table_est = float(
        robot_cfg.get("z_table_m", cal_cfg["nominal_z_table_for_dry_run_m"])
    )
    if "rest_pose" in robot_cfg:
      rest_pose = dict(robot_cfg["rest_pose"])
    else:
      snap = json.loads(
          Path(check_cfg["readonly_snapshot"]).read_text(encoding="utf-8")
      )
      rest_pose = ticks_to_norm(
          {k: int(v) for k, v in snap["raw_ticks"].items()}, calibs
      )
    grid = build_calibration_grid(rest_pose, z_table_est, cal_cfg, calibs, kin)
    print(
        f"[DRY-RUN] Hand-Eye Calibration Sweep: {len(grid)} poses at clearances "
        f"{cal_cfg['z_clearances_above_table_m']} above z_table={z_table_est:+.4f}m"
    )
    return

  cv = require_cv2()
  z_table_m = float(robot_cfg["z_table_m"])
  out_dir = Path(cal_cfg["output_dir"])
  out_dir.mkdir(parents=True, exist_ok=True)
  log_path = out_dir / "handeye_sweep_log.jsonl"
  samples_path = out_dir / "handeye_samples.json"
  cal_out_path = Path(cal_cfg["calibration_output"])
  grid_img_path = out_dir / "handeye_calibration_grid.png"

  if args.fit_from_log:
    samples: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
      row = json.loads(line)
      if row.get("event") == "calibration_sample":
        samples.append({
            "pose_idx": row["pose_idx"],
            "q_meas_deg": row["q_meas_deg"],
            "tip_xyz_m": row["tip_xyz_m"],
            "T_ee_world": row["T_ee_world"],
            "marker_uv_px": row["marker_uv_px"],
            "white_surround_fraction": row["white_surround_fraction"],
        })
    td_path = out_dir / "table_height.json"
    td_img_path = out_dir / "touchdown_contact_overhead.png"
    if td_path.exists() and td_img_path.exists():
      td_probe = json.loads(td_path.read_text(encoding="utf-8")).get(
          "contact_probe"
      )
      td_frame = cv.imread(str(td_img_path))
      if td_probe and td_frame is not None:
        det_td = detect_gripper_marker(td_frame, marker_cfg)
        if det_td["detected"]:
          t_td = kin.forward_kinematics(
              np.array(
                  [td_probe["q_meas_deg"][j] for j in BODY_JOINTS], dtype=float
              )
          )
          samples.append({
              "pose_idx": 0,
              "q_meas_deg": td_probe["q_meas_deg"],
              "tip_xyz_m": [round(float(v), 5) for v in t_td[:3, 3]],
              "T_ee_world": [
                  [round(float(v), 6) for v in r] for r in t_td
              ],
              "marker_uv_px": det_td["accepted"][0]["centroid_px"],
              "white_surround_fraction": det_td["accepted"][0][
                  "white_surround_fraction"
              ],
          })
    samples_path.write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )
    fit = fit_projective_handeye(samples)
    fit["z_table_m"] = z_table_m
    fit["calibrated_utc"] = datetime.now(timezone.utc).isoformat()
    cal_out_path.parent.mkdir(parents=True, exist_ok=True)
    cal_out_path.write_text(
        json.dumps(fit, indent=2) + "\n", encoding="utf-8"
    )
    ref_path = out_dir.parent / "stage4" / "reference_overhead.png"
    if ref_path.exists():
      ref_frame = cv.imread(str(ref_path))
      if ref_frame is not None:
        save_calibration_grid_image(ref_frame, fit, z_table_m, grid_img_path)
    print(
        f"Hand-Eye Calibration Fitted ({fit['num_samples']} samples): "
        f"RMSE = {fit['rmse_reprojection_px']:.2f} px ({fit['rmse_localization_mm']:.2f} mm)"
    )
    return

  cap_wr, cap_ov = run_preflight(robot_cfg, check_cfg)
  if cap_ov is None:
    if cap_wr is not None:
      cap_wr.release()
    raise RuntimeError("Stage 5 Hand-Eye calibration requires an active fixed/overhead camera.")
  robot = None
  rest_pose: dict[str, float] = {}
  limits = cal_cfg["limits"]
  control_hz = float(limits["control_hz"])
  dt = 1.0 / control_hz
  max_speed = float(limits["max_speed_deg_per_sec"])

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
      raise RuntimeError("Robot calibration missing.")
    robot.connect(calibrate=False)

    obs0 = robot.get_observation()
    rest_pose = {
        **{j: round(float(obs0[f"{j}.pos"]), 3) for j in BODY_JOINTS},
        "gripper": round(float(obs0["gripper.pos"]), 2),
    }
    grid = build_calibration_grid(rest_pose, z_table_m, cal_cfg, calibs, kin)
    travelled: list[dict[str, float]] = [dict(rest_pose)]
    collected_samples: list[dict[str, Any]] = []
    base_frame_for_grid = None
    consecutive_misses = 0

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

      curr_pose = dict(rest_pose)
      for item in grid:
        idx = int(item["pose_idx"])
        target_pose = dict(item["pose"])
        if not run_eased_move(curr_pose, target_pose, f"grid_pose_{idx}"):
          return
        curr_pose = target_pose

        time.sleep(float(limits["settle_sec"]))
        obs = robot.get_observation()
        q_meas = {j: round(float(obs[f"{j}.pos"]), 3) for j in BODY_JOINTS}
        t_ee = kin.forward_kinematics(
            np.array([q_meas[j] for j in BODY_JOINTS], dtype=float)
        )

        for _ in range(5):
          cap_ov.read()
        ok_ov, f_ov = cap_ov.read()
        if not ok_ov or f_ov is None:
          raise RuntimeError(
              f"Camera anomaly: failed to capture overhead frame at grid pose {idx}"
          )
        if base_frame_for_grid is None:
          base_frame_for_grid = f_ov.copy()

        det = detect_gripper_marker(f_ov, marker_cfg)
        if not det["detected"]:
          consecutive_misses += 1
          jf.write(
              json.dumps({
                  "event": "detector_disagreement",
                  "pose_idx": idx,
                  "det": det,
              })
              + "\n"
          )
          jf.flush()
          if consecutive_misses >= 3:
            execute_r3_retreat(robot, travelled, rest_pose, control_hz, jf)
            return
          continue

        consecutive_misses = 0
        uv = det["accepted"][0]["centroid_px"]
        sample = {
            "pose_idx": idx,
            "q_meas_deg": q_meas,
            "tip_xyz_m": [round(float(v), 5) for v in t_ee[:3, 3]],
            "T_ee_world": [[round(float(v), 6) for v in row] for row in t_ee],
            "marker_uv_px": uv,
            "white_surround_fraction": det["accepted"][0][
                "white_surround_fraction"
            ],
        }
        collected_samples.append(sample)
        jf.write(
            json.dumps({"event": "calibration_sample", **sample}) + "\n"
        )
        jf.flush()

      execute_r3_retreat(robot, travelled, rest_pose, control_hz, jf)
      samples_path.write_text(
          json.dumps(collected_samples, indent=2) + "\n", encoding="utf-8"
      )
      fit = fit_projective_handeye(collected_samples)
      fit["z_table_m"] = z_table_m
      fit["calibrated_utc"] = datetime.now(timezone.utc).isoformat()
      cal_out_path.parent.mkdir(parents=True, exist_ok=True)
      cal_out_path.write_text(
          json.dumps(fit, indent=2) + "\n", encoding="utf-8"
      )
      if base_frame_for_grid is not None:
        save_calibration_grid_image(
            base_frame_for_grid, fit, z_table_m, grid_img_path
        )
  finally:
    if cap_wr is not None:
      cap_wr.release()
    if cap_ov is not None and cap_ov is not cap_wr:
      cap_ov.release()
    safe_disconnect_robot(robot, rest_pose, float(limits["rest_tolerance_deg"]))


if __name__ == "__main__":
  main()
