# Copyright 2026 Google LLC
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

"""Stage 2 4-camera freshness check, 2x2 contact sheet, and MP4 recorder."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from fr3_arm_controller import read_arm_joints_or_rest
from fr3_camera_controller import (
    CANONICAL_CAMERAS,
    CameraFrame,
    Fr3DuoCameraController,
    render_contact_sheet,
)
from fr3_config import (
    RobotConfig,
    normalize_robot_config,
    open_mp4_writer,
    resolve_stage_output_dir,
    save_rgb_png,
)
import fr3_kinematics
import fr3_mjcf

if TYPE_CHECKING:
  import fr3_duo_robot


def _record_mosaic_mp4(
    rig: Fr3DuoCameraController,
    mp4_path: Path,
    jsonl_path: Path,
    *,
    duration_s: float,
    target_fps: float,
) -> dict[str, Any]:
  """Records 2x2 contact-sheet frames to MP4 and per-frame logs to JSONL."""
  mp4_path.parent.mkdir(parents=True, exist_ok=True)
  jsonl_path.parent.mkdir(parents=True, exist_ok=True)

  first_batch = rig.capture_all()
  first_mosaic = render_contact_sheet(first_batch)
  h, w = int(first_mosaic.shape[0]), int(first_mosaic.shape[1])

  writer = open_mp4_writer(mp4_path, width=w, height=h, fps=target_fps)

  total_frames = max(
      2, int(round(max(0.1, float(duration_s)) * float(target_fps)))
  )
  dt_s = 1.0 / max(1.0, float(target_fps))
  frames_written = 0

  try:
    with jsonl_path.open("w", encoding="utf-8") as log_f:
      for idx in range(total_frames):
        t0 = time.monotonic()
        batch: dict[str, CameraFrame] = (
            first_batch if idx == 0 else rig.capture_all()
        )
        mosaic_rgb = first_mosaic if idx == 0 else render_contact_sheet(batch)
        writer.write(np.ascontiguousarray(mosaic_rgb[:, :, ::-1]))
        frames_written += 1
        event = {
            "frame_index": idx,
            "captured_monotonic_s": round(t0, 6),
            "streams": {
                name: {
                    "timestamp_s": round(float(frm.timestamp_s), 6),
                    "brightness_mean": round(float(frm.brightness_mean), 2),
                }
                for name, frm in batch.items()
            },
        }
        log_f.write(json.dumps(event) + "\n")
        elapsed = time.monotonic() - t0
        if idx + 1 < total_frames and dt_s > elapsed:
          time.sleep(min(0.05, dt_s - elapsed))
  finally:
    writer.release()

  if not mp4_path.is_file() or mp4_path.stat().st_size == 0:
    raise RuntimeError(f"Failed to encode valid MP4 video at {mp4_path}")

  return {
      "mp4_path": str(mp4_path),
      "jsonl_path": str(jsonl_path),
      "frames_written": frames_written,
      "resolution": [w, h],
  }


def _render_fk_overlay(
    cfg: RobotConfig,
    snapshot: Mapping[str, CameraFrame],
    fk_file: Path,
    *,
    realtime: str,
) -> str | None:
  """Projects left/right pinch TCPs onto camera frames and writes mosaic."""
  try:
    left_q, right_q, fk_warns = read_arm_joints_or_rest(
        cfg, realtime_mode=realtime
    )
    for fw in fk_warns:
      print(f"Warning: {fw}", file=sys.stderr)
    mj_model, _ = fr3_mjcf.build_fr3_duo_mjcf(cfg)
    cal = fr3_kinematics.get_camera_calibration_opencv(
        mj_model, left_q, right_q
    )
    p_left, _ = fr3_kinematics.compute_arm_fk(mj_model, "left", left_q)
    p_right, _ = fr3_kinematics.compute_arm_fk(mj_model, "right", right_q)
    pinch_sites = {"L pinch": p_left, "R pinch": p_right}
    colors = {
        "L pinch": np.array([0, 255, 255], dtype=np.uint8),
        "R pinch": np.array([255, 128, 0], dtype=np.uint8),
    }

    overlay_snapshot: dict[str, CameraFrame] = {}
    for c_name in CANONICAL_CAMERAS:
      frm = snapshot.get(c_name)
      if frm is None:
        continue
      tile = frm.rgb.copy()
      h_tile, w_tile = tile.shape[:2]
      if c_name in cal:
        k_mat = np.array(cal[c_name]["K"], dtype=np.float64, copy=True)
        cal_w = int(cal[c_name].get("width", w_tile))
        cal_h = int(cal[c_name].get("height", h_tile))
        if cal_w > 0 and cal_h > 0 and (w_tile != cal_w or h_tile != cal_h):
          sx = float(w_tile) / float(cal_w)
          sy = float(h_tile) / float(cal_h)
          k_mat[0, 0] *= sx
          k_mat[0, 2] *= sx
          k_mat[1, 1] *= sy
          k_mat[1, 2] *= sy
        t_base_cam = cal[c_name]["T_base_cam"]
        for label, p_base in pinch_sites.items():
          (u, v), z_depth = fr3_kinematics.project_base_points_opencv(
              p_base, k_mat, t_base_cam
          )
          if z_depth > 0.04:
            ui, vi = int(round(u)), int(round(v))
            col = colors[label]
            r_c = 18
            if -r_c <= ui < w_tile + r_c and -r_c <= vi < h_tile + r_c:
              tile[
                  max(0, vi - 2) : min(h_tile, vi + 3),
                  max(0, ui - r_c) : min(w_tile, ui + r_c + 1),
              ] = col
              tile[
                  max(0, vi - r_c) : min(h_tile, vi + r_c + 1),
                  max(0, ui - 2) : min(w_tile, ui + 3),
              ] = col
      overlay_snapshot[c_name] = CameraFrame(
          name=frm.name,
          rgb=tile,
          timestamp_s=frm.timestamp_s,
          width=frm.width,
          height=frm.height,
          backend=frm.backend,
      )
    save_rgb_png(render_contact_sheet(overlay_snapshot), fk_file)
    return str(fk_file)
  except (RuntimeError, OSError, ValueError, KeyError) as exc:
    print(
        f"Warning: Failed to generate fk_projection_overlay_2x2.png: {exc}",
        file=sys.stderr,
    )
    return None


def run_camera_check(
    config: RobotConfig | Mapping[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
    output_mp4: Path | None = None,
    output_json: Path | None = None,
    duration_sec: float | None = None,
    fps: float = 15.0,
    max_age_sec: float = 0.30,
    num_reads: int = 3,
    save_references: bool = False,
    dry_run: bool = False,
    realtime: str = "auto",
    sim_robot: fr3_duo_robot.Fr3DuoRobot | None = None,
) -> tuple[int, dict[str, Any]]:
  """Runs Stage 2 4-camera freshness check, contact sheet, and MP4 recording."""
  cfg = normalize_robot_config(config)
  effective_dry_run = bool(dry_run)

  resolved_out_dir = resolve_stage_output_dir(
      "stage2_cameras", output_dir, dry_run=effective_dry_run
  )
  duration_s = (
      float(duration_sec)
      if duration_sec is not None
      else (0.25 if effective_dry_run else 2.0)
  )

  with Fr3DuoCameraController(
      specs=cfg.cameras,
      dry_run=effective_dry_run,
      sim_robot=sim_robot,
  ) as rig:
    freshness = rig.check_freshness(
        max_age_s=max_age_sec,
        num_reads=num_reads,
        interval_s=0.01 if effective_dry_run else 0.05,
    )
    snapshot = rig.capture_all()
    mosaic_rgb = render_contact_sheet(snapshot)

    contact_sheet_path: str | None = None
    fk_overlay_path: str | None = None
    recording_info: dict[str, Any] | None = None

    if resolved_out_dir is not None or output_mp4 is not None:
      if resolved_out_dir is not None:
        base_dir = resolved_out_dir
      else:
        assert output_mp4 is not None
        base_dir = output_mp4.parent
      sheet_file = base_dir / "contact_sheet_2x2.png"
      save_rgb_png(mosaic_rgb, sheet_file)
      contact_sheet_path = str(sheet_file)

      if not effective_dry_run:
        fk_overlay_path = _render_fk_overlay(
            cfg,
            snapshot,
            base_dir / "fk_projection_overlay_2x2.png",
            realtime=realtime,
        )

      if save_references:
        for cam_name in CANONICAL_CAMERAS:
          if cam_name in snapshot:
            save_rgb_png(
                snapshot[cam_name].rgb, base_dir / f"ref_{cam_name}.png"
            )

      mp4_file = output_mp4 or (base_dir / "cameras_recording.mp4")
      jsonl_file = base_dir / "camera_frames.jsonl"
      recording_info = _record_mosaic_mp4(
          rig,
          mp4_file,
          jsonl_file,
          duration_s=duration_s,
          target_fps=fps,
      )

  mode_label = "dry_run" if effective_dry_run else "hardware"
  report: dict[str, Any] = {
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "stage": "stage2_check_cameras",
      "mode": mode_label,
      "dry_run": effective_dry_run,
      "freshness": freshness,
      "contact_sheet_shape": list(mosaic_rgb.shape),
      "contact_sheet_path": contact_sheet_path,
      "fk_projection_overlay_path": fk_overlay_path,
      "recording": recording_info,
  }

  json_path = output_json or (
      resolved_out_dir / "camera_check_report.json"
      if resolved_out_dir is not None
      else None
  )
  if json_path is not None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(json_path)

  tag = "[DRY-RUN]" if effective_dry_run else "[HARDWARE]"
  print(
      f"{tag} Stage 2 Camera Check | ok={freshness['ok']} | "
      f"streams={sorted(freshness['cameras'].keys())} | "
      f"mosaic={list(mosaic_rgb.shape)}"
  )
  if contact_sheet_path is not None:
    print(f"  contact_sheet: {contact_sheet_path}")
  if fk_overlay_path is not None:
    print(f"  fk_overlay   : {fk_overlay_path}")
  if recording_info is not None and recording_info.get("mp4_path"):
    print(
        f"  recording_mp4: {recording_info['mp4_path']} "
        f"({recording_info.get('frames_written', 0)} frames)"
    )
  if json_path is not None:
    print(f"  report_json  : {json_path}")
  for err in freshness.get("errors", []):
    print(f"  ERROR: {err}", file=sys.stderr)

  return (0 if freshness["ok"] else 1), report
