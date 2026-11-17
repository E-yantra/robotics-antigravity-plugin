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
"""1- or 2-camera freshness check (R6), lighting/glare diagnostics, reference capture, and drift detection (R10)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_common import (  # noqa: E402
    get_configured_cameras,
    open_configured_camera,
    require_cv2,
)


def analyze_lighting(
    frame: np.ndarray, thresh_cfg: dict[str, Any], is_overhead: bool
) -> dict[str, Any]:
  """Evaluate global/ROI brightness, glare hotspots, RGB balance, and 4x4 spatial grid."""
  cv = require_cv2()
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  b_mean, g_mean, r_mean, _ = cv.mean(frame)
  glare_th = int(thresh_cfg.get("glare_threshold", 240))
  blown_th = int(thresh_cfg.get("blownout_threshold", 250))

  h, w = gray.shape
  roi_rect: tuple[int, int, int, int] = (0, 0, w, h)
  if is_overhead and "table_roi_norm" in thresh_cfg:
    x0f, y0f, x1f, y1f = thresh_cfg["table_roi_norm"]
    roi_rect = (int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h))

  rx0, ry0, rx1, ry1 = roi_rect
  eval_region = gray[ry0:ry1, rx0:rx1]
  glare_pct = (
      float(np.count_nonzero(eval_region > glare_th))
      * 100.0
      / float(max(eval_region.size, 1))
  )
  blown_pct = (
      float(np.count_nonzero(eval_region > blown_th))
      * 100.0
      / float(max(eval_region.size, 1))
  )

  patches_4x4: list[list[float]] = []
  for r in range(4):
    row_vals = []
    y0, y1 = int(r * h / 4), int((r + 1) * h / 4)
    for c in range(4):
      x0, x1 = int(c * w / 4), int((c + 1) * w / 4)
      row_vals.append(round(float(np.mean(gray[y0:y1, x0:x1])), 2))
    patches_4x4.append(row_vals)

  mean_bright = float(np.mean(gray))
  max_bright = int(np.max(gray))
  lap_var = float(cv.Laplacian(gray, cv.CV_64F).var())

  issues: list[str] = []
  if is_overhead:
    b_min = float(thresh_cfg.get("mean_brightness_min", 45.0))
    b_max = float(thresh_cfg.get("mean_brightness_max", 220.0))
    if mean_bright < b_min:
      issues.append(f"TOO_DARK (mean={mean_bright:.1f} < {b_min:.1f})")
    elif mean_bright > b_max:
      issues.append(f"TOO_BRIGHT (mean={mean_bright:.1f} > {b_max:.1f})")
    if glare_pct > float(thresh_cfg.get("max_table_glare_pct", 10.0)):
      issues.append(
          f"TABLE_GLARE ({glare_pct:.2f}% table pixels > {glare_th})"
      )
    if blown_pct > float(thresh_cfg.get("max_table_blownout_pct", 5.0)):
      issues.append(
          f"TABLE_BLOWN_OUT ({blown_pct:.2f}% table pixels > {blown_th})"
      )
  else:
    min_peak = int(thresh_cfg.get("min_peak_brightness", 80))
    if max_bright < min_peak:
      issues.append(
          f"LOW_DYNAMIC_RANGE (max={max_bright} < {min_peak})"
      )
    if glare_pct > float(thresh_cfg.get("max_glare_pct", 15.0)):
      issues.append(f"WRIST_GLARE ({glare_pct:.2f}% pixels > {glare_th})")
    if blown_pct > float(thresh_cfg.get("max_blownout_pct", 8.0)):
      issues.append(f"WRIST_BLOWN_OUT ({blown_pct:.2f}% pixels > {blown_th})")

  return {
      "resolution": [int(w), int(h)],
      "mean_brightness": round(mean_bright, 2),
      "median_brightness": round(float(np.median(gray)), 2),
      "p05_brightness": round(float(np.percentile(gray, 5)), 2),
      "p95_brightness": round(float(np.percentile(gray, 95)), 2),
      "max_brightness": max_bright,
      "eval_roi_rect": list(roi_rect),
      "glare_threshold": glare_th,
      "glare_pct": round(glare_pct, 3),
      "blownout_threshold": blown_th,
      "blownout_pct": round(blown_pct, 3),
      "rgb_means": {
          "R": round(r_mean, 2),
          "G": round(g_mean, 2),
          "B": round(b_mean, 2),
      },
      "sharpness_laplacian_var": round(lap_var, 2),
      "patches_4x4": patches_4x4,
      "lighting_ok": len(issues) == 0,
      "issues": issues,
  }


def save_diagnostic_image(
    frame: np.ndarray,
    out_path: Path,
    cam_label: str,
    stats: dict[str, Any],
) -> None:
  """Render ROI, glare masks, and lighting status summary onto a diagnostic PNG."""
  cv = require_cv2()
  vis = frame.copy()
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  glare_th = int(stats["glare_threshold"])
  rx0, ry0, rx1, ry1 = stats["eval_roi_rect"]
  if (rx0, ry0, rx1, ry1) != (0, 0, vis.shape[1], vis.shape[0]):
    cv.rectangle(vis, (rx0, ry0), (rx1, ry1), (255, 220, 40), 2)

  roi_gray = gray[ry0:ry1, rx0:rx1]
  glare_mask_full = np.zeros_like(gray, dtype=bool)
  glare_mask_full[ry0:ry1, rx0:rx1] = roi_gray > glare_th

  if np.any(glare_mask_full):
    highlight = np.zeros_like(vis)
    highlight[glare_mask_full] = (60, 0, 255)
    vis = cv.addWeighted(vis, 0.78, highlight, 0.45, 0.0)
    contours, _ = cv.findContours(
        glare_mask_full.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    cv.drawContours(vis, contours, -1, (0, 255, 255), 2)

  status_str = (
      "PASS"
      if stats["lighting_ok"]
      else "FIX_LIGHTING: " + ", ".join(stats["issues"])
  )
  rgb = stats["rgb_means"]
  scope_lbl = "TableROI" if cam_label in ("overhead", "fixed") else "FullFrame"
  lines = [
      f"{cam_label.upper()} DIAGNOSTIC | Mean={stats['mean_brightness']:.1f}/255 "
      f"(P05={stats['p05_brightness']:.0f}, P95={stats['p95_brightness']:.0f}, Max={stats['max_brightness']})",
      f"{scope_lbl} Glare(>{glare_th})={stats['glare_pct']:.2f}% | "
      f"Blown(>{stats['blownout_threshold']})={stats['blownout_pct']:.2f}% | "
      f"RGB=({rgb['R']:.0f},{rgb['G']:.0f},{rgb['B']:.0f})",
      f"Status: {status_str}",
  ]

  bar = vis.copy()
  cv.rectangle(bar, (0, 0), (vis.shape[1], 76), (10, 10, 10), -1)
  cv.addWeighted(bar, 0.72, vis, 0.28, 0.0, vis)
  scale = 0.52 if vis.shape[1] >= 1000 else 0.35
  color = (90, 255, 160) if stats["lighting_ok"] else (60, 180, 255)
  for idx, line in enumerate(lines):
    cv.putText(
        vis,
        line,
        (10, 22 + idx * 24),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv.LINE_AA,
    )

  out_path.parent.mkdir(parents=True, exist_ok=True)
  cv.imwrite(str(out_path), vis)


def check_scene_drift(
    current_ov_stats: dict[str, Any],
    ref_payload: dict[str, Any],
    drift_cfg: dict[str, Any],
) -> dict[str, Any]:
  """Compare current fixed/overhead frame statistics against locked reference_scene.json."""
  ref_ov = ref_payload["overhead_baseline"]
  bright_shift = abs(
      float(current_ov_stats["mean_brightness"])
      - float(ref_ov["mean_brightness"])
  )
  rgb_cur = current_ov_stats["rgb_means"]
  rgb_ref = ref_ov["rgb_means"]
  chan_shift = max(
      abs(float(rgb_cur[c]) - float(rgb_ref[c])) for c in ("R", "G", "B")
  )
  cur_grid = np.array(current_ov_stats["patches_4x4"], dtype=float)
  ref_grid = np.array(ref_ov["patches_4x4"], dtype=float)
  patch_mad = float(np.mean(np.abs(cur_grid - ref_grid)))

  reasons = []
  if bright_shift > float(drift_cfg["max_brightness_shift"]):
    reasons.append(
        f"brightness_shift={bright_shift:.2f} > {drift_cfg['max_brightness_shift']}"
    )
  if chan_shift > float(drift_cfg["max_channel_shift"]):
    reasons.append(
        f"channel_shift={chan_shift:.2f} > {drift_cfg['max_channel_shift']}"
    )
  if patch_mad > float(drift_cfg["max_patch_mad"]):
    reasons.append(
        f"patch_mad={patch_mad:.2f} > {drift_cfg['max_patch_mad']}"
    )
  return {
      "passed": len(reasons) == 0,
      "brightness_shift": round(bright_shift, 2),
      "max_channel_shift": round(chan_shift, 2),
      "patch_mad": round(patch_mad, 2),
      "reasons": reasons,
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description="1- or 2-camera pre-flight, lighting check, reference save, and drift check."
  )
  parser.add_argument(
      "--config", type=Path, default=Path("config/cameras_stage4.json")
  )
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument(
      "--save-references",
      "--save-reference",
      dest="save_references",
      action="store_true",
  )
  parser.add_argument("--check-drift", action="store_true")
  parser.add_argument("--output-json", type=Path, default=None)
  args = parser.parse_args()

  cv = require_cv2()
  cfg = json.loads(args.config.read_text(encoding="utf-8"))
  out_dir = Path(cfg.get("output_dir", "logs/stage4"))
  ref_cfg_path = Path(cfg.get("reference_config", "config/reference_scene.json"))
  pf_cfg = cfg.get(
      "preflight", {"max_frame_age_sec": 0.3, "num_checks": 3, "interval_sec": 0.1}
  )
  cams = get_configured_cameras(cfg)

  if args.dry_run:
    cam_desc = ", ".join(
        f"{role} ({c['width']}x{c['height']}@{c['fps']}fps)"
        for role, c in cams.items()
    )
    print(
        f"[DRY-RUN] Would verify configured camera(s) [{cam_desc}] across "
        f"{pf_cfg['num_checks']} freshness checks (max_age <= {pf_cfg['max_frame_age_sec']}s) "
        f"and evaluate lighting diagnostics in {out_dir}."
    )
    return

  caps: dict[str, Any] = {}
  max_age = float(pf_cfg["max_frame_age_sec"])

  try:
    for role, ccfg in cams.items():
      caps[role] = open_configured_camera(ccfg)

    last_frames: dict[str, np.ndarray] = {}
    for idx in range(1, int(pf_cfg["num_checks"]) + 1):
      for role, cap in caps.items():
        t0 = time.monotonic()
        ok, frm = cap.read()
        age = time.monotonic() - t0
        if not ok or frm is None or age > max_age:
          raise RuntimeError(
              f"PRE-FLIGHT FAIL ({role}) check {idx}: ok={ok}, age={age:.3f}s"
          )
        last_frames[role] = frm
      if idx < int(pf_cfg["num_checks"]):
        time.sleep(float(pf_cfg["interval_sec"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cameras_checked": list(cams.keys()),
    }
    thresh_all = cfg.get("lighting_thresholds", {})

    primary_stats: dict[str, Any] | None = None
    for role, frm in last_frames.items():
      is_ov = role in ("overhead", "fixed")
      th = thresh_all.get(role) or thresh_all.get(
          "overhead" if is_ov else "wrist", {}
      )
      stats = analyze_lighting(frm, th, is_overhead=is_ov)
      report[f"{role}_lighting"] = stats
      save_diagnostic_image(
          frm, out_dir / f"{role}_diagnostics.png", role, stats
      )
      if is_ov or primary_stats is None:
        primary_stats = stats

    if args.save_references and primary_stats is not None:
      ref_payload: dict[str, Any] = {
          "saved_utc": datetime.now(timezone.utc).isoformat(),
          "overhead_baseline": {
              "resolution": primary_stats["resolution"],
              "mean_brightness": primary_stats["mean_brightness"],
              "rgb_means": primary_stats["rgb_means"],
              "patches_4x4": primary_stats["patches_4x4"],
          },
      }
      for role, frm in last_frames.items():
        ref_path = out_dir / f"reference_{role}.png"
        cv.imwrite(str(ref_path), frm)
        ref_payload[f"reference_{role}_png"] = str(ref_path)
      ref_cfg_path.parent.mkdir(parents=True, exist_ok=True)
      ref_cfg_path.write_text(
          json.dumps(ref_payload, indent=2) + "\n", encoding="utf-8"
      )
      report["references_saved"] = True

    if args.check_drift and primary_stats is not None:
      ref_payload = json.loads(ref_cfg_path.read_text(encoding="utf-8"))
      drift_res = check_scene_drift(
          primary_stats,
          ref_payload,
          cfg.get(
              "drift_thresholds",
              {
                  "max_brightness_shift": 18.0,
                  "max_channel_shift": 22.0,
                  "max_patch_mad": 18.0,
              },
          ),
      )
      report["drift_check"] = drift_res
      if not drift_res["passed"]:
        raise RuntimeError(
            f"Scene lighting/camera drift detected: {drift_res['reasons']}"
        )

    out_json = args.output_json or (out_dir / "lighting_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
  finally:
    for cap in caps.values():
      cap.release()


if __name__ == "__main__":
  main()
