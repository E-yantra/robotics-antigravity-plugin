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
"""Two-cue detector (R9) for a unique SO-101 gripper marker (blue sticker or custom HSV Post-it triangle + surround contrast)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_common import open_configured_camera, require_cv2  # noqa: E402


def detect_gripper_marker(
    frame_bgr: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any]:
  """Detect a unique marker on the gripper after the wrist using two independent cues (R9).

  Cue 1: Chromatic BGR excess ((B - G) >= min_b_minus_g AND (G - R) >= min_g_minus_r AND B >= min_b)
         OR custom HSV range (hsv_lower .. hsv_upper) for a Post-it / colored triangle marker,
         with 3x3 morphological dilation plus compact area & solidity checks.
  Cue 2: Concentric annulus r in [inner_radius_px, outer_radius_px] around the candidate centroid
         verifying matte-white mount fraction (white_surround_fraction >= min_white_fraction)
         OR strong luminance contrast against the surrounding gripper body.
  """
  cv = require_cv2()
  cfg = cfg.get("marker_detect", cfg)
  c1 = cfg["cue1_blue_triangle"]
  c2 = cfg["cue2_white_mount_surround"]

  hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
  if "hsv_lower" in c1 and "hsv_upper" in c1:
    lower = np.array(c1["hsv_lower"], dtype=np.uint8)
    upper = np.array(c1["hsv_upper"], dtype=np.uint8)
    color_mask = cv.inRange(hsv, lower, upper)
  else:
    b_ch = frame_bgr[:, :, 0].astype(np.int16)
    g_ch = frame_bgr[:, :, 1].astype(np.int16)
    r_ch = frame_bgr[:, :, 2].astype(np.int16)
    color_mask = (
        ((b_ch - g_ch) >= int(c1["min_b_minus_g"]))
        & ((g_ch - r_ch) >= int(c1["min_g_minus_r"]))
        & (b_ch >= int(c1["min_b"]))
    ).astype(np.uint8) * 255

  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
  color_mask = cv.dilate(color_mask, kernel, iterations=1)

  contours, _ = cv.findContours(
      color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
  )
  h_img, w_img = frame_bgr.shape[:2]

  white_mask = (hsv[:, :, 2] >= int(c2.get("white_v_min", 130))) & (
      hsv[:, :, 1] <= int(c2.get("white_s_max", 75))
  )
  gray_img = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
  r_in = int(c2["inner_radius_px"])
  r_out = int(c2["outer_radius_px"])
  min_white_frac = float(c2.get("min_white_fraction", 0.35))
  min_contrast_delta = float(c2.get("min_contrast_delta", 999.0))

  cue1_candidates: list[dict[str, Any]] = []
  accepted: list[dict[str, Any]] = []

  yy, xx = np.ogrid[:h_img, :w_img]
  for cnt in contours:
    area = float(cv.contourArea(cnt))
    if area < float(c1["min_area_px"]) or area > float(c1["max_area_px"]):
      continue
    hull = cv.convexHull(cnt)
    hull_area = float(cv.contourArea(hull))
    solidity = area / max(hull_area, 1e-6)
    if solidity < float(c1["min_solidity"]):
      continue

    moments = cv.moments(cnt)
    if moments["m00"] <= 1e-6:
      continue
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    x, y, bw, bh = cv.boundingRect(cnt)

    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    annulus = (dist_sq >= r_in**2) & (dist_sq <= r_out**2)
    inner_disk = dist_sq < r_in**2
    annulus_px = int(np.count_nonzero(annulus))
    white_frac = float(np.count_nonzero(white_mask & annulus)) / max(
        annulus_px, 1
    )
    inner_mean = float(np.mean(gray_img[inner_disk])) if np.any(inner_disk) else 0.0
    outer_mean = float(np.mean(gray_img[annulus])) if annulus_px > 0 else 0.0
    contrast_delta = abs(inner_mean - outer_mean)
    cue2_ok = (white_frac >= min_white_frac) or (
        contrast_delta >= min_contrast_delta
    )

    cand = {
        "centroid_px": [round(cx, 2), round(cy, 2)],
        "bbox_px": [int(x), int(y), int(bw), int(bh)],
        "area_px": round(area, 1),
        "solidity": round(solidity, 3),
        "white_surround_fraction": round(white_frac, 3),
        "surround_contrast_delta": round(contrast_delta, 2),
        "cue2_passed": cue2_ok,
    }
    cue1_candidates.append(cand)
    if cand["cue2_passed"]:
      accepted.append(cand)

  return {
      "detected": len(accepted) == 1,
      "num_cue1_candidates": len(cue1_candidates),
      "num_accepted": len(accepted),
      "accepted": accepted,
      "all_candidates": cue1_candidates,
      "reason": (
          "OK: unique marker confirmed by both Cue 1 (marker colour/solidity) and Cue 2 (mount surround contrast)"
          if len(accepted) == 1
          else (
              "NONE: no candidate passed both Cue 1 (marker colour/solidity) and Cue 2 (mount surround contrast)"
              if len(accepted) == 0
              else f"AMBIGUOUS: {len(accepted)} markers passed both cues"
          )
      ),
  }


def annotate_marker_result(
    frame_bgr: np.ndarray,
    res: dict[str, Any],
    cfg: dict[str, Any],
    out_path: Path,
) -> None:
  """Save an annotated verification PNG highlighting accepted markers and annuli."""
  cv = require_cv2()
  cfg = cfg.get("marker_detect", cfg)
  vis = frame_bgr.copy()
  c2 = cfg["cue2_white_mount_surround"]
  r_in = int(c2["inner_radius_px"])
  r_out = int(c2["outer_radius_px"])

  for cand in res["accepted"]:
    cx, cy = int(round(cand["centroid_px"][0])), int(
        round(cand["centroid_px"][1])
    )
    x, y, bw, bh = cand["bbox_px"]
    ok = cand["cue2_passed"] and res["detected"]
    color = (80, 255, 140) if ok else (0, 140, 255)
    cv.circle(vis, (cx, cy), r_in, (255, 220, 60), 1)
    cv.circle(vis, (cx, cy), r_out, (255, 220, 60), 1)
    cv.rectangle(vis, (x - 4, y - 4), (x + bw + 4, y + bh + 4), color, 2)
    cv.drawMarker(vis, (cx, cy), (0, 0, 255), cv.MARKER_CROSS, 16, 2)
    label = (
        f"Gripper Marker ({cand['centroid_px'][0]:.1f}, {cand['centroid_px'][1]:.1f}) "
        f"| area={cand['area_px']:.0f}px | white={cand['white_surround_fraction']*100:.0f}%"
    )
    cv.putText(
        vis,
        label,
        (cx + 18, cy - 12),
        cv.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv.LINE_AA,
    )

  bar = vis.copy()
  cv.rectangle(bar, (0, 0), (vis.shape[1], 54), (10, 10, 10), -1)
  cv.addWeighted(bar, 0.72, vis, 0.28, 0.0, vis)
  status_color = (90, 255, 160) if res["detected"] else (60, 140, 255)
  cv.putText(
      vis,
      "Two-Cue Gripper Marker Verification "
      f"(Cue1Rejected={res['num_cue1_candidates'] - res['num_accepted']}, Accepted={res['num_accepted']})",
      (12, 22),
      cv.FONT_HERSHEY_SIMPLEX,
      0.54,
      (240, 240, 240),
      1,
      cv.LINE_AA,
  )
  cv.putText(
      vis,
      f"Status: {res['reason']}",
      (12, 46),
      cv.FONT_HERSHEY_SIMPLEX,
      0.50,
      status_color,
      1,
      cv.LINE_AA,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  cv.imwrite(str(out_path), vis)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Verify unique SO-101 gripper marker via two independent cues (R9)."
  )
  parser.add_argument(
      "--config", type=Path, default=Path("config/marker_detect.json")
  )
  parser.add_argument(
      "--image",
      type=Path,
      default=None,
      help="Optional image file; if omitted, captures one fresh frame from fixed/overhead camera.",
  )
  parser.add_argument(
      "--overhead-index",
      type=int,
      default=1,
      help="OpenCV index for live fixed/overhead capture when --image is omitted.",
  )
  parser.add_argument(
      "--output-image",
      type=Path,
      default=Path("data/stage5/marker_verification.png"),
  )
  parser.add_argument(
      "--output-json",
      type=Path,
      default=Path("data/stage5/marker_verification.json"),
  )
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  cfg = json.loads(args.config.read_text(encoding="utf-8"))
  if args.dry_run:
    src_desc = (
        str(args.image)
        if args.image is not None
        else f"live fixed/overhead camera (index={args.overhead_index}, 1280x720@30)"
    )
    print(
        f"[DRY-RUN] Would run two-cue marker detector on {src_desc} and save "
        f"annotated output to {args.output_image}."
    )
    return

  cv = require_cv2()
  if args.image is not None:
    frame = cv.imread(str(args.image))
    if frame is None:
      raise RuntimeError(f"Could not read image {args.image}")
  else:
    cap = open_configured_camera({
        "opencv_index": args.overhead_index,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "fourcc": "MJPG",
    })
    try:
      ok, frame = cap.read()
      if not ok or frame is None:
        raise RuntimeError(
            f"Failed to read live frame from fixed/overhead camera index {args.overhead_index}"
        )
    finally:
      cap.release()

  res = detect_gripper_marker(frame, cfg)
  annotate_marker_result(frame, res, cfg, args.output_image)
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(
      json.dumps(res, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
