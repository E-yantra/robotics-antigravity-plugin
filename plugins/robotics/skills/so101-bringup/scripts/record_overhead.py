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
"""Timestamped camera MP4 recorder with JSON-lines event logging (supports 1 or 2 configured cameras)."""

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


def burn_timestamp_overlay(
    frame: np.ndarray,
    stage_name: str,
    frame_idx: int,
    elapsed_sec: float,
    total_sec: float,
    written_count: int,
    skipped_count: int,
) -> np.ndarray:
  """Render top banner with local/UTC timestamp and frame counters onto a frame."""
  cv = require_cv2()
  out = frame.copy()
  now_utc = datetime.now(timezone.utc)
  now_local = datetime.now().astimezone()
  clock_str = (
      f"{now_local.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {now_local.tzname()} | "
      f"UTC {now_utc.strftime('%H:%M:%S.%f')[:-3]}"
  )
  line1 = f"{stage_name} | {clock_str}"
  line2 = (
      f"t={elapsed_sec:05.2f}s / {total_sec:04.1f}s | frame #{frame_idx:04d} | "
      f"written={written_count} skipped={skipped_count}"
  )

  bar = out.copy()
  cv.rectangle(bar, (0, 0), (out.shape[1], 56), (8, 8, 8), -1)
  cv.addWeighted(bar, 0.70, out, 0.30, 0.0, out)
  cv.putText(
      out,
      line1,
      (12, 22),
      cv.FONT_HERSHEY_SIMPLEX,
      0.56,
      (90, 255, 160),
      1,
      cv.LINE_AA,
  )
  cv.putText(
      out,
      line2,
      (12, 46),
      cv.FONT_HERSHEY_SIMPLEX,
      0.52,
      (240, 240, 240),
      1,
      cv.LINE_AA,
  )
  return out


def record_session(
    cfg: dict[str, Any],
    duration_sec: float,
    stage_name: str,
    mp4_path: Path,
    jsonl_path: Path,
    sample_png_path: Path | None = None,
    preferred_role: str | None = None,
) -> dict[str, Any]:
  """Run pre-flight freshness checks across all configured cameras and record timestamped MP4 + JSONL log."""
  cv = require_cv2()
  cams = get_configured_cameras(cfg)
  pf_cfg = cfg.get(
      "preflight", {"max_frame_age_sec": 0.3, "num_checks": 3, "interval_sec": 0.1}
  )
  rec_cfg = cfg.get("recording", {})

  rec_role = preferred_role
  if rec_role is None or rec_role not in cams:
    for candidate in ("overhead", "fixed", "wrist"):
      if candidate in cams:
        rec_role = candidate
        break
    if rec_role is None:
      rec_role = next(iter(cams))

  rec_cam_cfg = cams[rec_role]
  max_age = float(pf_cfg["max_frame_age_sec"])
  target_fps = float(rec_cfg.get("target_fps", rec_cam_cfg.get("fps", 30)))

  caps: dict[str, Any] = {}
  for role, ccfg in cams.items():
    caps[role] = open_configured_camera(ccfg)

  mp4_path.parent.mkdir(parents=True, exist_ok=True)
  jsonl_path.parent.mkdir(parents=True, exist_ok=True)

  fourcc = cv.VideoWriter_fourcc(*"mp4v")
  writer = cv.VideoWriter(
      str(mp4_path),
      fourcc,
      target_fps,
      (int(rec_cam_cfg["width"]), int(rec_cam_cfg["height"])),
  )
  if not writer.isOpened():
    for cap in caps.values():
      cap.release()
    raise RuntimeError(f"Failed to open VideoWriter for {mp4_path}")

  frames_written = 0
  frames_skipped = 0
  saved_sample = False

  try:
    with jsonl_path.open("w", encoding="utf-8") as jf:
      for chk_idx in range(1, int(pf_cfg["num_checks"]) + 1):
        ages: dict[str, float] = {}
        passed = True
        for role, cap in caps.items():
          t0 = time.monotonic()
          ok, frm = cap.read()
          age = time.monotonic() - t0
          ages[f"{role}_age_sec"] = round(age, 4)
          if not ok or frm is None or age > max_age:
            passed = False

        jf.write(
            json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "preflight_check",
                "check": chk_idx,
                **ages,
                "passed": passed,
            })
            + "\n"
        )
        jf.flush()
        if not passed:
          raise RuntimeError(f"Pre-flight failed on check {chk_idx}: {ages}")
        if chk_idx < int(pf_cfg["num_checks"]):
          time.sleep(float(pf_cfg["interval_sec"]))

      rec_cap = caps[rec_role]
      t_start = time.monotonic()
      attempt_idx = 0
      while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= duration_sec:
          break

        attempt_idx += 1
        t_read = time.monotonic()
        ok, frame = rec_cap.read()
        read_age = time.monotonic() - t_read

        if not ok or frame is None or read_age > max_age:
          frames_skipped += 1
          jf.write(
              json.dumps({
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "event": "frame_skipped",
                  "attempt": attempt_idx,
                  "elapsed_sec": round(elapsed, 4),
                  "read_age_sec": round(read_age, 4),
              })
              + "\n"
          )
          continue

        frames_written += 1
        annotated = burn_timestamp_overlay(
            frame=frame,
            stage_name=stage_name,
            frame_idx=frames_written,
            elapsed_sec=elapsed,
            total_sec=duration_sec,
            written_count=frames_written,
            skipped_count=frames_skipped,
        )
        writer.write(annotated)

        if (
            sample_png_path is not None
            and not saved_sample
            and elapsed >= duration_sec * 0.5
        ):
          sample_png_path.parent.mkdir(parents=True, exist_ok=True)
          cv.imwrite(str(sample_png_path), annotated)
          saved_sample = True

      actual_duration = time.monotonic() - t_start
      summary = {
          "timestamp_utc": datetime.now(timezone.utc).isoformat(),
          "stage_name": stage_name,
          "recorded_role": rec_role,
          "mp4_path": str(mp4_path),
          "jsonl_path": str(jsonl_path),
          "duration_requested_sec": duration_sec,
          "duration_actual_sec": round(actual_duration, 3),
          "frames_written": frames_written,
          "frames_skipped_failed_or_stale": frames_skipped,
          "effective_written_fps": round(
              frames_written / max(actual_duration, 1e-6), 2
          ),
      }
      jf.write(
          json.dumps({"event": "recording_complete", **summary}) + "\n"
      )
      jf.flush()
      return summary
  finally:
    writer.release()
    for cap in caps.values():
      cap.release()


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Record timestamped camera MP4 + JSONL event log."
  )
  parser.add_argument(
      "--config", type=Path, default=Path("config/cameras_stage4.json")
  )
  parser.add_argument("--role", type=str, default=None)
  parser.add_argument("--duration-sec", type=float, default=None)
  parser.add_argument("--output-mp4", type=Path, default=None)
  parser.add_argument("--output-jsonl", type=Path, default=None)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  cfg = json.loads(args.config.read_text(encoding="utf-8"))
  rec = cfg.get("recording", {})
  out_dir = Path(cfg.get("output_dir", "logs/stage4"))
  duration_sec = float(args.duration_sec or rec.get("duration_sec", 10.0))
  target_fps = float(rec.get("target_fps", 30.0))
  mp4_path = args.output_mp4 or (
      out_dir / str(rec.get("mp4_filename", "stage4_camera_10s.mp4"))
  )
  jsonl_path = args.output_jsonl or (
      out_dir / str(rec.get("jsonl_filename", "stage4_camera_10s.jsonl"))
  )
  sample_png = (
      out_dir / str(rec["mp4_sample_frame"])
      if rec.get("mp4_sample_frame")
      else None
  )

  if args.dry_run:
    print(
        f"[DRY-RUN] Would verify pre-flight freshness and record "
        f"{duration_sec}s @ {target_fps}fps to {mp4_path} and {jsonl_path}."
    )
    return

  summary = record_session(
      cfg=cfg,
      duration_sec=duration_sec,
      stage_name=str(cfg.get("stage_name", "STAGE 4")),
      mp4_path=mp4_path,
      jsonl_path=jsonl_path,
      sample_png_path=sample_png,
      preferred_role=args.role,
  )
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
