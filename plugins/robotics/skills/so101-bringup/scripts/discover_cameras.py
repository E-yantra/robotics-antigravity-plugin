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
"""Read-only OS and OpenCV camera enumeration, role guidance, still capture, and lighting check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_common import compute_lighting_stats, require_cv2  # noqa: E402


def list_os_cameras() -> dict[str, str]:
  """Query OS USB/camera utilities without altering device state."""
  info: dict[str, str] = {"system": platform.system()}
  sys_name = platform.system()
  if sys_name == "Darwin":
    try:
      sp = subprocess.run(
          ["system_profiler", "SPCameraDataType"],
          capture_output=True,
          text=True,
          timeout=10,
          check=False,
      )
      info["system_profiler"] = sp.stdout.strip()
    except Exception as exc:
      info["system_profiler_error"] = str(exc)

    try:
      ff = subprocess.run(
          ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
          capture_output=True,
          text=True,
          timeout=10,
          check=False,
      )
      info["ffmpeg_avfoundation"] = (ff.stderr or ff.stdout).strip()
    except Exception as exc:
      info["ffmpeg_avfoundation_error"] = str(exc)
  else:
    try:
      v4l = subprocess.run(
          ["v4l2-ctl", "--list-devices"],
          capture_output=True,
          text=True,
          timeout=10,
          check=False,
      )
      info["v4l2_ctl"] = v4l.stdout.strip()
    except Exception as exc:
      info["v4l2_ctl_error"] = str(exc)
  return info


def scan_opencv_cameras(
    max_index: int, warmup_frames: int, output_dir: Path
) -> list[dict]:
  """Probe OpenCV indices 0..max_index, save a still per active camera, and evaluate lighting (0-255)."""
  cv = require_cv2()
  output_dir.mkdir(parents=True, exist_ok=True)
  discovered: list[dict] = []

  for idx in range(max_index + 1):
    cap = cv.VideoCapture(idx)
    if not cap.isOpened():
      cap.release()
      continue

    try:
      backend = cap.getBackendName()
      width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
      fps = float(cap.get(cv.CAP_PROP_FPS))

      frame = None
      for _ in range(max(1, warmup_frames)):
        ok, frm = cap.read()
        if ok and frm is not None:
          frame = frm
        time.sleep(0.03)

      if frame is None:
        print(f"[index {idx}] Opened ({backend}), but failed to read a frame.")
        continue

      actual_h, actual_w = frame.shape[:2]
      lighting = compute_lighting_stats(frame)
      out_path = output_dir / f"camera_{idx}.png"
      cv.imwrite(str(out_path), frame)

      entry = {
          "index": idx,
          "backend": backend,
          "reported_width": width,
          "reported_height": height,
          "actual_width": actual_w,
          "actual_height": actual_h,
          "reported_fps": fps,
          "mean_brightness": lighting["mean_brightness"],
          "lighting": lighting,
          "image_path": str(out_path),
      }
      discovered.append(entry)
      print(
          f"[index {idx}] {actual_w}x{actual_h} @ {fps:.1f} fps ({backend}) | "
          f"mean_brightness={lighting['mean_brightness']:.1f}/255 ({lighting['status']}) | "
          f"saved -> {out_path}"
      )
    finally:
      cap.release()

  return discovered


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Discover OS and OpenCV cameras and check lighting (read-only)."
  )
  parser.add_argument(
      "--config",
      type=Path,
      default=None,
      help="Optional path to JSON config file.",
  )
  parser.add_argument("--max-index", type=int, default=8)
  parser.add_argument("--warmup-frames", type=int, default=10)
  parser.add_argument("--output-dir", type=Path, default=Path("data/stage1"))
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print planned actions without opening cameras.",
  )
  args = parser.parse_args()

  max_index = args.max_index
  warmup_frames = args.warmup_frames
  output_dir = args.output_dir
  if args.config is not None and args.config.exists():
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    max_index = int(cfg.get("max_index", max_index))
    warmup_frames = int(cfg.get("warmup_frames", warmup_frames))
    output_dir = Path(cfg.get("output_dir", output_dir))

  if args.dry_run:
    print(
        f"[DRY-RUN] Would query OS camera devices, scan OpenCV indices 0..{max_index} "
        f"({warmup_frames} warmup frames each), check lighting (0-255), and save stills to {output_dir}/."
    )
    return

  print("=== Camera Setup & Role Guidance ===")
  print(
      "- Minimal setup (Stages 0-3 through Hello World): 1 camera (e.g., 'wrist' or 'fixed').\n"
      "- Recommended setup (Stages 0-5 + Phase 2): 'fixed' (overhead) + 'wrist' camera.\n"
      "- Laptop built-in camera note: Not recommended; if used as 'fixed', the laptop lid must "
      "face the arm and stay completely stationary."
  )

  print("\n=== OS Camera Enumeration ===")
  os_info = list_os_cameras()
  for k, v in os_info.items():
    if v:
      print(f"--- {k} ---\n{v}")

  print("\n=== OpenCV Camera Scan & Lighting Check ===")
  discovered = scan_opencv_cameras(
      max_index=max_index, warmup_frames=warmup_frames, output_dir=output_dir
  )
  summary_path = output_dir / "cameras_discovered.json"
  summary_path.write_text(
      json.dumps({"os_info": os_info, "opencv_cameras": discovered}, indent=2)
      + "\n",
      encoding="utf-8",
  )
  print(
      f"\nSaved discovery report -> {summary_path} ({len(discovered)} cameras captured)."
  )


if __name__ == "__main__":
  main()
