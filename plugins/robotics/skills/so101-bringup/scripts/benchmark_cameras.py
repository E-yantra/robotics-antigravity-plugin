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
"""Inspect USB bus topology/bandwidth and benchmark single- or dual-camera streaming FPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_common import (  # noqa: E402
    estimate_usb_bandwidth_mbps,
    get_camera_index,
    get_configured_cameras,
    inspect_usb_topology,
    require_cv2,
)


def benchmark_single_camera(
    role: str, cam_cfg: dict[str, Any], warmup: int, duration: float
) -> dict[str, Any]:
  """Stream a single configured camera for `duration` seconds and measure FPS & max frame gap."""
  cv = require_cv2()
  idx = get_camera_index(cam_cfg)
  cap = cv.VideoCapture(idx)
  if not cap.isOpened():
    cap.release()
    return {"role": role, "index": idx, "opened": False}

  try:
    fourcc = str(cam_cfg.get("fourcc", "MJPG"))
    w = int(cam_cfg.get("width", 640))
    h = int(cam_cfg.get("height", 480))
    fps = float(cam_cfg.get("fps", 30))
    if fourcc:
      cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*fourcc))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv.CAP_PROP_FPS, fps)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(warmup):
      cap.read()

    ok_count = fail_count = 0
    shape = None
    last_t = time.monotonic()
    max_gap = 0.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
      ok, frm = cap.read()
      t_now = time.monotonic()
      if ok and frm is not None:
        ok_count += 1
        shape = list(frm.shape)
        max_gap = max(max_gap, t_now - last_t)
        last_t = t_now
      else:
        fail_count += 1
    elapsed = max(time.monotonic() - t0, 1e-6)
    bw = estimate_usb_bandwidth_mbps(w, h, fps, fourcc)
    return {
        "role": role,
        "index": idx,
        "opened": True,
        "fourcc": fourcc,
        "requested": f"{w}x{h}@{fps:.0f}",
        "actual_shape": shape,
        "measured_fps": round(ok_count / elapsed, 2),
        "frames_ok": ok_count,
        "frames_fail": fail_count,
        "max_gap_sec": round(max_gap, 4),
        "bandwidth_estimate": bw,
    }
  finally:
    cap.release()


def benchmark_profile(
    wrist_idx: int, overhead_idx: int, prof: dict, warmup: int, duration: float
) -> dict:
  """Stream both cameras concurrently for `duration` seconds and measure FPS & inter-frame gaps."""
  cv = require_cv2()
  cap_wr = cv.VideoCapture(wrist_idx)
  cap_ov = cv.VideoCapture(overhead_idx)

  if not cap_wr.isOpened() or not cap_ov.isOpened():
    cap_wr.release()
    cap_ov.release()
    return {
        "name": prof["name"],
        "opened": False,
        "wrist_opened": cap_wr.isOpened(),
        "overhead_opened": cap_ov.isOpened(),
    }

  try:
    fourcc = prof.get("fourcc", "MJPG")
    for cap, w, h, fps in [
        (cap_wr, prof["wr_w"], prof["wr_h"], prof["wr_fps"]),
        (cap_ov, prof["ov_w"], prof["ov_h"], prof["ov_fps"]),
    ]:
      if fourcc:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*str(fourcc)))
      cap.set(cv.CAP_PROP_FRAME_WIDTH, int(w))
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(h))
      cap.set(cv.CAP_PROP_FPS, float(fps))
      cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(warmup):
      cap_wr.read()
      cap_ov.read()

    wr_ok = wr_fail = ov_ok = ov_fail = 0
    wr_shape = ov_shape = None
    last_wr_t = last_ov_t = time.monotonic()
    max_wr_gap = max_ov_gap = 0.0

    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
      ok0, f0 = cap_wr.read()
      t_now = time.monotonic()
      if ok0 and f0 is not None:
        wr_ok += 1
        wr_shape = list(f0.shape)
        max_wr_gap = max(max_wr_gap, t_now - last_wr_t)
        last_wr_t = t_now
      else:
        wr_fail += 1

      ok1, f1 = cap_ov.read()
      t_now = time.monotonic()
      if ok1 and f1 is not None:
        ov_ok += 1
        ov_shape = list(f1.shape)
        max_ov_gap = max(max_ov_gap, t_now - last_ov_t)
        last_ov_t = t_now
      else:
        ov_fail += 1

    elapsed = max(time.monotonic() - t0, 1e-6)
    ov_bw = estimate_usb_bandwidth_mbps(
        int(prof["ov_w"]), int(prof["ov_h"]), float(prof["ov_fps"]), str(fourcc)
    )
    wr_bw = estimate_usb_bandwidth_mbps(
        int(prof["wr_w"]), int(prof["wr_h"]), float(prof["wr_fps"]), str(fourcc)
    )
    result = {
        "name": prof["name"],
        "opened": True,
        "fourcc": fourcc,
        "elapsed_sec": round(elapsed, 3),
        "combined_active_mbps": round(
            ov_bw["active_estimated_mbps"] + wr_bw["active_estimated_mbps"], 2
        ),
        "combined_yuyv_uncompressed_mbps": round(
            ov_bw["yuyv_uncompressed_mbps"] + wr_bw["yuyv_uncompressed_mbps"], 2
        ),
        "overhead": {
            "requested": f"{prof['ov_w']}x{prof['ov_h']}@{prof['ov_fps']}",
            "actual_shape": ov_shape,
            "measured_fps": round(ov_ok / elapsed, 2),
            "frames_ok": ov_ok,
            "frames_fail": ov_fail,
            "max_gap_sec": round(max_ov_gap, 4),
            "bandwidth_estimate": ov_bw,
        },
        "wrist": {
            "requested": f"{prof['wr_w']}x{prof['wr_h']}@{prof['wr_fps']}",
            "actual_shape": wr_shape,
            "measured_fps": round(wr_ok / elapsed, 2),
            "frames_ok": wr_ok,
            "frames_fail": wr_fail,
            "max_gap_sec": round(max_wr_gap, 4),
            "bandwidth_estimate": wr_bw,
        },
    }
    print(
        f"[{prof['name']}] "
        f"OV {result['overhead']['requested']} -> {result['overhead']['measured_fps']:.1f} fps "
        f"(shape={ov_shape}, fail={ov_fail}, max_gap={max_ov_gap:.3f}s) | "
        f"WR {result['wrist']['requested']} -> {result['wrist']['measured_fps']:.1f} fps "
        f"(shape={wr_shape}, fail={wr_fail}, max_gap={max_wr_gap:.3f}s)"
    )
    return result
  finally:
    cap_wr.release()
    cap_ov.release()


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Inspect USB topology/bandwidth and benchmark single- or dual-camera streaming."
  )
  parser.add_argument("--config", type=Path, default=None)
  parser.add_argument("--wrist-index", type=int, default=0)
  parser.add_argument("--overhead-index", type=int, default=1)
  parser.add_argument("--duration-sec", type=float, default=4.0)
  parser.add_argument("--warmup-frames", type=int, default=10)
  parser.add_argument(
      "--output",
      "--output-json",
      dest="output",
      type=Path,
      default=Path("data/stage4/bandwidth_benchmark.json"),
  )
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  usb_topology = inspect_usb_topology()
  configured_cams: dict[str, dict[str, Any]] | None = None
  if args.config is not None and args.config.exists():
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if "cameras" in cfg:
      configured_cams = get_configured_cameras(cfg)

  if args.dry_run:
    if configured_cams and len(configured_cams) == 1:
      role, ccfg = next(iter(configured_cams.items()))
      bw = estimate_usb_bandwidth_mbps(
          int(ccfg.get("width", 640)),
          int(ccfg.get("height", 480)),
          float(ccfg.get("fps", 30)),
          str(ccfg.get("fourcc", "MJPG")),
      )
      print(
          f"[DRY-RUN] Minimal 1-camera setup ('{role}'): estimated active={bw['active_estimated_mbps']} Mbps "
          f"(YUYV={bw['yuyv_uncompressed_mbps']} Mbps) vs USB 2.0 480 Mbps theoretical / 320 Mbps usable."
      )
    else:
      print(
          f"[DRY-RUN] Would inspect USB topology ({usb_topology['command']}) and benchmark "
          f"MJPG camera streaming for {args.duration_sec}s."
      )
    return

  if configured_cams and len(configured_cams) == 1:
    role, ccfg = next(iter(configured_cams.items()))
    results = [
        benchmark_single_camera(
            role=role,
            cam_cfg=ccfg,
            warmup=args.warmup_frames,
            duration=args.duration_sec,
        )
    ]
  else:
    wrist_idx = args.wrist_index
    overhead_idx = args.overhead_index
    if configured_cams:
      if "wrist" in configured_cams:
        wrist_idx = get_camera_index(configured_cams["wrist"])
      for ov_key in ("overhead", "fixed"):
        if ov_key in configured_cams:
          overhead_idx = get_camera_index(configured_cams[ov_key])
          break
    profiles = [
        {
            "name": "mjpg_hd_overhead_vga_wrist",
            "fourcc": "MJPG",
            "ov_w": 1280,
            "ov_h": 720,
            "ov_fps": 30,
            "wr_w": 640,
            "wr_h": 480,
            "wr_fps": 30,
        },
    ]
    results = [
        benchmark_profile(
            wrist_idx=wrist_idx,
            overhead_idx=overhead_idx,
            prof=prof,
            warmup=args.warmup_frames,
            duration=args.duration_sec,
        )
        for prof in profiles
    ]

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps({"usb_topology": usb_topology, "results": results}, indent=2)
      + "\n",
      encoding="utf-8",
  )
  print(f"Saved benchmark results -> {args.output}")


if __name__ == "__main__":
  main()
