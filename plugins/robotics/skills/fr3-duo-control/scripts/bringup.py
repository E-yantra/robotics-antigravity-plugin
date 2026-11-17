#!/usr/bin/env python3
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
#
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#     "mujoco",
#     "numpy",
#     "opencv-python",
#     "pylibfranka; sys_platform == 'linux' and platform_machine == 'x86_64'",
#     "pyserial",
# ]
# ///

"""Interactive terminal bring-up CLI runner for Franka Research 3 Duo."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import check_cameras
import check_robot
import discover_hardware
from fr3_config import (
    DEFAULT_CONFIG_PATH,
    RobotConfig,
    load_robot_config,
    normalize_robot_config,
    resolve_stage_output_dir,
    save_robot_config,
    validate_robot_config_complete,
)
from fr3_mjcf import (
    build_fr3_duo_mjcf,
    find_menagerie_dir,
)

STAGE_ORDER: tuple[str, ...] = (
    "env",
    "preflight",
    "geometry",
    "cameras",
    "gripper",
    "nudge",
    "abort-test",
    "hello-world",
    "all",
)
STEP_CHOICES: tuple[str, ...] = (
    "env",
    "preflight",
    "geometry",
    "cameras",
    "gripper",
    "nudge",
    "abort-test",
    "hello-world",
    "home",
)

_GEOMETRY_CLI_ALLOW_ZERO: dict[str, bool] = {
    "base_height_m": False,
    "table_height_m": True,
    "table_front_offset_m": True,
    "table_width_m": False,
    "table_depth_m": False,
    "tcp_offset_z_m": False,
}
_COMPLETE_CONFIG_REQUIRED_STEPS: frozenset[str] = frozenset({
    "gripper",
    "nudge",
    "abort-test",
    "hello-world",
})

_ACTUATION_DESCRIPTIONS: dict[str, str] = {
    "gripper": (
        "Step 3a: Robotiq 2F-85 Gripper Activation & Cycling"
        " (open -> close -> open)"
    ),
    "nudge": (
        "Step 3b: Single-Joint J1 Nudge (5 deg slow stroke on left & right"
        " arms)"
    ),
    "abort-test": (
        "Step 3c: Abort-and-Unwind Test (10 deg stroke interrupted to test"
        " Hermite C^2 stop & rewind)"
    ),
    "hello-world": (
        "Step 4: Bimanual Hello World (synchronized wave + gripper cycle +"
        " 4-camera video)"
    ),
    "home": "Step 4h: Smooth Bimanual Homing to Canonical Rest Pose",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments for the FR3 Duo bring-up runner."""
  parser = argparse.ArgumentParser(
      description=(
          "Interactive terminal bring-up runner for Franka Research 3 Duo"
          " across Stages 0-4 (env check, preflight discovery, geometry"
          " calibration, camera verification, staged actuation, and homing)."
      )
  )
  parser.add_argument(
      "--config",
      type=Path,
      default=Path(DEFAULT_CONFIG_PATH),
      help="Path to robot configuration JSON.",
  )
  parser.add_argument(
      "--no-write-config",
      action="store_true",
      help="If set, runs preflight without saving robot_config.json.",
  )
  parser.add_argument(
      "--read-only",
      action="store_true",
      help=(
          "Run in read-only mode without writing config or commanding"
          " physical actuation."
      ),
  )
  parser.add_argument(
      "--left-ip", type=str, default=None, help="Left arm FCI IP."
  )
  parser.add_argument(
      "--right-ip", type=str, default=None, help="Right arm FCI IP."
  )
  parser.add_argument(
      "--base-height-m",
      type=float,
      default=None,
      help="Floor to flat center bracket underside height in meters.",
  )
  parser.add_argument(
      "--table-height-m",
      type=float,
      default=None,
      help="Floor to tabletop height in meters.",
  )
  parser.add_argument(
      "--table-front-offset-m",
      type=float,
      default=None,
      help=(
          "Horizontal gap from bracket underside front lip to table front [m]."
      ),
  )
  parser.add_argument(
      "--table-width-m",
      type=float,
      default=None,
      help="Tabletop width along Y in meters.",
  )
  parser.add_argument(
      "--table-depth-m",
      type=float,
      default=None,
      help="Tabletop depth along X in meters.",
  )
  parser.add_argument(
      "--tcp-offset-z-m",
      type=float,
      default=None,
      help="Robotiq fingertip Z offset from flange in meters (0.145 or 0.185).",
  )
  parser.add_argument(
      "--confirm-geometry",
      action="store_true",
      help="Mark physical cell geometry & fingertip measurements confirmed.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=None,
      help="Base directory for bring-up step outputs and summary JSON.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help=(
          "Verify MJCF kinematics and trajectory limits without opening live"
          " hardware or commanding motion."
      ),
  )
  parser.add_argument(
      "--realtime",
      choices=("auto", "enforce", "ignore"),
      default="auto",
      help="FCI realtime scheduling policy.",
  )
  target_group = parser.add_mutually_exclusive_group()
  target_group.add_argument(
      "--up-to",
      choices=STAGE_ORDER,
      default="all",
      help="Run progressively up to and including this stage.",
  )
  target_group.add_argument(
      "--step",
      choices=STEP_CHOICES,
      default=None,
      help="Run only this single bring-up step (or 'home').",
  )
  parser.add_argument(
      "--yes",
      "-y",
      action="store_true",
      help="Auto-confirm interactive actuation prompts without prompting.",
  )
  return parser.parse_args(argv)


def _write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
  try:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "bringup_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
  except OSError as exc:
    print(
        f"Warning: Failed to write bringup_summary.json: {exc}",
        file=sys.stderr,
    )


def _call_stage(
    fn: Callable[..., tuple[int, dict[str, Any]]], *args: Any, **kwargs: Any
) -> tuple[int, dict[str, Any], str | None]:
  """Calls a stage `run_*` function and traps unexpected runtime errors."""
  try:
    rc, report = fn(*args, **kwargs)
    return int(rc), report, None
  except Exception as exc:  # pylint: disable=broad-exception-caught
    err_msg = f"{type(exc).__name__}: {exc}"
    print(f"  ERROR: {err_msg}", file=sys.stderr)
    return 1, {}, err_msg


def _fmt_opt(val: Any) -> Any:
  return "N/A" if val is None else val


def _fail_step(
    out_dir: Path,
    summary: dict[str, Any],
    step_key: str,
    rc: int,
    error: str | None = None,
) -> int:
  summary["status"] = "failed"
  info: dict[str, Any] = {"status": "failed", "exit_code": rc}
  if error:
    info["error"] = error
  summary["steps"][step_key] = info
  _write_summary(out_dir, summary)
  return rc


def _resolve_steps_to_run(args: argparse.Namespace) -> list[str]:
  if args.step is not None:
    return [str(args.step)]
  up_to = str(args.up_to or "all")
  progressive = (
      "env",
      "preflight",
      "geometry",
      "cameras",
      "gripper",
      "nudge",
      "abort-test",
      "hello-world",
  )
  if up_to == "all":
    return list(progressive)
  idx = progressive.index(up_to)
  return list(progressive[: idx + 1])


def _run_env_step(
    cfg: RobotConfig,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    is_hw: bool,
    mode_label: str,
) -> int:
  """Executes Step 0 module, MJCF, and dry-run trajectory pre-checks."""
  print("\n--- Step 0: Environment & Pre-Check ---")
  optional_notes = {
      "serial": "pyserial hardware driver; only required for live hardware",
      "pylibfranka": "Franka FCI binding; only required for live hardware",
  }
  missing_required: list[str] = []
  for mod_name, is_req in (
      ("numpy", True),
      ("mujoco", True),
      ("cv2", True),
      ("serial", is_hw),
      ("pylibfranka", is_hw),
  ):
    try:
      mod = importlib.import_module(mod_name)
      ver = getattr(mod, "__version__", "")
      print(
          f"  [PASS] Python module '{mod_name}'" + (f" ({ver})" if ver else "")
      )
    except ImportError:
      if is_req:
        print(
            f"  [FAIL] Python module '{mod_name}' missing (required for"
            f" mode={mode_label})"
        )
        missing_required.append(mod_name)
      else:
        print(
            f"  [WARN] Python module '{mod_name}' missing"
            f" ({optional_notes.get(mod_name, 'optional dependency')})"
        )

  menagerie_path = find_menagerie_dir()
  if missing_required:
    print(
        f"\n[FAIL] Step 0 environment check failed: missing {missing_required}",
        file=sys.stderr,
    )
    summary["status"] = "failed"
    summary["steps"]["env"] = {
        "status": "failed",
        "missing_modules": missing_required,
    }
    _write_summary(out_dir, summary)
    return 1

  step0_dir = out_dir / "step0_precheck"
  step0_dir.mkdir(parents=True, exist_ok=True)
  mjcf_err: str | None = None
  try:
    build_fr3_duo_mjcf(cfg)
    if menagerie_path is None:
      menagerie_path = find_menagerie_dir()
    if menagerie_path is not None:
      print(f"  [PASS] Canonical Duo MJCF ready (Menagerie: {menagerie_path})")
    else:
      print("  [PASS] Canonical Duo MJCF ready (Menagerie)")
  except (RuntimeError, OSError, ValueError) as exc:
    mjcf_err = f"{type(exc).__name__}: {exc}"
    print(f"  ERROR: MJCF compile check failed: {mjcf_err}", file=sys.stderr)

  dry_check_rc, _, dry_check_err = _call_stage(
      check_robot.run_robot_check,
      cfg,
      mode="nudge",
      dry_run=True,
      output_dir=step0_dir,
      output_json=step0_dir / "dry_run_check.json",
  )
  if mjcf_err is not None or dry_check_rc != 0:
    print(
        "\n[FAIL] Step 0 MuJoCo & trajectory pre-check failed.", file=sys.stderr
    )
    summary["status"] = "failed"
    step0_info: dict[str, Any] = {
        "status": "failed",
        "mjcf_ok": mjcf_err is None,
        "dry_check_rc": dry_check_rc,
    }
    if mjcf_err or dry_check_err:
      step0_info["error"] = mjcf_err or dry_check_err
    summary["steps"]["env"] = step0_info
    _write_summary(out_dir, summary)
    return 1

  print(
      "  [PASS] MuJoCo & dry-run pre-check succeeded"
      " (kinematics & safety gates ok)"
  )
  summary["steps"]["env"] = {"status": "ok", "menagerie": bool(menagerie_path)}
  return 0


def _run_preflight_step(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    effective_dry_run: bool,
    is_read_only: bool,
) -> tuple[int, RobotConfig]:
  """Runs Step 1 read-only hardware discovery and updates `cfg` if saved."""
  print("\n--- Step 1: Hardware Discovery & Preflight Audit ---")
  step1_dir = out_dir / "step1_preflight"
  step1_dir.mkdir(parents=True, exist_ok=True)
  s1_json = step1_dir / "discovery_report.json"
  write_cfg_in_place = bool(
      not effective_dry_run and not args.no_write_config and not is_read_only
  )
  rc1, d_rep, err1 = _call_stage(
      discover_hardware.run_discovery,
      cfg,
      realtime=args.realtime,
      output_dir=step1_dir,
      output_json=s1_json,
      write_config=args.config if write_cfg_in_place else None,
      read_only=not (write_cfg_in_place and args.config is not None),
      strict_preflight=not (effective_dry_run or is_read_only),
      yes=args.yes,
  )
  if rc1 != 0:
    print(
        f"\n[FAIL] Step 1 preflight blocked (exit code {rc1}) — fix warnings"
        " above before continuing.",
        file=sys.stderr,
    )
    return _fail_step(out_dir, summary, "preflight", rc1, err1), cfg

  if write_cfg_in_place and args.config is not None and args.config.is_file():
    cfg = load_robot_config(args.config)
  elif d_rep and isinstance(d_rep.get("discovered_config"), Mapping):
    cfg = normalize_robot_config(d_rep["discovered_config"])
  elif d_rep:
    grip_sel = d_rep.get("gripper_selection") or {}
    for side in ("left", "right"):
      if not cfg.arms[side].gripper_port and grip_sel.get(side):
        cfg.arms[side].gripper_port = str(grip_sel[side])
    uvc_devs = d_rep.get("uvc_devices") or []
    if uvc_devs:
      discover_hardware._auto_bind_devices(cfg, grip_sel, list(uvc_devs))
  if args.left_ip:
    cfg.arms.left.ip = str(args.left_ip)
  if args.right_ip:
    cfg.arms.right.ip = str(args.right_ip)

  if d_rep:
    if d_rep.get("report_path"):
      summary["artifacts"]["discovery_report"] = d_rep["report_path"]
    elif s1_json.is_file():
      summary["artifacts"]["discovery_report"] = str(s1_json)
    rt_cfg = (d_rep.get("realtime_profile") or {}).get(
        "realtime_config", "unknown"
    )
    is_rt = d_rep.get("preempt_rt_detected", False)
    print(f"  Realtime Profile: {rt_cfg} (PREEMPT_RT={is_rt})")
    arm_ro = d_rep.get("arm_readout") or {}
    fci_targets = d_rep.get("fci_targets") or {}
    for arm in ("left", "right"):
      rd = arm_ro.get(arm, {})
      da = rd.get("desk_audit") or {}
      reach = (fci_targets.get(arm) or {}).get("tcp_1337_reachable")
      fk_err_mm = float(rd.get("fk_pos_error_m", 0.0)) * 1000.0
      f_norm = float(rd.get("free_air_force_norm_n", 0.0))
      print(
          f"  {arm.capitalize()} Arm: tcp_1337={reach} |"
          f" mode={_fmt_opt(da.get('robot_mode'))} |"
          f" m_total={_fmt_opt(da.get('m_total_kg'))} kg |"
          f" grav_err={_fmt_opt(da.get('gravity_angle_err_deg'))} deg |"
          f" fk_err={fk_err_mm:.2f} mm | free_air={f_norm:.2f} N"
      )
    grippers = d_rep.get("gripper_selection") or {}
    print(
        f"  Robotiq Grippers: left={grippers.get('left')} |"
        f" right={grippers.get('right')}"
    )
    print(
        f"  UVC Cameras: {len(d_rep.get('uvc_devices') or [])} device(s) found"
    )
    notes = [
        *list(d_rep.get("warnings") or []),
        *list(d_rep.get("advisories") or []),
    ]
    if notes:
      print(f"  Advisories/Warnings ({len(notes)}):")
      for note in notes:
        print(f"    - {note}")

  summary["steps"]["preflight"] = {"status": "ok", "exit_code": 0}
  return 0, cfg


_GEOMETRY_PROMPTS: tuple[tuple[str, str, bool], ...] = (
    (
        "base_height_m",
        "  1) base_height_m (floor -> flat center bracket underside) [m]"
        " (default {val:.4f}): ",
        False,
    ),
    (
        "table_height_m",
        "  2) table_height_m (floor -> tabletop, ALOHA 0.750) [m]"
        " (default {val:.3f}): ",
        True,
    ),
    (
        "table_front_offset_m",
        "  3) table_front_offset_m (bracket front lip -> table front edge)"
        " [m] (default {val:.3f}): ",
        True,
    ),
    (
        "table_width_m",
        "  4a) table_width_m (ALOHA 1.220) [m] (default {val:.3f}): ",
        False,
    ),
    (
        "table_depth_m",
        "  4b) table_depth_m (ALOHA 0.740) [m] (default {val:.3f}): ",
        False,
    ),
)


def _geometry_reference_image_path() -> Path:
  """Resolves `assets/fr3_duo_dimensions_reference.png` relative to skill."""
  rel = Path("assets") / "fr3_duo_dimensions_reference.png"
  for base in (os.path.abspath(__file__), Path(__file__).resolve()):
    candidate = Path(base).parents[1] / rel
    if candidate.is_file():
      return candidate
  return Path(__file__).resolve().parents[1] / rel


def _validate_geom_value(
    name: str, val: float, *, allow_zero: bool = False
) -> float:
  if not math.isfinite(val):
    raise ValueError(f"{name} must be finite, got {val!r}")
  if (allow_zero and val < 0.0) or (not allow_zero and val <= 0.0):
    op = ">=" if allow_zero else ">"
    raise ValueError(f"{name} must be {op} 0, got {val!r}")
  return float(val)


def _extract_geometry_cli_updates(
    args: argparse.Namespace,
) -> tuple[dict[str, float], str | None]:
  """Extracts and validates geometry CLI overrides from parsed `args`."""
  updates: dict[str, float] = {}
  for field_name, allow_zero in _GEOMETRY_CLI_ALLOW_ZERO.items():
    raw_val = getattr(args, field_name, None)
    if raw_val is None:
      continue
    try:
      updates[field_name] = _validate_geom_value(
          field_name, float(raw_val), allow_zero=allow_zero
      )
    except ValueError as exc:
      return {}, str(exc)
  return updates, None


def _print_geometry_summary(cfg: RobotConfig, ref_img: Path) -> None:
  """Prints current physical geometry, derived `z_table_m`, and table box."""
  table_center, table_half_extents = cfg.table_box_bounds()
  print(f"  Reference diagram : {ref_img}")
  print(
      f"  base_height_m     : {cfg.base_height_m:.6f} m"
      " (floor -> flat center bracket underside)"
  )
  print(f"  table_height_m    : {cfg.table_height_m:.6f} m (floor -> tabletop)")
  print(
      f"  table_front_offset: {cfg.table_front_offset_m:.4f} m"
      " (bracket underside front lip -> table front edge)"
  )
  print(
      f"  table_width/depth : {cfg.table_width_m:.3f} m x"
      f" {cfg.table_depth_m:.3f} m"
  )
  print(f"  tcp_offset_z_m    : {cfg.tcp_offset_z_m:.3f} m")
  print(f"  Derived z_table_m : {cfg.z_table_m:+.6f} m")
  print(
      f"  Table box bounds  : center={table_center},"
      f" half_extents={table_half_extents}"
  )
  print(f"  geometry_confirmed: {bool(cfg.geometry_confirmed)}")


def _prompt_geometry_calibration(
    cfg: RobotConfig,
    ref_img: Path,
    input_fn: Callable[[str], str],
) -> tuple[str, RobotConfig, str | None]:
  """Interactively prompts for physical cell geometry & fingertip offset."""
  print(
      "  Measure your floor-mounted setup per reference diagram:\n"
      f"    {ref_img}\n"
      "  Press ENTER to accept the default value in parentheses."
  )
  try:
    updates: dict[str, Any] = {"geometry_confirmed": True}
    for field_name, tmpl, allow_zero in _GEOMETRY_PROMPTS:
      default_val = float(getattr(cfg, field_name))
      raw = input_fn(tmpl.format(val=default_val)).strip()
      updates[field_name] = (
          _validate_geom_value(field_name, float(raw), allow_zero=allow_zero)
          if raw
          else default_val
      )
    tcp_raw = (
        input_fn(
            "  5) Fingertips: [1/stock] 2F-85 pads (0.145) | [2/extended]"
            f" ALOHA (0.185) | float [m] (default {cfg.tcp_offset_z_m:.3f}): "
        )
        .strip()
        .lower()
    )
    if not tcp_raw:
      updates["tcp_offset_z_m"] = float(cfg.tcp_offset_z_m)
    elif tcp_raw in ("1", "stock", "s", "pads", "2f85", "2f-85"):
      updates["tcp_offset_z_m"] = 0.145
    elif tcp_raw in ("2", "extended", "ext", "aloha"):
      updates["tcp_offset_z_m"] = 0.185
    else:
      updates["tcp_offset_z_m"] = _validate_geom_value(
          "tcp_offset_z_m", float(tcp_raw)
      )

    candidate = normalize_robot_config({**cfg.to_dict(), **updates})
    table_center, table_half_extents = candidate.table_box_bounds()
    print(f"  -> Computed z_table_m : {candidate.z_table_m:+.6f} m")
    print(
        f"  -> Table box bounds   : center={table_center},"
        f" half_extents={table_half_extents}"
    )
    print(f"  -> tcp_offset_z_m     : {candidate.tcp_offset_z_m:.3f} m")
    confirm = (
        input_fn("  Confirm geometry & fingertip measurements? [y/N]: ")
        .strip()
        .lower()
    )
  except (EOFError, KeyboardInterrupt):
    return "stopped", cfg, "geometry_prompt_aborted"
  except ValueError as exc:
    return "invalid", cfg, str(exc)

  if confirm in ("y", "yes"):
    return "confirmed", candidate, None
  return "stopped", cfg, "geometry_confirmation_declined"


def _run_geometry_step(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    effective_dry_run: bool,
    is_read_only: bool,
    input_fn: Callable[[str], str],
    cli_updates: Mapping[str, float] | None = None,
) -> tuple[int, RobotConfig]:
  """Runs Step 1b physical cell geometry & fingertip calibration."""
  print("\n--- Step 1b: Physical Cell Geometry & Fingertip Calibration ---")
  ref_img = _geometry_reference_image_path()
  if cli_updates is None:
    updates, cli_err = _extract_geometry_cli_updates(args)
    if cli_err is not None:
      print(f"\n[FAIL] Invalid geometry CLI flag: {cli_err}", file=sys.stderr)
      return _fail_step(out_dir, summary, "geometry", 2, cli_err), cfg
  else:
    updates = cli_updates

  confirmed_via_cli = bool(args.confirm_geometry) or (
      len(updates) == len(_GEOMETRY_CLI_ALLOW_ZERO)
  )
  write_cfg = (
      not effective_dry_run and not args.no_write_config and not is_read_only
  )

  if updates or confirmed_via_cli:
    cfg = normalize_robot_config({
        **cfg.to_dict(),
        **updates,
        "geometry_confirmed": bool(cfg.geometry_confirmed) or confirmed_via_cli,
    })

  if (
      confirmed_via_cli
      or effective_dry_run
      or (cfg.geometry_confirmed and args.step != "geometry")
      or is_read_only
      or args.no_write_config
      or args.yes
      or (input_fn is input and not sys.stdin.isatty())
  ):
    _print_geometry_summary(cfg, ref_img)
    step_entry: dict[str, Any] = {
        "status": "ok",
        "exit_code": 0,
        "geometry_confirmed": bool(cfg.geometry_confirmed),
        "z_table_m": cfg.z_table_m,
    }
    if (updates or confirmed_via_cli) and write_cfg and args.config is not None:
      save_robot_config(cfg, args.config)
      if cfg.geometry_confirmed:
        print(f"  Saved confirmed geometry to {args.config}")
      else:
        print(
            f"  Saved partial geometry updates (unconfirmed) to {args.config}"
        )
    elif effective_dry_run:
      missing = validate_robot_config_complete(cfg)
      print(
          "  Dry-run completeness preview missing:"
          f" {missing or 'NONE (complete)'}"
      )
      if not confirmed_via_cli:
        step_entry["missing_for_live"] = missing
    elif cfg.geometry_confirmed:
      print("  [PASS] Cell geometry & fingertip dimensions already confirmed.")
    elif is_read_only or args.no_write_config:
      print(
          "  [WARN] geometry_confirmed is False (skipping interactive prompt in"
          " read-only/--no-write-config mode). Run '--step geometry' or pass"
          " explicit measurement flags + '--confirm-geometry' before live"
          " actuation."
      )
      step_entry["read_only"] = is_read_only
    elif args.yes:
      print(
          "  [WARN] geometry_confirmed is False; --yes will NOT auto-confirm"
          " unverified default dimensions. Run '--step geometry' interactively"
          " or pass explicit measurement flags + '--confirm-geometry'."
      )
      step_entry["warning"] = "geometry_unconfirmed_with_yes"
    else:
      print(
          "  [WARN] geometry_confirmed is False and stdin is not interactive."
          " Run '--step geometry' in an interactive TTY or pass explicit"
          " measurement flags + '--confirm-geometry'."
      )
      step_entry["warning"] = "geometry_unconfirmed_non_tty"
    summary["steps"]["geometry"] = step_entry
    return 0, cfg

  outcome, new_cfg, prompt_err = _prompt_geometry_calibration(
      cfg, ref_img, input_fn
  )
  if outcome == "invalid":
    print(
        f"\n[FAIL] Invalid geometry input ({prompt_err}); geometry remains"
        " unconfirmed.",
        file=sys.stderr,
    )
    return _fail_step(
        out_dir, summary, "geometry", 2, f"invalid_geometry_input: {prompt_err}"
    ), cfg
  if outcome == "stopped":
    print("Geometry calibration stopped; geometry remains unconfirmed.")
    summary["status"] = "stopped_by_operator"
    summary["steps"]["geometry"] = {
        "status": "stopped_by_operator",
        "geometry_confirmed": False,
    }
    _write_summary(out_dir, summary)
    return 0, cfg

  cfg = new_cfg
  if write_cfg and args.config is not None:
    save_robot_config(cfg, args.config)
    print(f"  Saved confirmed geometry to {args.config}")
  summary["steps"]["geometry"] = {
      "status": "ok",
      "exit_code": 0,
      "geometry_confirmed": True,
      "z_table_m": cfg.z_table_m,
  }
  return 0, cfg


def _run_cameras_step(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    effective_dry_run: bool,
) -> int:
  """Runs Step 2 4-camera freshness check, contact sheet, and MP4 recording."""
  print("\n--- Step 2: Camera Verification & Contact Sheet ---")
  step2_dir = out_dir / "step2_cameras"
  step2_dir.mkdir(parents=True, exist_ok=True)
  rc2, c_rep, err2 = _call_stage(
      check_cameras.run_camera_check,
      cfg,
      realtime=args.realtime,
      dry_run=effective_dry_run,
      output_dir=step2_dir,
      output_json=step2_dir / "camera_check_report.json",
  )
  if rc2 != 0:
    print(
        f"\n[FAIL] Step 2 camera check failed (exit code {rc2}).",
        file=sys.stderr,
    )
    return _fail_step(out_dir, summary, "cameras", rc2, err2)

  for key, art_key, label in (
      ("contact_sheet_path", "contact_sheet_2x2", "Contact Sheet"),
      ("fk_projection_overlay_path", "fk_projection_overlay_2x2", "FK Overlay"),
  ):
    if c_rep.get(key):
      summary["artifacts"][art_key] = c_rep[key]
      print(f"  {label:<14}: {c_rep[key]}")
  mp4_path = (c_rep.get("recording") or {}).get("mp4_path")
  if mp4_path:
    summary["artifacts"]["cameras_recording_mp4"] = mp4_path
    print(f"  MP4 Recording : {mp4_path}")

  summary["steps"]["cameras"] = {"status": "ok", "exit_code": 0}
  return 0


def _confirm_actuation(
    step_key: str,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    will_actuate: bool,
    input_fn: Callable[[str], str],
) -> tuple[str, int]:
  """Handles interactive `[E-STOP READY]` prompt before live actuation."""
  if not will_actuate:
    return "run", 0

  if args.yes:
    if step_key == "abort-test" and sys.stdin.isatty():
      print(
          "  NOTE: Type 'q' + ENTER (or press Ctrl+C) while the left arm"
          " slowly nudges (10 deg @ 4 deg/s) to trigger smooth C^2"
          " abort-and-unwind."
      )
    return "run", 0

  if input_fn is input and not sys.stdin.isatty():
    print(
        f"[FAIL] Physical actuation for '{step_key}' requires an"
        " interactive TTY or --yes.",
        file=sys.stderr,
    )
    return "fail", _fail_step(
        out_dir, summary, step_key, 2, "non_interactive_without_yes"
    )

  print(f"[E-STOP READY] Operator confirmation required for '{step_key}':")
  if step_key == "abort-test":
    print(
        "  NOTE: Type 'q' + ENTER (or press Ctrl+C) while the left arm"
        " slowly nudges (10 deg @ 4 deg/s) to trigger smooth C^2"
        " abort-and-unwind."
    )

  try:
    ans = (
        input_fn("Proceed with actuation? [y = run / s = skip / N,q = quit]: ")
        .strip()
        .lower()
    )
  except (EOFError, KeyboardInterrupt):
    ans = "q"

  if ans in ("s", "skip"):
    print(f"Skipping '{step_key}'.")
    summary["steps"][step_key] = {"status": "skipped"}
    return "skip", 0
  if ans not in ("y", "yes"):
    msg = (
        f"Bring-up stopped by operator before '{step_key}'."
        if ans in ("q", "quit", "n", "no", "")
        else f"Unrecognized response {ans!r}; stopping bring-up."
    )
    print(msg)
    summary["status"] = "stopped_by_operator"
    summary["steps"][step_key] = {"status": "stopped_by_operator"}
    _write_summary(out_dir, summary)
    return "stop", 0
  return "run", 0


def _validate_actuation_result(
    step_key: str,
    sub_rep: Mapping[str, Any],
    sub_dir: Path,
    out_dir: Path,
    summary: dict[str, Any],
) -> tuple[str, int]:
  """Checks post-step success/abort invariants for live actuation steps."""
  if step_key == "gripper":
    bad_arms = [
        arm_name
        for arm_name, arm_res in (sub_rep.get("results") or {}).items()
        if isinstance(arm_res, dict)
        and not bool(
            (arm_res.get("gripper_cycle") or {}).get("stroke_ok", True)
        )
    ]
    if bad_arms:
      print(
          f"\n[FAIL] Step 3a gripper stroke check failed on {bad_arms}.",
          file=sys.stderr,
      )
      return "fail", _fail_step(
          out_dir, summary, step_key, 1, "gripper_stroke_failed"
      )
    return "ok", 0

  if step_key in ("nudge", "home"):
    if any(
        isinstance(arm_res, dict)
        and bool((arm_res.get(step_key) or {}).get("aborted", False))
        for arm_res in (sub_rep.get("results") or {}).values()
    ):
      print(f"Bring-up stopped by operator abort during '{step_key}'.")
      summary["status"] = "stopped_by_operator"
      summary["steps"][step_key] = {
          "status": "stopped_by_operator",
          "aborted": True,
      }
      _write_summary(out_dir, summary)
      return "stop", 0
    return "ok", 0

  if step_key == "abort-test":
    abort_info = next(
        (
            arm_res.get("abort_test")
            for arm_res in (sub_rep.get("results") or {}).values()
            if isinstance(arm_res, dict)
            and isinstance(arm_res.get("abort_test"), dict)
        ),
        None,
    )
    if abort_info is not None:
      aborted_ok = bool(abort_info.get("aborted", False))
      unwound_ok = bool(abort_info.get("unwound_to_start", False))
      grip_ok = bool(abort_info.get("gripper_preserved", True))
      if not (aborted_ok and unwound_ok and grip_ok):
        print(
            "\n[FAIL] Step 3c abort-test did not verify smooth"
            f" abort-and-unwind (aborted={aborted_ok},"
            f" unwound_to_start={unwound_ok},"
            f" gripper_preserved={grip_ok}). Re-run and type 'q' +"
            " ENTER while the left arm moves.",
            file=sys.stderr,
        )
        return "fail", _fail_step(
            out_dir, summary, step_key, 1, "abort_unwind_not_verified"
        )
    return "ok", 0

  if step_key == "hello-world":
    mp4_cand = sub_rep.get("mp4_path") or str(
        sub_dir / "stage4_hello_world.mp4"
    )
    if Path(mp4_cand).is_file():
      summary["artifacts"]["stage4_hello_world_mp4"] = str(mp4_cand)
      print(f"  Stage 4 MP4 Recording: {mp4_cand}")
    if bool(sub_rep.get("aborted", False)):
      print("Bring-up stopped by operator abort during 'hello-world'.")
      summary["status"] = "stopped_by_operator"
      summary["steps"][step_key] = {
          "status": "stopped_by_operator",
          "aborted": True,
      }
      _write_summary(out_dir, summary)
      return "stop", 0
  return "ok", 0


def main(
    argv: Sequence[str] | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
  """Executes the progressive bring-up pipeline or a single bring-up step."""
  args = parse_args(argv)
  effective_dry_run = bool(args.dry_run)
  is_read_only = bool(args.read_only)
  mode_label = (
      "dry_run"
      if effective_dry_run
      else ("read_only" if is_read_only else "hardware")
  )

  resolved_out_dir = resolve_stage_output_dir(
      "bringup", args.output_dir, dry_run=effective_dry_run
  )
  out_dir = (
      resolved_out_dir
      if resolved_out_dir is not None
      else Path(tempfile.mkdtemp(prefix="fr3_bringup_"))
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  steps_to_run = _resolve_steps_to_run(args)
  target_desc = (
      f"step={args.step}" if args.step is not None else f"up_to={args.up_to}"
  )
  summary: dict[str, Any] = {
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "mode": mode_label,
      "up_to": None if args.step is not None else args.up_to,
      "step": args.step,
      "read_only": is_read_only,
      "status": "in_progress",
      "steps": {},
      "artifacts": {},
  }

  print(
      f"=== Franka Research 3 Duo Bring-Up (mode={mode_label},"
      f" {target_desc}) ==="
  )
  print(f"Output directory: {out_dir}")

  try:
    cfg = load_robot_config(args.config)
    if args.left_ip:
      cfg.arms.left.ip = str(args.left_ip)
    if args.right_ip:
      cfg.arms.right.ip = str(args.right_ip)
  except (OSError, ValueError, TypeError) as exc:
    err_msg = f"{type(exc).__name__}: {exc}"
    print(f"\n[FAIL] Invalid robot configuration: {err_msg}", file=sys.stderr)
    return _fail_step(out_dir, summary, steps_to_run[0], 1, err_msg)

  cli_geom_updates, cli_geom_err = _extract_geometry_cli_updates(args)
  if cli_geom_err is not None:
    print(
        f"\n[FAIL] Invalid geometry CLI flag: {cli_geom_err}", file=sys.stderr
    )
    err_step = "geometry" if "geometry" in steps_to_run else steps_to_run[0]
    return _fail_step(out_dir, summary, err_step, 2, cli_geom_err)
  if cli_geom_updates:
    cfg = normalize_robot_config({**cfg.to_dict(), **cli_geom_updates})

  if "env" in steps_to_run:
    rc0 = _run_env_step(
        cfg,
        out_dir,
        summary,
        is_hw=not effective_dry_run,
        mode_label=mode_label,
    )
    if rc0 != 0:
      return rc0

  if "preflight" in steps_to_run:
    rc1, cfg = _run_preflight_step(
        cfg,
        args,
        out_dir,
        summary,
        effective_dry_run=effective_dry_run,
        is_read_only=is_read_only,
    )
    if rc1 != 0:
      return rc1

  if "geometry" in steps_to_run:
    rc_geom, cfg = _run_geometry_step(
        cfg,
        args,
        out_dir,
        summary,
        effective_dry_run=effective_dry_run,
        is_read_only=is_read_only,
        input_fn=input_fn,
        cli_updates=cli_geom_updates,
    )
    if rc_geom != 0 or summary.get("status") == "stopped_by_operator":
      return rc_geom
  elif cli_geom_updates or args.confirm_geometry:
    confirmed_via_cli = bool(args.confirm_geometry) or (
        len(cli_geom_updates) == len(_GEOMETRY_CLI_ALLOW_ZERO)
    )
    cfg = normalize_robot_config({
        **cfg.to_dict(),
        **cli_geom_updates,
        "geometry_confirmed": bool(cfg.geometry_confirmed) or confirmed_via_cli,
    })
    if (
        not effective_dry_run
        and not args.no_write_config
        and not is_read_only
        and args.config is not None
    ):
      save_robot_config(cfg, args.config)

  if "cameras" in steps_to_run:
    rc2 = _run_cameras_step(
        cfg,
        args,
        out_dir,
        summary,
        effective_dry_run=effective_dry_run,
    )
    if rc2 != 0:
      return rc2

  for step_key in steps_to_run:
    desc = _ACTUATION_DESCRIPTIONS.get(step_key)
    if desc is None:
      continue
    print(f"\n--- {desc} ---")

    will_actuate = not effective_dry_run and not is_read_only
    if will_actuate and step_key in _COMPLETE_CONFIG_REQUIRED_STEPS:
      missing_fields = validate_robot_config_complete(cfg)
      if missing_fields:
        missing_str = ", ".join(missing_fields)
        print(
            f"\n[FAIL] Cannot run live actuation '{step_key}': incomplete"
            f" robot configuration. Missing/unconfirmed: {missing_str}."
            " Run '--step preflight' and '--step geometry' (or pass"
            " measurement flags + '--confirm-geometry') first.",
            file=sys.stderr,
        )
        return _fail_step(
            out_dir,
            summary,
            step_key,
            2,
            f"incomplete_config: {missing_str}",
        )

    action, prompt_rc = _confirm_actuation(
        step_key,
        args,
        out_dir,
        summary,
        will_actuate=will_actuate,
        input_fn=input_fn,
    )
    if action == "skip":
      continue
    if action in ("fail", "stop"):
      return prompt_rc

    sub_dir = out_dir / f"step3_{step_key}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    rc, sub_rep, sub_err = _call_stage(
        check_robot.run_robot_check,
        cfg,
        mode=step_key,
        read_only=is_read_only,
        dry_run=effective_dry_run,
        realtime=args.realtime,
        output_dir=sub_dir,
    )
    if rc != 0:
      print(
          f"\n[FAIL] {desc} returned non-zero exit code {rc}.", file=sys.stderr
      )
      return _fail_step(out_dir, summary, step_key, rc, sub_err)

    if will_actuate:
      outcome, out_rc = _validate_actuation_result(
          step_key, sub_rep, sub_dir, out_dir, summary
      )
      if outcome in ("fail", "stop"):
        return out_rc

    step_entry: dict[str, Any] = {"status": "ok", "exit_code": 0}
    if is_read_only:
      step_entry["read_only"] = True
    summary["steps"][step_key] = step_entry

  summary["status"] = "ok"
  _write_summary(out_dir, summary)

  print("\n=======================================================")
  print("  FR3 DUO BRING-UP COMPLETED SUCCESSFULLY!")
  print(f"  Mode     : {mode_label}")
  print(f"  Target   : {target_desc}")
  print(f"  Summary  : {out_dir / 'bringup_summary.json'}")
  for art_name, art_p in summary.get("artifacts", {}).items():
    print(f"  Artifact : [{art_name}] -> {art_p}")
  print("=======================================================")
  return 0


if __name__ == "__main__":
  sys.exit(main())
