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

"""Unified hardware/simulation backends and `Fr3DuoRobot` robot controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import threading
import time
from typing import Any
import warnings
import weakref

from fr3_arm_controller import (
    ArmWorkerProcess,
    _recovery_result,
    _state_has_errors,
    audit_arm_desk_config,
    compute_ext_wrench_base_6d,
    compute_fk_pos_error_m,
    compute_link0_in_base_transform_for_model,
    disconnect_pylibfranka_robot,
    read_pylibfranka_state,
    recover_pylibfranka_errors,
)
import fr3_camera_controller
from fr3_config import (
    RobotConfig,
    normalize_robot_config,
    resolve_realtime_config,
    validate_robot_config_complete,
)
import fr3_kinematics
import fr3_mjcf
from fr3_safety import (
    AbortWatcher,
    Fr3DuoSafetyChecker,
    clamp_q_to_padded_limits,
    generate_quintic_trajectory,
    simplify_joint_polyline,
    unpack_o_t_ee,
)
import numpy as np
import robotiq_gripper_controller

try:
  import mujoco  # pytype: disable=import-error
except ImportError:
  mujoco: Any = None

__all__ = [
    "Fr3DuoRobot",
    "Fr3DuoRealBackend",
    "Fr3DuoSimBackend",
]


def _build_bimanual_unwind_trajs(
    left_waypoints: Sequence[Sequence[float]],
    right_waypoints: Sequence[Sequence[float]],
    max_vel: float,
    max_acc: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Builds smooth quintic unwind trajectories from recorded path samples."""
  unwind_l = generate_quintic_trajectory(
      simplify_joint_polyline(left_waypoints),
      dt_s=0.001,
      max_vel_rad_s=max_vel,
      max_acc_rad_s2=max_acc,
  )
  unwind_r = generate_quintic_trajectory(
      simplify_joint_polyline(right_waypoints),
      dt_s=0.001,
      max_vel_rad_s=max_vel,
      max_acc_rad_s2=max_acc,
  )
  return unwind_l, unwind_r


def _resolve_sides(side: str | None) -> tuple[str, ...]:
  if side is None:
    return ("left", "right")
  if side not in ("left", "right"):
    raise ValueError(f"Invalid side={side!r}; expected 'left' or 'right'")
  return (side,)


class Fr3DuoRealBackend:
  """Manages 1 kHz `ArmWorkerProcess` workers and Robotiq RS-485 grippers."""

  def __init__(
      self,
      config: RobotConfig,
      *,
      realtime_cfg_name: str,
      robot_factory: Any = None,
      gripper_factory: Any = None,
  ) -> None:
    """Initializes hardware arm workers and optional gripper factories."""
    self.config = config
    self.realtime_cfg_name = realtime_cfg_name
    self._robot_factory = robot_factory
    self._gripper_factory = gripper_factory
    self.workers: dict[str, ArmWorkerProcess] = {}
    self.injected_robots: dict[str, Any] = {}
    self.grippers: dict[str, Any] = {}

  def connect(self, realtime_cfg_name: str) -> None:
    """Connects enabled arm workers/factories and Robotiq serial grippers."""
    self.realtime_cfg_name = realtime_cfg_name
    try:
      for side, arm_cfg in self.config.arms.items():
        if not arm_cfg.enabled:
          continue
        if self._robot_factory is not None:
          self.injected_robots[side] = self._robot_factory(
              side, arm_cfg.ip, realtime_cfg_name
          )
        else:
          worker = ArmWorkerProcess(side, arm_cfg.ip, realtime_cfg_name)
          worker.start()
          self.workers[side] = worker

        g_port = arm_cfg.gripper_port.strip()
        if self._gripper_factory is not None:
          g_inst = self._gripper_factory(side, g_port)
          connect_fn = getattr(g_inst, "connect", None)
          if callable(connect_fn):
            connect_fn()
          self.grippers[side] = g_inst
        elif g_port:
          gripper = robotiq_gripper_controller.RobotiqGripper(
              port=g_port,
              baudrate=int(arm_cfg.gripper_baudrate),
              slave_id=int(arm_cfg.gripper_slave_id),
          )
          gripper.connect()
          self.grippers[side] = gripper

        rec = self.recover_errors(side).get(side)
        if rec and rec.get("had_errors") and not rec.get("recovered"):
          warnings.warn(
              f"FR3 {side} arm has unrecovered FCI errors on connect: "
              f"errors_after={rec.get('errors_after')}, "
              f"error={rec.get('error')}",
              RuntimeWarning,
              stacklevel=2,
          )
    except Exception:
      self.close()
      raise

  def close(self) -> None:
    """Stops child arm workers, disconnects robots, and closes grippers."""
    try:
      for worker in list(self.workers.values()):
        try:
          worker.stop()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      for robot in list(self.injected_robots.values()):
        disconnect_pylibfranka_robot(robot)
      for gripper in list(self.grippers.values()):
        close_fn = getattr(gripper, "close", None)
        if callable(close_fn):
          try:
            close_fn()
          except Exception:  # pylint: disable=broad-exception-caught
            pass
    finally:
      self.workers.clear()
      self.injected_robots.clear()
      self.grippers.clear()

  def has_arm(self, side: str) -> bool:
    """Returns True if `side` has an active worker or injected robot."""
    return side in self.workers or side in self.injected_robots

  def read_arm_state(self, side: str) -> dict[str, Any]:
    """Reads the latest FCI state dictionary for arm `side`."""
    if side in self.workers:
      return self.workers[side].read_state()
    return read_pylibfranka_state(self.injected_robots[side])

  def recover_errors(
      self, side: str | None = None
  ) -> dict[str, dict[str, Any]]:
    """Checks for FCI errors/reflex and runs automatic recovery if needed."""
    out: dict[str, dict[str, Any]] = {}
    for arm_side in _resolve_sides(side):
      if arm_side in self.workers:
        worker = self.workers[arm_side]
        recover_fn = getattr(worker, "recover_errors", None)
        if callable(recover_fn):
          out[arm_side] = recover_fn()
        else:
          has_err, errs = _state_has_errors(worker.read_state())
          out[arm_side] = (
              _recovery_result(
                  had_errors=True,
                  errors_before=errs,
                  recovered=False,
                  errors_after=errs,
                  error="worker.recover_errors() is unavailable",
              )
              if has_err
              else _recovery_result()
          )
      elif arm_side in self.injected_robots:
        out[arm_side] = recover_pylibfranka_errors(
            self.injected_robots[arm_side]
        )
    return out

  def activate_grippers(
      self,
  ) -> dict[str, robotiq_gripper_controller.RobotiqStatus]:
    """Activates all connected Robotiq grippers."""
    return {side: g.activate() for side, g in sorted(self.grippers.items())}

  def move_gripper(
      self,
      arm: str,
      closure_0_to_1: float,
      *,
      speed: float = 0.5,
      force: float = 0.5,
      wait: bool = True,
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Commands normalized gripper closure `[0.0, 1.0]` on `arm`."""
    return self.grippers[arm].move_normalized(
        closure_0_to_1, speed=speed, force=force, wait=wait
    )

  def execute_bimanual_trajectory(
      self,
      traj_left: np.ndarray,
      traj_right: np.ndarray,
      q_left_start: np.ndarray,
      q_right_start: np.ndarray,
      *,
      max_vel: float,
      max_acc: float,
      abort_watcher: AbortWatcher | None = None,
  ) -> bool:
    """Executes dual-arm 1 kHz splines and unwinds cleanly on abort."""
    for side in tuple(self.workers.keys()):
      rec = self.recover_errors(side).get(side)
      if rec and rec.get("had_errors") and not rec.get("recovered"):
        raise RuntimeError(
            f"FR3 {side} arm has unrecoverable FCI errors before trajectory "
            f"execution: errors_after={rec.get('errors_after')}, "
            f"error={rec.get('error')}"
        )

    if "left" in self.workers:
      self.workers["left"].send_trajectory_async(traj_left)
    if "right" in self.workers:
      self.workers["right"].send_trajectory_async(traj_right)

    replies: dict[str, dict[str, Any]] = {}
    aborted = False
    while len(replies) < len(self.workers):
      if (
          not aborted
          and abort_watcher is not None
          and abort_watcher.poll()
      ):
        aborted = True
        for w in self.workers.values():
          w.signal_abort()
      for side, w in self.workers.items():
        if side not in replies:
          rep = w.poll_done(0.005)
          if rep is not None:
            replies[side] = rep

    for side, rep in replies.items():
      if not rep.get("ok"):
        raise RuntimeError(
            f"Hardware trajectory error on {side} arm: {rep.get('error')}"
        )

    if aborted:
      l_rep = replies.get("left", {})
      r_rep = replies.get("right", {})
      l_trav = list(l_rep.get("travelled", []))
      r_trav = list(r_rep.get("travelled", []))
      l_qd = (l_rep.get("state") or {}).get("q_d")
      r_qd = (r_rep.get("state") or {}).get("q_d")
      l_raw = ([l_qd] if l_qd else []) + l_trav[::-1] + [q_left_start.tolist()]
      r_raw = ([r_qd] if r_qd else []) + r_trav[::-1] + [q_right_start.tolist()]
      unwind_l, unwind_r = _build_bimanual_unwind_trajs(
          l_raw, r_raw, max_vel, max_acc
      )
      if "left" in self.workers:
        self.workers["left"].send_trajectory_async(unwind_l)
      if "right" in self.workers:
        self.workers["right"].send_trajectory_async(unwind_r)
      unwind_replies: dict[str, dict[str, Any]] = {}
      while len(unwind_replies) < len(self.workers):
        for side, w in self.workers.items():
          if side not in unwind_replies:
            rep = w.poll_done(0.01)
            if rep is not None:
              unwind_replies[side] = rep
      for side, u_rep in unwind_replies.items():
        if not u_rep.get("ok"):
          raise RuntimeError(
              f"Hardware unwind trajectory error on {side} arm: "
              f"{u_rep.get('error')}"
          )

    return aborted


class Fr3DuoSimBackend:
  """Headless MuJoCo simulation and offscreen rendering backend."""

  def __init__(self, mj_model: Any, mj_data: Any) -> None:
    """Indexes MuJoCo arm joints, actuators, and initial gripper status."""
    if mujoco is None:
      raise RuntimeError(
          "`mujoco` is required for `Fr3DuoSimBackend`. "
          "Install it with `pip install mujoco`."
      )
    self.mj_model = mj_model
    self.mj_data = mj_data
    self._renderer: Any = None
    self._renderer_hw: tuple[int, int] = (0, 0)
    self._arm_qpos_addrs: dict[str, list[int]] = {}
    self._arm_dof_addrs: dict[str, list[int]] = {}
    self._arm_act_ids: dict[str, list[int]] = {}
    self._gripper_act_id: dict[str, int] = {}
    self._gripper_status: dict[
        str, robotiq_gripper_controller.RobotiqStatus
    ] = {}

    for side in ("left", "right"):
      q_addrs, d_addrs = fr3_mjcf.arm_joint_qpos_and_dof_addrs(mj_model, side)
      a_ids = [
          int(
              mujoco.mj_name2id(
                  mj_model,
                  mujoco.mjtObj.mjOBJ_ACTUATOR,
                  f"{side}/fr3_joint{idx + 1}",
              )
          )
          for idx in range(7)
      ]
      self._arm_qpos_addrs[side] = q_addrs
      self._arm_dof_addrs[side] = d_addrs
      self._arm_act_ids[side] = a_ids

      self._gripper_act_id[side] = int(
          mujoco.mj_name2id(
              mj_model,
              mujoco.mjtObj.mjOBJ_ACTUATOR,
              fr3_mjcf.gripper_actuator_name(side),
          )
      )
      self._gripper_status[side] = robotiq_gripper_controller.RobotiqStatus(
          activated=True,
          goto_active=False,
          obj_status=3,
          fault=0,
          requested_pos_raw=0,
          measured_pos_raw=0,
          current_ma=0.0,
      )

    self.sync_ctrl_to_qpos()

  def sync_ctrl_to_qpos(self) -> None:
    """Copies current joint positions into position actuators and forwards."""
    for side in ("left", "right"):
      for qadr, aid in zip(
          self._arm_qpos_addrs[side], self._arm_act_ids[side]
      ):
        if aid >= 0:
          self.mj_data.ctrl[aid] = float(self.mj_data.qpos[qadr])
    mujoco.mj_forward(self.mj_model, self.mj_data)

  def get_arm_q(self, arm: str) -> np.ndarray:
    """Returns simulated 7-DOF joint positions (rad) for `arm`."""
    return np.array(
        [self.mj_data.qpos[i] for i in self._arm_qpos_addrs[arm]],
        dtype=np.float64,
    )

  def get_arm_dq(self, arm: str) -> np.ndarray:
    """Returns simulated 7-DOF joint velocities (rad/s) for `arm`."""
    return np.array(
        [self.mj_data.qvel[i] for i in self._arm_dof_addrs[arm]],
        dtype=np.float64,
    )

  def get_arm_tau(self, arm: str) -> np.ndarray:
    """Returns simulated bias torques (Nm) for `arm`."""
    return np.array(
        [self.mj_data.qfrc_bias[i] for i in self._arm_dof_addrs[arm]],
        dtype=np.float64,
    )

  def get_gripper_status(
      self, arm: str
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Returns simulated Robotiq status for `arm`."""
    return self._gripper_status[arm]

  def activate_grippers(
      self,
  ) -> dict[str, robotiq_gripper_controller.RobotiqStatus]:
    """Marks both simulated grippers as activated."""
    for side in ("left", "right"):
      prev = self._gripper_status[side]
      self._gripper_status[side] = robotiq_gripper_controller.RobotiqStatus(
          activated=True,
          goto_active=False,
          obj_status=3,
          fault=0,
          requested_pos_raw=prev.requested_pos_raw,
          measured_pos_raw=prev.measured_pos_raw,
          current_ma=10.0,
      )
    return dict(self._gripper_status)

  def move_gripper(
      self, arm: str, closure_0_to_1: float
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Steps simulated gripper actuator toward `closure_0_to_1` in `[0, 1]`."""
    raw_target = robotiq_gripper_controller.closure_to_raw(closure_0_to_1)
    aid = self._gripper_act_id[arm]
    if aid >= 0:
      self.mj_data.ctrl[aid] = float(raw_target)
      for _ in range(80):
        mujoco.mj_step(self.mj_model, self.mj_data)
    self._gripper_status[arm] = robotiq_gripper_controller.RobotiqStatus(
        activated=True,
        goto_active=False,
        obj_status=3,
        fault=0,
        requested_pos_raw=raw_target,
        measured_pos_raw=raw_target,
        current_ma=20.0 if raw_target > 0 else 0.0,
    )
    return self._gripper_status[arm]

  def step_bimanual_targets(
      self, q_left: np.ndarray, q_right: np.ndarray
  ) -> None:
    """Sets both arms' target qpos/ctrl and updates MuJoCo kinematics."""
    for side, q_vec in (("left", q_left), ("right", q_right)):
      for val, qadr, dadr, aid in zip(
          q_vec,
          self._arm_qpos_addrs[side],
          self._arm_dof_addrs[side],
          self._arm_act_ids[side],
      ):
        if aid >= 0:
          self.mj_data.ctrl[aid] = float(val)
        self.mj_data.qpos[qadr] = float(val)
        self.mj_data.qvel[dadr] = 0.0
    mujoco.mj_forward(self.mj_model, self.mj_data)

  def sync_qpos_no_actuators(self, side: str, q_rad: Sequence[float]) -> None:
    """Mirrors live hardware `q_rad` onto `mj_data.qpos` without stepping."""
    for val, qadr in zip(q_rad, self._arm_qpos_addrs[side]):
      self.mj_data.qpos[qadr] = float(val)
    mujoco.mj_forward(self.mj_model, self.mj_data)

  def render_cameras(
      self, width: int = 1280, height: int = 720
  ) -> dict[str, np.ndarray]:
    """Renders the 4 canonical Duo cameras offscreen via `mujoco.Renderer`."""
    w, h = int(width), int(height)
    frames: dict[str, np.ndarray] = {}
    mujoco.mj_forward(self.mj_model, self.mj_data)
    if self._renderer is None or self._renderer_hw != (h, w):
      self.close_renderer()
      self._renderer = mujoco.Renderer(self.mj_model, height=h, width=w)
      self._renderer_hw = (h, w)

    for cam_name in fr3_camera_controller.CANONICAL_CAMERAS:
      mjcf_cam = fr3_mjcf.resolve_mjcf_camera_name(self.mj_model, cam_name)
      self._renderer.update_scene(self.mj_data, camera=mjcf_cam)
      rgb = np.ascontiguousarray(self._renderer.render(), dtype=np.uint8)
      frames[cam_name] = rgb
    return frames

  def close_renderer(self) -> None:
    """Closes the offscreen MuJoCo GL renderer if active."""
    if self._renderer is not None:
      close_fn = getattr(self._renderer, "close", None)
      if callable(close_fn):
        try:
          close_fn()
        except (RuntimeError, OSError, AttributeError):
          # Ignore GL context teardown errors when closing renderer.
          pass
      self._renderer = None
      self._renderer_hw = (0, 0)


class Fr3DuoRobot:
  """Unified live hardware and MuJoCo simulation controller for FR3 Duo."""

  def __init__(
      self,
      config: RobotConfig | Mapping[str, Any] | None = None,
      *,
      mode: str = "hardware",
      dry_run: bool = False,
      realtime_mode: str = "auto",
      menagerie_dir: str | Path | None = None,
      robot_factory: Any = None,
      gripper_factory: Any = None,
  ) -> None:
    """Builds MJCF safety/sim models and selects real or simulated backend."""
    norm_mode = mode.strip().lower()
    if norm_mode not in ("sim", "hardware"):
      raise ValueError(
          f"Invalid robot mode={mode!r}; expected 'sim' or 'hardware'"
      )
    if mujoco is None:
      raise RuntimeError(
          "`mujoco` is required for `Fr3DuoRobot`. "
          "Install it with `pip install mujoco`."
      )
    self.mode = norm_mode
    self.dry_run = bool(dry_run)
    self.config = normalize_robot_config(config)
    if self.mode == "hardware" and not self.dry_run:
      missing_cfg = validate_robot_config_complete(self.config)
      if missing_cfg:
        warnings.warn(
            "Incomplete/unconfirmed RobotConfig for live hardware mode "
            f"(missing: {', '.join(missing_cfg)}). Run `bringup.py --step "
            "geometry` and bind all hardware before live actuation.",
            UserWarning,
            stacklevel=2,
        )
    self.realtime_mode = str(realtime_mode or self.config.realtime_mode)
    self.realtime_info = resolve_realtime_config(
        self.realtime_mode
        if (self.mode == "hardware" and not self.dry_run)
        else (
            "ignore" if self.realtime_mode == "enforce" else self.realtime_mode
        )
    )
    self._sim_lock = threading.RLock()

    self._mj_model, self._mjcf_xml = fr3_mjcf.build_fr3_duo_mjcf(
        self.config, menagerie_dir=menagerie_dir
    )
    self._mj_data = mujoco.MjData(self._mj_model)
    self._sim_backend = Fr3DuoSimBackend(self._mj_model, self._mj_data)
    self._mj_data.qpos[:] = self._mj_model.qpos0
    for side in ("left", "right"):
      rest_q = self.config.arms[side].rest_pose_rad
      for qadr, val in zip(self._sim_backend._arm_qpos_addrs[side], rest_q):
        self._mj_data.qpos[qadr] = float(val)
    self._sim_backend.sync_ctrl_to_qpos()

    self._safety = Fr3DuoSafetyChecker(
        self.config, mj_model=self._mj_model, mjcf_xml=self._mjcf_xml
    )
    self._link0_in_base_cache: dict[str, np.ndarray] = {}
    self._link0_inv_cache: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
      t_bl = compute_link0_in_base_transform_for_model(
          self._mj_model, side, self._mj_data
      )
      self._link0_in_base_cache[side] = t_bl
      self._link0_inv_cache[side] = np.linalg.inv(t_bl)

    self._hw = (
        Fr3DuoRealBackend(
            self.config,
            realtime_cfg_name=str(self.realtime_info["realtime_config"]),
            robot_factory=robot_factory,
            gripper_factory=gripper_factory,
        )
        if (self.mode == "hardware" and not self.dry_run)
        else None
    )
    self._hw_finalizer: weakref.finalize | None = (
        weakref.finalize(self, self._hw.close)
        if self._hw is not None
        else None
    )
    self._connected = False

  @property
  def mj_model(self) -> Any:
    """Returns the compiled MuJoCo `MjModel` for the Duo cell."""
    return self._mj_model

  @property
  def mj_data(self) -> Any:
    """Returns the active MuJoCo `MjData` state."""
    return self._mj_data

  @property
  def mjcf_xml(self) -> str:
    """Returns the resolved MJCF XML string for the Duo cell."""
    return self._mjcf_xml

  def connect(self) -> None:
    """Connects hardware workers/grippers when `mode == 'hardware'`."""
    if self._connected:
      return
    if self.dry_run or self._hw is None:
      self._connected = True
      return

    if self._hw_finalizer is None or not self._hw_finalizer.alive:
      self._hw_finalizer = weakref.finalize(self, self._hw.close)
    self.realtime_info = resolve_realtime_config(self.realtime_mode)
    self._hw.connect(str(self.realtime_info["realtime_config"]))
    self._connected = True

  def close(self) -> None:
    """Cleanly shuts down child processes, serial ports, and GL renderers."""
    if self._hw_finalizer is not None:
      self._hw_finalizer.detach()
      self._hw_finalizer = None
    if self._hw is not None:
      self._hw.close()
    with self._sim_lock:
      self._sim_backend.close_renderer()
    self._connected = False

  def __enter__(self) -> Fr3DuoRobot:
    self.connect()
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self.close()

  def _ensure_connected(self) -> None:
    if not self._connected:
      self.connect()

  def recover_errors(
      self, side: str | None = None
  ) -> dict[str, dict[str, Any]]:
    """Checks for FCI errors/reflex and runs automatic recovery if needed."""
    sides = _resolve_sides(side)
    if self.dry_run or self._hw is None:
      return {arm_side: _recovery_result() for arm_side in sides}
    self._ensure_connected()
    return self._hw.recover_errors(side=side)

  def read_state(self) -> dict[str, Any]:
    """Reads dual-arm joints, TCP, external wrench, gripper, and 16-D state."""
    if not self._connected and self.mode == "hardware" and not self.dry_run:
      self.connect()

    ts = time.monotonic()
    result: dict[str, Any] = {"timestamp_s": ts}
    state_16d: list[float] = []

    for side in ("left", "right"):
      t_base_link0 = self._link0_in_base_cache[side]
      r_base_link0 = t_base_link0[:3, :3]
      t_link0_base = self._link0_inv_cache[side]

      is_hw = self._hw is not None and self._hw.has_arm(side)
      if is_hw and self._hw is not None:
        hw = self._hw.read_arm_state(side)
        q_rad = np.asarray(hw["q"], dtype=np.float64)
        dq_rad = np.asarray(hw["dq"], dtype=np.float64)
        tau_j = np.asarray(hw["tau_j"], dtype=np.float64)
        o_t_ee_4x4 = unpack_o_t_ee(hw["o_t_ee"])

        with self._sim_lock:
          self._sim_backend.sync_qpos_no_actuators(side, q_rad)

        if side in self._hw.grippers:
          g_status = self._hw.grippers[side].read_status()
        else:
          g_status = self._sim_backend.get_gripper_status(side)
      else:
        hw = None
        with self._sim_lock:
          q_rad = self._sim_backend.get_arm_q(side)
          dq_rad = self._sim_backend.get_arm_dq(side)
          tau_j = self._sim_backend.get_arm_tau(side)
          g_status = self._sim_backend.get_gripper_status(side)
        o_t_ee_4x4 = None

      tcp_pos_base, tcp_rot_base = fr3_kinematics.compute_arm_fk(
          self._mj_model, side, q_rad
      )
      t_base_tcp = np.eye(4, dtype=np.float64)
      t_base_tcp[:3, :3] = tcp_rot_base
      t_base_tcp[:3, 3] = tcp_pos_base

      if o_t_ee_4x4 is None:
        o_t_ee_4x4 = t_link0_base @ t_base_tcp
        fk_pos_error_m = 0.0
      else:
        fk_pos_error_m = compute_fk_pos_error_m(
            tcp_pos_base,
            t_base_link0,
            o_t_ee_4x4,
            float(self.config.tcp_offset_z_m),
            hw.get("f_t_ee") if hw is not None else None,
            mj_model=self._mj_model,
            side=side,
            q_rad=q_rad,
        )

      wrench_6d = compute_ext_wrench_base_6d(
          hw if is_hw else None,
          self._mj_model,
          side,
          q_rad,
          tau_j,
          r_base_link0,
      )
      g_ratio = float(g_status.position_ratio)

      result[side] = {
          "q_rad": [float(x) for x in q_rad],
          "dq_rad": [float(x) for x in dq_rad],
          "tau_j": [float(x) for x in tau_j],
          "tau_ext_base_6d": [float(x) for x in wrench_6d],
          "tcp_pos_base": [float(x) for x in tcp_pos_base],
          "tcp_rot_base": [
              [float(v) for v in row] for row in tcp_rot_base.tolist()
          ],
          "o_t_ee_4x4": [
              [float(v) for v in row] for row in o_t_ee_4x4.tolist()
          ],
          "fk_pos_error_m": float(fk_pos_error_m),
          "gripper_ratio": g_ratio,
          "gripper_status": g_status.to_dict(),
          "current_errors": list(hw.get("current_errors") or [])
          if hw is not None
          else [],
          "desk_audit": audit_arm_desk_config(
              side, hw if is_hw else None, r_base_link0
          ),
      }
      state_16d.extend(float(x) for x in q_rad)
      state_16d.append(g_ratio)

    result["state_16d"] = state_16d
    return result

  def render_sim_cameras(
      self,
      width: int = 1280,
      height: int = 720,
  ) -> dict[str, np.ndarray]:
    """Renders the 4 canonical Duo cameras offscreen via `Fr3DuoSimBackend`."""
    with self._sim_lock:
      return self._sim_backend.render_cameras(width=width, height=height)

  def activate_grippers(
      self,
  ) -> dict[str, robotiq_gripper_controller.RobotiqStatus]:
    """Activates both Robotiq grippers (no-op on HW when `dry_run=True`)."""
    self._ensure_connected()
    if self.dry_run:
      with self._sim_lock:
        return {
            side: self._sim_backend.get_gripper_status(side)
            for side in ("left", "right")
        }
    if self._hw is None:
      with self._sim_lock:
        return self._sim_backend.activate_grippers()
    if not self._hw.grippers:
      raise RuntimeError(
          "No hardware Robotiq grippers configured/connected in "
          "mode='hardware'."
      )
    return self._hw.activate_grippers()

  def move_gripper(
      self,
      arm: str,
      closure_0_to_1: float,
      *,
      speed: float = 0.5,
      force: float = 0.5,
      wait: bool = True,
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Moves `arm`'s Robotiq gripper to normalized closure `[0.0, 1.0]`."""
    if arm not in ("left", "right"):
      raise ValueError(f"Invalid arm={arm!r}; expected 'left' or 'right'")
    self._ensure_connected()
    if self.dry_run:
      with self._sim_lock:
        return self._sim_backend.get_gripper_status(arm)
    if self._hw is not None:
      if arm not in self._hw.grippers:
        raise RuntimeError(
            f"Hardware Robotiq gripper for '{arm}' is not configured/connected."
        )
      return self._hw.move_gripper(
          arm, closure_0_to_1, speed=speed, force=force, wait=wait
      )
    with self._sim_lock:
      return self._sim_backend.move_gripper(arm, closure_0_to_1)

  def _resolve_speed_caps(
      self, speed_deg_s: float | None
  ) -> tuple[float, float]:
    if speed_deg_s is not None:
      default_deg_s = float(speed_deg_s)
    elif self.config.max_joint_speed_deg_s is not None:
      default_deg_s = float(self.config.max_joint_speed_deg_s)
    else:
      default_deg_s = float(
          self.realtime_info.get("max_joint_speed_deg_s", 15.0)
      )
    chosen_deg_s = min(max(float(default_deg_s), 1.0), 45.0)
    max_vel_rad_s = math.radians(chosen_deg_s)
    max_acc_rad_s2 = float(
        self.realtime_info.get("max_joint_acc_rad_s2", 2.0)
    )
    return max_vel_rad_s, max_acc_rad_s2

  def move_arm_waypoints(
      self,
      arm: str,
      waypoints_rad: Sequence[Sequence[float]],
      *,
      abort_watcher: AbortWatcher | None = None,
      speed_deg_s: float | None = None,
  ) -> dict[str, Any]:
    """Executes a safety-checked single-arm trajectory with abort unwind."""
    if arm not in ("left", "right"):
      raise ValueError(f"Invalid arm={arm!r}; expected 'left' or 'right'")
    self._ensure_connected()
    if self._hw is not None and arm not in self._hw.workers:
      raise RuntimeError(
          f"Hardware arm worker for '{arm}' is not configured/connected."
      )

    state_now = self.read_state()
    other_arm = "right" if arm == "left" else "left"
    q_other = clamp_q_to_padded_limits(state_now[other_arm]["q_rad"])
    if arm == "left":
      res = self._move_bimanual_waypoints_impl(
          waypoints_rad,
          [q_other.tolist()],
          abort_watcher=abort_watcher,
          speed_deg_s=speed_deg_s,
          required_arms=("left",),
      )
      res["final_q_rad"] = res["final_left_q_rad"]
      return res

    res = self._move_bimanual_waypoints_impl(
        [q_other.tolist()],
        waypoints_rad,
        abort_watcher=abort_watcher,
        speed_deg_s=speed_deg_s,
        required_arms=("right",),
    )
    res["final_q_rad"] = res["final_right_q_rad"]
    return res

  def _execute_sim_bimanual_trajectory(
      self,
      traj_left: np.ndarray,
      traj_right: np.ndarray,
      q_left_start: np.ndarray,
      q_right_start: np.ndarray,
      *,
      max_vel: float,
      max_acc: float,
      abort_watcher: AbortWatcher | None,
  ) -> tuple[bool, int]:
    n_max = int(traj_left.shape[0])
    num_chunks = min(40, n_max)
    chunk_indices = np.unique(
        np.linspace(0, n_max - 1, num_chunks, dtype=np.intp)
    )
    travelled_left: list[list[float]] = [q_left_start.tolist()]
    travelled_right: list[list[float]] = [q_right_start.tolist()]
    aborted = False
    steps_done = 0

    for c_idx in chunk_indices:
      if abort_watcher is not None and abort_watcher.poll():
        aborted = True
        break
      ql = traj_left[c_idx]
      qr = traj_right[c_idx]
      with self._sim_lock:
        self._sim_backend.step_bimanual_targets(ql, qr)
      travelled_left.append(ql.tolist())
      travelled_right.append(qr.tolist())
      steps_done = int(c_idx) + 1
      if abort_watcher is not None:
        time.sleep(0.004)

    if not aborted and abort_watcher is not None and abort_watcher.poll():
      aborted = True

    if aborted:
      unwind_l, unwind_r = _build_bimanual_unwind_trajs(
          travelled_left[::-1] + [q_left_start.tolist()],
          travelled_right[::-1] + [q_right_start.tolist()],
          max_vel,
          max_acc,
      )
      u_max = max(len(unwind_l), len(unwind_r))
      u_indices = np.unique(
          np.linspace(0, u_max - 1, min(16, u_max), dtype=np.intp)
      )
      for u_idx in u_indices:
        ql = unwind_l[min(u_idx, len(unwind_l) - 1)]
        qr = unwind_r[min(u_idx, len(unwind_r) - 1)]
        with self._sim_lock:
          self._sim_backend.step_bimanual_targets(ql, qr)
    return aborted, steps_done

  def move_bimanual_waypoints(
      self,
      left_waypoints_rad: Sequence[Sequence[float]],
      right_waypoints_rad: Sequence[Sequence[float]],
      *,
      abort_watcher: AbortWatcher | None = None,
      speed_deg_s: float | None = None,
  ) -> dict[str, Any]:
    """Executes synchronized dual-arm splines with safety & abort unwind."""
    return self._move_bimanual_waypoints_impl(
        left_waypoints_rad,
        right_waypoints_rad,
        abort_watcher=abort_watcher,
        speed_deg_s=speed_deg_s,
        required_arms=("left", "right"),
    )

  def _move_bimanual_waypoints_impl(
      self,
      left_waypoints_rad: Sequence[Sequence[float]],
      right_waypoints_rad: Sequence[Sequence[float]],
      *,
      abort_watcher: AbortWatcher | None = None,
      speed_deg_s: float | None = None,
      required_arms: tuple[str, ...] = ("left", "right"),
  ) -> dict[str, Any]:
    self._ensure_connected()

    state_before = self.read_state()
    q_left_start = clamp_q_to_padded_limits(state_before["left"]["q_rad"])
    q_right_start = clamp_q_to_padded_limits(state_before["right"]["q_rad"])

    l_wps = [list(map(float, wp)) for wp in left_waypoints_rad]
    r_wps = [list(map(float, wp)) for wp in right_waypoints_rad]
    if not l_wps or not r_wps:
      raise ValueError("Waypoints for both arms must be non-empty")

    if float(np.max(np.abs(np.asarray(l_wps[0]) - q_left_start))) > 1e-4:
      l_wps = [q_left_start.tolist()] + l_wps
    if float(np.max(np.abs(np.asarray(r_wps[0]) - q_right_start))) > 1e-4:
      r_wps = [q_right_start.tolist()] + r_wps

    max_vel, max_acc = self._resolve_speed_caps(speed_deg_s)
    traj_left = generate_quintic_trajectory(
        l_wps, dt_s=0.001, max_vel_rad_s=max_vel, max_acc_rad_s2=max_acc
    )
    traj_right = generate_quintic_trajectory(
        r_wps, dt_s=0.001, max_vel_rad_s=max_vel, max_acc_rad_s2=max_acc
    )

    n_max = max(int(traj_left.shape[0]), int(traj_right.shape[0]))
    if traj_left.shape[0] < n_max:
      pad = np.repeat(traj_left[-1:], n_max - traj_left.shape[0], axis=0)
      traj_left = np.vstack([traj_left, pad])
    if traj_right.shape[0] < n_max:
      pad = np.repeat(traj_right[-1:], n_max - traj_right.shape[0], axis=0)
      traj_right = np.vstack([traj_right, pad])

    with self._sim_lock:
      self._safety.verify_bimanual_trajectory(
          traj_left,
          traj_right,
          z_table_m=float(self.config.z_table_m),
          min_clearance_m=float(self.config.min_inter_arm_clearance_m),
      )

    if self.dry_run:
      return {
          "ok": True,
          "aborted": False,
          "dry_run": True,
          "reason": "",
          "unwound_to_start": False,
          "steps_executed": 0,
          "final_left_q_rad": state_before["left"]["q_rad"],
          "final_right_q_rad": state_before["right"]["q_rad"],
      }

    if self._hw is not None:
      missing = [s for s in required_arms if s not in self._hw.workers]
      if missing:
        raise RuntimeError(
            f"Hardware arm worker(s) {missing} not configured/connected in "
            "mode='hardware'."
        )
      aborted = self._hw.execute_bimanual_trajectory(
          traj_left,
          traj_right,
          q_left_start,
          q_right_start,
          max_vel=max_vel,
          max_acc=max_acc,
          abort_watcher=abort_watcher,
      )
      steps_done = n_max
    else:
      aborted, steps_done = self._execute_sim_bimanual_trajectory(
          traj_left,
          traj_right,
          q_left_start,
          q_right_start,
          max_vel=max_vel,
          max_acc=max_acc,
          abort_watcher=abort_watcher,
      )

    after = self.read_state()
    return {
        "ok": not aborted,
        "aborted": aborted,
        "reason": abort_watcher.reason if (aborted and abort_watcher) else "",
        "unwound_to_start": aborted,
        "steps_executed": steps_done,
        "final_left_q_rad": after["left"]["q_rad"],
        "final_right_q_rad": after["right"]["q_rad"],
    }
