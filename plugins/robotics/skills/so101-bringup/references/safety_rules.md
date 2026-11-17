<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
# SO-101 Working & Hardware Safety Rules Reference (`R1–R16`)

This reference summarizes the implementation contract for rules `R1–R16` enforced by [`so101_common.py`](../scripts/so101_common.py).

## Summary of Rules `R1–R16`

| Rule | Title | Technical Enforcement in Scripts |
| :--- | :--- | :--- |
| **R1** | Ask only before live motion | Any action enabling STS3215 motor torque (including `robot.connect()`) or commanding joint positions counts as motion. Never ask for confirmation on `--dry-run`, `--read-only`, `--fit-from-log`, camera scans/benchmarks, or software setup. Ask for explicit `"go"` (reporting start/end poses, `z_min`, peak joint speed, and abort instructions) only before energizing torque: once in Stage 2 before first powered bring-up motion (covering the single `abort-test` nudge and Stage 3 `hello_world.py` while the workspace stays clear) and once in Stage 5 (`measure_table.py` + `calibrate_handeye.py`). |
| **R2** | Auto-run dry-run first | Automatically run `--dry-run` (validates bounds, step limits, and FK with zero hardware contact) before requesting a live motion `"go"`, then run smallest real motion (`--mode abort-test` -> `hello_world.py`), changing one variable per trial. |
| **R3** | Abort key on every motion (`q` + `Enter`) | `AbortWatcher` monitors stdin for `q` + `Enter` and `SIGINT` (`Ctrl-C`); triggers `execute_r3_retreat()` to unwind smoothly backwards along `travelled_waypoints` to `rest_pose`. |
| **R4** | Torque off at rest | `safe_disconnect_robot()` returns smoothly to `rest_pose`, verifies `Present_Position` directly from `SerialMotorsBus`, and sets `Torque_Enable = 0` inside `try ... finally`; if away from `rest_pose` or unreadable, leave `Torque_Enable = 1` and warn the human to support the arm before cutting power. |
| **R5** | Speed, step, joint, and height limits | `<= 45.0°/step` ceiling (`20.0–30.0°` default), cosine-eased velocity `<= 90.0°/s` ceiling (`30.0°/s` default; Stage 5 touch-down `<= 12°/s` and hand-eye sweep `<= 15°/s`), `>= 5.0°` margin inside calibrated `[range_min, range_max]` bounds on moved joints that never blocks moving away from or returning to a `rest_pose` inside the margin, `-0.05 m` hard Cartesian Z floor (`z_table_m` after Stage 5 Part B), default gripper current/torque limits. |
| **R6** | Pre-flight check before every motion | `run_preflight()` verifies all 6 STS3215 servo IDs (`1..6`) answer with `Torque_Enable = 0` and every configured camera (`>= 1` for Stages 0–3; fixed + wrist for Stage 5) delivers fresh frames (`age <= 0.30 s`) across repeated reads before enabling torque. |
| **R7** | Record video of every motion | Continuous MP4 + JSONL recording from before leaving `rest_pose` until after returning to `rest_pose` and disabling torque. |
| **R8** | Forward kinematics (`SO101Follower.urdf`) | `RobotKinematics` computes 3D tip positions using `SO101Follower.urdf` / `so101_new_calib.urdf`. |
| **R9** | Two-cue visual verification | `detect_marker.py` requires both colour/polygon solidity and surround contrast against the wrist mount plate. |
| **R10** | Re-verify if anything shifts | `check_scene_drift()` compares live fixed-camera frames against `reference_scene.json` and requires `< 15 px` Hand-Eye residual. |
| **R11** | `tmux` session `robot` | Interactive CLI tools (`lerobot-find-port`, `lerobot-calibrate`) and camera scripts run inside `tmux` session `robot`. |
| **R12** | Keep `NOTES.md` current | Read `NOTES.md` before each stage; update `NOTES.md` and `config/robot_config.json` after each stage. |
| **R13** | Clean scripts & git commits | Single-purpose scripts with `--dry-run` and JSON config; commit to git after every stage. |
| **R14** | 6-field stage report | Every stage report includes `Result`, `Evidence`, `Numbers measured`, `Done-when checks`, `Concerns`, and `Need from you`. |
| **R15** | Auto-advance non-motion stages | Auto-advance across software, `--dry-run`, and read-only stages without waiting for sign-off; pause only for physical human actions (USB re-plug only if port auto-detection is ambiguous, `lerobot-calibrate` unless a valid calibration is reused, marker placement) or `R1` live-motion `"go"`. |
| **R16** | Human safety first | Ensure `Torque_Enable = 0` whenever the human touches the arm, cables, marker, or workspace. |
