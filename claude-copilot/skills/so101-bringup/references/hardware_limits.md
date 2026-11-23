# SO-101 Hardware Limits & Configuration Schema

## 1. Servo Bus & Joint Map (`Feetech STS3215`)

*   **Actuators:** `6 × Feetech STS3215` serial bus servos (`1,000,000` baud, 12-bit magnetic encoder `0..4095` ticks = `360.0°`, `0.087890625°/tick`).
*   **Joint Order (`IDs 1..6`):**
    1. `shoulder_pan` (base yaw)
    2. `shoulder_lift` (shoulder pitch)
    3. `elbow_flex` (elbow pitch)
    4. `wrist_flex` (wrist pitch)
    5. `wrist_roll` (wrist roll)
    6. `gripper` (jaw opening, `0.0` = closed, `100.0` = open)

## 2. Camera Configurations

*   **Minimal Setup (`1` camera):**
    *   Configure at least one camera in `cameras` (e.g., `wrist` at `640×480 @ 30 fps`, or `fixed`/`overhead` at `1280×720 @ 30 fps`).
    *   Sufficient for Stages 0–3 (including Stage 3 "Hello World").
*   **Recommended Setup (`2` cameras):**
    *   `fixed` (or `overhead`): Stationary camera viewing the arm and tabletop workspace (`1280×720 @ 30 fps`). Required for Stage 5 (Hand-Eye & Table Height Calibration) and Phase 2 Pick-and-Place.
    *   `wrist`: Gripper-mounted eye-in-hand camera (`640×480 @ 30 fps`).
*   **USB Bandwidth Guidelines:**
    *   A single USB 2.0 root hub shares `480 Mbps` theoretical (`~320 Mbps` effective isochronous/bulk payload).
    *   Uncompressed `1280×720 @ 30 fps` in `YUYV` (`16 bpp`) consumes `~442.4 Mbps` by itself, saturating a USB 2.0 bus if paired with a second camera.
    *   Request `MJPG` (`fourcc: "MJPG"`) or place the two cameras on separate USB controllers when running dual streams.

## 3. Canonical `config/robot_config.json` Schema

```json
{
  "robot_id": "my_follower",
  "robot_type": "so101_follower",
  "serial_port": "/dev/tty.usbmodem5B3E1187881",
  "baudrate": 1000000,
  "use_degrees": true,
  "calibration_file": "~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower.json",
  "urdf_path": "repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
  "target_frame_name": "gripper_frame_link",
  "cameras": {
    "wrist": {
      "index": 1,
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    },
    "fixed": {
      "index": 0,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "fourcc": "MJPG",
      "reference_image": "config/reference_fixed.png"
    }
  },
  "rest_pose": {
    "shoulder_pan": -2.1,
    "shoulder_lift": -88.5,
    "elbow_flex": 92.0,
    "wrist_flex": 64.5,
    "wrist_roll": 0.5,
    "gripper": 12.0
  },
  "safety": {
    "max_step_deg": 20.0,
    "r5_hard_max_step_deg": 45.0,
    "max_speed_deg_per_sec": 30.0,
    "r5_hard_max_speed_deg_per_sec": 90.0,
    "joint_margin_deg": 5.0,
    "max_frame_gap_sec": 0.3,
    "z_floor_search_m": -0.05,
    "z_table_m": null,
    "drift_threshold_mean_diff": 18.0
  }
}
```
