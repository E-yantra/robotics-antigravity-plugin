<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
# SO-101 Stage 5 Calibration Notes & Geometric Model

## 1. Fixed Camera Requirement (Step 0)

Stage 5 (Hand-Eye & Table Height Calibration) requires a **fixed camera** (`"fixed"` or `"overhead"` in `config/robot_config.json`) with a stationary view of the SO-101 arm and tabletop workspace.
*   If the user is running the **minimal 1-camera setup** (wrist camera only), Stage 3 ("Hello World") is the final bring-up milestone and Stage 5 is skipped unless a fixed camera is added.
*   If a laptop built-in camera is used as the fixed camera, moving the laptop lid even slightly invalidates the calibration (`R10`).

## 2. Two-Cue Gripper Marker Detection (`R9`)

`detect_gripper_marker` in [`scripts/detect_marker.py`](../scripts/detect_marker.py) locates a unique marker placed on the gripper **after the wrist joint** (e.g., a Post-it triangle or a royal-blue triangle sticker on the matte-white wrist camera mount plate) using two independent cues:
*   **Cue 1 (Colour + Polygon/Solidity):** Chromatic BGR excess (`(B - G) >= min_b_minus_g`, `(G - R) >= min_g_minus_r`, `B >= min_b`) or custom HSV range (`hsv_lower`..`hsv_upper`) with `3×3` dilation, bounded contour area, and convex-hull solidity (`>= min_solidity`).
*   **Cue 2 (Surround Contrast):** Concentric annulus `r ∈ [inner_radius_px, outer_radius_px]` around the candidate centroid verifying either bright matte-white mount pixels (`white_surround_fraction >= min_white_fraction`) or high local luminance contrast against the surrounding gripper surface.

## 3. Perspective-Regularized `3×4` Projective Model

`fit_projective_handeye` fits a homogeneous `3×4` camera projection matrix `M` mapping 3D robot base-frame end-effector tip coordinates `[X_tip, Y_tip, Z_tip, 1]^T` to fixed/overhead camera pixel coordinates `[u, v, 1]^T`:

```text
[u * w, v * w, w]^T = M @ [X_tip, Y_tip, Z_tip, 1]^T,   with M[2, 3] = 1.0
```

*   **Why `reg_lambda = 500.0` on `M[2, 0..2]`:** Over a compact tabletop workspace (`~15 cm × 15 cm × 3 cm`) viewed from `~0.75 m`, unregularized DLT can place a near-zero projective denominator (`w -> 0`) just outside the calibration box, skewing oblique grid projections. Adding Tikhonov regularization `reg_lambda * M[2, j] = 0` (`j = 0, 1, 2`) eliminates spurious poles across `Z` while keeping full perspective accuracy (`< 3 mm` localization RMSE).
*   **Closed-Form Ray-Plane Unprojection (`pixel_to_world_at_z`):** Given any pixel `(u, v)` and horizontal plane height `Z = z_plane_m` (such as `z_table_m` or `z_table_m + object_height_m`), solving the `2×2` linear system yields exact base-frame `(X, Y)` coordinates for top-down grasp planning.
