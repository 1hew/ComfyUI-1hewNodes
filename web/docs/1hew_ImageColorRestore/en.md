# Image Color Restore - Restore Reference Colors

**Node Purpose:** `Image Color Restore` restores the colors of an aligned AI edit back to the original reference image while locally smoothing occlusion seams from object removal. It uses a trusted affine color fit and a continuous seam field (structural unchanged-mask cleanup disabled), preserving newly generated regions. Ported from ComfyUI-RH-Nodes' `Reference Color Restore (Occlusion Seam Advanced) V0.8`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `edit_image` | - | IMAGE | - | - | AI-edited image on an approximately solid background; must be pixel-aligned with `org_image` (resized to the reference size when dimensions differ). |
| `org_image` | - | IMAGE | - | - | Aligned original/reference image used as the color source (no solid-background requirement). |
| `background_threshold` | - | FLOAT | 0.0 | 0–80 (step 0.1) | LAB distance threshold for foreground/background split; `0` = auto. Foreground must land in 0.5%–80% or the node errors. |
| `unchanged_threshold` | - | FLOAT | 10.0 | 0–80 (step 0.1) | Max RGB residual (0–255) for a pixel to count as unchanged; `0` = auto. Higher restores the reference more aggressively. |
| `seam_max_distance` | - | FLOAT | 96.0 | 1–1024 (step 1) | Max propagation distance (px) from the trusted unchanged region. |
| `seam_decay` | - | FLOAT | 64.0 | 1–1024 (step 1) | Distance falloff `exp(-d/decay)`; larger reaches farther. |
| `seam_residual_blur` | - | FLOAT | 24.0 | 0.1–128 (step 0.5) | Gaussian sigma for smoothing the residual field. |
| `seam_color_sigma` | - | FLOAT | 20.0 | 0.1–128 (step 0.5) | Lab color-similarity gate sigma; only propagates to similar colors. |
| `occlusion_residual_blur` | - | FLOAT | 48.0 | 0–128 (step 0.5) | Extra residual blur applied only in the detected occlusion zone; `0` disables the local pass. |
| `occlusion_bridge_width` | - | FLOAT | 48.0 | 0–128 (step 0.5) | Occlusion detection reach / bridge width; `0` disables the local pass. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `corrected_image` | IMAGE | Color-restored image; preserves `edit_image`'s alpha channel when present. |
| `safe_unchanged_mask` | MASK | Mask of confidently unchanged pixels, useful for diagnostics or downstream compositing. |

## Features

- Trusted fit: an affine (quadratic) color fit is solved only on confidently unchanged pixels, keeping real edits from contaminating the model.
- Occlusion seams: a continuous seam field locally smooths seams left by removed/occluded objects.
- Alpha preservation: restoration operates on RGB only; the `edit_image` alpha channel is kept intact.
- Batch support: batches are supported; a single-image side is broadcast across all frames.
- Safe failure: errors are raised when foreground segmentation or the unchanged ratio is implausible, avoiding silently corrupted output.

## Typical Usage

- Recolor a local inpaint/AI edit back to the original palette while keeping the new content.
- Compensate for slight global color drift introduced by editing models beyond the local edit.
- Combine with `Image Align Change Mask`: align first, then restore colors with this node.

## Notes & Tips

- `edit_image` must place the subject on an approximately solid background, otherwise foreground segmentation (and the whole fit) fails.
- The two images must be essentially pixel-aligned; on size mismatch `edit_image` is resized to `org_image`.
- Defaults suit most cases; if restoration is too aggressive, lower `unchanged_threshold` (`0` = auto).
