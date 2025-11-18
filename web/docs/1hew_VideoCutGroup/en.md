# Video Cut Group - Detect scene cuts and group frames

**Node Purpose:** `Video Cut Group` detects hard scene changes in a sequence of frames and groups the video into segments. Supports fast simplified SSIM mode, multi-kernel blurred SSIM detection with dynamic thresholds, post-grouping rules (`min_frame_count`, `max_frame_count`), and user edits via `add_frame`/`delete_frame`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input video frames as an image batch. |
| `threshold_base` | - | FLOAT | 0.8 | 0.0-1.0 | Base threshold applied to `1-SSIM`; larger values mean stricter cuts. |
| `threshold_range` | - | FLOAT | 0.05 | 0.01-0.2 | Range around `threshold_base` to generate multiple thresholds. |
| `threshold_count` | - | INT | 2 | 1-10 | Number of thresholds to use within the range. |
| `kernel` | - | STRING | `3, 7, 11` | odd sizes | Comma-separated Gaussian kernel sizes; non-negative odd integers ≥3. |
| `min_frame_count` | - | INT | 10 | 1-1000 | Minimum segment length; merges cuts that are too close. |
| `max_frame_count` | - | INT | 0 | 0-10000 | Maximum segment length; `0` disables splitting of long segments. |
| `fast` | - | BOOLEAN | `False` | - | Enable simplified SSIM detection for speed. |
| `add_frame` | - | STRING | `` | comma list | Manually add cut frames; accepts English/Chinese commas. |
| `delete_frame` | - | STRING | `` | comma list | Remove cut frames (except `0`); accepts English/Chinese commas. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Frames at each group start index (keyframes). |
| `group_total` | INT | Number of groups (segments). |
| `start_index` | LIST(INT) | Start frame indices for each segment (always includes `0`). |
| `batch_count` | LIST(INT) | Frame counts per segment.

## Features

- Preprocessing: converts to `float32`, scales `[0,1]→[0,255]` if needed, and reduces to grayscale.
- Multi-kernel detection: computes blurred SSIM across adjacent frames for each kernel; evaluates multiple thresholds.
- Fast mode: simplified SSIM for quick detection; still applies grouping rules afterward.
- Unified fusion: merges detected cut points from all configurations; applies `min_frame_count` and `max_frame_count` consistently.
- User edits: add or delete cuts using comma-separated indices; `0` is always preserved.
- Robust outputs: returns keyframe batch along with segment metadata.

## Typical Usage

- Scene segmentation for downstream processing: tune `threshold_base` and `min_frame_count` to balance sensitivity and stability.
- Keyframe extraction: use the output `image` (start frames) as thumbnails or anchors.
- Manual refinement: adjust cuts using `add_frame` and `delete_frame`.

## Notes & Tips

- Larger `threshold_base` values imply stricter detection (fewer cuts) because comparison uses `1-SSIM > threshold`.
- When `max_frame_count > 0`, very long segments are split at fixed intervals within the limit.