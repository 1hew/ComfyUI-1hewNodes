# Detect Guide Line - Vanishing-point guided line detection

**Node Purpose:** `Detect Guide Line` detects line segments, estimates the vanishing point via intersection clustering, and draws guide lines from segment endpoints to the vanishing point. Outputs the annotated image, a lines-only image, and a binary mask.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `canny_low` | - | FLOAT | 0.2 | 0.0-1.0 | Lower Canny threshold (scaled to 0–255). |
| `canny_high` | - | FLOAT | 0.8 | 0.0-1.0 | Upper Canny threshold; clamped ≥ `canny_low`. |
| `seg_min_len` | - | INT | 40 | 1-300 | Minimum line length for Hough (`minLineLength`). |
| `seg_max_gap` | - | INT | 8 | 1-100 | Maximum gap for Hough (`maxLineGap`). |
| `guide_filter` | - | FLOAT | 0.6 | 0.1-1.0 | Line selection strength (percentile-based). |
| `guide_width` | - | INT | 2 | 1-100 | Drawing thickness (pixels). |
| `cluster_eps` | - | INT | 30 | 1-100 | DBSCAN `eps` for intersection clustering. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Annotated image with red guide lines and vanishing point. |
| `line_image` | IMAGE | Lines-only image on black, with vanishing point. |
| `line_mask` | MASK | Binary mask (`B×H×W`) of guide lines and vanishing point. |

## Features

- Edge detection: Canny on blurred grayscale; HoughLinesP to get segments.
- Vanishing point: extend segments, compute pairwise intersections, cluster with DBSCAN to pick the dominant vanishing point.
- Line selection: scores lines by alignment to vanishing point directions; selects by percentile controlled by `guide_filter`.
- Rendering: draws lines from each endpoint to the vanishing point; adds a red dot at the vanishing point.
- Batch async: per-frame processing offloaded to threads with bounded concurrency.

## Typical Usage

- Highlight perspective lines and estimate a vanishing point for composition guidance.
- Increase `seg_min_len`/decrease `seg_max_gap` to reduce short/noisy segments.
- Tune `cluster_eps` according to image scale so intersection clusters converge.

## Notes & Tips

- When no lines or intersections are found, outputs are valid tensors with zeros.
- `guide_filter` affects the percentile: stronger values select fewer, stronger-aligned lines.