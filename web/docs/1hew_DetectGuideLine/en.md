# Detect Guide Line

**Node Function:** The `Detect Guide Line` node detects prominent perspective guide lines and estimates a vanishing point using edge detection, line segment extraction, intersection clustering, and percentile-based line filtering. Outputs the annotated image, a lines-only overlay, and a binary mask of the guide lines.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image tensor |
| `canny_low` | Required | FLOAT | 0.2 | 0.0–1.0 | Canny lower threshold (scaled to 0–255) |
| `canny_high` | Required | FLOAT | 0.8 | 0.0–1.0 | Canny upper threshold (scaled to 0–255, clamped ≥ lower) |
| `seg_min_len` | Required | INT | 40 | 1–300 | Minimum line length for `HoughLinesP` |
| `seg_max_gap` | Required | INT | 8 | 1–100 | Maximum gap allowed in line segments |
| `guide_filter` | Required | FLOAT | 0.6 | 0.1–1.0 | Line selection strictness; higher keeps more lines (percentile threshold = `100 - guide_filter*60`) |
| `guide_width` | Required | INT | 2 | 1–100 | Render width of guide lines and vanishing point marker |
| `cluster_eps` | Required | INT | 30 | 1–100 | DBSCAN `eps` (pixels) for intersection clustering |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Input image annotated with guide lines to the vanishing point |
| `line_image` | IMAGE | Lines-only overlay (black background with red guide lines and vanishing point) |
| `line_mask` | MASK | Binary mask of guide lines (255 along lines, 0 elsewhere) |
