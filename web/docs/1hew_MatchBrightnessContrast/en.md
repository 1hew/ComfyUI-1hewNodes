# Match Brightness Contrast

**Node Purpose:** Adjusts the brightness and contrast of the source image to match the reference image. Supports matching methods based on histogram or standard statistics, with control over the calculation area (full image or edges).

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `source_image` | Required | IMAGE | - | - | Source image batch to be adjusted |
| `reference_image` | Required | IMAGE | - | - | Reference image batch providing brightness and contrast info |
| `edge_amount` | - | FLOAT | 0.2 | 0.0-8192.0 | Calculation area control. <=0: Full image; <1.0: Edge percentage; >=1.0: Edge pixel width |
| `consistency` | - | COMBO | `lock_first` | `lock_first` / `frame_match` | Temporal consistency. `lock_first`: Lock first frame params for sequence; `frame_match`: Match frame by frame |
| `method` | - | COMBO | `histogram` | `standard` / `histogram` | Matching algorithm. `histogram`: Histogram matching (more precise); `standard`: Mean/Std matching (softer) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Adjusted image batch |

## Features

- **Matching Algorithms**:
  - `histogram`: Mapping based on Cumulative Distribution Function (CDF) of channel histograms, reproducing reference tone distribution more precisely.
  - `standard`: Linear transformation based on mean and standard deviation, preserving more source texture features, yielding softer results.
- **Area Control (`edge_amount`)**:
  - 0: Statistics from the full image.
  - >0: Statistics only from the peripheral edge areas, ignoring the center. Useful for matching ambient atmosphere while ignoring subject differences.
- **Temporal Consistency (`consistency`)**:
  - `lock_first`: Calculates matching parameters only from the first frame pair and applies them to all subsequent frames. Ideal for video processing to prevent flickering.
  - `frame_match`: Calculates matching parameters for each frame individually. Suitable for unrelated image batches.

## Typical Usage

- **Video Style Unification**: Connect video frames to `source_image` and a style reference to `reference_image`, set `consistency=lock_first` to unify video tone with the reference stably.
- **Compositing Blending**: For image compositing, use foreground as `source_image` and background as `reference_image`, set appropriate `edge_amount` (e.g., 0.2) to blend foreground edges with the background tone.

## Notes & Tips

- Always use `lock_first` for video processing to ensure stability.
- Proper use of `edge_amount` prevents subject colors (e.g., clothing) from interfering with the overall tone matching.
