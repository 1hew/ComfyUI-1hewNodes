# Match Brightness Contrast - Brightness and Contrast Matching

**Node Purpose:** `Match Brightness Contrast` adjusts the brightness and contrast of a source image to match a reference image. It provides options to calculate statistics based only on edge areas to ignore central content changes, making it suitable for seamless image blending or style transfer tasks.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `source_image` | required | IMAGE | - | - | The source image batch to be adjusted. |
| `reference_image` | required | IMAGE | - | - | The reference image batch used as the target for brightness and contrast. |
| `edge_amount` | - | FLOAT | 0.2 | 0.0 - 8192.0 | Controls the margin for statistics calculation. <= 0: Use full image; 0 < amount < 1.0: Percentage of short side; >= 1.0: Pixel count. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | The adjusted image batch with brightness and contrast matched to the reference. |

## Features

- **Brightness & Contrast Matching**: Automatically adjusts the source image's mean (brightness) and standard deviation (contrast) to align with the reference image.
- **Edge-Based Statistics**: The `edge_amount` parameter allows users to specify whether to use the entire image or only the edge regions for calculating color statistics. This is particularly useful when the central content differs significantly but the background/lighting should match.
- **Flexible Margin Control**: Supports both percentage-based (relative to the shortest side) and absolute pixel-based margin definitions.
- **Batch Processing**: Supports batch processing for both source and reference images. If batch sizes differ, the reference images are cycled.

## Usage Tips

- **Seamless Blending**: When compositing an object into a new background, use the background as the `reference_image` and the object as the `source_image`. Setting `edge_amount` to 0 (full image) or a small value can help align their lighting conditions.
- **Style Transfer**: Use a reference image with a specific mood or lighting style to transfer its global brightness and contrast characteristics to the source image.
- **Edge Amount**:
    - Set to `0` or less to use the global statistics of the entire image.
    - Set to a value between `0` and `1` (e.g., `0.2`) to use a percentage of the image's shortest side as the margin width.
    - Set to a value `>= 1` to specify the exact margin width in pixels.
