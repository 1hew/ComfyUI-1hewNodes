# Image Resize Universal - Universal Image Resizer

**Node Purpose:** `Image Resize Universal` performs unified resizing under multiple aspect ratio sources and fit modes, with complete and consistent mask output. It supports ratio presets, target-side control, resampling method selection, size multiple constraints, and multiple padding background strategies.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | optional | IMAGE | - | - | Input image batch; when absent, target size can be inferred via `preset_ratio` / `get_image_size`. |
| `mask` | optional | MASK | - | - | Input mask batch; when size matches the image, it is resized in sync; when absent, a default mask is generated. |
| `get_image_size` | optional | IMAGE | - | - | Reference image (first frame only) used to infer target size. |
| `preset_ratio` | - | COMBO | `origin` | `origin` / `custom` / `1:1` / `3:2` / `4:3` / `16:9` / `21:9` / `2:3` / `3:4` / `9:16` / `9:21` | Aspect ratio source; `origin` uses input size, `custom` uses the proportional parameters below. |
| `proportional_width` | - | INT | 1 | 1-8192 | Ratio width in `custom` mode. |
| `proportional_height` | - | INT | 1 | 1-8192 | Ratio height in `custom` mode. |
| `method` | - | COMBO | `lanczos` | `nearest` / `bilinear` / `lanczos` / `bicubic` / `hamming` / `box` | Resampling method. |
| `scale_to_side` | - | COMBO | `None` | `None` / `longest` / `shortest` / `width` / `height` / `length_to_sq_area` | Target-side control; e.g., set the longest/shortest side or explicit width/height. |
| `scale_to_length` | - | INT | 1024 | 1-8192 | Target length for the selected `scale_to_side` mode. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode: crop, pad, or stretch. |
| `pad_color` | - | STRING | 1.0 | grayscale/HEX/RGB/`edge`/`average`/`extend`/`mirror` | Padding background strategy; see “Padding Strategies”. |
| `divisible_by` | - | INT | 8 | 1-1024 | Ensure output width/height round up to a multiple of this value (often required by models). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch; can generate solid background images in no-image scenarios. |
| `mask` | MASK | Mask batch matched to output size; generated according to rules below. |

## Features

- Ratio inference: supports original size, standard ratios, and custom ratios; combined with target-side control to compute target width/height.
- Size constraints: `divisible_by` ensures the output dimensions are integer multiples of the given value.
- Resampling methods: `nearest`, `bilinear`, `lanczos`, `bicubic`, `hamming`, `box` to balance quality and performance.
- Fit modes:
  - `crop`: keep aspect, then center-crop to target size.
  - `pad`: keep aspect, then center-pad to target size; strictly distinguishes original and padded areas.
  - `stretch`: directly stretch to target size.
- No-image input: infer target size via `preset_ratio` / `scale_to_side` or `get_image_size`, and generate a background image.
- Mask generation: outputs meaningful masks for all valid input combinations; synchronized resizing when a mask is provided.

## Mask Rules

- `pad` mode: output mask matches target size; original image area is white (255), padded area is black (0).
- `crop` mode: mask expresses the cropped region (white) based on the original, then converted to output.
- `stretch` mode: mask matches the output size and is fully white (255).
- Default masks: when no input mask is provided, a corresponding mask is generated per output image.

## Padding Strategies (`pad_color`)

- Grayscale value: e.g., `0.5` indicates 50% gray and is converted to RGB.
- HEX: e.g., `#FF0000` or `FF0000`.
- RGB: e.g., `255,0,0` or `0.5,0.2,0.8` (0–1 values are auto-converted).
- `edge`: fill with average edge color (top/bottom or left/right depending on orientation).
- `average`: fill with the global average color of the image.
- `extend`: replicate edge pixels (`replicate`).
- `mirror`: reflect edge pixels (`reflect`, with segmented extension support).

## Typical Usage

- Preserve aspect while controlling the longest side: set `preset_ratio=origin`, `scale_to_side=longest`, and `scale_to_length` as needed.
- Align inputs to model constraints: set `fit=pad`, and `divisible_by=8/16` to meet model size alignment.
- Batch resize with synchronized masks: connect both `image` and `mask` so the node resizes them consistently and outputs corresponding batches.

## Notes & Tips

- Matching image and mask sizes yields the most stable synchronized resizing; providing only masks can also serve as a size reference.
- In no-image scenarios, provide `get_image_size` or explicit ratio/target-side settings to ensure deterministic output size.