# Image Crop Square - Square Cropper

**Node Purpose:** `Image Crop Square` crops an image to a square region centered on the mask’s bounding box (or the image center when the mask is absent), with `scale_factor`, extra padding, and flexible background filling strategies. Supports optional masked pasting (`apply_mask`) and enforces final size to be a multiple of `divisible_by`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Guides crop center and size via its bounding box; when absent/empty, center crop is used. |
| `scale_factor` | - | FLOAT | 1.0 | 0.1–3.0 | Scales square size relative to the bbox or center-based square. |
| `extra_padding` | - | INT | 0 | 0–512 | Additional padding around the cropped region before forming the final square. |
| `fill_color` | - | STRING | 1.0 | grayscale/HEX/RGB/name/`edge`/`average`/`extend`/`mirror`/`mk`/`mask` | Background fill; see “Filling Strategies”. |
| `apply_mask` | - | BOOLEAN | false | - | Paste the cropped region using the mask for per-pixel transparency. |
| `divisible_by` | - | INT | 8 | 1–1024 | Ensures final square width/height is a multiple of this value. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Cropped square image batch; batch shapes are auto-aligned. |

## Features

- Mask-aware square: computes square from mask bbox; falls back to centered square when the mask is empty.
- Size control: `scale_factor` grows/shrinks the square; `extra_padding` adds symmetric borders; final size snaps to `divisible_by`.
- Filling strategies: `extend` (replicate), `mirror` (reflect), `edge` (average color from four edges), `average` (global average), or explicit grayscale/HEX/RGB/name. When `apply_mask` is true, `mk`/`mask` selects average color from the mask region.
- Masked paste: with `apply_mask`, the cropped region is pasted using the local mask for soft edges.
- Batch robustness: if output sizes differ across the batch, images are padded to a common size before stacking.

## Typical Usage

- Face-centric crops: provide a face mask; set `scale_factor≈1.2` to include context; choose `fill_color=extend` for seamless borders.
- Centered squares: omit `mask` to crop around image center; use `average` for harmonized backgrounds.
- Soft edges: enable `apply_mask` to blend the cropped region using the mask’s transparency.

## Filling Strategies (`fill_color`)

- Grayscale: `0.5` becomes mid-gray; auto-converted to RGB.
- HEX: `#FF0000` or `FF0000`.
- RGB: `255,0,0` or `0.5,0.2,0.8` (0–1 auto-converted to 0–255).
- Names: standard color names (e.g., `red`, `white`).
- `extend`: replicate edge pixels to pad.
- `mirror`: reflect edge pixels to pad.
- `edge`: fill using average colors computed from the four edges adjacent to the crop.
- `average`: fill using the global average of the input image.
- `mk`/`mask` (with `apply_mask`): fill using the average color inside the mask region.

## Notes & Tips

- When mask and image sizes differ, the mask is centered and resized/canvas-adjusted before bbox calculation.
- If the computed `final_size` would be non-positive, it falls back to `divisible_by`.
- Edge colors are sampled per side to avoid corner color bias.