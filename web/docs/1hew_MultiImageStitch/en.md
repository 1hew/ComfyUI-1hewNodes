# Multi Image Stitch - Directional Concatenation with Spacing

**Node Purpose:** `Multi Image Stitch` stitches multiple images in a chosen direction (`top`, `bottom`, `left`, `right`) with configurable spacing width and color. Supports match-by-resize or match-by-padding and handles batches safely.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `direction` | - | COMBO | `right` | `top` / `bottom` / `left` / `right` | Stitch direction. |
| `match_image_size` | - | BOOLEAN | True | - | Match sizes by resizing when `True`; otherwise pad to unify. |
| `spacing_width` | - | INT | 10 | 0–1000 | Spacer thickness between images. |
| `spacing_color` | - | STRING | `1.0` | grayscale/HEX/RGB/name | Spacer color. |
| `pad_color` | - | STRING | `1.0` | color strategy for padding | Padding strategy or color.
| `image_1` | - | IMAGE | - | - | First image. |
| `image_2…image_N` | optional | IMAGE | - | - | Additional images recognized by numeric suffix ordering.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Stitched image batch; clamped to `[0,1]` float.

## Features

- Iterative stitching: stitches images pairwise in order `image_1..N`.
- Batch broadcasting: repeats smaller batches to match larger ones.
- Two matching modes:
- Resize keep ratio: when `match_image_size=True`.
- Pad to unify: when `match_image_size=False` with `pad_color` strategies.
- Spacers: vertical or horizontal strips with `spacing_color`; `spacing_width=0` yields no gap.
- Advanced padding: `extend`, `mirror`, `edge`, `average`, or explicit colors.

## Typical Usage

- Collage creation: concatenate images into banners or columns with controlled spacing.
- Consistent layouts: use resize matching to keep proportions; use padding to preserve entire content.
- Batch-safe stitching: combine batched images with automatic broadcasting.

## Notes & Tips

- `spacing_color` supports grayscale floats, `R,G,B` in `0..1` or `0..255`, `#hex`, and common names.
- For precise alignment in left/right, target heights are unified; for top/bottom, target widths are unified.