# Image Solid - Generate solid-color canvas

**Node Purpose:** `Image Solid` creates solid-color images with optional alpha scaling, inversion, and mask opacity. Supports preset sizes or custom width/height, optional divisibility constraints, and size inference from a reference image.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `get_image_size` | optional | IMAGE | - | - | Reference image for size inference when generating per-frame canvases. |
| `preset_size` | - | COMBO | `custom` | presets | Target size preset; when not `custom`, overrides `width`/`height`. |
| `width` | - | INT | 1024 | 1–8192 | Target width when `preset_size=custom`. |
| `height` | - | INT | 1024 | 1–8192 | Target height when `preset_size=custom`. |
| `color` | - | STRING | `1.0` | Gray/HEX/RGB | Base color; supports `0..1` gray, `R,G,B`, HEX, and names. |
| `alpha` | - | FLOAT | 1.0 | 0.0–1.0 | Global alpha applied to `color` channels. |
| `invert` | - | BOOLEAN | False | - | Invert color channels before applying `alpha`. |
| `mask_opacity` | - | FLOAT | 1.0 | 0.0–1.0 | Mask intensity for output mask. |
| `divisible_by` | - | INT | 8 | 1–1024 | Enforce size to be multiples of this value. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Solid-color image batch. |
| `mask` | MASK | Corresponding mask batch at `mask_opacity`.

## Features

- Flexible sizing: use `preset_size` or custom `width`/`height`; optionally infer size from `get_image_size` per frame.
- Color parsing: supports gray (`0..1`), HEX (`#RRGGBB`), `R,G,B`, and named colors.
- Alpha/invert: applies `alpha` scaling; `invert=True` flips channels before alpha.
- Divisibility: rounds up dimensions to be divisible by `divisible_by`.

## Typical Usage

- Create a background canvas with a specific color and mask intensity.
- Generate per-frame solid canvases matching a reference batch via `get_image_size`.
- Prepare model-sized inputs quickly using presets and `divisible_by`.

## Notes & Tips

- Mask is full ones scaled by `mask_opacity` and matches the image size.
- Color values are applied in channels-last layout and clamped to `[0,1]`.