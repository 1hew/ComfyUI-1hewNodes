# Image Resize FluxKontext - Preset model resolutions

**Node Purpose:** `Image Resize FluxKontext` resizes images and masks to model-aligned preset resolutions used by FluxKontext. Supports `auto` closest aspect selection, fit modes (`crop`, `pad`, `stretch`), and rich `pad_color` strategies.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto` | presets | Target resolution; `auto` picks closest aspect preset. |
| `fit` | - | COMBO | `crop` | `crop`/`pad`/`stretch` | Fit strategy to reach target size. |
| `pad_color` | - | STRING | `1.0` | Gray/HEX/RGB/`edge`/`average`/`extend`/`mirror` | Background strategy for padding. |
| `image` | optional | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Input mask batch; strictly aligned to image or used standalone. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch at preset size. |
| `mask` | MASK | Resized mask aligned to output geometry.

## Features

- `auto` preset: selects the preset whose aspect ratio is closest to the input image/mask.
- Fit modes:
  - `crop`: center-crop to target aspect then resize; outputs an origin-space mask marking the crop window.
  - `pad`: preserve aspect, center-pad with configurable background and output a mask marking original content region.
  - `stretch`: direct resize for both image and mask.
- Mask handling: converts masks to 3D (`B×H×W`), uses nearest-neighbor for mask resampling, and preserves `[0,1]`.
- Color parsing: supports gray (`0..1`), HEX, `R,G,B`, and named strategies (`edge`/`average`/`extend`/`mirror`).

## Typical Usage

- Align to model: set `preset_size` to a FluxKontext preset or leave `auto`.
- Preserve content: use `fit=pad` and `pad_color=edge`/`average` for natural-looking borders.
- Strict content region: use `fit=crop` to remove excess content cleanly.

## Notes & Tips

- When only `mask` is provided, the node produces a solid-color image using `pad_color` and resizes the mask accordingly.
- All outputs are clamped to `[0,1]` and returned in channels-last layout.