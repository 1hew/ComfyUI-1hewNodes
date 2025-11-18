# Image Resize QwenImage - Preset-Driven Resize with Fit Modes

**Node Purpose:** `Image Resize QwenImage` resizes images and/or masks to predefined resolutions optimized for QwenImage workflows. Supports `auto` preset selection based on aspect ratio, three fit modes (`crop`, `pad`, `stretch`), advanced `pad_color` strategies, and synchronized mask output.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto` | `auto` / `928×1664 [1:1.79]` / `1056×1584 [1:1.50] (2:3)` / `1140×1472 [1:1.29]` / `1328×1328 [1:1.00] (1:1)` / `1472×1140 [1.29:1]` / `1584×1056 [1.50:1] (3:2)` / `1664×928 [1.79:1]` | Target resolution preset. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Resize mode. |
| `pad_color` | - | STRING | 1.0 | grayscale/HEX/RGB/name/`edge`/`average`/`extend`/`mirror` | Background strategy used in `pad` mode. |
| `image` | optional | IMAGE | - | - | Input image batch; can be omitted when resizing masks only or generating solid backgrounds. |
| `mask` | optional | MASK | - | - | Input mask batch; synchronized to output size. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch; solid background when no image is provided. |
| `mask` | MASK | Synchronized mask batch according to fit mode.

## Features

- Auto preset: selects the closest preset to the input aspect ratio from image or mask; defaults to `1328×1328` when no inputs.
- Fit modes:
- `stretch`: direct resize via bicubic; mask resized via nearest; outputs full-white mask when image-only.
- `pad`: center-pad preserving aspect; image padded via `pad_color`; mask marks original content as white and padding as black.
- `crop`: center-crop to target aspect, then resize; mask marks the cropped area on the original and is resized to output.
- Mask-only paths: produce a solid background image using `pad_color` and resize/synchronize the mask.
- Advanced padding: `extend` (replicate), `mirror` (reflect with segmented extension), `edge` (per-side averages), `average` (global average), or explicit colors.

## Typical Usage

- Preset alignment: set `preset_size` to a listed resolution to match model expectations; use `auto` to infer from input aspect.
- Safe padding backgrounds: choose `pad_color=extend/mirror` for natural borders, `edge` to harmonize colors, or `average` for uniform tone.
- Mask synchronization: when resizing masks only, leverage `pad` to center and mark content appropriately.

## Notes & Tips

- Masks are normalized to 3D `[B,H,W]` via `_ensure_mask_3d` before processing.
- `mirror` performs segmented reflections to handle large padding sizes without artifacts.
- All outputs are clamped to `[0,1]` float32 and remain device-safe.