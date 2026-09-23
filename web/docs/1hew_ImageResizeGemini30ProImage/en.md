# Image Resize Gemini 3.0 Pro Image - Gemini 3.0 Pro Size Adapter

**Node Purpose:** `Image Resize Gemini 3.0 Pro Image` adapts image/mask inputs to Gemini 3.0 Pro preset resolutions. It supports nearest preset auto-selection and synchronized image/mask outputs under `crop` / `pad` / `stretch` fit modes.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto (2k \| 4k)` | `auto` / `auto (1k)` / `auto (2k)` / `auto (4k)` / `auto (1k \| 2k)` / `auto (2k \| 4k)` / preset resolution entries | Target size selector; `auto` picks the `1k`/`2k`/`4k` tier by input area, then the closest-aspect preset within it; `auto (1k)` etc. fix the tier. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode: crop, pad, or stretch. |
| `pad_color` | - | STRING | `1.0` | grayscale/HEX/RGB/color name/`edge`/`average`/`extend`/`mirror` | Background fill strategy for `pad` mode. |
| `image` | optional | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Input mask batch; transformed in sync with image when provided. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch. |
| `mask` | MASK | Output mask batch aligned to output size. |
| `native_image` | IMAGE | Native-resolution copy of the main output; identical content/crop/padding, different resolution. |
| `native_mask` | MASK | Mask aligned with `native_image`. |

## Features

- Auto preset matching: `auto` picks the `1k`/`2k`/`4k` tier by input area, then matches the closest-aspect preset within it; `auto (1k)` etc. fix the tier.
- Works with partial inputs: accepts image-only or mask-only; when both are absent, outputs a target-sized background image and full-white mask.
- Three fit modes:
  - `crop`: center-crop with aspect preservation, then resize.
  - `pad`: resize with aspect preservation, then center-pad.
  - `stretch`: direct resize to target dimensions.
- Synchronized mask transform: input masks follow the same geometric transform strategy.

## Typical Usage

- Adapt to common Gemini 3.0 Pro sizes: `preset_size=auto (2k | 4k)` with `fit=crop`.
- Preserve full framing with padding: use `fit=pad` and set `pad_color` as needed (e.g. `1.0`, `#000000`, `mirror`).
- Mask-only alignment: connect only `mask`; the node generates a background image and aligned output mask.

## Notes & Tips

- `pad_color` has the strongest effect in `pad` mode; `crop`/`stretch` are primarily geometric transforms.
- If `image` and `mask` sizes differ, pre-aligning dimensions before this node usually gives cleaner edge consistency.
