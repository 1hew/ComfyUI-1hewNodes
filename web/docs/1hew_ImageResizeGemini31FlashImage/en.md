# Image Resize Gemini31FlashImage - Gemini 3.1 Flash Size Adapter

**Node Purpose:** `Image Resize Gemini31FlashImage` reuses the Gemini30 resizing behavior while extending preset coverage (including `0.5k` / `1k` / `2k` / `4k` tiers). It is intended for image/mask size normalization before Gemini 3.1 Flash workflows.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto (2k \| 4k)` | `auto` / `auto (0.5k)` / `auto (1k \| 2k)` / `auto (2k \| 4k)` / preset resolution entries | Target size selector; `auto (0.5k)` matches within `[512]` presets first. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode: crop, pad, or stretch. |
| `pad_color` | - | STRING | `1.0` | grayscale/HEX/RGB/color name/`edge`/`average`/`extend`/`mirror` | Background fill strategy for `pad` mode. |
| `image` | optional | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Input mask batch. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch. |
| `mask` | MASK | Output mask batch aligned to output size. |

## Features

- Extended presets: adds more extreme aspect ratios and `0.5k` auto tier compared with Gemini30 version.
- Auto selection: `auto*` modes pick nearest presets by aspect ratio and area.
- Same fit behavior: `crop` / `pad` / `stretch` remain consistent, including synchronized mask transforms.
- Flexible inputs: supports image-only, mask-only, and no-input fallback.

## Typical Usage

- Fast low-resolution preprocessing: `preset_size=auto (0.5k)`.
- Broad size coverage: `preset_size=auto (2k | 4k)` with `fit=crop`.
- Preserve full composition: use `fit=pad` and choose an appropriate `pad_color`.

## Notes & Tips

- Prefer `auto (0.5k)` or `auto (1k | 2k)` for speed-sensitive pipelines.
- For detail retention, use higher tiers and avoid aggressive `stretch` when possible.
