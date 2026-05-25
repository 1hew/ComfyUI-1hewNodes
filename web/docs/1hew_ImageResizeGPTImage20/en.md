# Image Resize GPT Image 2.0 - GPT Image 2.0 Size Adapter

**Node Purpose:** `Image Resize GPT Image 2.0` adapts image/mask inputs to GPT Image 2.0 compatible sizes. `auto*` stays consistent with Gemini resize nodes and picks the nearest fixed preset, while `dynamic*` follows GPT Image 2.0 rules to preserve the input aspect ratio as much as possible.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto (2k)` | `auto*` / `dynamic*` / preset size entries | Target size selector; `auto*` matches fixed presets, `dynamic*` preserves input aspect ratio with dynamic scaling, and concrete preset entries output fixed sizes. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode: crop, pad, or stretch. |
| `pad_color` | - | STRING | `1.0` | grayscale/HEX/RGB/color name/`edge`/`average`/`extend`/`mirror` | Background fill strategy for `pad` mode. |
| `image` | optional | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Input mask batch; transformed in sync with image when provided. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image batch. |
| `mask` | MASK | Output mask batch aligned to output size. |

## Features

- Preset auto: `auto` matches across all presets; `auto (1k)` / `auto (2k)` / `auto (4k)` restricts matching to the selected preset tier.
- Dynamic mode: `dynamic` selects the closest `1k` / `2k` / `4k` tier by input area and dynamically scales while preserving input aspect ratio; `dynamic (1k)` / `dynamic (2k)` / `dynamic (4k)` fixes the tier.
- Fixed presets: provides stable common-aspect sizes for `1k`, `2k`, and `4k` tiers when a precise output size is needed.
- GPT Image 2.0 rules: both dynamic and preset sizes are handled with 16-pixel alignment, safe aspect coverage, and longest-edge limit of 3840.
- Works with partial inputs: accepts image-only or mask-only; when both are absent, outputs a target-sized background image and full-white mask.
- Three fit modes:
  - `crop`: center-crop with aspect preservation, then resize.
  - `pad`: resize with aspect preservation, then center-pad.
  - `stretch`: direct resize to target dimensions.

## Typical Usage

- Prepare an input with common presets for GPT Image 2.0 editing: `preset_size=auto (2k)` with `fit=crop`.
- Preserve the source aspect ratio as much as possible: `preset_size=dynamic (2k)` with `fit=pad` or `fit=crop`.
- Use a fixed landscape output: choose `[2k] 2048x1152 (16:9)` or `[4k] 3840x2160 (16:9)`.
- Preserve full framing with padding: use `fit=pad` and set `pad_color` as needed (e.g. `1.0`, `#000000`, `mirror`).
- Mask-only alignment: connect only `mask`; the node generates a background image and aligned output mask.

## Fixed Preset Size Table

| Ratio | 1k Size | 2k Size | 4k Size |
| ----- | ------- | ------- | ------- |
| `9:21` | `432x1008` | `864x2016` | `1648x3840` |
| `9:16` | `576x1024` | `1152x2048` | `2160x3840` |
| `2:3` | `688x1024` | `1376x2048` | `2368x3488` |
| `3:4` | `768x1024` | `1536x2048` | `2496x3312` |
| `4:5` | `816x1024` | `1632x2048` | `2576x3216` |
| `1:1` | `1024x1024` | `1920x1920` | `2880x2880` |
| `5:4` | `1024x816` | `2048x1632` | `3216x2576` |
| `4:3` | `1024x768` | `2048x1536` | `3312x2496` |
| `3:2` | `1024x688` | `2048x1376` | `3488x2368` |
| `16:9` | `1024x576` | `2048x1152` | `3840x2160` |
| `21:9` | `1008x432` | `2016x864` | `3840x1648` |

## Notes & Tips

- This table is a node preset list derived from GPT Image 2.0 sizing rules, not an official fixed enumeration.
- `auto*` picks the nearest size from this table; `dynamic*` computes sizes dynamically from the input aspect ratio.
- `4k` outputs may take longer in downstream API nodes; consider increasing `timeout_sec`.
