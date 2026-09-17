# Image Resize GPT Image 2.0 - GPT Image 2.0 Size Adapter

**Node Purpose:** `Image Resize GPT Image 2.0` adapts image/mask inputs to GPT Image 2.0 compatible sizes. `auto` first picks the `1k`/`2k`/`4k` tier by input area, then selects the closest-aspect preset within that tier (`auto (1k)` etc. fix the tier); `dynamic*` follows GPT Image 2.0 rules to preserve the input aspect ratio as much as possible.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto (2k)` | `auto*` / `dynamic*` / preset size entries | Target size selector; `auto` picks a tier by area then the closest aspect, `auto (1k)` etc. fix the tier, `dynamic*` preserves input aspect ratio with dynamic scaling, and concrete preset entries output fixed sizes. |
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

- Preset auto: `auto` first picks the `1k`/`2k`/`4k` tier by input area, then matches the closest-aspect preset within it; `auto (1k)` / `auto (2k)` / `auto (4k)` restrict matching to that tier.
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
- Use a fixed landscape output: choose `[2k] 2560x1440 (16:9)` or `[4k] 3840x2160 (16:9)`.
- Preserve full framing with padding: use `fit=pad` and set `pad_color` as needed (e.g. `1.0`, `#000000`, `mirror`).
- Mask-only alignment: connect only `mask`; the node generates a background image and aligned output mask.

## Fixed Preset Size Table

| Ratio | 1k Size | 2k Size | 4k Size |
| ----- | ------- | ------- | ------- |
| `3:1` | `1760x592` | `3456x1152` | `3840x1280` |
| `21:9` | `1552x672` | `2688x1152` | `3840x1648` |
| `2:1` | `1424x736` | `2880x1440` | `3840x1920` |
| `16:9` | `1328x784` | `2560x1440` | `3840x2160` |
| `3:2` | `1232x848` | `2496x1664` | `3520x2352` |
| `4:3` | `1168x896` | `2304x1728` | `3312x2496` |
| `5:4` | `1136x912` | `2240x1792` | `3216x2576` |
| `1:1` | `1024x1024` | `2048x2048` | `2880x2880` |
| `4:5` | `912x1136` | `1792x2240` | `2576x3216` |
| `3:4` | `896x1168` | `1728x2304` | `2496x3312` |
| `2:3` | `848x1232` | `1664x2496` | `2352x3520` |
| `9:16` | `784x1328` | `1440x2560` | `2160x3840` |
| `1:2` | `736x1424` | `1440x2880` | `1920x3840` |
| `9:21` | `672x1552` | `1152x2688` | `1648x3840` |
| `1:3` | `592x1760` | `1152x3456` | `1280x3840` |

## Notes & Tips

- This table is a node preset list derived from GPT Image 2.0 sizing rules, not an official fixed enumeration.
- `auto` picks a tier by input area and the nearest size within it; `auto (1k)` etc. fix the tier; `dynamic*` computes sizes dynamically from the input aspect ratio.
- `4k` outputs may take longer in downstream API nodes; consider increasing `timeout_sec`.
