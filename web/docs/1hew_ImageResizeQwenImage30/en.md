# Image Resize Qwen Image 3.0 - Qwen Image 3.0 Size Adapter

**Node Purpose:** `Image Resize Qwen Image 3.0` adapts image/mask inputs to Qwen Image 3.0 compatible sizes. `auto` first picks the `1k`/`2k` tier by input area, then selects the closest-aspect preset within that tier (`auto (1k)` etc. fix the tier); `dynamic*` follows Qwen Image 3.0 rules to preserve the input aspect ratio as much as possible.

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

- Preset auto: `auto` first picks the `1k`/`2k` tier by input area, then matches the closest-aspect preset within it; `auto (1k)` / `auto (2k)` restrict matching to that tier.
- Dynamic mode: `dynamic` selects the closest `1k` / `2k` tier by input area and dynamically scales while preserving input aspect ratio; `dynamic (1k)` / `dynamic (2k)` fixes the tier.
- Fixed presets: provides stable common-aspect sizes for the `1k` and `2k` tiers when a precise output size is needed.
- Qwen Image 3.0 rules: both dynamic and preset sizes are handled with 16-pixel alignment, safe aspect coverage (1:8 ~ 8:1), and a longest-edge limit of 4096.
- Works with partial inputs: accepts image-only or mask-only; when both are absent, outputs a target-sized background image and full-white mask.
- Three fit modes:
  - `crop`: center-crop with aspect preservation, then resize.
  - `pad`: resize with aspect preservation, then center-pad.
  - `stretch`: direct resize to target dimensions.

## Typical Usage

- Prepare an input with common presets for Qwen Image 3.0 text-to-image: `preset_size=auto (2k)` with `fit=crop`.
- Preserve the source aspect ratio as much as possible: `preset_size=dynamic (2k)` with `fit=pad` or `fit=crop`.
- Use a fixed landscape output: choose `[2k] 2560x1440 (16:9)` or `[1k] 1280x720 (16:9)`.
- Preserve full framing with padding: use `fit=pad` and set `pad_color` as needed (e.g. `1.0`, `#000000`, `mirror`).
- Mask-only alignment: connect only `mask`; the node generates a background image and aligned output mask.

## Fixed Preset Size Table

| Ratio | 1k Size | 2k Size |
| ----- | ------- | ------- |
| `3:1` | `1728x576` | `3456x1152` |
| `21:9` | `1344x576` | `2688x1152` |
| `2:1` | `1440x720` | `2880x1440` |
| `16:9` | `1280x720` | `2560x1440` |
| `3:2` | `1248x832` | `2496x1664` |
| `4:3` | `1152x864` | `2304x1728` |
| `5:4` | `1120x896` | `2240x1792` |
| `1:1` | `1024x1024` | `2048x2048` |
| `4:5` | `896x1120` | `1792x2240` |
| `3:4` | `864x1152` | `1728x2304` |
| `2:3` | `832x1248` | `1664x2496` |
| `9:16` | `720x1280` | `1440x2560` |
| `1:2` | `720x1440` | `1440x2880` |
| `9:21` | `576x1344` | `1152x2688` |
| `1:3` | `576x1728` | `1152x3456` |

## Notes & Tips

- This table is a node preset list derived from Qwen Image 3.0 sizing rules, not an official fixed enumeration.
- Qwen Image 3.0 official limits: pixel area `512x512` ~ `2048x2048` and aspect ratio `1:8` ~ `8:1`, so this node intentionally has **no `4k` tier**.
- The `1k` / `2k` presets are **identical** to `Image Resize GPT Image 2.0` (the extra `4k` tier of that node is not available on Qwen).
- Differences from `Image Resize GPT Image 2.0` are limited to dynamic parameters: safe aspect range is `1:8` ~ `8:1` (GPT uses `1:3` ~ `3:1`) and the longest-edge limit is `4096` (GPT uses `3840`).
- Note: Qwen Image 3.0 **image editing (edits) ignores the `size` parameter**; the output resolution is decided upstream (~2K). In that case this node is still useful for shaping the reference image aspect ratio, which indirectly influences the output aspect ratio.
- `auto` picks a tier by input area and then the closest aspect from that tier's table; `auto (1k)` etc. fix the tier; `dynamic*` computes from the input aspect ratio.
