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
| `3:1` | `1728x576` | `3456x1152` | `3840x1280` |
| `21:9` | `1344x576` | `2688x1152` | `3808x1632` |
| `2:1` | `1440x720` | `2880x1440` | `3840x1920` |
| `16:9` | `1280x720` | `2560x1440` | `3840x2160` |
| `3:2` | `1248x832` | `2496x1664` | `3504x2336` |
| `4:3` | `1152x864` | `2304x1728` | `3264x2448` |
| `5:4` | `1120x896` | `2240x1792` | `3200x2560` |
| `1:1` | `1024x1024` | `2048x2048` | `2880x2880` |
| `4:5` | `896x1120` | `1792x2240` | `2560x3200` |
| `3:4` | `864x1152` | `1728x2304` | `2448x3264` |
| `2:3` | `832x1248` | `1664x2496` | `2336x3504` |
| `9:16` | `720x1280` | `1440x2560` | `2160x3840` |
| `1:2` | `720x1440` | `1440x2880` | `1920x3840` |
| `9:21` | `576x1344` | `1152x2688` | `1632x3808` |
| `1:3` | `576x1728` | `1152x3456` | `1280x3840` |

## Notes & Tips

- This table is a node preset list derived from GPT Image 2.0 sizing rules, not an official fixed enumeration.
- All three tiers are **exactly proportional**: `1k` / `2k` are built as `(a×k, b×k)` with `k` a multiple of 16; `4k` takes the **largest exact value** within the official constraints **and** this node's own `3840` longest-edge cap.
- The three official custom-size rules (`GPT Image 2`): both sides a **multiple of 16**, aspect ratio **≤ 3:1**, total pixels **655,360 – 8,294,400**. Every entry in this table satisfies them; the `3840` longest-edge cap is an additional node-level limit (not one of the three official rules), which is what stops a `4k` entry such as `2:1` from growing past `3840x1920`.
- `auto` picks a tier by input area and the nearest size within it; `auto (1k)` etc. fix the tier; `dynamic*` computes sizes dynamically from the input aspect ratio.
- `4k` outputs may take longer in downstream API nodes; consider increasing `timeout_sec`.
