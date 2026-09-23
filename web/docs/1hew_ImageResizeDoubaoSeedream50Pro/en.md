# Image Resize Doubao Seedream 5.0 Pro - Doubao Seedream 5.0 Pro Size Adapter

**Node Purpose:** `Image Resize Doubao Seedream 5.0 Pro` adapts image/mask inputs to exact pixels from the Doubao Seedream 5.0 Pro official size table, and is designed to pair with the 1hewOffice `Doubao Seedream 5.0 Pro` API node. `auto` first picks the `1k`/`1.5k`/`2k` tier by input area, then selects the closest-aspect preset within that tier (`auto (1k)` etc. fix the tier); `dynamic*` preserves the input aspect ratio as much as possible.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `preset_size` | - | COMBO | `auto (2k)` | `auto*` / `dynamic*` / preset size entries | Target size selector; `auto` picks a tier by area then the closest aspect, `auto (1k)` etc. fix the tier, `dynamic*` scales dynamically, explicit presets output a fixed size |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode: crop, pad, stretch |
| `pad_color` | - | STRING | `1.0` | gray / HEX / RGB / color name / `edge` / `average` / `extend` / `mirror` | Background fill strategy for `pad` |
| `image` | optional | IMAGE | - | - | Input image batch |
| `mask` | optional | MASK | - | - | Input mask batch, transformed together with the image |

## Outputs

| Name | Type | Description |
| ---- | ---- | ----------- |
| `image` | IMAGE | Resized image batch |
| `mask` | MASK | Mask batch aligned to the output size |
| `native_image` | IMAGE | Native-resolution copy of the main output; identical content/crop/padding, different resolution only |
| `native_mask` | MASK | Mask aligned to `native_image` |

## Behavior

- Preset auto: `auto` picks the `1k` / `1.5k` / `2k` tier by input area, then the closest aspect within it; `auto (1k)` / `auto (1.5k)` / `auto (2k)` fix the tier.
- Dynamic: `dynamic` picks the nearest tier by area and scales while preserving the input aspect ratio; `dynamic (1k)` etc. fix the tier.
- Fixed presets: stable sizes for 8 aspect ratios across the `1k` / `1.5k` / `2k` tiers.
- Rules: dynamic sizes are 16-pixel aligned, clamped to the `1:16` ~ `16:1` aspect range and a `4096` longest edge.
- Works without image or mask: either input alone is enough; with neither, it outputs a target-size background and an all-white mask.
- Fit modes: `crop` (center-crop then scale), `pad` (scale then center-pad), `stretch` (direct resize).

## Typical Usage

- Align a reference image to official sizes: pick a `preset_size` here, wire `image` to the API node's `image_1`, set `aspect_ratio=auto` and the same `resolution` tier there, and the generated size matches this node exactly.
- Fixed landscape size: choose `[2k] 2816x1584 (16:9)` or `[1k] 1424x800 (16:9)`.
- Keep the whole frame with padding: `fit=pad` with a `pad_color` such as `1.0`, `#000000` or `mirror`.
- Mask-only canvas: connect only `mask` to get a background plus an aligned mask.

## Fixed Preset Size Table

| Ratio | 1k | 1.5k | 2k |
| ----- | -- | ---- | -- |
| `21:9` | `1568x672` | `2352x1008` | `3136x1344` |
| `16:9` | `1424x800` | `2048x1152` | `2816x1584` |
| `3:2` | `1248x832` | `1872x1248` | `2496x1664` |
| `4:3` | `1152x864` | `1792x1344` | `2368x1776` |
| `1:1` | `1024x1024` | `1536x1536` | `2048x2048` |
| `3:4` | `864x1152` | `1344x1792` | `1776x2368` |
| `2:3` | `832x1248` | `1248x1872` | `1664x2496` |
| `9:16` | `800x1424` | `1152x2048` | `1584x2816` |

(Ordered widest to tallest, matching the API node's `aspect_ratio` option order.)

## Notes

- This table is **item-for-item identical** to the `SIZE_MAP` class attribute of `DoubaoSeedream50Pro` in the external 1hewOffice `Doubao Seedream 5.0 Pro` API node: 3 tiers x 8 ratios = 24 entries, verified one by one. That API node ships with 1hewOffice, so its source file is not present in this repository.
- The Doubao Seedream 5.0 Pro API node has **no `get_image_size` input** (unlike Qwen Image 3.0 Pro / GPT Image 2.0). So this node can only feed `image` into the API node's `image_1` and let `aspect_ratio=auto` infer the ratio from the reference image; for **text-to-image (no reference) you cannot set the output size here** - use the API node's `aspect_ratio` + `resolution` instead.
- Only 8 ratios exist: `21:9` / `16:9` / `3:2` / `4:3` / `1:1` / `3:4` / `2:3` / `9:16`, matching the API node's `aspect_ratio` options. There is no `5:4`, `3:1` or `2:1`.
- `dynamic*` produces sizes outside the official table; fed back into the API node, `auto` snaps the ratio to the nearest official tier and switches to a table size, so the **generated size will not exactly match this node** (the ratio is roughly preserved). Use `auto (...)` or an explicit preset for an exact match.
- The `1k` / `2k` presets here **differ** from Qwen Image 3.0 Pro / GPT Image 2.0 (each model has its own official table); do not mix them.
