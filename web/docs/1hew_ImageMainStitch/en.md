# Image Main Stitch - Main-image anchored stitching

**Node Purpose:** `Image Main Stitch` stitches `image_2..image_N` into a group, then attaches the group to `image_1` along a chosen direction with optional spacing. It outputs a mask that marks the `image_1` region as `1` and the stitched-group/spacing region as `0`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image_1` | - | IMAGE | - | - | Main image batch used as the anchor. |
| `image_2` | optional | IMAGE | - | - | Additional image batch; stitched into the group. |
| `image_3` | optional | IMAGE | - | - | Additional image batch; stitched into the group. |
| `image_4..image_N` | optional | IMAGE | - | - | Dynamic extra inputs; discovered by name and appended in numeric order. |
| `direction` | - | COMBO | `left` | `top` / `bottom` / `left` / `right` | Side on which the stitched group is placed relative to `image_1`. |
| `match_image_size` | - | BOOLEAN | True | - | When True, uses aspect-preserving resize to match edge length for stitching; when False, uses centered padding to reconcile sizes. |
| `spacing_width` | - | INT | 10 | 0–1000 | Spacing strip width/height between the main image and the group, and between images inside the group. |
| `spacing_color` | - | STRING | `1.0` | Gray/HEX/RGB | Spacing strip color. |
| `pad_color` | - | STRING | `1.0` | Gray/HEX/RGB/`edge`/`average`/`extend`/`mirror` | Padding strategy used when reconciling sizes. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Stitched image batch. |
| `mask` | MASK | Region mask: `image_1` area is `1`, other areas are `0`. |

## Features

- Dynamic inputs: accepts `image_2..image_N` and stitches them in ascending index order.
- Two-stage layout: builds the group first, then attaches it to `image_1` based on `direction`.
- Batch broadcasting: repeats smaller batches to the maximum batch size across all inputs.
- Size reconciliation:
  - `match_image_size=True`: resizes with bicubic interpolation while preserving aspect ratio.
  - `match_image_size=False`: uses centered padding with `pad_color` for size alignment.
- Color parsing:
  - `spacing_color`: grayscale, HEX, RGB, and named colors.
  - `pad_color`: adds `edge`, `average`, `extend`, `mirror` strategies.

## Typical Usage

- Build a main canvas with reference strips: use `direction=left/right` to attach a vertical reference group.
- Create a top/bottom comparison panel: use `direction=top/bottom` to attach a horizontal group.
- Produce a “main region mask” for downstream compositing: use the `mask` output as a selector for operations applied only to `image_1`.

## Notes & Tips

- When only `image_1` is provided, the node returns `image_1` and a full-white mask.
- `spacing_width=0` yields a direct adjacency between regions.
