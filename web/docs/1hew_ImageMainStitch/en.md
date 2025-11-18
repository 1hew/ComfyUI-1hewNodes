# ImageMainStitch - Main stitch

**Node Purpose:** `ImageMainStitch` first builds a composition from `image_2..image_N` (horizontally for `top/bottom`, vertically for `left/right`), then attaches this composition to one side of `image_1` with optional spacing and size reconciliation.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image_1` | - | IMAGE | - | - | Primary image batch. |
| `image_2` | - | IMAGE | - | - | Secondary image batch. |
| `image_3` | - | IMAGE | - | - | Tertiary image batch. |
| `direction` | - | COMBO | `left` | `top`/`bottom`/`left`/`right` | Side to attach the combined pair to `image_1`. |
| `match_image_size` | - | BOOLEAN | True | - | When True, scales along the target axis; when False, pads to unify dimensions without resizing originals. |
| `spacing_width` | - | INT | 10 | 0–1000 | Spacing width used for both the inner pair and the outer attachment. |
| `spacing_color` | - | STRING | `1.0` | Gray/HEX/RGB | Spacing color. |
| `pad_color` | - | STRING | `1.0` | Gray/HEX/RGB/`edge`/`average`/`extend`/`mirror` | Padding strategy when unifying sizes. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Stitched image batch.

## Features

- Two-stage composition: builds `image_2+image_3` horizontally for `top/bottom` or vertically for `left/right`, then attaches to `image_1`.
- Size reconciliation: `match_image_size=True` uses ratio-preserving resize on the target axis; otherwise uses centered padding or cropping.
- Spacing control: single `spacing_width` applies to both pair and attachment stages.
- Color strategies: supports edge-, average-, extend-, and mirror-based padding; spacing uses fixed RGB.
- Batch broadcasting: repeats inputs to the maximum batch size for consistent output.

## Typical Usage

- Side composition: set `direction=right` to attach the pair to the right of `image_1`.
- Top/bottom stack: set `direction=top/bottom` to place the pair above/below with spacing.
- Preserve originals: set `match_image_size=False` to avoid resizing and use padding.

## Notes & Tips

- Internal helpers `_resize_keep_ratio` and `_pad_to_rgb` ensure natural scaling and padding.