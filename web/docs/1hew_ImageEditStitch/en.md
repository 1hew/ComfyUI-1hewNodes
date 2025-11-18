# Image Edit Stitch - Stitch reference and edit images

**Node Purpose:** `Image Edit Stitch` stitches a reference image and an edit image along a chosen side with optional spacing. Preserves reference aspect ratio when `match_edit_size=False`, aligns edit mask to the edit image, and outputs both a combined mask and a split mask indicating edit vs reference regions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `reference_image` | - | IMAGE | - | - | Reference image batch. |
| `edit_image` | - | IMAGE | - | - | Edit image batch. |
| `edit_mask` | optional | MASK | - | - | Mask aligned to edit image; auto-created as white if absent. |
| `edit_image_position` | - | COMBO | `right` | `top`/`bottom`/`left`/`right` | Side on which the edit image is placed. |
| `match_edit_size` | - | BOOLEAN | False | - | When True, resizes reference to match edit image size (with padding). When False, preserves reference aspect ratio. |
| `spacing` | - | INT | 0 | 0–1000 | Spacing width/height between images. |
| `spacing_color` | - | STRING | `1.0` | Gray/HEX/RGB | Spacing color (strict RGB 0..1). |
| `pad_color` | - | STRING | `1.0` | Gray/HEX/RGB/`edge`/`average`/`extend`/`mirror` | Padding strategy for size reconciliation. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Stitched image batch. |
| `mask` | MASK | Combined mask, edit region preserved, other regions set to 0. |
| `split_mask` | MASK | Two-region mask: edit region `1`, reference region `0`. Includes spacing region as `0`.

## Features

- Aspect handling: preserves reference aspect ratio when not matching edit size; otherwise pads to match target side.
- Mask alignment: resizes edit mask to edit image using nearest-neighbor to preserve binary nature.
- Color parsing: supports `edge`, `average`, `extend`, `mirror` for padding; spacing color uses strict RGB.
- Batch broadcasting: automatically repeats inputs to maximum batch size across `reference`, `edit`, and `mask`.
- Position variants: `top`, `bottom`, `left`, `right` with consistent spacing behavior and mask composition.

## Typical Usage

- A/B view: place `edit_image` on `right` with `spacing>0` for visual comparison.
- Vertical comparison: use `top/bottom` when stacking; spacing color provides a separator.
- Preserve reference look: set `match_edit_size=False` to maintain reference proportions.

## Notes & Tips

- When only one image is provided, the node returns that image with a corresponding full white mask and a split mask indicating region semantics.
- Spacing strips are constant color tensors expanded per batch for efficiency.