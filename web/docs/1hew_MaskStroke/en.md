# Mask Stroke - Outline a Mask

**Node Purpose:** `Mask Stroke` creates an outer stroke mask from the input mask. It can optionally fill enclosed holes first, accepts stroke width in pixels or percentage, and can output either the stroke ring alone or the union of the stroke and the base mask.

## Inputs

| Parameter | Type | Default | Range | Description |
| --------- | ---- | ------- | ----- | ----------- |
| `mask` | MASK | - | - | Input mask batch |
| `fill_hole` | BOOLEAN | False | - | When `True`, fill enclosed holes first using the same logic as `Mask Fill Hole`, then build the stroke from that processed mask |
| `stroke_width` | FLOAT | 20.0 | 0–8192; `0<x<1` treated as percentage | Stroke width; values below `1` are converted from a ratio of the mask short side into pixels |
| `include_mask` | BOOLEAN | True | - | When `True`, output the union of the stroke and the base mask; when `False`, output only the stroke ring |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| `mask` | MASK | Generated stroke mask, or the union of stroke and base mask |

## Behavior

- Shared hole-fill source of truth: `fill_hole=True` reuses `Mask Fill Hole` internals so the preprocessing result stays consistent.
- Stroke generation: uses elliptical dilation on the base mask and subtracts the base mask to isolate the outer stroke band.
- Percentage width: when `stroke_width` is less than `1`, it is resolved against the mask short side.
- Batch support: processes masks item by item with bounded concurrency for standard `[B,H,W]` batches.

## Typical Uses

- Edge-only guide masks: disable `include_mask` to get only the stroke ring.
- Expanded edit regions: enable `include_mask` to keep the original area and add an outer stroke.
- Hollow-shape cleanup: enable `fill_hole` before stroking to avoid inner voids affecting the outline.

## Notes

- When `stroke_width=0`, no new stroke is generated. With `include_mask=True`, the output is the base mask; otherwise it is an empty mask.
- `fill_hole=True` binary-fills the base mask before stroking, which is best for enclosed region masks.
