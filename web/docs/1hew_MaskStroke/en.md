# Mask Stroke - Outline a Mask

**Node Purpose:** `Mask Stroke` creates an outer stroke mask from the input mask. It can optionally fill enclosed holes first, accepts stroke width in pixels or percentage, and can output either the stroke ring alone or the union of the stroke and the base mask.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch |
| `fill_hole` | - | BOOLEAN | False | - | When `True`, fill enclosed holes first using the same logic as `Mask Fill Hole`, then build the stroke from that processed mask |
| `stroke_width` | - | FLOAT | 20.0 | 0–8192; `0<x<1` treated as percentage | Stroke width; values below `1` are converted from a ratio of the mask short side into pixels |
| `include_mask` | - | BOOLEAN | False | - | When `True`, output the union of the stroke and the base mask; when `False` (default), output only the stroke ring |
| `close_border` | - | BOOLEAN | False | - | When `True`, the edge side is completed from the mask's own inner band (the part that would fall outside is drawn inside), so an edge-touching outline closes inside the frame and joins the rounded corners continuously; `False` keeps the clipped result |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Generated stroke mask, or the union of stroke and base mask |

## Features

- Shared hole-fill source of truth: `fill_hole=True` reuses `Mask Fill Hole` internals so the preprocessing result stays consistent.
- Border handling: a full 1px frame around the canvas is detected and ignored while stroking, so it does not dilate into a ring around the image, while real edge content is still stroked — a line or outline touching the wall is closed by the inward half of the dilation.
- Edge closing: `close_border=True` completes the clipped side from the mask's own inner band — the added part follows the original rounded contour and joins the existing stroke as one continuous line, instead of a straight bar butted against the edge; `False` keeps the clipped result.
- Stroke generation: uses elliptical dilation on the base mask and subtracts the base mask to isolate the outer stroke band.
- Percentage width: when `stroke_width` is less than `1`, it is resolved against the mask short side.
- Batch support: processes masks item by item with bounded concurrency for standard `[B,H,W]` batches.

## Typical Usage

- Edge-only guide masks: disable `include_mask` to get only the stroke ring.
- Expanded edit regions: enable `include_mask` to keep the original area and add an outer stroke.
- Hollow-shape cleanup: enable `fill_hole` before stroking to avoid inner voids affecting the outline.

## Notes & Tips

- When `stroke_width=0`, no new stroke is generated. With `include_mask=True`, the output is the base mask; otherwise it is an empty mask.
- `fill_hole=True` binary-fills the base mask before stroking, which is best for enclosed region masks.
- A solid region cut by the canvas has no outside space at the wall, so its outward stroke can only stay open there; `auto` / `stroke` close the stroke for a line touching the wall via the inward half of the dilation.
