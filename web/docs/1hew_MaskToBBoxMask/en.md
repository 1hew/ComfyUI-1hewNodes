# Mask To BBox Mask - Convert Mask to BBox Mask

**Node Purpose:** `Mask To BBox Mask` converts foreground regions in a mask into minimum bounding rectangle masks. It is useful for turning an object-shaped mask into a square/rectangular bbox mask that encloses the object.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch; 2D inputs are expanded to `[B,H,W]`. |
| `output_mode` | - | COMBO | `merge` | `merge` / `separate` | `merge` outputs one bbox per input mask; `separate` outputs one bbox mask per connected region. |
| `divisible_by` | - | INT | `8` | 1-1024 | Expands the rectangle width and height outward until they are divisible by this value; it never shrinks the original foreground area. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bbox_mask` | MASK | Converted rectangular bbox masks. |

## Features

- Bounding rectangle: computes the minimum enclosing rectangle from all non-zero foreground pixels and fills it white.
- Divisible sizing: uses `divisible_by` to expand bbox width and height to the requested multiple; near image edges, expansion shifts toward available space.
- Merged output: `merge` combines all valid foreground in each input mask into one bbox.
- Separate output: `separate` creates bbox masks per 8-connected region, so the output batch may be larger than the input batch.

## Typical Usage

- Object boxing: convert a detailed object mask into a rectangular mask around it.
- Crop/paste workflows: feed the result into nodes such as `Mask Crop by BBox Mask` or `Image Paste by BBox Mask`.
- Multi-object handling: use `Mask Separate` first, then convert each object mask into a stable rectangular area.

## Notes & Tips

- If an input mask has no non-zero foreground, `merge` outputs an all-black mask.
- If the image itself is smaller than the aligned target size, the bbox expands within image bounds and prioritizes keeping the original foreground intact.
- If one mask contains multiple distant objects, `merge` creates one large rectangle around all of them. Use `separate` or `Mask Separate` when each object needs its own bbox.
