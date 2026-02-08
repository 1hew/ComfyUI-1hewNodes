# Mask to SAM3 Box - Mask to SAM3 box prompt

**Node Purpose:** `Mask to SAM3 Box` converts an input mask into bounding-box prompts and outputs a `SAM3_BOXES_PROMPT` structure for SAM3 workflows that require box prompts. It supports positive/negative prompts, merged output, and connected-component split output.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask (batch supported). Pixels with value `> 0.5` are treated as foreground |
| `condition` | - | COMBO | `positive` | `positive` / `negative` | Prompt label; `positive` means foreground boxes, `negative` means exclusion boxes |
| `output_mode` | - | COMBO | `merge` | `merge` / `separate` | `merge` outputs a single merged box; `separate` outputs multiple boxes by connected components |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `sam3_box` | SAM3_BOXES_PROMPT | SAM3 box prompt; a dict for single mask, and a list of dicts for batched masks |

## Features

- Box extraction:
  - `merge`: computes one minimal enclosing rectangle over all foreground pixels.
  - `separate`: computes an enclosing rectangle per foreground connected component.
- Normalized format: each box is output as `[cx, cy, bw, bh]` normalized to 0-1 relative to mask width/height.
- Label output: `labels` is a boolean list where `positive` maps to `True` and `negative` maps to `False`, aligned with `boxes`.

## Typical Usage

- Connect a `MASK` from segmentation/detection to this node, then connect `sam3_box` to nodes that accept SAM3 box prompts.
- Use `condition=negative` to exclude areas; use `output_mode=merge` to reduce the number of boxes.

## Notes & Tips

- Mask noise can produce many small boxes in `separate` mode. Consider upstream cleanup or use `merge`.
