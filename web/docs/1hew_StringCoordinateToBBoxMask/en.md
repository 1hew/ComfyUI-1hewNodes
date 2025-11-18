# String Coordinate to BBox Mask - Build masks from bbox strings

**Node Purpose:** `String Coordinate to BBox Mask` parses a text string of bounding boxes and generates binary masks. Supports merged mask per frame or separate masks per bbox.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Reference batch (`B×H×W×C`) to set output shape. |
| `coordinates_string` | - | STRING | `` | multiline | Lines of bbox coordinates; separators can be spaces or commas; brackets `[]()` are ignored. Each line should contain at least four numbers `x1 y1 x2 y2`. |
| `output_mode` | - | COMBO | `merge` | `separate` / `merge` | Output strategy: per-bbox masks or merged union per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bbox_mask` | MASK | For `merge`: shape `B×H×W` (union of all boxes per frame). For `separate`: shape `(B×N)×H×W`, one mask per bbox for each frame. |

## Features

- Robust parsing: strips brackets and tolerates commas/spaces; converts floats to ints.
- Bounds clamping: coordinates are clamped to image size; boxes where `x2>x1` and `y2>y1` are filled.
- Mode control: choose merged union or separate masks for each bbox.

## Typical Usage

- Provide bbox lines like `12,34,200,180` or `12 34 200 180` per line.
- Use `merge` for a single mask covering all regions; use `separate` to get one mask per bbox.

## Notes & Tips

- When no valid boxes are found, outputs an all-zero mask batch of shape `B×H×W`.