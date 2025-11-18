# Image BBox Overlay by Mask - Draw bboxes from mask

**Node Purpose:** `Image BBox Overlay by Mask` draws bounding boxes on images using connected components from the input mask or a single merged bbox. Supports color, stroke width, optional fill, and extra padding.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch (`B×H×W×3`). |
| `mask` | - | MASK | - | - | Mask batch (`B×H×W`). Automatically aligned and broadcast to image batch. |
| `bbox_color` | - | COMBO | `green` | options | `red`/`green`/`blue`/`yellow`/`cyan`/`magenta`/`white`/`black`. |
| `stroke_width` | - | INT | 4 | 1–100 | Line width for outline mode. |
| `fill` | - | BOOLEAN | True | - | When True, fill bbox region; otherwise draw outline. |
| `padding` | - | INT | 0 | 0–1000 | Expand bbox outward by this many pixels. |
| `output_mode` | - | COMBO | `separate` | `separate`/`merge` | Multiple bboxes per component or a single merged bbox for the whole mask. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image batch with drawn bboxes. |

## Features

- Batch alignment: broadcasts `mask` to match image batch length and aligns size by LANCZOS.
- Modes: `separate` finds connected components via `regionprops`; `merge` uses min/max over all positive mask pixels.
- Padding: expands each bbox by `padding` in all directions, clamped to image bounds.
- Async per-frame: processes each sample in worker threads for responsiveness.

## Typical Usage

- Visualize detection masks: set `output_mode=separate` to draw bboxes for each component.
- Generate a single ROI: set `output_mode=merge` and tune `padding` to include context.
- Choose display style: switch `fill` to fill the rectangle or use outline with `stroke_width`.

## Notes & Tips

- Mask thresholding uses `>128` on 8-bit representation when computing components and merged bbox.
- Color options are mapped to fixed RGB values; defaults to `green` if not found.