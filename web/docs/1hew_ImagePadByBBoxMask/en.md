# Image Pad By BBox Mask - Pad Canvas by Bounding Mask

**Node Purpose:** `Image Pad By BBox Mask` extracts the white-region bbox from `bbox_mask`, fits `paste_image` into that bbox while preserving aspect ratio, and outputs an image with the same size as `bbox_mask`. Areas not covered by the fitted image are filled with `pad_color`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `paste_image` | - | IMAGE | - | - | Image to place inside the bbox; RGBA alpha is preserved. |
| `bbox_mask` | - | MASK | - | - | Mask used to determine the output size and white-region bounding box. |
| `pad_color` | - | STRING | `1.0` | grayscale/HEX/RGB/color name/`edge`/`average`/`extend`/`mirror` | Fill color or fill strategy for areas outside the placed image. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image placed by bbox and padded to the mask canvas. |

## Features

- BBox detection: follows the same `bbox_mask` logic as `Image Paste By BBox Mask`, extracting a bounding rectangle from the white area.
- Aspect-preserving placement: fits `paste_image` into the bbox and centers it inside the bbox.
- Canvas size: output width and height come from `bbox_mask`, making it useful for restoring a crop into the original mask coordinate space.
- Fill strategies: supports solid colors, grayscale, RGB, HEX, color names, plus `edge`, `average`, `extend`, and `mirror`.
- Batch handling: cycles `paste_image` and `bbox_mask` batches by modulo when their batch counts differ.

## Typical Usage

- Pad a cropped local image back to the original mask canvas so downstream nodes receive a full-coordinate image.
- Add padding around a local edit before image editing, compositing, or alignment.
- Pair with `Mask To BBox Mask` to generate a rectangular bbox first, then restore the local image position on the full canvas.

## Notes & Tips

- `bbox_mask` is only used to compute the bounding rectangle; arbitrary mask shapes are not blended pixel by pixel.
- If `bbox_mask` is empty, the node outputs a `pad_color` background image with the same size as the mask.
- When `paste_image` and the bbox have different aspect ratios, the uncovered edges inside the bbox are also filled with `pad_color`.
