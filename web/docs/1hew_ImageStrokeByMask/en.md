# Image Stroke by Mask - Outline mask regions

**Node Purpose:** `Image Stroke by Mask` creates an outline (stroke) around the mask region on a black canvas in the stroke color, then pastes original image content inside the original mask. Outputs an updated mask which is the union of stroke and original mask.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `mask` | - | MASK | - | - | Input mask batch. |
| `stroke_width` | - | INT | 20 | 0–1000 | Stroke width in pixels. |
| `stroke_color` | - | STRING | `1.0` | Gray/HEX/RGB/named | Stroke color; supports `average` (`a`) of image or `mask`-weighted (`mk`). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image with stroke around mask and original content restored inside mask. |
| `mask` | MASK | Stroke mask unioned with original mask.

## Features

- Size alignment: resizes `mask` to image size (LANCZOS) to ensure consistent placement.
- Color parsing: supports gray, HEX, `R,G,B`, named colors, `average` of image, and `mask`-weighted average color.
- Stroke creation: uses morphological dilation (`cv2.MORPH_ELLIPSE`) and subtracts original mask to form the ring.
- Batch support: processes batches by cycling indices when `image` and `mask` lengths differ.

## Typical Usage

- Highlight masked regions with a colored outline while keeping original content inside.
- Compute stroke color from image average or mask-weighted average (`stroke_color=mk`).

## Notes & Tips

- Output image is black outside stroke and mask union; content within original mask is pasted from the input image.
- Stroke kernel size is `(2×stroke_width+1)`, tuned for smooth elliptical dilation.