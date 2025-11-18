# Image Rotate with Mask - Rotate with advanced padding

**Node Purpose:** `Image Rotate with Mask` rotates images with optional mask alignment, advanced padding strategies, and center selection. Supports `expand` canvas, rotation around mask centroid, and consistent mask transformation.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Optional mask batch aligned to `image`. |
| `angle` | - | FLOAT | 0.0 | -3600.0–3600.0 | Rotation angle in degrees (clockwise positive). |
| `pad_color` | - | STRING | `0.0` | Gray/HEX/RGB/`edge`/`average`/`extend`/`mirror` | Background strategy when expanding/cropping. |
| `expand` | - | BOOLEAN | True | - | Whether to expand canvas to contain full rotated image. |
| `mask_center` | - | BOOLEAN | False | - | Rotate around the centroid of the white mask area when provided. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Rotated image batch. |
| `mask` | MASK | Rotated mask; if absent, a mask of original image region is generated.

## Features

- Center control: when `mask_center=True`, computes weighted centroid of the white mask area and rotates around it.
- Advanced padding: supports color fill, `edge` replicate, `mirror` reflect, and `extend` replicate borders using OpenCV when available; falls back to PIL otherwise.
- Expand vs crop: `expand=True` enlarges canvas; `False` crops back to original size after rotation.
- Robust mask handling: aligns mask to image size, rotates with nearest-neighbor, and generates a full-content mask when none is provided.

## Typical Usage

- Precise rotation: enable `mask_center` to rotate around a subject defined by mask.
- Natural borders: use `pad_color=edge`/`mirror` to minimize artifacts when expanding.
- Cropped output: set `expand=False` to keep original dimensions.

## Notes & Tips

- Angle sign is internally inverted to match PIL/OpenCV conventions; input `angle` rotates clockwise.
- When OpenCV is unavailable, the node uses PIL equivalents with consistent behavior.