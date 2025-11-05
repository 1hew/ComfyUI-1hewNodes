# Image Rotate with Mask

## Description
The `Image Rotate with Mask` node provides advanced image rotation capabilities with comprehensive mask support. It enables arbitrary angle rotation with multiple fill modes and offers the option to rotate around the mask's white region center, making it ideal for precise image transformations.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to be rotated |
| `angle` | Required | FLOAT | 0.0 | -3600.0 to 3600.0 | Rotation angle in degrees (negative for clockwise) |
| `fill_mode` | Required | COMBO[STRING] | color | color, edge_extend, mirror | Fill mode for empty areas after rotation |
| `fill_color` | Required | STRING | "0.0" | - | Fill color for empty areas (color mode only) |
| `expand` | Required | BOOLEAN | True | - | Whether to expand canvas to contain the full rotated image |
| `use_mask_center` | Required | BOOLEAN | False | - | Whether to rotate around the mask's white region center |
| `mask` | Optional | MASK | - | - | Optional mask that undergoes the same transformation |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Rotated image with applied transformations |
| `mask` | MASK | Rotated mask or rotation area mask |