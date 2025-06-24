# Image Paste by BBox Mask

**Node Function:** The `Image Paste by BBox Mask` node is used to paste processed cropped images back to their original positions in base images based on bounding box mask information, supporting multiple blend modes and opacity control.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `base_image` | Required | IMAGE | - | - | Base image to paste onto |
| `cropped_image` | Required | IMAGE | - | - | Cropped image to be pasted |
| `bbox_mask` | Required | MASK | - | - | Bounding box mask indicating paste position |
| `blend_mode` | - | COMBO[STRING] | normal | normal, multiply, screen, overlay, soft_light, difference | Blending mode for pasting |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Opacity level for pasting |
| `cropped_mask` | Optional | MASK | - | - | Optional mask for the cropped image |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Final image with cropped content pasted back |