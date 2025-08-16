# Image Paste by BBox Mask

## Description
The `Image Paste by BBox Mask` node pastes processed cropped images back to their original positions in base images based on bounding box mask information, with advanced transformation capabilities including position adjustment, scaling, and rotation.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `paste_image` | Required | IMAGE | - | - | Image to be pasted |
| `base_image` | Required | IMAGE | - | - | Base image to paste onto |
| `bbox_mask` | Required | MASK | - | - | Bounding box mask indicating paste position |
| `position_x` | Required | INT | 0 | -1000 to 1000 | Horizontal position offset |
| `position_y` | Required | INT | 0 | -1000 to 1000 | Vertical position offset (inverted) |
| `scale` | Required | FLOAT | 1.0 | 0.1 to 5.0 | Scale factor for the pasted image |
| `rotation` | Required | FLOAT | 0.0 | -3600 to 3600 | Rotation angle in degrees |
| `opacity` | Required | FLOAT | 1.0 | 0.0 to 1.0 | Opacity/transparency of the pasted image |
| `paste_mask` | Optional | MASK | - | - | Optional mask for the paste image |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Final image with transformed content pasted back |
| `mask` | MASK | Output mask showing the processed areas in white, same size as base_image |