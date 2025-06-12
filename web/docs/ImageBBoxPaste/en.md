# Image BBox Paste

**Node Function:** The `Image BBox Paste` node pastes processed cropped images back to specified positions in the original image, supporting multiple blend modes and opacity control, commonly used for image editing and compositing.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `base_image` | Required | IMAGE | - | - | Base image as the paste background |
| `cropped_image` | Required | IMAGE | - | - | Cropped image to be pasted |
| `bbox_meta` | Required | DICT | - | - | Bounding box metadata specifying paste position |
| `blend_mode` | - | COMBO[STRING] | normal | normal, multiply, screen, overlay, soft_light, difference | Blend mode selection |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Opacity controlling transparency of pasted image |
| `cropped_mask` | Optional | MASK | - | - | Optional mask for precise control of paste area |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Composite image after pasting |

## Function Description

### Blend Modes
- **normal**: Normal mode, directly overlays the original image
- **multiply**: Multiply mode, color multiplication produces darker effects
- **screen**: Screen mode, produces brighter effects
- **overlay**: Overlay mode, combines multiply and screen effects
- **soft_light**: Soft light mode, produces soft lighting effects
- **difference**: Difference mode, calculates color differences