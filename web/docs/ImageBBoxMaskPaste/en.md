# Image BBox Mask Paste

**Node Function:** The `Image BBox Mask Paste` node pastes processed cropped images back to their original positions in the base image using bounding box mask information, supporting multiple blend modes and opacity settings.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `base_image` | Required | IMAGE | - | - | Base image to paste onto |
| `paste_image` | Required | IMAGE | - | - | Image to be pasted |
| `bbox_mask` | Required | MASK | - | - | Bounding box mask defining paste position |
| `blend_mode` | - | COMBO[STRING] | normal | normal, multiply, screen, overlay, soft_light, hard_light, color_dodge, color_burn, darken, lighten, difference, exclusion | Blend mode for pasting |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Opacity of the pasted image |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Final composite image |

## Features

### Blend Modes
- **Normal**: Direct overlay without blending
- **Multiply**: Darkens by multiplying colors
- **Screen**: Lightens by inverting, multiplying, and inverting again
- **Overlay**: Combines multiply and screen based on base color
- **Soft Light**: Subtle lighting effect
- **Hard Light**: Strong lighting effect
- **Color Dodge/Burn**: Brightens or darkens colors
- **Darken/Lighten**: Takes darker or lighter of the two colors
- **Difference/Exclusion**: Creates contrast effects

### Position Control
- **Precise Positioning**: Uses bbox mask for exact placement
- **Batch Support**: Handles multiple images and masks
- **Size Matching**: Automatically handles size differences
- **Edge Handling**: Properly clips content at image boundaries