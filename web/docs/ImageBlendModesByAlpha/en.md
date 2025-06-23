# Image Blend Modes By Alpha

**Node Function:** The `Image Blend Modes By Alpha` node provides comprehensive layer blending capabilities with support for base layer input, blend mode control, and opacity adjustment with optional mask application.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `base_image` | Required | IMAGE | - | - | Base layer image |
| `overlay_image` | Required | IMAGE | - | - | Overlay layer image |
| `blend_mode` | - | COMBO[STRING] | normal | normal, dissolve, darken, multiply, color burn, linear burn, add, lighten, screen, color dodge, linear dodge, overlay, soft light, hard light, linear light, vivid light, pin light, hard mix, difference, exclusion, subtract, divide, hue, saturation, color, luminosity | Blending mode selection |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Opacity of the overlay layer |
| `overlay_mask` | Optional | MASK | - | - | Optional mask for selective blending |
| `invert_mask` | - | BOOLEAN | False | True/False | Whether to invert the overlay mask |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Blended result image |

## Features

### Blend Mode Categories
- **Normal Modes**: normal, dissolve
- **Darken Modes**: darken, multiply, color burn, linear burn
- **Lighten Modes**: add, lighten, screen, color dodge, linear dodge
- **Contrast Modes**: overlay, soft light, hard light, linear light, vivid light, pin light, hard mix
- **Comparative Modes**: difference, exclusion, subtract, divide
- **Color Modes**: hue, saturation, color, luminosity

### Advanced Features
- **RGBA Support**: Automatic conversion of RGBA images to RGB
- **Batch Processing**: Handle multiple images with different batch sizes
- **Size Adaptation**: Automatic resizing of overlay to match base layer
- **Mask Integration**: Optional mask for selective blending areas
- **Quality Processing**: High-quality blending algorithms