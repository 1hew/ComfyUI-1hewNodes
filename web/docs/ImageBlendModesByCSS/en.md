# Image Blend Modes By CSS

**Node Function:** The `Image Blend Modes By CSS` node implements CSS standard image blend modes based on the Pilgram library, providing image compositing effects consistent with Web CSS blend modes.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `overlay_image` | Required | IMAGE | - | - | Overlay image (top layer) |
| `base_image` | Required | IMAGE | - | - | Base image (bottom layer) |
| `blend_mode` | - | COMBO[STRING] | normal | 16 CSS blend modes | CSS blend mode selection |
| `blend_percentage` | - | FLOAT | 1.0 | 0.0-1.0 | Blend intensity percentage |
| `overlay_mask` | Optional | MASK | - | - | Mask for overlay area |
| `invert_mask` | Optional | BOOLEAN | False | True/False | Whether to invert mask |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | CSS blended image |

## Function Description

### CSS Blend Mode Categories
#### Basic Modes
- **normal**: Normal overlay
- **multiply**: Multiply
- **screen**: Screen mode
- **overlay**: Overlay mode

#### Darken/Lighten Modes
- **darken**: Darken, takes darker value
- **lighten**: Lighten, takes brighter value
- **color_dodge**: Color dodge
- **color_burn**: Color burn

#### Contrast Modes
- **hard_light**: Hard light effect
- **soft_light**: Soft light effect

#### Difference Modes
- **difference**: Difference mode
- **exclusion**: Exclusion mode

#### Color Modes
- **hue**: Hue mode
- **saturation**: Saturation mode
- **color**: Color mode
- **luminosity**: Luminosity mode