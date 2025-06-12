# Image Blend Modes By Alpha

**Node Function:** The `Image Blend Modes By Alpha` node provides multiple professional image blend modes, supporting opacity control and mask application, achieving Photoshop-like layer blending effects.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `overlay_image` | Required | IMAGE | - | - | Overlay image (top layer) |
| `base_image` | Required | IMAGE | - | - | Base image (bottom layer) |
| `blend_mode` | - | COMBO[STRING] | normal | Multiple blend modes | Blend mode selection |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Opacity of overlay image |
| `overlay_mask` | Optional | MASK | - | - | Mask for overlay area |
| `invert_mask` | Optional | BOOLEAN | False | True/False | Whether to invert mask |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Blended image |

## Function Description

### Blend Mode Categories

#### Basic Modes
- **normal**: Normal overlay
- **dissolve**: Dissolve effect, randomly discards pixels

#### Darken Modes
- **darken**: Darken, takes darker value
- **multiply**: Multiply, color multiplication
- **color burn**: Color burn
- **linear burn**: Linear burn

#### Lighten Modes
- **lighten**: Lighten, takes brighter value
- **screen**: Screen mode
- **color dodge**: Color dodge
- **linear dodge**: Linear dodge
- **add**: Add mode

#### Contrast Modes
- **overlay**: Overlay mode
- **soft light**: Soft light effect
- **hard light**: Hard light effect
- **linear light**: Linear light
- **vivid light**: Vivid light mode
- **pin light**: Pin light mode
- **hard mix**: Hard mix

#### Difference Modes
- **difference**: Difference mode
- **exclusion**: Exclusion mode
- **subtract**: Subtract mode
- **divide**: Divide mode

#### Color Modes
- **hue**: Hue mode
- **saturation**: Saturation mode
- **color**: Color mode
- **luminosity**: Luminosity mode