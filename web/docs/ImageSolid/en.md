# Image Solid

**Node Function:** The `Image Solid` node is used to generate solid color images based on input color and dimensions, supporting multiple preset sizes and custom dimensions, can be used as background images or mask generation.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `reference_image` | Optional | IMAGE | - | - | Reference image, if provided uses reference image dimensions |
| `preset_size` | - | COMBO[STRING] | custom | Preset size options | Preset size selection, includes various common ratios like 1:1, 16:9, 9:16, etc., or select "custom" for custom dimensions |
| `width` | - | INT | 1024 | 1-8192 | Custom image width in pixels |
| `height` | - | INT | 1024 | 1-8192 | Custom image height in pixels |
| `color` | - | COLOR | #FFFFFF | Color value | Image color, default is white |
| `alpha` | - | FLOAT | 1.0 | 0.0-1.0 | Transparency / brightness adjustment |
| `invert` | - | BOOLEAN | False | True/False | Whether to invert color |
| `mask_opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Mask opacity |
| `divisible_by` | - | INT | 8 | 1-1024 | Divisibility number, ensures output dimensions are divisible by specified number |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Generated solid color image |
| `mask` | MASK | Corresponding mask image |

## Function Description

### Size Settings
- **Preset sizes**: Provides multiple common ratio preset sizes like 1:1, 3:2, 4:3, 16:9, 21:9, etc.
- **Custom dimensions**: When selecting "custom", can freely set width and height
- **Reference image**: If reference image is provided, will use reference image dimensions
- **Divisibility constraint**: Ensures output dimensions are divisible by specified number for easier subsequent processing

### Color Control
- **Color selection**: Supports standard color picker, default is white
- **Brightness adjustment**: Adjust image brightness through alpha parameter
- **Color inversion**: Can invert selected color
- **Mask transparency**: Independent control of mask opacity