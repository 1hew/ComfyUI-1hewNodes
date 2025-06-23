# Image Luma Matte

**Node Function:** The `Image Luma Matte` node creates luminance-based composites by applying masks to images, supporting batch processing with customizable background options.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to process |
| `mask` | Required | MASK | - | - | Mask defining the matte area |
| `invert_mask` | - | BOOLEAN | False | True/False | Whether to invert the mask |
| `add_background` | - | BOOLEAN | True | True/False | Whether to add background or create transparent output |
| `background_color` | - | STRING | "1.0" | Color values | Background color: grayscale (0.0-1.0), RGB ("r,g,b"), hex ("#RRGGBB"), or "average" for mask area average color |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Processed image with luma matte applied |

## Features

### Background Options
- **Solid Colors**: Support for grayscale, RGB, and hex color formats
- **Average Color**: Automatically calculate average color from masked area
- **Transparent Output**: Create RGBA output when background is disabled
- **Flexible Input**: Multiple color format support for easy use