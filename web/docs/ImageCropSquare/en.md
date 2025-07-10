# Image Crop Square

**Node Function:** The `Image Crop Square` node is used to crop images into squares based on masks, supporting scale factors, fill colors, and mask cutout functionality, commonly used for image preprocessing and region extraction. When no mask is provided, performs center square cropping.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Required | IMAGE | - | - | Input image |
| `mask` | Optional | MASK | - | - | Mask to determine crop area, if not provided performs center square cropping |
| `scale_factor` | - | FLOAT | 1.0 | 0.1-3.0 | Scale factor controlling crop area size |
| `apply_mask` | - | BOOLEAN | False | True/False | Whether to apply mask for cutout |
| `extra_padding` | - | INT | 0 | 0-512 | Extra padding in pixels |
| `fill_color` | - | STRING | 1.0 | Grayscale/HEX/RGB/edge | Background color, supports multiple formats or "edge" for automatic edge color |
| `divisible_by` | - | INT | 8 | 1-1024 | Ensures output dimensions are divisible by specified value, commonly used for AI model size requirements |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Cropped square image |

## Function Description

### Smart Cropping
- **Mask-driven**: Automatically identifies regions of interest based on input mask, performs center cropping when no mask provided
- **Square output**: Automatically calculates optimal square bounding box ensuring square output
- **Center alignment**: Crop area is aligned based on mask center or image center
- **Boundary handling**: Intelligently handles cases exceeding image boundaries
- **Size constraints**: Supports divisible_by parameter to ensure output dimensions meet specific requirements

### Fill Options
- **Grayscale value**: e.g., "0.5" represents 50% gray
- **HEX format**: e.g., "#FF0000" represents red
- **RGB format**: e.g., "255,0,0" represents red
- **Edge color (edge)**: Automatically uses average color of image edges for filling
- **Smart filling**: Chooses different color calculation strategies based on whether mask is applied