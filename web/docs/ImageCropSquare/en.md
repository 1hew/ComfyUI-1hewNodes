# Image Crop Square

**Node Function:** The `Image Crop Square` node is used to crop images into squares based on masks, supporting scale factors, fill colors, and mask cutout functionality, commonly used for image preprocessing and region extraction.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Required | IMAGE | - | - | Input image |
| `mask` | Required | MASK | - | - | Mask to determine crop area |
| `scale_factor` | - | FLOAT | 1.0 | 0.1-3.0 | Scale factor controlling crop area size |
| `apply_mask` | - | BOOLEAN | False | True/False | Whether to apply mask for cutout |
| `extra_padding` | - | INT | 0 | 0-512 | Extra padding in pixels |
| `fill_color` | - | STRING | 1.0 | Grayscale/HEX/RGB/edge | Background color, supports multiple formats or "edge" for automatic edge color |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Cropped square image |

## Function Description

### Smart Cropping
- **Mask-driven**: Automatically identifies regions of interest based on input mask
- **Square output**: Automatically calculates optimal square bounding box ensuring square output
- **Center alignment**: Crop area is aligned based on mask center
- **Boundary handling**: Intelligently handles cases exceeding image boundaries

### Fill Options
- **Grayscale value**: e.g., "0.5" represents 50% gray
- **HEX format**: e.g., "#FF0000" represents red
- **RGB format**: e.g., "255,0,0" represents red
- **Edge color (edge)**: Automatically uses average color of image edges for filling
- **Smart filling**: Chooses different color calculation strategies based on whether mask is applied

### Application Scenarios
- **Portrait cropping**: Crop square avatars based on face detection results
- **Object extraction**: Extract specific objects and generate square thumbnails
- **Data preprocessing**: Prepare standard-sized input images for AI models
- **Batch processing**: Uniformly process multiple images to the same size