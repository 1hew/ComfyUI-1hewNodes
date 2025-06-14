# Image Tile Merge

**Node Function:** The `Image Tile Merge` node intelligently merges multiple image tiles back into a complete image using an advanced weight mask system and cosine function gradient algorithm, ensuring perfect seamless stitching effects in merged images.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `tiles` | Required | IMAGE | - | - | Image tile batch to be merged |
| `tile_meta` | Required | DICT | - | - | Tile metadata from ImageTileSplit node |
| `blend_strength` | - | FLOAT | 1.0 | 0.0-1.0 | Blend strength controlling overlap area blending degree using cosine function gradient algorithm |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `merged_image` | IMAGE | Complete merged image |

## Function Description

### Core Technical Features
- **Weight Mask System**: Uses weight map accumulation for perfect blending in overlap areas
- **Cosine Function Gradient**: Employs mathematically smoother cosine functions for natural transition effects
- **High-Precision Calculation**: Uses 64-bit floating point for accumulation, avoiding precision loss and numerical errors
- **Smart Boundary Handling**: Automatically handles edge tile special cases ensuring complete coverage

### Optimal Configuration Recommendations
- **Recommended Settings**: `blend_strength = 1.0` (default value)
- **Effect**: Achieves completely seamless image stitching suitable for all types of image processing tasks

### Application Scenarios
- **AI processing result merging**: Merge AI-processed image tiles back into complete image
- **Super-resolution reconstruction**: Merge super-resolution processed image tiles
- **Large image processing**: Reconstruct large-sized images after tile processing
- **Batch image stitching**: High-quality image stitching and panoramic composition
- **Lossless image reconstruction**: Ensure processed images perfectly match original images