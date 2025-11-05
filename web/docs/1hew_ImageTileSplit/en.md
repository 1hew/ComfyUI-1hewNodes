# Image Tile Split

**Node Function:** The `Image Tile Split` node is used to intelligently split large images into multiple small tiles, supporting automatic grid division, custom split modes, and overlap area settings, commonly used for large image tile processing and AI inference optimization.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to be split |
| `split_mode` | - | COMBO[STRING] | auto | auto, custom, 2x2, 2x3, 2x4, 3x2, 3x3, 3x4, 4x2, 4x3, 4x4 | Split mode: auto, custom, or preset grids |
| `overlap_amount` | - | FLOAT | 0.05 | 0.0-512.0 | Overlap amount, ≤1.0 for ratio mode, >1.0 for pixel mode |
| `custom_rows` | Optional | INT | 2 | 1-10 | Number of rows in custom mode |
| `custom_cols` | Optional | INT | 2 | 1-10 | Number of columns in custom mode |
| `divisible_by` | Optional | INT | 8 | 1-64 | Divisibility number, ensures output dimensions are divisible by specified number |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `tiles` | IMAGE | Split image tile batch |
| `tiles_meta` | DICT | Tile metadata information including position, size, etc. |

## Function Description

### Application Scenarios
- **Large image AI processing**: Split large images into tiles for AI model inference
- **Memory optimization**: Reduce memory usage when processing large images
- **Parallel processing**: Support parallel processing of multiple tiles
- **Super-resolution**: Work with AI super-resolution models to process ultra-large images