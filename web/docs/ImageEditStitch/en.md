# Image Edit Stitch

**Node Function:** The `Image Edit Stitch` node is used to stitch reference images and edited images together, supporting four stitching directions (top, bottom, left, right), commonly used for comparison display of original and edited effects.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `reference_image` | Required | IMAGE | - | - | Reference image (original image) |
| `edit_image` | Required | IMAGE | - | - | Edited image |
| `edit_mask` | Optional | MASK | - | - | Mask of edited area |
| `position` | - | COMBO[STRING] | right | top, bottom, left, right | Stitch position: top, bottom, left, right |
| `match_size` | - | BOOLEAN | True | True/False | Whether to match size, when enabled will adjust image size for matching |
| `fill_color` | - | FLOAT | 1.0 | 0.0-1.0 | Fill color, range 0.0 (black) - 1.0 (white), used for filling when sizes don't match |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Stitched image |
| `mask` | MASK | Merged mask |
| `split_mask` | MASK | Split mask identifying reference and edit areas |

## Function Description

### Size Handling
- **Size matching**: When enabled, automatically adjusts image sizes for stitching
- **Fill color**: Fill color used when image sizes don't match
- **Proportional scaling**: Maintains image proportions during scaling adjustment