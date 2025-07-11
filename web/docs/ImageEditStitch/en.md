# Image Edit Stitch

**Node Function:** The `Image Edit Stitch` node is used to stitch reference images and edited images together, supporting four stitching directions (top, bottom, left, right), commonly used for comparison display of original and edited effects.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `reference_image` | Required | IMAGE | - | - | Reference image (original image) |
| `edit_image` | Required | IMAGE | - | - | Edited image |
| `edit_mask` | Optional | MASK | - | - | Mask of edited area |
| `edit_image_position` | - | COMBO[STRING] | right | top, bottom, left, right | Edit image stitch position: top, bottom, left, right |
| `match_edit_size` | - | BOOLEAN | False | True/False | Whether to match edit image size, when enabled adjusts reference image size to match edit image, when disabled maintains reference image aspect ratio |
| `spacing` | - | INT | 0 | 0-1000 | Stitch spacing, controls pixel spacing between two images |
| `fill_color` | - | FLOAT | 1.0 | 0.0-1.0 | Fill color, range 0.0 (black) - 1.0 (white), used for filling when sizes don't match and spacing areas |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Stitched image |
| `mask` | MASK | Merged mask |
| `split_mask` | MASK | Split mask identifying reference and edit areas |

## Function Description

### Size Handling
- **Size matching**: When enabled, automatically adjusts reference image size to match edit image size, when disabled maintains reference image aspect ratio with intelligent adjustment based on stitch direction
- **Fill color**: Fill color used when image sizes don't match and for spacing area filling
- **Smart adaptation**: Automatically matches corresponding dimensions based on stitch direction (horizontal or vertical) while maintaining image quality

### Stitching Control
- **Edit image position**: Controls the stitching position of edit image relative to reference image
- **Stitch spacing**: Adds specified pixel width spacing between two images, filled with fill color
- **Spacing effect**: When spacing > 0, inserts spacing strips between stitched images for clearer visual separation