# Image Crop With BBox Mask

**Node Function:** The `Image Crop With BBox Mask` node crops images based on masks and returns bounding box mask information for subsequent pasting back to original positions.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to be cropped |
| `mask` | Required | MASK | - | - | Mask for cropping guidance |
| `preset_ratio` | - | COMBO[STRING] | mask_ratio | mask_ratio, 1:1, 3:2, 4:3, 16:9, 21:9, 2:3, 3:4, 9:16, 9:21 | Target preset ratio |
| `scale_factor` | - | FLOAT | 1.0 | 0.1-5.0 | Scale factor for crop area |
| `extra_padding` | - | INT | 0 | 0-512 | Additional padding around crop area |
| `invert_mask` | - | BOOLEAN | False | True/False | Whether to invert the mask |
| `divisible_by` | - | INT | 8 | 1-1024 | Ensure output dimensions are divisible by this value |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Cropped image |
| `bbox_mask` | MASK | Bounding box mask for pasting back |
| `cropped_mask` | MASK | Cropped mask |

## Features

### Aspect Ratio Control
- **Multiple Ratios**: Support various aspect ratios including mask-based ratio
- **Flexible Scaling**: Adjust crop area size with scale factor
- **Smart Padding**: Add extra padding around the crop area

### Mask Processing
- **Mask Inversion**: Option to invert mask before processing
- **Batch Support**: Handle multiple images and masks
- **Position Tracking**: Generate bbox mask for accurate repositioning