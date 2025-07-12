# Image Crop with BBox Mask

**Node Function:** The `Image Crop with BBox Mask` node crops images based on masks and returns bounding box mask information for subsequent pasting back to original positions.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to be cropped |
| `mask` | Required | MASK | - | - | Mask for cropping guidance |
| `preset_ratio` | - | COMBO[STRING] | mask_ratio | mask_ratio, image_ratio, 1:1, 3:2, 4:3, 16:9, 21:9, 2:3, 3:4, 9:16, 9:21 | Target preset ratio |
| `scale_strength` | - | FLOAT | 0.0 | 0.0-1.0 | Scale strength: 0.0 for minimal mask-based crop, 1.0 for maximum crop within image |
| `divisible_by` | - | INT | 8 | 1-64 | Ensure output dimensions are divisible by this value |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Cropped image |
| `bbox_mask` | MASK | Bounding box mask for pasting back |
| `cropped_mask` | MASK | Cropped mask |

## Features

### Smart Aspect Ratio Control
- **Multiple Ratios**: Support mask-based ratio, image ratio, and preset aspect ratios
- **Precise Calculation**: Use fraction arithmetic for exact ratio precision
- **Adaptive Adjustment**: Automatically adjust crop size based on image boundaries

### Scale Strength Control
- **Minimal Crop**: scale_strength=0.0 performs minimal cropping based on mask bounding box
- **Maximum Crop**: scale_strength=1.0 gets maximum area within image that fits the ratio
- **Progressive Interpolation**: Values between 0.0-1.0 provide smooth size transitions