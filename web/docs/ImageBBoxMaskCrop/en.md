# Image BBox Mask Crop

**Node Function:** The `Image BBox Mask Crop` node performs batch image cropping based on bounding box mask information, supporting two output modes: complete region or white region only (with alpha channel).

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to be cropped |
| `mask` | Required | MASK | - | - | Mask defining the bounding box |
| `output_mode` | - | COMBO[STRING] | bbox_rgb | bbox_rgb, mask_rgba | Output mode: bbox_rgb (complete region) or mask_rgba (masked region with alpha) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Cropped image |

## Features

### Output Modes
- **BBox RGB Mode**: Output the complete bounding box region as RGB image
- **Mask RGBA Mode**: Output only the masked region with alpha channel for transparency