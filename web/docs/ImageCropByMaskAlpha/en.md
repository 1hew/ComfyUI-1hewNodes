# Image Crop by Mask Alpha

**Node Function:** The `Image Crop by Mask Alpha` node is used to batch crop images based on bounding box mask information, supporting two output modes: complete region or white region only (with alpha channel), commonly used for intelligent image cropping and region extraction.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to be cropped (supports both RGB and RGBA formats) |
| `mask` | Required | MASK | - | - | Mask for determining crop boundaries |
| `output_mode` | - | COMBO[STRING] | bbox_rgb | bbox_rgb, mask_rgba | Output mode: bbox_rgb (complete crop region in RGB format), mask_rgba (white region only with alpha channel in RGBA format) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Cropped image result (RGB format for bbox_rgb mode, RGBA format for mask_rgba mode) |
| `cropped_mask` | MASK | Cropped mask corresponding to the bounding box region |

## Features

- **Smart Channel Handling**: Automatically converts 4-channel RGBA input to 3-channel RGB output in bbox_rgb mode
- **Dual Output Modes**: 
  - `bbox_rgb`: Outputs complete cropped region in RGB format (3 channels)
  - `mask_rgba`: Outputs masked region with alpha channel in RGBA format (4 channels)
- **Batch Processing**: Supports processing multiple images and masks simultaneously
- **Automatic Padding**: Handles images of different sizes by padding to uniform dimensions
- **Mask Output**: Always provides the cropped mask region for both output modes