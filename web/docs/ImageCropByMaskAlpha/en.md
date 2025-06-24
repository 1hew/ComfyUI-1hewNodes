# Image Crop by Mask Alpha

**Node Function:** The `Image Crop by Mask Alpha` node is used to batch crop images based on bounding box mask information, supporting two output modes: complete region or white region only (with alpha channel), commonly used for intelligent image cropping and region extraction.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to be cropped |
| `mask` | Required | MASK | - | - | Mask for determining crop boundaries |
| `output_mode` | - | COMBO[STRING] | bbox_rgb | bbox_rgb, mask_rgba | Output mode: bbox_rgb (complete crop region), mask_rgba (white region only with alpha channel) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Cropped image result |