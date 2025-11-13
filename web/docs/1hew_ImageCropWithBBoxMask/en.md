# Image Crop with BBox Mask

**Node Function:** The `Image Crop with BBox Mask` node crops images based on masks and returns bounding box mask information for subsequent pasting back to original positions.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to be cropped (supports both 3-channel RGB and 4-channel RGBA) |
| `mask` | Required | MASK | - | - | Mask for cropping guidance |
| `preset_ratio` | - | COMBO[STRING] | mask | mask, image, 1:1, 3:2, 4:3, 16:9, 21:9, 2:3, 3:4, 9:16, 9:21 | Target preset ratio |
| `scale_strength` | - | FLOAT | 0.0 | 0.0-1.0 | Scale strength: 0.0 for minimal mask-based crop, 1.0 for maximum crop within image |
| `crop_to_side` | - | COMBO[STRING] | None | None, longest, shortest, width, height | Precise dimension control: None preserves original behavior, others enable exact size control |
| `crop_to_length` | - | INT | 512 | 8-4096 | Target length for the specified crop_to_side dimension (only effective when crop_to_side is not 'None') |
| `divisible_by` | - | INT | 8 | 1-1024 | Ensure output dimensions are divisible by this value |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Cropped image |
| `bbox_mask` | MASK | Bounding box mask for pasting back |
| `cropped_mask` | MASK | Cropped mask |