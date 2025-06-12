# Image Luma Matte

**Node Function:** The `Image Luma Matte` node is used to create luma matte effects based on masks, supporting mask inversion, background addition, and multiple background color options, commonly used for image compositing and cutout processing.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `images` | Required | IMAGE | - | - | Input image batch |
| `mask` | Required | MASK | - | - | Mask for matting |
| `invert_mask` | Optional | BOOLEAN | False | True/False | Whether to invert mask |
| `add_background` | Optional | BOOLEAN | True | True/False | Whether to add background |
| `background_color` | Optional | STRING | 1.0 | Grayscale/HEX/RGB/average | Background color, supports multiple formats or "average" for automatic calculation |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Image after matte application |

## Function Description

### Application Scenarios
- **Portrait cutout**: Background replacement based on portrait segmentation masks
- **Object extraction**: Extract specific objects and add new backgrounds
- **Image compositing**: Prepare matte materials for image compositing
- **Batch processing**: Batch process matte effects for multiple images