# Image Crop Edge

**Node Function:** The `Image Crop Edge` node is used to crop edge areas of images, supporting separate settings for four-side crop amounts or uniform cropping, and ensuring output dimensions meet divisibility constraints.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Required | IMAGE | - | - | Input image |
| `left_amount` | - | FLOAT | 0 | 0.0-8192 | Left crop amount, ≤1 for ratio, >1 for pixels |
| `right_amount` | - | FLOAT | 0 | 0.0-8192 | Right crop amount, ≤1 for ratio, >1 for pixels |
| `top_amount` | - | FLOAT | 0 | 0.0-8192 | Top crop amount, ≤1 for ratio, >1 for pixels |
| `bottom_amount` | - | FLOAT | 0 | 0.0-8192 | Bottom crop amount, ≤1 for ratio, >1 for pixels |
| `uniform_amount` | - | FLOAT | 0 | 0.0-8192 | Uniform crop amount for all sides, higher priority than individual side settings |
| `divisible_by` | - | INT | 8 | 1-1024 | Divisibility number, ensures output dimensions are divisible by specified number |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Cropped image |

## Function Description

### Flexible Cropping
- **Individual Side Control**: Can separately set crop amounts for left, right, top, and bottom sides
- **Uniform Cropping**: uniform_amount setting overrides all individual side settings
- **Dual Mode**: Supports ratio mode (0-1) and pixel mode (≥1)
- **Priority Handling**: Uniform cropping has the highest priority

### Application Scenarios
- **Image Preprocessing**: Prepare input dimensions that meet AI model requirements
- **Border Removal**: Remove unwanted areas around images
- **Size Standardization**: Crop different sized images to standard dimensions
- **Batch Processing**: Uniformly process edge areas of multiple images