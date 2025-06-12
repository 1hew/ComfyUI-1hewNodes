# Image Detail HL Freq Separation

**Node Function:** The `Image Detail HL Freq Separation` node implements detail-preserving high-low frequency separation technology, preserving detail information and optimizing image quality through complex image processing workflows.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `generate_image` | Required | IMAGE | - | - | Generated base image |
| `detail_image` | Required | IMAGE | - | - | Detail reference image |
| `detail_mask` | Required | MASK | - | - | Detail area mask |
| `gaussian_blur` | - | FLOAT | 10.00 | 0.00-1000.00 | Gaussian blur radius |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Image after high-low frequency separation processing |

## Function Description

### Parameter Description
- **gaussian_blur**: Controls Gaussian blur intensity, affecting frequency separation effect
  - Smaller values: Preserve more details but may produce noise
  - Larger values: Smoother effect but may lose details
  - Recommended range: 5.0-20.0