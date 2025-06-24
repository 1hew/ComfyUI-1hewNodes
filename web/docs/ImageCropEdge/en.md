# Image Crop Edge

**Node Function:** The `Image Crop Edge` node supports cropping all four edges simultaneously or setting individual crop amounts for each edge, with support for percentage and pixel modes.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to be cropped |
| `left_amount` | - | FLOAT | 0 | 0.0-8192 | Left edge crop amount (0-1 for percentage, >=1 for pixels) |
| `right_amount` | - | FLOAT | 0 | 0.0-8192 | Right edge crop amount (0-1 for percentage, >=1 for pixels) |
| `top_amount` | - | FLOAT | 0 | 0.0-8192 | Top edge crop amount (0-1 for percentage, >=1 for pixels) |
| `bottom_amount` | - | FLOAT | 0 | 0.0-8192 | Bottom edge crop amount (0-1 for percentage, >=1 for pixels) |
| `uniform_amount` | - | FLOAT | 0 | 0.0-8192 | Uniform crop amount for all edges (0-1 for percentage, >=1 for pixels) |
| `divisible_by` | - | INT | 8 | 1-1024 | Ensure output dimensions are divisible by this value |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Cropped image |