# Image Edge Crop Pad

**Node Function:** The `Image Edge Crop Pad` node is used to perform edge cropping or padding operations on images, supporting negative values for inward cropping and positive values for outward padding, with multiple color formats and edge filling modes, commonly used for image size adjustment and edge processing.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to be processed |
| `left_amount` | - | FLOAT | 0 | -8192~8192 | Left crop/pad amount, negative for crop, positive for pad |
| `right_amount` | - | FLOAT | 0 | -8192~8192 | Right crop/pad amount, negative for crop, positive for pad |
| `top_amount` | - | FLOAT | 0 | -8192~8192 | Top crop/pad amount, negative for crop, positive for pad |
| `bottom_amount` | - | FLOAT | 0 | -8192~8192 | Bottom crop/pad amount, negative for crop, positive for pad |
| `uniform_amount` | - | FLOAT | 0 | -8192~8192 | Uniform crop/pad amount, overrides other directional settings when non-zero |
| `pad_color` | - | STRING | 0.0 | Multiple formats | Padding color, supports grayscale, HEX, RGB, color names and special values |
| `divisible_by` | - | INT | 8 | 1-1024 | Ensure output dimensions are divisible by this value, commonly used for AI model size requirements |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Processed image |
| `mask` | MASK | Operation area mask, cropped or padded areas are white, original image area is black |

## Features

### Value Modes
- **Percentage Mode**: When absolute value is less than 1 (e.g., 0.1, -0.2), calculated as percentage of image dimensions
- **Pixel Mode**: When absolute value is greater than or equal to 1 (e.g., 50, -100), used directly as pixel values
- **Negative Values**: Inward cropping, reducing image size
- **Positive Values**: Outward padding, increasing image size

### Uniform Mode
- **uniform_amount Priority**: When uniform_amount is non-zero, it overrides other four directional settings
- **Smart Distribution**: Negative values crop all edges inward, positive values pad all edges outward
- **Percentage Handling**: Percentages in uniform mode are automatically distributed to each edge (divided by 2)

### Padding Color Support
- **Grayscale Values**: e.g., "0.5" for 50% gray, "1.0" for white
- **HEX Format**: e.g., "#FF0000" or "FF0000" for red
- **RGB Format**: e.g., "255,0,0" or "1.0,0.0,0.0" for red
- **Color Names**: e.g., "red", "blue", "white" and other standard color names
- **Edge Color**: Use "edge", "e" or "ed" to automatically calculate average color of each edge for filling
- **Average Color**: Use "average", "avg" or "a" to calculate average color of entire image