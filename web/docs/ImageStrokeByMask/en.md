# Image Stroke by Mask

## Description
The `Image Stroke by Mask` node applies stroke effects to specified mask regions in input images. It creates outline borders around masked areas with customizable width and color, commonly used for highlighting objects or creating visual emphasis effects.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Required | IMAGE | - | - | Input image for stroke processing |
| `mask` | Required | MASK | - | - | Input mask defining stroke regions |
| `stroke_width` | - | INT | 20 | 0-1000 | Stroke border width in pixels |
| `stroke_color` | - | STRING | "1.0" | Multiple formats | Stroke color specification |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Processed image with stroke effects applied |
| `mask` | MASK | Combined mask including original mask and stroke regions |

## Features

### Advanced Color Parsing
- **Multiple color formats**: Supports grayscale values (0-1 and 0-255), RGB tuples, HEX codes, and color names
- **Flexible input**: Accepts formats like "1.0", "255,128,64", "#FF8040", "red", etc.
- **Bracket support**: Handles bracketed RGB values like "(127,126,69)"
- **Fallback handling**: Returns white color for unrecognized inputs