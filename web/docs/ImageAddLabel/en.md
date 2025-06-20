# Image Add Label

**Node Function:** The `Image Add Label` node is used to add text labels to images, supporting custom fonts, colors, positions and other attributes, with dynamic input value referencing, commonly used for image annotation and description.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to add labels to |
| `text` | - | STRING | "" | Multi-line text | Label text content, supports {input1} and {input2} placeholders, each line corresponds to one image when multi-line |
| `font` | - | COMBO[STRING] | arial.ttf | Font file list | Font selection, supports multiple font files |
| `font_size` | - | INT | 36 | 1-256 | Font size |
| `height` | - | INT | 60 | 1-1024 | Label area height (top/bottom) or width (left/right) |
| `direction` | - | COMBO[STRING] | top | top, bottom, left, right | Label position: top, bottom, left, right |
| `invert_colors` | - | BOOLEAN | True | True/False | Whether to invert colors, True for black text on white background, False for white text on black background |
| `input1` | Optional | STRING | "" | - | Dynamic input value 1, can be referenced in text using {input1} |
| `input2` | Optional | STRING | "" | - | Dynamic input value 2, can be referenced in text using {input2} |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Image with labels added |

## Features

### Text Processing
- **Multi-line Support**: Supports multi-line text, separated by line breaks
- **Batch Processing**: Each line corresponds to one image when multi-line text
- **Text Cycling**: Text cycles when there are more images than text lines
- **Placeholder Support**: Text can use {input1} and {input2} placeholders

### Layout Control
- **Position Selection**: Can choose to add labels in four directions of the image
- **Area Size**: Adjust label area height or width based on label position
- **Color Scheme**: Supports switching between black and white color schemes
- **Auto Adaptation**: Label area automatically adapts to image size