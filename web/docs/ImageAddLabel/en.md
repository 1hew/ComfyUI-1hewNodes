# Image Add Label

**Node Function:** The `Image Add Label` node is used to add text labels to images, supporting custom fonts, colors, positions and other attributes, commonly used for image annotation and description.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Image to add labels to |
| `text` | - | STRING | "" | Multi-line text | Label text content, each line corresponds to one image when multi-line |
| `font` | - | COMBO[STRING] | arial.ttf | Font file list | Font selection, supports multiple font files |
| `font_size` | - | INT | 36 | 1-256 | Font size |
| `height` | - | INT | 60 | 1-1024 | Label area height (top/bottom) or width (left/right) |
| `direction` | - | COMBO[STRING] | top | top, bottom, left, right | Label position: top, bottom, left, right |
| `invert_colors` | - | BOOLEAN | True | True/False | Whether to invert colors, True for black text on white background, False for white text on black background |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Image with labels added |

## Features

### Text Processing
- **Multi-line Support**: Supports multi-line text, separated by line breaks
- **Batch Processing**: Each line corresponds to one image when multi-line text
- **Text Cycling**: Text cycles when there are more images than text lines

### Font System
- **Font Files**: Supports font files in the project fonts folder
- **System Fonts**: Automatically searches for Windows, Linux, macOS system fonts
- **Font Formats**: Supports TTF and OTF format font files
- **Font Size**: Adjustable font size to meet different needs

### Layout Control
- **Position Selection**: Can choose to add labels in four directions of the image
- **Area Size**: Adjust label area height or width based on label position
- **Color Scheme**: Supports switching between black and white color schemes
- **Auto Adaptation**: Label area automatically adapts to image size