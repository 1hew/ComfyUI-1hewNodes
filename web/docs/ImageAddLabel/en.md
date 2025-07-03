# Image Add Label

**Node Function:** The `Image Add Label` node is used to add text labels to images, supporting custom fonts, colors, positions and other attributes, with dynamic input value referencing, commonly used for image annotation and description.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Image to add labels to |
| `height_pad` | - | INT | 24 | 1-1024 | Total top and bottom padding for label area, final height auto-calculated based on actual text rendering height |
| `font_size` | - | INT | 36 | 1-256 | Font size |
| `invert_color` | - | BOOLEAN | True | True/False | Whether to invert colors, True for black text on white background, False for white text on black background |
| `font` | - | COMBO[STRING] | Alibaba-PuHuiTi-Regular.otf | Font file list | Font selection, supports multiple font files |
| `text` | - | STRING | "" | Multi-line text | Label text content, supports {input1} and {input2} placeholders, supports -- separator functionality |
| `direction` | - | COMBO[STRING] | top | top, bottom, left, right | Label position: top, bottom, left, right |
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
- **Dash Separator**: When lines containing only dashes exist, content between -- becomes complete labels, other splitting methods are disabled