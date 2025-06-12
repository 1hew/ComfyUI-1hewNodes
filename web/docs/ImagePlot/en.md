# Image Plot

**Node Function:** The `Image Plot` node is used to plot multiple images into one large image, supporting horizontal, vertical, and grid arrangements, commonly used for creating image collections or comparison displays.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `images` | Required | IMAGE | - | - | Image batch to be combined |
| `layout` | - | COMBO[STRING] | horizontal | horizontal, vertical, grid | Arrangement: horizontal, vertical, grid |
| `gap` | - | INT | 10 | 0-100 | Gap between images in pixels |
| `columns` | - | INT | 2 | 1-10 | Number of images per row in grid mode |
| `background_color` | - | STRING | 1.0 | Grayscale/HEX/RGB | Background color, supports multiple formats |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Combined image |

## Function Description

### Arrangement Methods
- **horizontal**: All images arranged horizontally in one row
- **vertical**: All images arranged vertically in one column
- **grid**: Arranged in grid pattern with specified number of columns

### Layout Control
- **Gap control**: Can set spacing between images
- **Background fill**: Empty areas filled with specified color
- **Grid layout**: Can control number of images displayed per row

### Color Format Support
- **Grayscale value**: 0.0-1.0 (e.g., "0.5" represents gray)
- **Hexadecimal**: #RRGGBB (e.g., "#FF0000" represents red)
- **RGB values**: R,G,B (e.g., "255,0,0" represents red)