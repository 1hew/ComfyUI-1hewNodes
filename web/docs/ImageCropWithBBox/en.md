# Image Crop With BBox

**Node Function:** The `Image Crop With BBox` node crops images based on masks and returns bounding box information, supporting multiple aspect ratios, scale factors, and fill options, commonly used for object extraction and subsequent paste operations.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to be cropped |
| `mask` | Required | MASK | - | - | Mask to determine crop area |
| `aspect_ratio` | - | COMBO[STRING] | mask_ratio | mask_ratio, 1:1, 3:2, 4:3, 16:9, 21:9, 2:3, 3:4, 9:16, 9:21 | Output aspect ratio selection |
| `scale_factor` | - | FLOAT | 1.0 | 0.1-5.0 | Scale factor controlling crop area size |
| `extra_padding` | - | INT | 0 | 0-512 | Extra padding in pixels |
| `exceed_image` | - | BOOLEAN | False | True/False | Whether to allow crop area to exceed original image boundaries |
| `invert_mask` | - | BOOLEAN | False | True/False | Whether to invert mask |
| `fill_color` | - | STRING | 1.0 | Grayscale/HEX/RGB/edge | Background fill color, supports multiple formats |
| `divisible_by` | - | INT | 8 | 1-1024 | Divisibility number, ensures output dimensions are divisible by specified number |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `cropped_image` | IMAGE | Cropped image |
| `cropped_mask` | MASK | Cropped mask |
| `bbox_meta` | DICT | Bounding box metadata containing position and size information |

## Function Description

### Aspect Ratio Control
- **Multiple ratios**: Supports common aspect ratio selections (1:1, 16:9, 4:3, etc.)
- **Original ratio**: mask_ratio maintains the original aspect ratio of the mask
- **Auto adjustment**: Automatically adjusts crop area based on selected ratio
- **Center preservation**: Maintains mask center position during adjustment

### Scaling and Filling
- **Scale control**: Control crop area size through scale_factor
- **Padding setting**: extra_padding adds additional padding to crop area
- **Boundary handling**: exceed_image allows crop area to exceed original image boundaries
- **Smart filling**: Supports multiple fill color formats and automatic edge color acquisition

### Fill Color Formats
- **Grayscale value**: e.g., "0.5" represents 50% gray
- **HEX format**: e.g., "#FF0000" represents red
- **RGB format**: e.g., "255,0,0" represents red
- **Edge color (edge)**: Automatically uses average color of image edges for filling