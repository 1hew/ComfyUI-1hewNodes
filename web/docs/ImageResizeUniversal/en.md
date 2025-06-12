# Image Resize Universal

**Node Function:** `Image Resize Universal` is a powerful image resizing node that supports multiple aspect ratios, scaling modes, and fitting methods, intelligently adjusting image sizes to meet different requirements.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `image` | Optional | IMAGE | - | - | Input image to be scaled |
| `mask` | Optional | MASK | - | - | Input mask to be scaled |
| `get_image_size` | Optional | IMAGE | - | - | Reference image for target size, if provided uses reference image dimensions |
| `preset_ratio` | - | COMBO[STRING] | origin | origin, custom, 1:1, 3:2, 4:3, 16:9, 21:9, 2:3, 3:4, 9:16, 9:21 | Preset aspect ratio selection, origin maintains original ratio, custom uses custom ratio |
| `proportional_width` | - | INT | 1 | 1-1e8 | Custom ratio width value for custom mode |
| `proportional_height` | - | INT | 1 | 1-1e8 | Custom ratio height value for custom mode |
| `method` | - | COMBO[STRING] | lanczos | nearest, bilinear, lanczos, bicubic, hamming, box | Image scaling algorithm selection |
| `scale_to_side` | - | COMBO[STRING] | None | None, longest, shortest, width, height, mega_pixels_k | Scale by side mode, determines how to calculate target dimensions |
| `scale_to_length` | - | INT | 1024 | 4-1e8 | Target length value, used with scale_to_side |
| `fit` | - | COMBO[STRING] | crop | stretch, crop, pad | Fitting method: stretch, crop, pad |
| `pad_color` | - | STRING | 1.0 | Grayscale/HEX/RGB/edge | Padding color, supports multiple formats or "edge" for automatic edge color |
| `divisible_by` | - | INT | 8 | 1-1024 | Divisibility number, ensures output dimensions are divisible by specified number |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Scaled image |
| `mask` | MASK | Scaled mask |

## Function Description

### Aspect Ratio Modes
- **origin**: Maintains original image aspect ratio
- **custom**: Uses proportional_width and proportional_height for custom ratio
- **Preset ratios**: Common ratios like 1:1, 3:2, 4:3, 16:9, 21:9, 2:3, 3:4, 9:16, 9:21

### Scaling Modes
- **None**: Maintains original dimensions, only adjusts aspect ratio
- **longest**: Scales by longest side to specified length
- **shortest**: Scales by shortest side to specified length
- **width**: Scales by width to specified length
- **height**: Scales by height to specified length
- **mega_pixels_k**: Scales by total pixel count (in thousands of pixels)

### Fitting Methods
- **stretch**: Directly stretches to target dimensions, may change image ratio
- **crop**: Maintains ratio scaling then crops excess parts, center crop
- **pad**: Maintains ratio scaling then fills empty areas with specified color

### Scaling Algorithms
- **nearest**: Nearest neighbor interpolation, fast but lower quality
- **bilinear**: Bilinear interpolation, balances speed and quality
- **lanczos**: Lanczos interpolation, high quality scaling (default recommended)
- **bicubic**: Bicubic interpolation, high quality but slower
- **hamming**: Hamming window interpolation
- **box**: Box filter

### Padding Color Formats
- **Grayscale value**: e.g., "0.5" represents 50% gray
- **HEX format**: e.g., "#FF0000" represents red
- **RGB format**: e.g., "255,0,0" or "1.0,0.0,0.0"
- **edge**: Automatically uses average color of image edges for padding