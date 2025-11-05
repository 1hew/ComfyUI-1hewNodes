# Image Solid

**Node Function:** The `Image Solid` node is used to generate solid color images based on input color and dimensions, supporting multiple preset sizes and custom dimensions, can be used as background images or mask generation.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `reference_image` | Optional | IMAGE | - | - | Reference image, if provided uses reference image dimensions |
| `preset_size` | - | COMBO[STRING] | custom | Preset size options | Preset size selection, includes various common ratios like 1:1, 16:9, 9:16, etc., or select "custom" for custom dimensions |
| `width` | - | INT | 1024 | 1-8192 | Custom image width in pixels |
| `height` | - | INT | 1024 | 1-8192 | Custom image height in pixels |
| `color` | Optional | STRING | "1.0" | Multiple formats | Enhanced color input supporting multiple formats: grayscale values (0.0-1.0), RGB tuples ("0.5,0.7,0.9" or "128,192,255"), hex colors ("#FF0000" or "FF0000"), color names ("red", "blue"), and single-letter shortcuts ("r"=red, "g"=green, "b"=blue, "c"=cyan, "m"=magenta, "y"=yellow, "k"=black, "w"=white) |
| `alpha` | - | FLOAT | 1.0 | 0.0-1.0 | Transparency / brightness adjustment |
| `invert` | - | BOOLEAN | False | True/False | Whether to invert color |
| `mask_opacity` | - | FLOAT | 1.0 | 0.0-1.0 | Mask opacity |
| `divisible_by` | - | INT | 8 | 1-1024 | Divisibility number, ensures output dimensions are divisible by specified number |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Generated solid color image |
| `mask` | MASK | Corresponding mask image |