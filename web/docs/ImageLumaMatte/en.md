# Image Luma Matte

**Node Function:** The `Image Luma Matte` node creates luminance-based composites by applying masks to images, supporting batch processing, edge feathering, and customizable background options with multiple color formats and special values.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Input image to process |
| `mask` | Required | MASK | - | - | Mask defining the matte area |
| `invert_mask` | Optional | BOOLEAN | False | True/False | Whether to invert the mask |
| `feather` | Optional | INT | 0 | 0-50 | Feather radius for softening mask edges |
| `background_add` | Optional | BOOLEAN | True | True/False | Whether to add background or create transparent output |
| `background_color` | Optional | STRING | "1.0" | Multiple formats | Background color, supports multiple formats and special values |
| `out_alpha` | Optional | BOOLEAN | False | True/False | Whether to output RGBA format (with alpha channel) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Processed image with luma matte applied |