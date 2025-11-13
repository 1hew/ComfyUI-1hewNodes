# Multi Image Stitch

**Node Function:** The `Multi Image Stitch` node stitches dynamic `image_X` inputs in order along a selected direction, with configurable spacing width and color, optional canvas size matching, and center alignment.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image_1` | Optional | IMAGE | - | - | First dynamic image input; supports additional `image_2`, `image_3`, ... |
| `direction` | Required | COMBO[STRING] | right | top, bottom, left, right | Stitching direction |
| `match_image_size` | Required | BOOLEAN | True | True/False | Whether to match image sizes to a unified canvas before stitching |
| `spacing_width` | Required | INT | 10 | 0–1000 | Width of spacing between adjacent images |
| `spacing_color` | Required | STRING | 1.0 | Color string | Color of the spacing area between images |
| `pad_color` | Required | STRING | 1.0 | Color string | Canvas padding color for alignment |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Stitched image according to direction, spacing and alignment |
