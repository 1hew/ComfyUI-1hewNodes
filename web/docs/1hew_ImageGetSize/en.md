# Image Get Size

**Node Function:** The `Image Get Size` node is used to extract the width and height dimensions from an input image. This utility node is essential for workflows that need to know image dimensions for further processing, calculations, or conditional operations.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image tensor from which to extract dimensions |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `width` | INT | The width of the image in pixels |
| `height` | INT | The height of the image in pixels |