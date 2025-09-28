# Image Grid Split

**Node Function:** The `Image Grid Split` node is used to split images into grid-based sub-images according to specified rows and columns, supporting selective output of specific grid cells or all split images as a batch.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image to be split |
| `rows` | Required | INT | 2 | 1-10 | Number of rows for grid splitting |
| `columns` | Required | INT | 2 | 1-10 | Number of columns for grid splitting |
| `output_index` | Required | INT | 0 | 0-100 | Output index: 0 for all split images as batch, 1+ for specific grid cell (row-major order) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Split image(s) based on output_index selection |