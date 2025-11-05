# Image Batch Split

**Node Function:** The `Image Batch Split` node intelligently splits image batches into two separate groups based on configurable parameters, supporting both forward and backward splitting modes with enhanced boundary condition handling.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image batch to be split |
| `take_count` | Required | INT | 8 | 1-1024 | Number of images to take for splitting |
| `from_start` | Required | BOOLEAN | False | True/False | Split direction: True for taking from start, False for taking from end |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image_1` | IMAGE | First split result containing the specified portion of images |
| `image_2` | IMAGE | Second split result containing the remaining images |