# Image Batch to List

**Node Function:** The `Image Batch to List` node converts batch images to image lists, splitting the batch dimension into individual image elements, commonly used for batch processing workflows and individual image operations.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image_batch` | Required | IMAGE | - | - | Batch images to be converted to list |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image_list` | IMAGE | List of individual images (OUTPUT_IS_LIST=True) |