# Image List to Batch

**Node Function:** The `Image List to Batch` node converts image lists to batch images, combining individual image elements into a single batch tensor with automatic size normalization, commonly used for batch processing and model inference.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `image_list` | Required | IMAGE | - | - | List of images to be converted to batch (INPUT_IS_LIST=True) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image_batch` | IMAGE | Combined batch image tensor |