# Mask List to Batch

**Node Function:** The `Mask List to Batch` node converts mask lists to batch masks, combining individual mask elements into a single batch tensor with automatic size normalization, commonly used for batch mask processing and model operations.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `mask_list` | Required | MASK | - | - | List of masks to be converted to batch (INPUT_IS_LIST=True) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `mask_batch` | MASK | Combined batch mask tensor |