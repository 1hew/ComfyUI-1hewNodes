# Mask Batch to List

**Node Function:** The `Mask Batch to List` node converts batch masks to mask lists, splitting the batch dimension into individual mask elements, commonly used for mask processing workflows and individual mask operations.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `mask_batch` | Required | MASK | - | - | Batch masks to be converted to list |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `mask_list` | MASK | List of individual masks (OUTPUT_IS_LIST=True) |