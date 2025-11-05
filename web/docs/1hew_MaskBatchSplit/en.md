# Mask Batch Split

**Node Function:** The `Mask Batch Split` node intelligently splits mask batches into two separate groups based on configurable parameters, supporting both forward and backward splitting modes with enhanced boundary condition handling for mask processing workflows.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `mask` | Required | MASK | - | - | Input mask batch to be split |
| `take_count` | Required | INT | 8 | 1-1024 | Number of masks to take for splitting |
| `from_start` | Required | BOOLEAN | False | True/False | Split direction: True for taking from start, False for taking from end |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `mask_1` | MASK | First split result containing the specified portion of masks |
| `mask_2` | MASK | Second split result containing the remaining masks |