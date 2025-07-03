# Mask Fill Hole

**Node Function:** The `Mask Fill Hole` node is used to fill holes in enclosed areas of masks, supporting batch processing of multiple masks.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `mask` | Required | MASK | - | - | Mask to fill holes in |
| `invert_mask` | - | BOOLEAN | False | True/False | Whether to invert the mask, True for inversion, False for no inversion |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `mask` | MASK | Mask with holes filled |