# Mask Paste by BBox Mask

**Node Function:** The `Mask Paste by BBox Mask` node pastes a mask back to its original position in a base mask based on bounding box mask information. This node is designed for mask operations and provides a simplified interface for basic mask pasting without complex transformations.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
| --------- | -------- | --------- | ------- | ----- | ----------- |
| `paste_mask` | Required | MASK | - | - | Mask to be pasted |
| `bbox_mask` | Required | MASK | - | - | Bounding box mask indicating paste position |
| `base_mask` | Optional | MASK | - | - | Base mask to paste onto (defaults to black mask if not provided) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `mask` | MASK | Final mask with paste_mask pasted back to the bounding box area |