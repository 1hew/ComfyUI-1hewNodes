# Mask Batch Math Ops

**Node Function:** The `Mask Batch Math Ops` node supports unified OR and AND operations on batch masks, merging multiple masks into a single mask.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `masks` | Required | MASK | - | - | Input mask batch |
| `operation` | - | COMBO[STRING] | or | or, and | Batch operation type |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `mask` | MASK | Single mask after batch operation |

## Function Description

### Batch Operation Types
#### OR Operation
- **Function**: Performs logical OR operation on all masks
- **Effect**: Merges all mask regions, takes maximum value
- **Usage**: Creates union mask containing all input regions

#### AND Operation
- **Function**: Performs logical AND operation on all masks
- **Effect**: Only preserves regions covered by all masks, takes minimum value
- **Usage**: Creates intersection mask of all input regions