# Mask Math Ops

**Node Function:** The `Mask Math Ops` node supports mathematical operations between masks such as intersection, addition, subtraction, and XOR, with batch processing functionality.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `mask_a` | Required | MASK | - | - | First input mask |
| `mask_b` | Required | MASK | - | - | Second input mask |
| `operation` | - | COMBO[STRING] | or | or, and, subtract (a-b), subtract (b-a), xor | Mathematical operation type |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `mask` | MASK | Mask result after operation |

## Function Description

### Operation Types
#### Logical Operations
- **or**: Logical OR operation, takes maximum value of two masks
- **and**: Logical AND operation, takes minimum value of two masks
- **xor**: XOR operation, calculates absolute difference between two masks

#### Subtraction Operations
- **subtract (a-b)**: Subtracts mask B from mask A
- **subtract (b-a)**: Subtracts mask A from mask B