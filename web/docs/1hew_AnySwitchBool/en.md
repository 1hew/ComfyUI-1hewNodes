# Any Switch Bool

**Node Function:** The `Any Switch Bool` node is a universal boolean conditional switcher that supports any type of input and lazy evaluation. It selects output between `on_true` or `on_false` values based on boolean conditions, computing only the required branch to improve execution efficiency. Ideal for conditional branch control and dynamic data routing.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `boolean` | Required | BOOLEAN | True | True/False | Boolean condition value controlling the switch |
| `on_true` | Optional | * | - | - | Value to output when boolean is True (supports any type) |
| `on_false` | Optional | * | - | - | Value to output when boolean is False (supports any type) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `output` | * | Selected output value based on boolean condition (type matches selected input) |

## Functionality

### Lazy Evaluation Mechanism

- **Smart Computation**: Only computes the required branch, unselected branches are not executed
