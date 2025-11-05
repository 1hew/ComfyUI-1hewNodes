# Any Switch Int

**Node Function:** The `Any Switch Int` node is a multi-way integer switcher that supports switching between multiple input options with lazy evaluation. It selects the corresponding input output based on integer index (1-5), computing only the selected branch to provide flexible multi-way data routing capabilities. Ideal for multi-option conditional control and complex data flow management.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `select` | Required | INT | 1 | 1-5 | Index value for selecting input, step: 1 |
| `input_1` | Optional | * | - | - | First input option (supports any type) |
| `input_2` | Optional | * | - | - | Second input option (supports any type) |
| `input_3` | Optional | * | - | - | Third input option (supports any type) |
| `input_4` | Optional | * | - | - | Fourth input option (supports any type) |
| `input_5` | Optional | * | - | - | Fifth input option (supports any type) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `output` | * | Selected output value based on index (type matches selected input) |

## Functionality

### Lazy Evaluation Mechanism

- **Smart Computation**: Only computes the input corresponding to the selected index, other inputs are not executed
