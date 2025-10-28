# Any Empty Int

**Node Function:** The `Any Empty Int` node checks whether any type of input is empty and returns custom integer values based on the check result. It supports empty value detection for various data types and allows users to customize integer outputs for empty and non-empty values, providing flexible solutions for numerical conditional control.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `any` | Required | * | - | - | Input value of any type to be checked |
| `empty` | Required | INT | 0 | -999999 to 999999 | Integer value returned when input is empty, step: 1 |
| `not_empty` | Required | INT | 1 | -999999 to 999999 | Integer value returned when input is not empty, step: 1 |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `int` | INT | Integer value returned based on empty check result |
