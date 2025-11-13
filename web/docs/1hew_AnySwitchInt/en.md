# Any Switch Int

The `Any Switch Int` node is a multi-way integer switch with dynamic inputs and lazy evaluation. It selects the value from input_N according to an integer index, computing only the chosen branch. This is ideal for multi-option control and routing in complex graphs.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `select` | Required | INT | 1 | 1–999999 | Index for selecting input_N (step: 1). |
| `input_N` | Optional | * | - | - | Dynamic inputs named `input_1`, `input_2`, ... The UI automatically expands ports as you connect more. Supports any type. |

## Output

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `output` | * | Pass-through of the selected input value. Type matches the selected input. |
