# Int Split

**Node Function:** The `Int Split` node is designed to split a total value into two parts, supporting both percentage (0.0-1.0) and integer input modes for the split point. It's ideal for scenarios requiring proportional or fixed-value integer splitting, such as batch size allocation, dataset partitioning, and more.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `total` | Required | INT | 20 | 1-10000 | Total value to be split, step: 1 |
| `split_point` | Required | FLOAT | 0.5 | 0.0-10000.0 | Split point, supports percentage (0.0-1.0) or integer values, step: 0.01 |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `int_total` | INT | Original total value |
| `int_split` | INT | Split value result |
