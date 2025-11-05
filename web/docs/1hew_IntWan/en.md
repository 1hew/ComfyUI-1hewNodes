# Int Wan

**Node Function:** The `Int Wan` node generates integers following the 4n+1 sequence pattern (1, 5, 9, 13, 17...), providing a specialized integer input for workflows requiring specific arithmetic progressions.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `value` | Required | INT | 1 | 1-10000 | Integer value following 4n+1 sequence pattern with step size of 4 |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `value` | INT | The input integer value following 4n+1 sequence |