# Step Split

**Node Function:** The `Step Split` node is used for separating total sampling steps into high and low frequency parts, supporting both percentage (0.0-1.0) and integer step input modes, commonly used in high-low frequency sampling separation workflows.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `steps` | Required | INT | 10 | 1-10000 | Total sampling steps |
| `step_split` | Required | FLOAT | 0.6 | 0.0-10000.0 | Split step position, supports percentage mode (0.0-1.0) and integer mode (>1.0) |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `total_step` | INT | Total sampling steps (same as input) |
| `split_step` | INT | Split step position |

## Features

### Input Mode Support
- **Percentage Mode**: When `step_split` is between 0.0-1.0, it represents the percentage of total steps
- **Integer Mode**: When `step_split` is greater than 1.0, it represents the absolute step number
- **Special Case**: When `step_split` equals 1.0, it outputs 1 (not the total steps)