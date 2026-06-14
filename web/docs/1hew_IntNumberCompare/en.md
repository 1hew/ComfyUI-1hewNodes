# Int Number Compare - Integer Value Comparison

**Node Purpose:** `Int Number Compare` compares two integer inputs and outputs a boolean result based on the selected operator. It is useful for integer thresholds, counts, and index-based conditions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `a` | - | INT | 0 | -999999999–999999999 | Left integer value. |
| `operator` | - | COMBO | `==` | `==`, `!=`, `>`, `>=`, `<`, `<=` | Comparison operator. |
| `b` | - | INT | 0 | -999999999–999999999 | Right integer value. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bool` | BOOLEAN | `true` when the comparison is satisfied, otherwise `false`. |

## Features

- Supports common integer comparisons: equal, not equal, greater than, greater than or equal, less than, and less than or equal.
- Inputs are handled as integers, making this node suitable for counts, indices, switch selections, and other integer outputs.
- Robust fallback: on exceptions, outputs `false`.

## Typical Usage

- Threshold checks: test whether a count has reached a target value.
- Branch control: connect the output to `Any Switch Bool` for workflow routing.
- Index checks: trigger logic when the current index matches a target value.
