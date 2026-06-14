# Float Number Compare - Float Value Comparison

**Node Purpose:** `Float Number Compare` compares two float inputs and outputs a boolean result based on the selected operator. It is useful for decimal parameters, ratios, and threshold-based conditions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `a` | - | FLOAT | 0.0 | -999999999.0–999999999.0 | Left float value. |
| `operator` | - | COMBO | `==` | `==`, `!=`, `>`, `>=`, `<`, `<=` | Comparison operator. |
| `b` | - | FLOAT | 0.0 | -999999999.0–999999999.0 | Right float value. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bool` | BOOLEAN | `true` when the comparison is satisfied, otherwise `false`. |

## Features

- Supports common float comparisons: equal, not equal, greater than, greater than or equal, less than, and less than or equal.
- Inputs are handled as floats, making this node suitable for ratios, opacity, scale values, thresholds, and other decimal parameters.
- Robust fallback: on exceptions, outputs `false`.

## Typical Usage

- Threshold checks: test whether a decimal parameter exceeds a target value.
- Branch control: connect the output to `Any Switch Bool` for workflow routing.
- Parameter validation: check whether ratio, opacity, or strength-like parameters are in a target range.
