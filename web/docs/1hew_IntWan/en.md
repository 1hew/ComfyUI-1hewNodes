# Int Wan - Integer Constant

**Node Purpose:** `Int Wan` outputs an integer constant, converting the input to `int` while honoring the configured range and UI step.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `value` | - | INT | `1` | `1–10000` | Integer value to output. The UI step is `4` for coarse adjustments. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | Pass-through integer constant. |

## Features

- Type-safe conversion: input is cast to `int` at execution.
- Simple constant provider: useful for parameterization in graphs.
- Core logic: internal type coercion and constant emission.

## Typical Usage

- Provide a stable integer constant for downstream nodes (e.g., counts, indices).

## Notes & Tips

- Use the UI step (`4`) to adjust values quickly when fine precision is not required.