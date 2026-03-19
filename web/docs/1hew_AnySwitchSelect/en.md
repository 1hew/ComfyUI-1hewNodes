# Any Switch Select - Indexed Pass-Through (1…10)

**Node Purpose:** `Any Switch Select` chooses one of up to ten inputs by integer `select`, emits the chosen value, and also returns the effective clamped `select`. It uses lazy evaluation so only the selected input branch is computed.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `select` | - | INT | 1 | 1-10 | Input number to emit; `1` maps to `input_1`. |
| `input_1…input_10` | optional | ANY (`*`) | - | - | Candidate inputs; only the selected one is evaluated lazily. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | ANY (`*`) | Selected input value. |
| `select` | INT | Effective clamped `select` value used for routing. |

## Features

- Select range: `select` is clamped to `[1, 10]`.
- Lazy ports: evaluation is restricted to the selected input.
- Robust behavior: emits `None` if the selected input is not connected.
- Consistent naming: node, file, and parameter names all use `select`.

## Typical Usage

- Mode multiplexer: route to one of many alternative subgraphs with a single `select`.
- Parameter bank: choose between preset values or resources by index.
- Presence-based routing: combine with `Any Empty Int` to derive `select` values.

## Notes & Tips

- Provide the specific `input_k` that matches `select`; otherwise `output` will be `None`.
- Out-of-range values are automatically corrected before routing.
