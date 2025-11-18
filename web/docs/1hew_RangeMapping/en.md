# Range Mapping - Map 0–1 to [min, max]

**Node Purpose:** `Range Mapping` maps a normalized value `value ∈ [0,1]` to a target range `[min, max]`, with configurable decimal rounding or integer casting.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `value` | - | FLOAT | `1.0` | `0.0–1.0` | Normalized input value; slider control for precise mapping. |
| `min` | - | FLOAT | `0.0` | large range | Lower bound of mapping range. |
| `max` | - | FLOAT | `1.0` | large range | Upper bound of mapping range. |
| `rounding` | - | INT | `3` | `0–10` | Decimal places; `0` casts to integer. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `float` | FLOAT | Mapped value after rounding/int casting. |
| `int` | INT | Integer form of the mapped value. |

## Features

- Linear mapping: `actual = min + value × (max − min)`.
- Rounding control: when `rounding>0`, rounds to `rounding` decimals; otherwise casts to `int`.
- Dual outputs: returns both float and int forms for flexible downstream usage.
- Wide bounds: accepts very large positive/negative bounds for `min`/`max`.

## Typical Usage

- Parameter scaling: drive model thresholds by mapping a slider to `[min, max]`.
- Integer steps: set `rounding=0` to generate enumerated indices from a normalized input.
- Inverted ranges: swap `min` and `max` to map decreasing scales.

## Notes & Tips

- Keep `value` within `0–1` for well-defined behavior; the UI range enforces this.
- Choose `rounding` according to downstream precision requirements; integer casting helps produce discrete categories.