# Int Split - Integer Split

**Node Purpose:** `Int Split` divides a total integer into a split count determined by `split_point`. It supports ratio-based splitting when `split_point` is within `0.0–1.0`, and absolute-value splitting otherwise.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `total` | - | INT | `20` | `1–10000` | Total integer to be divided. |
| `split_point` | - | FLOAT | `0.5` | `0.0–10000.0` | Ratio or absolute split value. `0.0–1.0` uses ratio mode; `1.0` yields `1`; values >1.0 use absolute mode and are clamped to `total`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int_total` | INT | Echo of the input `total`. |
| `int_split` | INT | Computed split count from `split_point`. |

## Features

- Ratio mode: `0.0–1.0` multiplies `total` by `split_point`, then casts to `int`.
- Special case: `split_point=1.0` produces `1` for single-block semantics.
- Absolute mode: values >`1.0` are cast to `int` and clamped to `total`.
- Non-negative guarantee: final value is clamped at minimum `0`.
- Core logic: ratio vs absolute split computation and clamping.

## Typical Usage

- Balanced split: set `split_point=0.25` to obtain a quarter-sized split from `total`.
- Fixed count: set `split_point=8` to request eight units (clamped to `total`).
- Single unit: set `split_point=1.0` to produce `1` regardless of `total`.

## Notes & Tips

- Provide `total≥1` to keep the split meaningful.
- Large absolute `split_point` values are safely clamped to `total`.