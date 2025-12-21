# Any Switch Int - Indexed Pass-Through (1…10)

**Node Purpose:** `Any Switch Int` selects one of up to ten inputs by an integer index and emits that value. Implements lazy evaluation to compute only the selected input.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `select` | - | INT | 1 | 1–10 | Index of the input to emit. |
| `input_1…input_10` | optional | ANY (`*`) | - | - | Candidate inputs; only the selected is evaluated (lazy).

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | ANY (`*`) | Selected input value.
| `select` | INT | The raw input select value.

## Features

- Select range: clamped to `[1, SELECT_MAX]` where `SELECT_MAX=10`.
- Lazy ports: evaluation restricted to the selected input.
- Robust selection: emits `None` when the selected input is absent.

## Typical Usage

- Mode multiplexer: route to one of many alternative subgraphs with a single index.
- Parameter bank: choose between preset values or resources indexed by `select`.
- With emptiness checks: combine with `Any Empty Int` to compute indices from presence.

## Notes & Tips

- Provide the specific `input_{k}` that matches `select`; unspecified indices result in `None` output.
- Use `select` as a safe, clamped control signal; out-of-range values are automatically adjusted.