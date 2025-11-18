# Any Empty Int - Map Emptiness to Integer

**Node Purpose:** `Any Empty Int` evaluates emptiness and outputs `empty` or `not_empty` integer values accordingly. Uses the same type-aware rules as `Any Empty Bool` and provides configurable mappings.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `any` | - | ANY (`*`) | - | - | Value to test. |
| `empty` | - | INT | 0 | -999999–999999 | Output when input is empty. |
| `not_empty` | - | INT | 1 | -999999–999999 | Output when input is non-empty.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `int` | INT | `empty` if input is empty, else `not_empty`.

## Features

- Type-aware emptiness identical to `Any Empty Bool`.
- Configurable mapping: set `empty` and `not_empty` to suit downstream logic.
- Robust fallback: on exceptions, outputs `empty`.

## Typical Usage

- Switch indices: drive integer-controlled selectors or loop counts based on input presence.
- Numeric flags: emit `0/1` or custom codes for downstream condition checks.
- Pipeline defaults: pair with `Any Switch Int` or arithmetic nodes to route flows.

## Notes & Tips

- Use wide ranges to encode richer states if needed (e.g., `-1` for empty, `+1` for non-empty).
- Shares the same recursive and zero-check behavior for tensors/arrays and containers.