# Any Switch Bool - Conditional Pass-Through

**Node Purpose:** `Any Switch Bool` selects between `on_true` and `on_false` inputs based on a boolean, emitting one value. Implements lazy evaluation so only the chosen branch is computed.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `boolean` | - | BOOLEAN | True | - | Control flag. |
| `on_true` | optional | ANY (`*`) | - | - | Value emitted when `boolean=True` (lazy). |
| `on_false` | optional | ANY (`*`) | - | - | Value emitted when `boolean=False` (lazy).

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | ANY (`*`) | Selected input value.

## Features

- Lazy ports: only the active input port is evaluated.
- Simple selection: returns `on_true` when `boolean=True`, otherwise `on_false`.
- Robust fallback: on exceptions, returns `None`.

## Typical Usage

- Gate expensive operations: compute heavy branches only when enabled by a condition.
- Default routing: pair with `Any Empty Bool` to route present vs empty inputs.
- Mode toggles: quickly switch between two alternative processing paths.

## Notes & Tips

- Provide either or both branch inputs; the node emits whichever is active.
- Combine with validation nodes to create safe, branch-controlled pipelines.