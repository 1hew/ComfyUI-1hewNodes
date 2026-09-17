# Multi Mask Math Ops - Binary Mask Operations

**Node Purpose:** `Mask Math Ops` performs per-pixel binary-like operations across multiple masks: `or`, `and`, `subtract (a-b)`, `subtract (b-a)`, and `xor`. Supports dynamic `mask_1..mask_N` inputs, batch cycling, and size alignment.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask_1` | - | MASK | - | - | First mask batch. |
| `mask_2..mask_N` | optional (dynamic) | MASK | - | - | Additional mask batches; resized to `mask_1` size when needed. |
| `operation` | - | COMBO | `or` | `or` / `and` / `subtract (a-b)` / `subtract (b-a)` / `xor` | Operation type. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Resulting mask batch. |

## Features

- Dynamic ports: connecting the last `mask_X` port automatically appends the next mask port.
- Batch cycling: aligns differing batch sizes by indexing modulo per item.
- Size alignment: resizes additional masks to `mask_1` with Lanczos when shapes differ.
- Operations:
- `and`: min of arrays.
- `or`: max of arrays.
- `subtract (a-b)`: `clip(a-b, 0, 1)`.
- `subtract (b-a)`: `clip(b-a, 0, 1)`.
- `xor`: absolute difference.

## Typical Usage

- Combine masks: union/intersection of regions for composite selection.
- Cutouts: subtract one region from another to refine selections.
- Edge emphasis: `xor` highlights differences between masks.

## Notes & Tips

- Inputs are treated as float `[0,1]` arrays internally; operations are per-channel on single-channel masks.
- Ensure consistent semantic meaning of mask intensity (white=selected) across inputs before combining.
