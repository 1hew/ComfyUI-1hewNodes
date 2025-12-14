# Multi Mask Math Ops - Binary Mask Operations

**Node Purpose:** `Mask Math Ops` performs per-pixel binary-like operations between two masks: `or`, `and`, `subtract (a-b)`, `subtract (b-a)`, and `xor`. Handles batch cycling and resizes masks to match when sizes differ.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask_1` | - | MASK | - | - | First mask batch. |
| `mask_2` | - | MASK | - | - | Second mask batch; resized to `mask_1` size when needed. |
| `operation` | - | COMBO | `or` | `or` / `and` / `subtract (a-b)` / `subtract (b-a)` / `xor` | Operation type.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Resulting mask batch.

## Features

- Batch cycling: aligns differing batch sizes by indexing modulo per item.
- Size alignment: resizes `mask_2` to `mask_1` with Lanczos when shapes differ.
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