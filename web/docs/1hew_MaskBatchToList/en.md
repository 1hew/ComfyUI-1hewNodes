# Mask Batch to List - Split mask batch into single-frame list

**Node Purpose:** `Mask Batch to List` converts a multi-frame mask batch into a Python list where each element is a 1-frame batch (`B=1`). Preserves order and tensor properties.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask_batch` | - | MASK | - | - | Input mask batch (`B×H×W`). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask_list` | MASK_LIST | List of single-frame mask batches, in original order. |

## Features

- Deterministic splitting: slices by index and keeps per-frame tensors as `B=1` batches.
- Empty handling: returns an empty list when input is `None` or has `B=0`.

## Typical Usage

- Convert a mask sequence to a list for nodes that operate per-frame.

## Notes & Tips

- Each list item is shaped as a batch (`1×H×W`).