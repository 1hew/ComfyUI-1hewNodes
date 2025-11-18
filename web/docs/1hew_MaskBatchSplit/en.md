# Mask Batch Split - Split mask batch into two parts

**Node Purpose:** `Mask Batch Split` splits a mask batch into two parts by a count, optionally taking from the start or the end. Handles edge cases and performs non-blocking slicing.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch. |
| `take_count` | - | INT | 8 | 1-1024 | Number of frames in the taken part. |
| `from_start` | - | BOOLEAN | `False` | - | If `True`, take from the start; otherwise take from the end. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask_1` | MASK | First part of the split. |
| `mask_2` | MASK | Second part of the split. |

## Features

- Directional split: `from_start=True` takes the first `take_count`; otherwise the last `take_count`.
- Edge handling: when `take_count >= batch_size`, one output is full and the other is empty.
- Async slicing: uses worker threads to avoid blocking.

## Typical Usage

- Separate mask prefixes from remaining segments.
- Keep last N masks while retaining the rest.

## Notes & Tips

- Prints debug information on shapes and parameters, aiding graph debugging.