# Image Batch Split - Split batch into two parts

**Node Purpose:** `Image Batch Split` splits an image batch into two parts by a count, either taking from the start or the end. Handles edge cases when `take_count` ≥ batch size and performs non-blocking slicing.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `take_count` | - | INT | 8 | 1-1024 | Number of frames in the taken part. |
| `from_start` | - | BOOLEAN | `False` | - | If `True`, take from the start; otherwise take from the end. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_1` | IMAGE | First part of the split. |
| `image_2` | IMAGE | Second part of the split. |

## Features

- Directional split: `from_start=True` takes the first `take_count` frames; otherwise the last `take_count` frames.
- Edge handling: when `take_count >= batch_size`, one output is the full batch and the other is empty.
- Async slicing: uses worker threads for slicing to avoid UI blocking.

## Typical Usage

- Separate a prefix vs the remainder: `from_start=True`, `take_count=N`.
- Extract last N frames while keeping the rest: `from_start=False`, `take_count=N`.

## Notes & Tips

- Empty outputs preserve dtype/device and shape semantics for compatibility downstream.