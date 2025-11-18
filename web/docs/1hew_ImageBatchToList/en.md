# Image Batch to List - Split batch into single-frame list

**Node Purpose:** `Image Batch to List` converts a multi-frame image batch into a Python list where each element is a 1-frame batch (`B=1`). Preserves order and tensor properties.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image_batch` | - | IMAGE | - | - | Input image batch (`B×H×W×C`). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_list` | IMAGE_LIST | List of single-frame batches, in original order. |

## Features

- Deterministic splitting: slices by index and keeps per-frame tensors as `B=1` batches.
- Empty handling: returns an empty list when input is `None` or has `B=0`.
- Compatibility: downstream nodes that accept list inputs can consume the output directly.

## Typical Usage

- Convert a sequence to a list for nodes that iterate per-frame.
- Build composite workflows that branch per-frame from a batch source.

## Notes & Tips

- Each list item is shaped as a batch (`1×H×W×C`), not a raw image tensor.