# Mask Batch Range - Slice masks by start and count

**Node Purpose:** `Mask Batch Range` extracts a contiguous segment from a mask batch using `start_index` and `num_frame`. When `start_index` is out of range (≥ total), the output is an empty batch. When `num_frame` exceeds remaining masks, only the remaining masks are returned.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch. |
| `start_index` | - | INT | 0 | 0-8192 | Starting frame index; if `start_index ≥ total`, returns empty. |
| `num_frame` | - | INT | 1 | 1-8192 | Number of masks to take; clipped to `total - start`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Selected contiguous segment of masks; empty when out-of-range or zero remain. |

## Features

- Bounds-safe: returns empty when `start_index ≥ total`; clips `num_frame` to `total - start`.
- Async slicing: performs slicing in a worker thread.
- Empty handling: returns an empty batch with matching dtype/device when needed.

## Typical Usage

- Align mask segments with corresponding image ranges.

## Notes & Tips

- Mirrors the behavior of `Image Batch Range` for mask tensors.
- If `start_index ≥ total`, output is empty; if `num_frame > total - start`, only remaining are returned; if `total = 0`, output is empty.