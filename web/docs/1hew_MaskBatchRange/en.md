# Mask Batch Range - Sample masks by start and step

**Node Purpose:** `Mask Batch Range` samples masks from a batch using `start_index`, `step`, and `num_frame`. `num_frame` means the final number of masks to output. When `num_frame = 0`, it keeps sampling from `start_index` to the end using `step`. If `start_index` is out of range (≥ total), the output is an empty batch.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch. |
| `start_index` | - | INT | 0 | 0-8192 | Starting frame index; if `start_index ≥ total`, returns empty. |
| `step` | - | INT | 1 | 1-8192 | Sampling stride. `1` means contiguous, `2` means take every other mask. |
| `num_frame` | - | INT | 1 | 0-8192 | Final number of masks to take. `0` means keep taking with `step` until the end. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Sampled mask batch based on start/step settings, or empty when out-of-range. |

## Features

- Bounds-safe: returns empty when `start_index ≥ total`.
- Step sampling: starts at `start_index` and samples using `step`.
- To-end mode: when `num_frame = 0`, sampling continues until the final mask.
- Async slicing: performs slicing in a worker thread.
- Empty handling: returns an empty batch with matching dtype/device when needed.

## Typical Usage

- Take the first N masks contiguously: `start_index=0`, `step=1`, `num_frame=N`.
- Sample every K masks from a point: `start_index=S`, `step=K`, `num_frame=N`.
- Sample from a point to the end: `start_index=S`, `step=K`, `num_frame=0`.

## Notes & Tips

- Mirrors the behavior of `Image Batch Range` for mask tensors.
- If `start_index ≥ total`, output is empty.
- If `num_frame > 0` and fewer masks are available, only the available masks are returned.
- When `step = 1`, behavior remains compatible with the old contiguous slice behavior.
- If `total = 0`, output is empty.