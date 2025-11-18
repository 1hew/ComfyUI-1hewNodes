# Image Batch Range - Slice frames by start and count

**Node Purpose:** `Image Batch Range` extracts a contiguous segment from an image batch using `start_index` and `num_frame`. When `start_index` is out of range (≥ total), the output is an empty batch. When `num_frame` exceeds the remaining frames, only the remaining frames are returned.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `start_index` | - | INT | 0 | 0-8192 | Starting frame index. If `start_index ≥ total`, returns empty. |
| `num_frame` | - | INT | 1 | 1-8192 | Number of frames to take. Clipped to `total - start`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Selected contiguous segment of frames, or empty batch when out-of-range or zero remain. |

## Features

- Bounds-safe: returns empty when `start_index ≥ total`; clips `num_frame` to `total - start`.
- Async slicing: performs the slice on a worker thread to avoid blocking.
- Empty handling: returns matching dtype/device empty batch when no frames are available.

## Typical Usage

- Extract a scene segment from a sequence: set `start_index=S`, `num_frame=N`.
- Chain multiple ranges to build subclips or training windows.

## Notes & Tips

- If `start_index ≥ total`, output is empty.
- If `num_frame > total - start`, only `total - start` frames are returned.
- If `total = 0`, output is an empty batch.