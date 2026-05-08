# Image Batch Uniform - Uniform sampling across the batch

**Node Purpose:** `Image Batch Uniform` samples evenly across the entire image batch, using `num_frame` as the final output count. `num_frame=0` means return all images.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `num_frame` | - | INT | 4 | 0-8192 | Final output count; `0` means return all images. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Uniformly sampled image batch. |

## Features

- `num_frame=0`: return all images.
- `num_frame=1`: return only the first image.
- `num_frame=2`: return the first and last images.
- `num_frame>=batch_size`: return all images.
- Otherwise: sample uniformly across the whole batch.
- Empty-safe: returns an empty batch when the input batch is empty.

## Typical Usage

- Build representative thumbnails: set `num_frame=4`
- Keep only the first frame: set `num_frame=1`
- Keep only the first and last frames: set `num_frame=2`
- Keep everything: set `num_frame=0`

## Notes & Tips

- This node always samples across the full batch and does not accept a sub-range.
- When `num_frame` is smaller than the batch size, sampling tries to cover the start, end, and middle evenly.
