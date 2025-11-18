# Image List to Batch - Merge list into padded batch

**Node Purpose:** `Image List to Batch` merges a list of images into a single batch. Normalizes shape to channels-last and pads each image to the maximum height/width using zeros so concatenation is possible.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image_list` | list | IMAGE/LIST | - | - | Input images as a list or single image. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_batch` | IMAGE | Combined batch (`B×H×W×C`) after zero padding to common size. |

## Features

- Flexible input: accepts a single tensor or a list/tuple; non-tensor items are ignored.
- Shape normalization: converts `H×W×C` tensors to `1×H×W×C` for batching.
- Size reconciliation: pads each image to the maximum `H` and `W` across inputs using zeros (black).
- Fallback: returns an empty batch of shape `(0, 64, 64, 3)` when no valid images are found; returns the single image unchanged when only one is present.

## Typical Usage

- Combine variable-sized frames into a batch for downstream model consumption.
- Merge results from different branches into one contiguous batch.

## Notes & Tips

- Padding uses constant value `0` and preserves device/dtype of each input before concatenation.