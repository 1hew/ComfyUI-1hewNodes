# Mask Separate - Separate Mask Regions

**Node Purpose:** `Mask Separate` separates the input mask based on connected regions, outputting a batch of separated masks and their count. It supports area filtering and multiple sorting modes.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch. |
| `threshold` | - | FLOAT | `0.5` | 0.0–1.0 | Binarization threshold. |
| `min_area` | - | INT | `1` | 1-100000000 | Minimum connected region area. Regions smaller than this will be ignored. |
| `sort_mode` | - | COMBO | `top_to_bottom` | `top_to_bottom` / `left_to_right` / `area_desc` | Sorting mode: top to bottom, left to right, or area descending. |
| `connectivity` | - | COMBO | `8` | `8` / `4` | Connectivity: 8-connected or 4-connected. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Batch of separated masks. |
| `count` | INT | The number of separated masks. |

## Features

- Connected component analysis: extracts independent connected regions from the binarized mask.
- Area filtering: filters out small noise or fragments using `min_area`.
- Sorted output: sorts the separated masks by spatial position (top-to-bottom/left-to-right) or area size (descending).
- Batch processing: separates each frame in the input batch and concatenates all results into a single output batch.

## Typical Usage

- Instance segmentation: splits a single mask containing multiple objects into individual object masks.
- Noise removal: combined with `min_area` to extract main targets and remove tiny noise.
- Sequential processing: ensures separated masks have a stable order via `sort_mode` for downstream sequential processing.

## Notes & Tips

- The `threshold` determines which pixels are considered foreground; the default 0.5 is usually sufficient.
- `connectivity` determines whether diagonally adjacent pixels are considered part of the same region; 8-connected is more lenient.