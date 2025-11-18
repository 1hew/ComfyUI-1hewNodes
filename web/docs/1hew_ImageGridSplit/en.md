# Image Grid Split - Grid-Based Image Tiling

**Node Purpose:** `Image Grid Split` divides each image into a grid of `rows × columns` tiles and outputs either all tiles as a batch or a single tile per image based on `output_index`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `rows` | - | INT | 2 | 1–10 | Number of rows to split. |
| `columns` | - | INT | 2 | 1–10 | Number of columns to split. |
| `output_index` | - | INT | 0 | 0–100 | 0: output all tiles. >0: output the Nth tile per image (row-major order).

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Grid tiles batch or the selected tile per input image.

## Features

- Deterministic tiling: computes tile sizes via integer division.
- Row-major ordering: tiles are ordered by rows, then columns; `output_index=1` selects the first tile.
- Batch handling: for `output_index=0`, all tiles across the batch are stacked; otherwise one tile per input is returned.

## Typical Usage

- Patch processing: split large images into manageable tiles for model inference; set `output_index=0` to process all.
- Targeted tile: set `output_index` to a specific tile index when only a particular region is needed per image.

## Notes & Tips

- Ensure image dimensions are divisible by `rows` and `columns` for exact tiling; otherwise the last pixels in each axis are dropped due to integer division.
- RGBA inputs are converted to RGB tiles internally to ensure consistent shape.