# Image Grid Split - Grid-Based Image Tiling

**Node Purpose:** `Image Grid Split` divides each image into a grid of `rows × columns` tiles. You can output all tiles as a batch or select a specific tile per image using `index`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `rows` | - | INT | 2 | 1–10 | Number of rows to split. |
| `columns` | - | INT | 2 | 1–10 | Number of columns to split. |
| `index` | - | INT | 0 | -100–100 | Python-style index to select a specific tile per image (e.g., 0 for first, -1 for last). Ignored if `all` is True. |
| `all` | - | BOOL | False | - | If True, outputs all tiles from the split. If False, outputs only the tile selected by `index`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Grid tiles batch or the selected tile per input image.

## Features

- Deterministic tiling: computes tile sizes via integer division.
- Row-major ordering: tiles are ordered by rows, then columns.
- Python-style indexing: Use `index` to select tiles (0-based, negative indexing supported).
- Batch handling: Toggle `all` to True to stack all tiles across the batch; otherwise one tile per input is returned.

## Typical Usage

- Patch processing: split large images into manageable tiles for model inference; set `all=True` to process all.
- Targeted tile: set `all=False` and specify `index` to retrieve a particular region from each image.

## Notes & Tips

- Ensure image dimensions are divisible by `rows` and `columns` for exact tiling; otherwise the last pixels in each axis are dropped due to integer division.
- RGBA inputs are converted to RGB tiles internally to ensure consistent shape.
