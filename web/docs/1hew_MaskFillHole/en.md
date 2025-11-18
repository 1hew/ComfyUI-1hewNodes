# Mask Fill Hole - Fill Internal Holes in Masks

**Node Purpose:** `Mask Fill Hole` fills internal holes in binary-like masks using morphological hole filling. Supports optional inversion to flip filled regions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch; 2D masks are expanded to batch `[B,H,W]`. |
| `invert_mask` | - | BOOLEAN | False | - | Invert the filled result when `True`.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK | Hole-filled mask batch.

## Features

- Batch processing: runs per item with controlled concurrency.
- Hole filling: uses `scipy.ndimage.binary_fill_holes` with a 2D connectivity structure.
- Inversion: optional inversion of the filled binary mask.

## Typical Usage

- Clean segmentation: fill internal voids in masks from segmentation outputs.
- Prepare for compositing: ensure contiguous regions for subsequent paste/blend operations.

## Notes & Tips

- Input masks are thresholded at `>127` internally to define binary regions.
- Requires SciPy; ensure the environment includes `scipy.ndimage` for hole filling.