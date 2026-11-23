# Mask Fill Hole - Fill Internal Holes in Masks

**Node Purpose:** `Mask Fill Hole` fills internal holes in binary-like masks using morphological hole filling. Supports optional inversion to flip filled regions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `mask` | - | MASK | - | - | Input mask batch; 2D masks are expanded to batch `[B,H,W]`. |
| `invert_mask` | - | BOOLEAN | False | - | Invert the filled result when `True`.
| `threshold` | - | FLOAT | 0.20 | 0.00-1.00 | Binarization threshold with 0.01 step: pixels above it are foreground, the rest are background. |

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

- Input masks are binarized with `threshold` (default 0.20): values above it are foreground, the rest background. Soft or low-opacity masks get hard-cut, so feather downstream if needed.
- Requires SciPy; ensure the environment includes `scipy.ndimage` for hole filling.