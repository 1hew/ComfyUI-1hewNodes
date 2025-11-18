# Image Tile Split - Grid Tiling with Overlap

**Node Purpose:** `Image Tile Split` splits a single image into a grid of tiles with optional overlap. Supports auto grid estimation, named presets like `2x3`, custom rows/cols, and tile sizes aligned to `divisible_by` multiples. Outputs a tile batch and a `tile_meta` dictionary for lossless merging.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image; only the first item of a batch is used. |
| `split_mode` | - | COMBO | `auto` | `auto` / `custom` / `2x2` / `2x3` / `2x4` / `3x2` / `3x3` / `3x4` / `4x2` / `4x3` / `4x4` | Grid selection strategy. |
| `overlap_amount` | - | FLOAT | 0.05 | 0.0–512.0 | Overlap as ratio (≤1.0) or pixels (>1.0). |
| `custom_rows` | optional | INT | 2 | 1–10 | Rows when `split_mode=custom`. |
| `custom_cols` | optional | INT | 2 | 1–10 | Cols when `split_mode=custom`. |
| `divisible_by` | optional | INT | 8 | 1–64 | Tile sizes rounded up to multiples; affects final overlap.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `tile` | IMAGE | Concatenated tile batch (`B = rows*cols`). |
| `tile_meta` | DICT | Metadata for merge: `tile_metas` (per-tile), `original_size`, `grid_size`, `tile_width`, `tile_height`, `rows`, `cols`, `overlap_amount`, `overlap_mode`, `overlap_width`, `overlap_height`, `split_mode`, `divisible_by`.

## Features

- Auto grid: estimates rows/cols so each tile is near `1024×1024` and square-ish.
- Overlap modes: ratio vs pixels, persisted as `overlap_mode` in metadata.
- Size alignment: base tile size plus overlap, then rounded to `divisible_by`; recomputes final overlap after rounding.
- Perfect coverage: computes precise tile positions so the last tile aligns with the image boundary.
- Edge padding: edge tiles smaller than tile size are padded to full size to keep uniform outputs.

## Typical Usage

- Patch workflows: split large images for per-tile processing; overlap `0.05` to reduce seams.
- Model constraints: set `divisible_by=8/16` to meet network stride restrictions.
- Controlled layouts: choose `2x3` or `custom` rows/cols to match downstream tiling logic.

## Notes & Tips

- `tile_meta['tile_metas']` includes `crop_region`, `position (col,row)`, and `actual_crop_size`; pass this directly to `Image Tile Merge`.
- If input is a batch, only the first image is used; feed single-image batches to avoid confusion.