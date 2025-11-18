# Image Tile Split Preset - Tiling by Predefined Sizes

**Node Purpose:** `Image Tile Split Preset` splits an image into tiles of predefined sizes. Supports a user-selected size or `auto` selection that optimizes coverage efficiency, aspect ratio match, and tile count. Produces tiles and a rich `tile_meta` for seamless merging.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image; only the first item of a batch is used. |
| `overlap_amount` | - | FLOAT | 0.05 | 0.0–512.0 | Overlap as ratio (≤1.0) or pixels (>1.0). |
| `tile_preset_size` | - | COMBO | `auto` | `auto` + listed sizes | Predefined tile size selection.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `tile` | IMAGE | Concatenated tile batch. |
| `tile_meta` | DICT | Metadata including `tile_metas`, `original_size`, `grid_size`, `tile_width`, `tile_height`, `rows`, `cols`, `overlap_amount`, `overlap_mode`, `overlap_width`, `overlap_height`, `tile_preset_size`, `selected_size_info`, `predefined_size`.

## Features

- Fixed selection: when a specific preset is chosen, computes grid and efficiency stats.
- Auto selection: scores each preset by efficiency, aspect difference, and tile count to pick the best.
- Coverage grid: calculates precise positions to perfectly cover the image, considering overlap.
- Edge padding: tiles at the border are padded to full size when partial.

## Typical Usage

- Constrain tile sizes: enforce known tile dimensions for downstream models that expect fixed inputs.
- Smart auto: prefer `auto` to balance coverage efficiency and aspect ratio match without manual tuning.

## Notes & Tips

- `selected_size_info` provides `efficiency`, `waste_pixels`, `total_tiles`, `aspect_ratio`, and `aspect_match_score` for insight.
- Only the first image in a batch is tiled; feed single-image batches to avoid ambiguity.