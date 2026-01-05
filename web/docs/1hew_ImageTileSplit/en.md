# Image Tile Split - Image Tile Splitter

**Node Purpose:** `Image Tile Split` splits a large image into smaller tiles based on various modes, supporting overlap, custom grids, and preset resolutions. It ensures seamless reconstruction by handling overlaps and precise positioning.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | required | IMAGE | - | - | Input image to be split. |
| `get_tile_size` | optional | IMAGE | - | - | Reference image; if connected, its dimensions are used as the tile size (overrides `mode`). |
| `mode` | - | COMBO | `auto` | `auto` / `grid` / Presets... | Split mode. `auto` calculates optimal grid; `grid` uses manual rows/cols; Presets use fixed resolutions. |
| `overlap_amount` | - | FLOAT | 0.05 | 0.0-512.0 | Overlap between tiles. <=1.0 is ratio, >1.0 is pixels. |
| `grid_row` | - | INT | 2 | 1-10 | Number of rows in `grid` mode. |
| `grid_col` | - | INT | 2 | 1-10 | Number of columns in `grid` mode. |
| `divisible_by` | - | INT | 8 | 1-1024 | Ensures tile dimensions are multiples of this value (ignored in Preset/Reference modes). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `tile` | IMAGE | Batch of image tiles. |
| `tile_meta` | DICT | Metadata containing tile positions, original size, overlap info, etc., for reconstruction. |
| `bbox_mask` | MASK | Mask batch aligned with `tile`; each mask marks the tile crop region on the original image size (crop region = 1, elsewhere = 0). |

## Features

- **Flexible Modes**:
  - `auto`: Automatically calculates the best grid layout targeting ~1024px tiles.
  - `grid`: Manually specify the number of rows and columns.
  - **Presets**: Choose from a list of predefined resolutions (e.g., "1024x1024").
  - **Reference Image**: Use the dimensions of another image (`get_tile_size`) as the tile size.
- **Overlap Handling**: Supports both proportional (e.g., 0.05 for 5%) and absolute pixel overlap.
- **Size Constraints**: `divisible_by` ensures tile sizes meet model requirements (e.g., multiples of 8 or 64) in auto/grid modes.
- **Precise Positioning**: Calculates exact crop coordinates to ensure full coverage and seamless merging.

## Notes & Tips

- **Reference Image Priority**: If `get_tile_size` is connected, it takes precedence over the `mode` selection.
- **Preset Behavior**: Preset resolutions and Reference Image sizes remain fixed; `divisible_by` is applied in `auto/grid` modes for size alignment.
- **Reconstruction**: The `tile_meta` output is essential for the `Image Tile Merge` node to reconstruct the original image correctly.
