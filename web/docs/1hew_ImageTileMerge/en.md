# Image Tile Merge - Seam-Aware Tile Composition

**Node Purpose:** `Image Tile Merge` reconstructs the full image from tiles produced by tile-splitting nodes using the provided `tile_meta`. Applies seam-aware cosine weight masks over overlaps controlled by `blend_strength` for smooth merging.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `tile` | - | IMAGE | - | - | Tile batch or single tile; accepts `B×H×W×3` or `H×W×3`. |
| `tile_meta` | - | DICT | - | - | Metadata dictionary from tile-split nodes. |
| `blend_strength` | - | FLOAT | 1.0 | 0.0–1.0 | Controls overlap weighting; 0 disables blending.

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Merged full image.

## Features

- Tile count handling: trims or pads tiles to expected `rows*cols` using last tile or zeros to ensure completeness.
- Cosine weight masks: generates smooth ramp weights along overlapped edges based on position and grid size.
- Weighted composition: accumulates weighted tile contributions and normalizes by the total weight to avoid seams.
- Overlap-aware: uses `overlap_width/height` from `tile_meta` to size blending ramps; scales by `blend_strength`.

## Typical Usage

- Merge after split: feed `tile` and `tile_meta` from `Image Tile Split` or `Image Tile Split Preset`, adjust `blend_strength` (e.g., 0.4–0.8) to minimize seams.
- Robust to variations: if tiles exceed or are fewer than expected, automatic trim/pad maintains grid integrity.

## Notes & Tips

- `tile_meta['tile_metas'][i]` includes `crop_region`, `position (col,row)`, and `actual_crop_size`, which determine placement and crop trimming.
- For large overlaps, reduce `blend_strength` when sharper boundaries are desired; increase for softer blending.