# Image Plot - Arrange images by layout

**Node Purpose:** `Image Plot` arranges images in a horizontal, vertical, or grid layout. Also supports a video-collection style input (list of batches) to produce frame-wise collages across multiple sequences.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE or LIST | - | - | Input image batch or a list of batches for video-collection display. |
| `layout` | - | COMBO | `horizontal` | `horizontal`/`vertical`/`grid` | Arrangement mode. |
| `spacing` | - | INT | 10 | 0–1000 | Space between images. |
| `grid_columns` | - | INT | 2 | 1–100 | Number of columns when using `grid`. |
| `background_color` | - | STRING | `1.0` | Gray/HEX/RGB | Background color for the canvas. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Combined image batch; for video-collection input, outputs a sequence batch (`T×H×W×3`). |

## Features

- Standard plotting: converts batch frames to PIL, arranges per `layout`, and returns a single composed image.
- Video collection: accepts a Python list of batches and builds per-frame collages across groups, preserving device and dtype.
- Size normalization: bilinear resize to the minimal common size across images for clean alignment.
- Color parsing: supports gray (`0.0–1.0`), HEX (`#RRGGBB`), and `R,G,B` integer tuples.

## Typical Usage

- Create a side-by-side comparison: set `layout=horizontal` and adjust `spacing`.
- Stack vertically: set `layout=vertical` to form a column.
- Build grids: set `layout=grid` and tune `grid_columns` for tiled views.
- Show multiple sequences: pass a list of batches to produce a frame-aligned collage across inputs.

## Notes & Tips

- Video-collection detection is based on Python list input; tensor input is treated as a single batch.
- Output tensor uses channels-last layout and clamps to `[0,1]`.