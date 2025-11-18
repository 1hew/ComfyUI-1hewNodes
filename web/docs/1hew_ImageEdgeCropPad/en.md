# Image Edge Crop/Pad - Per-edge crop and intelligent padding

**Node Purpose:** `Image Edge Crop/Pad` performs inward cropping (negative values) and outward padding (positive values) per edge, with a uniform mode for symmetric edits. It supports multiple padding strategies (`extend`, `mirror`, `edge`, `average`) and rich color formats, and ensures amounts align to a chosen multiple via `divisible_by`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch (`B×H×W×C`). |
| `uniform_amount` | - | FLOAT | 0.0 | -8192–8192 | Symmetric crop/pad applied to all edges; `<1` treated as ratio of size; negative → crop; positive → pad. |
| `top_amount` | - | FLOAT | 0.0 | -8192–8192 | Crop/pad amount for the top edge; `<1` treated as ratio of height. |
| `bottom_amount` | - | FLOAT | 0.0 | -8192–8192 | Crop/pad amount for the bottom edge; `<1` treated as ratio of height. |
| `left_amount` | - | FLOAT | 0.0 | -8192–8192 | Crop/pad amount for the left edge; `<1` treated as ratio of width. |
| `right_amount` | - | FLOAT | 0.0 | -8192–8192 | Crop/pad amount for the right edge; `<1` treated as ratio of width. |
| `pad_color` | - | STRING | `0.0` | formats | Padding color/strategy: grayscale/HEX/RGB/name or `extend`/`mirror`/`edge`/`average` (supports abbreviations). |
| `divisible_by` | - | INT | 8 | 1–1024 | Round each edge amount to a multiple of this value to keep aligned sizes. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Cropped/padded image batch with per-edge operations applied. |
| `mask` | MASK | Binary mask of edited areas; processed regions are white, original areas are black. |

## Features

- Per-edge control: combine `uniform_amount` with per-edge overrides for precise edits.
- Ratio or pixels: amounts `<1` are treated as ratios; otherwise interpreted as pixel values.
- Size alignment: quantizes each amount to `divisible_by` to maintain model-friendly dimensions.
- Rich padding strategies:
  - `extend`: replicate edge pixels (`replicate`).
  - `mirror`: reflect edge pixels (`reflect` with segmented extension for large amounts).
  - `edge`: fill with the average color of each edge (top/bottom/left/right).
  - `average`: fill with the global average color.
- Color formats: grayscale (`0.0`–`1.0`), HEX (`#RRGGBB` or `RGB`), RGB (`255,0,0` or `1.0,0.0,0.0`), and color names; single-letter aliases supported.

## Mask Rules

- Crop-only: the mask marks cropped-away regions as white and original areas as black.
- Pad-only: the mask matches output size; padded areas are white and original content areas are black.
- Mixed edits: the mask expresses the union of edited regions (cropped + padded) as white.
- All-zero amounts: returns the original image with an all-black mask.

## Typical Usage

- Uniform inward crop by 10%: set `uniform_amount=-0.1` and keep edge amounts at `0.0`.
- Outward padding by 32 px with mirror content: set `uniform_amount=32`, `pad_color=mirror`, `divisible_by=8`.
- Content-aware padding: set `pad_color=edge` or `average` to harmonize borders.
- Mixed per-edge edits: combine `top_amount`, `bottom_amount`, `left_amount`, `right_amount` with `uniform_amount` for asymmetric operations.

## Notes & Tips

- Negative values crop inward; positive values pad outward. Ratios are relative to width/height per edge.
- Use `divisible_by` to keep output dimensions aligned to model requirements (e.g., 8 or 16).
- When using `mirror`, large paddings are applied iteratively to respect `reflect` constraints.