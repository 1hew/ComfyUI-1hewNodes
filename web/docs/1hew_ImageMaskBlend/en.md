# Image Mask Blend - Mask-Guided Image Compositing

**Node Purpose:** `Image Mask Blend` composes images using a grayscale mask with professional control over hole filling, morphological expansion/erosion, Gaussian feather, inversion, opacity scaling, and rich background strategies. It outputs both the blended `image` and the processed `mask` with robust batch behavior.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `mask` | - | MASK | - | - | Input mask batch; resized to match `image` size when needed. |
| `fill_hole` | - | BOOLEAN | true | - | Fill mask holes to ensure continuous shapes. |
| `invert` | - | BOOLEAN | false | - | Invert the selection after morphology/feather steps. |
| `feather` | - | INT | 0 | 0–50 | Gaussian blur radius applied to the mask. |
| `opacity` | - | FLOAT | 1.0 | 0.0–1.0 | Scales mask strength; 0 disables, 1 preserves full mask. |
| `expansion` | - | INT | 0 | -100–100 | Positive dilates, negative erodes the mask (pixels). |
| `background_color` | - | STRING | 1.0 | grayscale/HEX/RGB/name/`edge`/`average`/`mk`/`mask` | Background color source for non-mask areas. |
| `background_opacity` | - | FLOAT | 1.0 | 0.0–1.0 | Mix strength between original image and background in non-mask areas. |
| `output_mask_invert` | - | BOOLEAN | false | - | Invert only the output `mask` (does not affect blending). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Final composite: `final = image*mask + mixed_bg*(1-mask)`. |
| `mask` | MASK | Processed mask (`mask_gray * opacity`), optionally inverted at output. |

## Features

- Robust batching: different image/mask counts are aligned by cycling the smaller batch; sizes are matched via Lanczos resizing.
- Morphology: optional hole filling, dilation (`expansion>0`), erosion (`expansion<0`), and Gaussian feathering for soft edges.
- Inversion and opacity: invert after morphology/feather; scale with `opacity` for graded selection.
- Background strategies: parse color from grayscale/HEX/RGB, named colors, global `average`, image `edge` color, or average within the `mask` region (`mk`/`mask`).
- Deterministic compositing: `mixed_bg = (1-background_opacity)*image + background_opacity*background`.

## Typical Usage

- Clean selections: enable `fill_hole`, then set `expansion` and `feather` to refine edges.
- Soft composites: use moderate `feather` and `opacity<1` for smooth transitions.
- Color-matched background: set `background_color=average` or `mk` to harmonize non-selected areas.
- Output control: toggle `output_mask_invert` when a complementary mask is required downstream.

## Notes & Tips

- Provide masks at or near image size for best fidelity; resizing is handled automatically.
- Negative `expansion` shrinks selections; combine with small `feather` to avoid aliasing.
- Named colors and shorthand codes are supported; invalid inputs fall back to white.