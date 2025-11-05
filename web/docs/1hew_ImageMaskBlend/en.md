# Image Mask Blend

**Node Function:** The `Image Mask Blend` node blends an image with a background color using a mask. It supports hole filling, expansion/erosion, feathering, inversion, opacity scaling, and background blending strength. Batch sizes can differ and will cycle to the max batch size.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image batch |
| `mask` | Required | MASK | - | - | Mask batch (grayscale 0–1) |
| `fill_hole` | - | BOOLEAN | True | True/False | Fill holes in the mask to ensure continuity |
| `invert` | - | BOOLEAN | False | True/False | Invert the selection after morphology/feathering |
| `feather` | - | INT | 0 | 0–50 | Gaussian blur radius for feathering (pixels) |
| `opacity` | - | FLOAT | 1.0 | 0.0–1.0 | Mask opacity scaling factor |
| `expansion` | - | INT | 0 | -100–100 | Positive dilates, negative erodes (pixels) |
| `background_color` | - | STRING | "1.0" | grayscale/HEX/RGB/name/edge/average | Background color for non-mask areas |
| `background_opacity` | - | FLOAT | 1.0 | 0.0–1.0 | Blend strength of background with base image outside the mask |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Blended image: inside mask shows original, outside shows mixed background |
| `mask` | MASK | Processed mask tensor (0–1), equals grayscale mask scaled by `opacity` |
