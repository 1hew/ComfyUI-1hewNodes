# Image Alpha Split - Extract alpha and flatten to background

**Node Purpose:** `Image Alpha Split` extracts the input image alpha as a `mask` output, while also compositing transparent areas onto `background_color` and returning the processed RGB image.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image, supports single image or batch; RGBA inputs extract alpha and composite against the background color; both outputs are empty when it is empty |
| `background_color` | - | STRING | `1.0` | named / `#RRGGBB` / `r,g,b` / `0~1` grayscale / single-letter shorthand / `sm` | Background color used to fill transparent areas; `sm` applies the complete stroke-aware refiner score per frame to select black, white, or `#808080` |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | RGB image flattened onto the selected background |
| `mask` | MASK | Alpha mask extracted from the input image; becomes a full-white mask when the input has no alpha |

## Features

- Empty `image` (unconnected or a zero-length batch): both the `image` and `mask` outputs are empty.
- Alpha-bearing RGBA or grayscale+alpha input: returns the original alpha as `mask` and composites transparency onto `background_color`; with `sm`, each frame uses the complete stroke-aware refiner algorithm: actual 8-bit sRGB compositing, OKLab distance, linear-luminance contrast, three alpha inner-boundary bands, grouped sampling, weighted low-tail/worst-group scoring, endpoint tie-breaking, and the gray-score margin to select black, white, or `#808080`.
- RGB input: passes `image` through unchanged and returns a full-white `mask`.
- Batch support: image batches are processed frame by frame and return synchronized `image` and `mask` outputs.
- Color parsing follows the same rules as the `Image Solid` node.

## Typical Usage

- Split alpha from transparent PNGs while also generating a background-filled preview or downstream-safe RGB image.
- Prepare white-background images for nodes that do not support alpha, while keeping the transparent area as a mask.
- Standardize batch workflows that need both a flattened image and the original alpha mask.

## Notes & Tips

- The `image` output is always RGB and does not preserve the original alpha channel.
- If the input image has no alpha channel, `mask` becomes fully white and `background_color` does not affect the image result.
- When `image` is empty no placeholder is built: both `image` and `mask` are returned empty so downstream branches can short-circuit.
