# Image Mask Crop - Mask-Boundary Cropping and Alpha Output

**Node Purpose:** `Image Mask Crop` crops by the mask’s bounding box or keeps original size, and optionally outputs alpha from the mask. Returns both the image and the processed mask, with robust batch size alignment.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `mask` | - | MASK | - | - | Input mask batch. |
| `output_crop` | - | BOOLEAN | true | - | Crop to the mask’s bbox; when false, keep original image size. |
| `output_alpha` | - | BOOLEAN | false | - | Output RGBA with alpha from the mask when true; otherwise output RGB. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Cropped or original-size image; RGBA if `output_alpha` is true. |
| `mask` | MASK | Cropped or padded mask aligned to the image output.

## Features

- BBox cropping: computes bbox from mask and crops image/mask accordingly.
- Alpha channel: when `output_alpha` is true, the mask is written into the image alpha.
- Size preservation: when `output_crop` is false, keeps original size and centers/pads the mask if needed.
- Batch robustness: mismatched image/mask counts are cycled; outputs are padded to uniform sizes before stacking.

## Typical Usage

- Matte extraction: set `output_alpha=true` to produce RGBA cutouts; downstream compositing can use alpha directly.
- Tight crops: set `output_crop=true` to focus on the mask region; `output_alpha=false` to keep RGB only.
- Full-frame effects: set `output_crop=false` to maintain the original canvas while applying mask alpha.

## Notes & Tips

- If the mask has no non-zero pixels, the node falls back to original-size outputs and applies alpha (if enabled).
- RGBA inputs are converted to RGB before cropping unless `output_alpha` is requested.
- Outputs are device-safe and clamped to [0,1].