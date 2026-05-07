# Image Mask Crop - Mask-Boundary Cropping and Alpha Output

**Node Purpose:** `Image Mask Crop` crops by the mask’s bounding box or keeps original size, and optionally outputs alpha from the mask. Returns both the image and the processed mask, with robust batch size alignment.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `mask` | optional | MASK | - | - | Input mask batch. If not provided, uses the image's alpha channel. |
| `output_alpha` | - | BOOLEAN | false | - | Output RGBA with alpha from the mask when true; otherwise preserve the input image's RGB/RGBA channel count. |
| `output_crop` | - | BOOLEAN | true | - | Crop to the mask's bbox; when false, keep original image size. |
| `pad_crop` | - | INT | `0` | 0-4096 | Pad the crop region by this many pixels (only applies when output_crop=true). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Cropped or original-size image; RGBA if `output_alpha` is true. |
| `mask` | MASK | Cropped or padded mask aligned to the image output.

## Features

- BBox cropping: computes bbox from mask and crops image/mask accordingly.
- Crop expansion: `pad_crop` expands the bbox before cropping, and the expanded area still comes from the original image rather than added black padding.
- Alpha channel: when `output_alpha` is true, output is RGBA with alpha from the mask. For RGBA input, output alpha is the input alpha multiplied by the mask.
- Channel preservation: when `output_alpha` is false, the image is only cropped and RGB is not darkened by the mask. RGB input stays RGB; RGBA input stays RGBA.
- Size preservation: when `output_crop` is false, keeps original size and centers/pads the mask if needed.
- Batch robustness: mismatched image/mask counts are cycled; outputs are padded to uniform sizes before stacking.

## Typical Usage

- Matte extraction: set `output_alpha=true` to produce RGBA cutouts; downstream compositing can use alpha directly.
- Tight crops: set `output_crop=true` to focus on the mask region; when `output_alpha=false`, the input channel count is preserved.
- BBox crops: when using rectangular bbox masks with `pad_crop`, the expanded area preserves original image content. If `output_alpha=true`, that area becomes transparent according to the mask.
- Full-frame effects: set `output_crop=false` to maintain the original canvas and let `output_alpha` decide whether mask alpha is output.

## Notes & Tips

- If the mask has no non-zero pixels, the node falls back to original-size outputs and applies alpha (if enabled).
- RGBA inputs keep their original alpha channel when `output_alpha=false`.
- Outputs are device-safe and clamped to [0,1].