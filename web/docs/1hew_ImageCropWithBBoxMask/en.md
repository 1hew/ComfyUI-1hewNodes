# Image Crop With BBox Mask - Mask-Aware Crop with BBox Return

**Node Purpose:** `Image Crop With BBox Mask` crops the image around a mask’s bounding box using configurable aspect ratios and side-length targeting. It returns the cropped image, a `bbox_mask` marking the crop region on the original image, and the `cropped_mask` aligned to the crop.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch. |
| `mask` | - | MASK | - | - | Mask guiding crop center and bbox; required. |
| `preset_ratio` | - | COMBO | `mask` | `mask` / `image` / `auto` / `1:1` / `3:2` / `4:3` / `16:9` / `21:9` / `2:3` / `3:4` / `9:16` / `9:21` | Aspect ratio source. |
| `scale_strength` | - | FLOAT | 0.0 | 0.0–1.0 | Chooses a target size between the min/max valid sizes; higher values favor larger crops. |
| `crop_to_side` | - | COMBO | `None` | `None` / `longest` / `shortest` / `width` / `height` | Side-length targeting mode. |
| `crop_to_length` | - | INT | 1024 | 8–8192 | Target length when using `crop_to_side`. |
| `divisible_by` | - | INT | 8 | 1–1024 | Target width/height must be multiples of this value. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `cropped_image` | IMAGE | Cropped image batch, padded to uniform size if needed. |
| `bbox_mask` | MASK | Original-size mask; the crop rectangle area is white (1.0). |
| `cropped_mask` | MASK | Mask cropped to the output image region.

## Features

- Ratio orientation: determines landscape/portrait/square orientation from `preset_ratio`.
- Flexible valid sizes: generates candidate sizes respecting aspect and `divisible_by`.
- Side targeting: select by `width`/`height` or `longest`/`shortest`, then fit to `crop_to_length`.
- BBox masking: emits `bbox_mask` with ones in the crop area and zeros elsewhere.
- Batch robustness: expands smaller batches, then pads images/masks to common sizes for stacking.

## Typical Usage

- Consistent ratios: set `preset_ratio=4:3` and `divisible_by=8/16` to align model expectations.
- Long-edge control: `crop_to_side=longest` with `crop_to_length` ensures uniform crop height/width across varied inputs.
- Downstream paste: use `bbox_mask` with `Image Paste By BBox Mask` to restore processed crops to their original positions.

## Notes & Tips

- If the mask is empty or invalid, the node returns the original image and a full-white `bbox_mask` as a fallback.
- Cropped outputs are clamped to device-safe ranges and converted to RGB if needed.
- Differences in output sizes across the batch are normalized by padding before stacking.