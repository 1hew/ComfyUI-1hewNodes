# Image Crop With BBox Mask

**Node Purpose:** Intelligently crops images based on the mask's bounding box. Supports various aspect ratio controls, size constraints, and batch processing. Outputs the cropped image, corresponding mask, and the bounding box mask of the original position for easy restoration.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Original image to be cropped |
| `mask` | Required | MASK | - | - | Mask defining the subject to crop |
| `preset_ratio` | - | COMBO | `mask` | `mask` / `image` / `auto` / `9:16` / `2:3` / `3:4` / `4:5` / `1:1` / `5:4` / `4:3` / `3:2` / `16:9` / `21:9` | Target aspect ratio preset. `mask` follows mask; `image` follows original; `auto` frames from the mask itself (ignores stray specks, `scale_strength` goes from mask-tight to almost the whole image) |
| `get_crop_ratio` | Optional | IMAGE | - | - | Optional reference image. If connected, auto-matches its aspect ratio, overriding `preset_ratio` |
| `scale_strength` | - | FLOAT | 0.0 | 0.0-1.0 | Scale strength. Under `auto`, `0` just contains the mask and `1` is the whole image, blending geometrically in between; the other modes step through their size ladder; ignored when `crop_to_side` is not `None` |
| `crop_to_side` | - | COMBO | `None` | `None` / `longest` / `shortest` / `width` / `height` | Target side control mode |
| `crop_to_length` | - | INT | 1024 | 8-8192 | Target side length value |
| `divisible_by` | - | INT | 8 | 1-1024 | Output size divisor constraint (often for model alignment) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `cropped_image` | IMAGE | Cropped and resized image |
| `bbox_mask` | MASK | Mask in original image dimensions, white area indicating where the crop occurred |
| `cropped_mask` | MASK | Mask corresponding to the cropped image |

## Features

- **Smart Ratio Control**:
  - `mask`: Fits the subject mask ratio as closely as possible.
  - `auto`: Frames from the mask, so long strips and dirty masks still work —
    1) sparse outliers are ignored first: only edges 10x lighter than the densest row/column and within 1% of the total mass are dropped, so a solid subject's edges are never shaved;
    2) the de-speckled bbox ratio is the base: it snaps to the nearest standard ratio only when the linear mismatch is within 30% (giving tidy 4:3 / 16:9 frames), otherwise the mask ratio is kept, so a thin subject is not letterboxed into a 16:9 sliver;
    3) the smallest crop box just contains the bbox (aligned to `divisible_by`) and always fits inside the image, unlike a fixed ratio that silently returns the whole frame when it cannot fit;
    4) `scale_strength` is a framing weight: `0` just contains the mask (mask ratio, subject fills the frame), `1` is the largest aligned box the image can provide (≈ the whole image, image ratio); in between it zooms **geometrically** (every 0.1 is a fixed multiplier) with width and height interpolated separately, so the ratio blends smoothly from the mask ratio to the image ratio. Example: a 431px circle in a 1663×521 frame with `divisible_by=32` gives 448×448, 864×480 and 1632×512 at `0` / `0.5` / `1`;
    5) the crop is centred on the mask bbox and shifted back inside the image, so a subject near an edge automatically gains context on the empty side; `crop_to_side` / `crop_to_length` take over the size whenever they are not `None` (still using the auto ratio ladder), and `scale_strength` is then ignored.
  - `get_crop_ratio`: Dynamically sets target ratio via input image.
- **Size Constraints & Adjustment**:
  - `scale_strength`: under `auto`, `0`→`1` goes from a mask-tight crop to (almost) the whole image while the ratio blends along; the other modes keep their ratio and only expand outwards.
  - `crop_to_side`/`crop_to_length`: Directly controls physical output resolution (e.g., fix long side to 1024).
  - `divisible_by`: Ensures output dimensions are multiples of a specific value (e.g., 8 for SDXL).
- **Batch Processing**: Supports batch inputs for Image and Mask, automatically broadcasting if batch sizes differ.

## Typical Usage

- **Inpainting Preprocessing**: Crop subject using mask for local upscaling or repainting, then use `bbox_mask` to paste back.
- **Dataset Preparation**: Batch crop subjects from large images and unify them to specific sizes (e.g., 512x512 or 1024x1024).

## Notes & Tips

- When `get_crop_ratio` is connected, `preset_ratio` (including `auto`) is ignored.
- Specks dropped by `auto` are not cropped in and are not covered by `bbox_mask`, so pasting back leaves them untouched; use `mask` or an explicit ratio if those pixels are real content.
- `auto` at `1.0` is bounded by `divisible_by` and is therefore a few pixels smaller than the image (1632×512 for 1663×521 with 32), keeping model input alignment intact.
- The old `auto` (image ratio snapped to a standard ratio) is now folded into this mode: the current `auto` frames from the mask and only reaches the image ratio at `scale_strength=1`. Workflows saved with the old `auto` will render differently after reload; that change is expected.
- If the mask is empty or invalid, the node returns default black/white images to prevent errors.
