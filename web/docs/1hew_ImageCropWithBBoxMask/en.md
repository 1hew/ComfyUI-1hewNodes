# Image Crop With BBox Mask

**Node Purpose:** Intelligently crops images based on the mask's bounding box. Supports various aspect ratio controls, size constraints, and batch processing. Outputs the cropped image, corresponding mask, and the bounding box mask of the original position for easy restoration.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | Required | IMAGE | - | - | Original image to be cropped |
| `mask` | Required | MASK | - | - | Mask defining the subject to crop |
| `preset_ratio` | - | COMBO | `mask` | `mask` / `image` / `auto` / `9:16` ... | Target aspect ratio preset. `mask` follows mask; `image` follows original; `auto` matches common ratios |
| `get_crop_ratio` | Optional | IMAGE | - | - | Optional reference image. If connected, auto-matches its aspect ratio, overriding `preset_ratio` |
| `scale_strength` | - | FLOAT | 0.0 | 0.0-1.0 | Scale strength. Extends the crop box outwards while maintaining aspect ratio |
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
  - `auto`: Automatically calculates and matches the closest standard ratio (e.g., 4:3, 16:9).
  - `get_crop_ratio`: Dynamically sets target ratio via input image.
- **Size Constraints & Adjustment**:
  - `scale_strength`: Allows the crop box to include more background context.
  - `crop_to_side`/`crop_to_length`: Directly controls physical output resolution (e.g., fix long side to 1024).
  - `divisible_by`: Ensures output dimensions are multiples of a specific value (e.g., 8 for SDXL).
- **Batch Processing**: Supports batch inputs for Image and Mask, automatically broadcasting if batch sizes differ.

## Typical Usage

- **Inpainting Preprocessing**: Crop subject using mask for local upscaling or repainting, then use `bbox_mask` to paste back.
- **Dataset Preparation**: Batch crop subjects from large images and unify them to specific sizes (e.g., 512x512 or 1024x1024).

## Notes & Tips

- When `get_crop_ratio` is connected, `preset_ratio` setting is ignored.
- If the mask is empty or invalid, the node returns default black/white images to prevent errors.
