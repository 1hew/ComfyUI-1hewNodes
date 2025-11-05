# Image Mask Crop

**Node Function:** The `Image Mask Crop` node crops the image and mask using the mask's bounding box or keeps the original size, with optional RGBA output (mask as alpha). Supports batch cycling and size alignment when image/mask sizes differ.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `image` | Required | IMAGE | - | - | Input image batch |
| `mask` | Required | MASK | - | - | Input mask batch |
| `output_crop` | Required | BOOLEAN | True | True/False | Crop to the mask bounding box when True; keep original size when False |
| `output_alpha` | Required | BOOLEAN | False | True/False | Output RGBA when True (alpha = mask); otherwise RGB |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Cropped or original-size image; RGBA when `output_alpha=True` |
| `mask` | MASK | Cropped or aligned mask (0–1) |
