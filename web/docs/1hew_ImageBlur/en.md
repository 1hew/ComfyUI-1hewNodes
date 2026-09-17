# Image Blur - Gaussian Blur

**Node Purpose:** `Image Blur` applies a per-channel Gaussian blur to the whole image, matching the behavior of LayerStyle's `LayerFilter: Gaussian Blur V2`. A `blur` value of `0` passes the image through unchanged. Processing is per-frame across the batch, preserving channel count.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch to blur. |
| `blur` | - | FLOAT | 20.0 | 0.0–1000.0 (step 0.01) | Gaussian blur radius; `0` or negative passes the image through unchanged. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Blurred image batch. |

## Features

- Fractional radius: `blur` is a FLOAT, semantically closer to the V2 Gaussian Blur behavior.
- Zero passthrough: when `blur <= 0`, the original image is returned untouched.
- Per-channel blur: each channel is blurred independently and re-stacked, preserving channel count.
- Batch-friendly: every frame in the batch is processed individually.

## Typical Usage

- Soften an image or defocus a background before compositing.
- Apply a quick whole-image smoothing pass in an image pipeline.

## Notes & Tips

- Blur is applied via PIL's `ImageFilter.GaussianBlur` with the requested radius.
- Large radii are progressively more expensive; prefer small-to-moderate values for routine smoothing.
