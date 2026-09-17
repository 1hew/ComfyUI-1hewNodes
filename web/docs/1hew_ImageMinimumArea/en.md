# Image Minimum Area - Upscale to Minimum Area

**Node Purpose:** `Image Minimum Area` upscales an image until its area reaches a minimum square reference area. `length_to_sq_area` defines the reference square side `N` (target area = `N²`). `fit` behaves like `Image Resize Universal` (`crop` / `pad` / `stretch`), `divisible_by` rounds both final dimensions up to a multiple, and `resize_bool` reports whether an actual upscale happened.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch to upscale. |
| `mask` | optional | MASK | - | - | Optional mask resized in sync; when absent, a default mask is generated. |
| `length_to_sq_area` | - | INT | 1024 | 1–65536 | Reference square side `N`; target area = `N²`. |
| `method` | - | COMBO | `lanczos` | `nearest` / `bilinear` / `lanczos` / `bicubic` / `hamming` / `box` | Resampling method. |
| `divisible_by` | - | INT | 1 | 1–1024 | Round both final dimensions up to a multiple of this value. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | Fit mode applied while scaling. |
| `pad_color` | - | STRING | `1.0` | grayscale/HEX/RGB | Padding background color when `fit=pad`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Upscaled image batch; passed through unchanged when the area already meets the target. |
| `mask` | MASK | Resized input mask, or a generated default mask matching the output size. |
| `resize_bool` | BOOLEAN | `True` when an upscale was performed, `False` otherwise. |

## Features

- Minimum-area target: scales both sides by a single factor so the result area reaches `length_to_sq_area²`.
- Universal-style fit: `crop` / `pad` / `stretch` behave like `Image Resize Universal`.
- Divisibility: `divisible_by` may add a few pixels so both dimensions satisfy a model size constraint.
- Synchronized mask: an input mask is resized with the image; otherwise a default mask is synthesized.
- Pass-through: when the input area already meets the target, everything is returned unchanged and `resize_bool` is `False`.

## Typical Usage

- Guarantee a minimum resolution for downstream models that need a certain pixel area.
- Upscale small crops to a standard area before further processing.

## Notes & Tips

- `divisible_by=1` disables the rounding constraint.
- A provided mask is resized with the same sampler and fit mode as the image.
