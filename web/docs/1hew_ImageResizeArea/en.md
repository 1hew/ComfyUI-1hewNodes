# Image Resize Area - Area-based automatic resize

**Node Purpose:** `Image Resize Area` measures the input image area, derives the reference square side `N = sqrt(width x height)`, clamps it into `[min_sq_area, max_sq_area]` and aligns it to `divisible_by`; the image and mask are then scaled to the aspect-preserving size whose area is about `N^2`, with both sides rounded up to a multiple of `divisible_by`. Images whose area already falls inside the range pass through untouched, oversized ones are downscaled with `method`, and undersized ones are upscaled with `method` or, when connected, first by an `upscale_model`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch (`B x H x W x C`); the first frame's `H` / `W` decide the target. Alpha is preserved. |
| `mask` | optional | MASK | - | - | Optional mask batch; must match the image size. Transformed with the same `fit` / `pad_color` rules. When absent, a default mask is generated. |
| `upscale_model` | optional | UPSCALE_MODEL | - | - | Optional upscaler (connect `Load Upscale Model`). Used only when the image is below `min_sq_area`; the model output is then resized to the target with `method`. Lazy: the loader is skipped entirely when not needed. |
| `min_sq_area` | - | INT | `1024` | `1-8192` | Lower bound of the reference square side `N`; images with `sqrt(area) < min_sq_area` are upscaled. Target area = `N^2`. |
| `max_sq_area` | - | INT | `2048` | `1-8192` | Upper bound of the reference square side `N`; images with `sqrt(area) > max_sq_area` are downscaled. |
| `method` | - | COMBO | `lanczos` | `nearest` / `bilinear` / `lanczos` / `bicubic` / `hamming` / `box` | Resampling method used for the resize. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | How the source fits the target; identical to `Image Resize Universal` / `Image Resize Square`. Only matters for the small ratio difference introduced by the `divisible_by` rounding. |
| `pad_color` | - | STRING | `1.0` | grayscale / HEX / RGB / `edge` / `average` / `extend` / `mirror` | Fill strategy for `fit = pad`; identical to `Image Resize Universal`. |
| `divisible_by` | - | INT | `32` | `1-1024` | Both output sides are rounded up to a multiple of this value. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image: scaled to the aspect-preserving target size (area about `N^2`, both sides multiples of `divisible_by`) when outside the range, otherwise the original image unchanged. |
| `mask` | MASK | Resized mask matching `image`, or a generated default mask when `mask` is not connected. |
| `int` | INT | The adjusted reference square side `N` (a multiple of `divisible_by`); the target area is `N^2`. |

## Features

- Aspect-preserving sizing: `N = sqrt(width x height)` normalises the pixel budget while the target keeps the source aspect ratio. The output is deliberately **not** forced square — e.g. a 4000x3000 input with `max_sq_area = 2048` becomes 2368x1792.
- Divisible alignment: both target sides are rounded up to a multiple of `divisible_by`, matching `Image Resize Universal`.
- Three-branch behaviour:
  - `sqrt(area)` inside `[min_sq_area, max_sq_area]`: image and mask pass through untouched;
  - above `max_sq_area`: downscale with `method`;
  - below `min_sq_area`: upscale with `method`, or first run `upscale_model` and then normalise with `method`.
- Shared resize core: `fit` / `pad_color` / mask transforms reuse `Image Resize Universal`, so crop/pad/stretch, alpha handling and default masks behave identically.
- Model upscaling: equivalent to the official `Upscale Image (using Model)`, using tiled scaling with automatic tile reduction on out-of-memory.

## Mask Rules

- `mask` connected: resized with the same `fit` and sampler as the image.
- `mask` absent, `fit = pad`: target size, original region white (255), padded region black (0).
- `mask` absent, `fit = crop` / `stretch`: target size, fully white (255).
- Inside the range (passthrough): a connected mask is returned unchanged; otherwise an all-white mask at the source size is generated.

## Padding Strategies (`pad_color`)

Same as `Image Resize Universal`: grayscale (`0.5`), HEX (`#FF0000`), RGB (`255,0,0` or `0.5,0.2,0.8`), `edge`, `average`, `extend`, `mirror`.

## Typical Usage

- Bounded normalisation: keep `sqrt(area)` between `min_sq_area` and `max_sq_area` and get an aspect-preserving, 32-aligned result for models.
- Quality upscaling: connect `Load Upscale Model`; it only runs for images below `min_sq_area`.
- Mask-synchronised workflow: feed image + mask and get both back at the same size.

## Notes & Tips

- Only the first frame's spatial size drives the target; the whole batch is resized together.
- Target size uses the same `sqrt(ratio) x N` formula as `Image Resize Universal`'s `length_to_sq_area`, then rounds each side up to `divisible_by`.
- `min_sq_area > max_sq_area` is normalised by swapping the two bounds.
- If no multiple of `divisible_by` exists inside `[min_sq_area, max_sq_area]` the node raises an error.
- Inside the range the image is not resized, so the reported `int` can differ from the actual image size; that is intentional.
- `upscale_model` is lazy: the upstream `Load Upscale Model` is skipped entirely for images that do not need it.
