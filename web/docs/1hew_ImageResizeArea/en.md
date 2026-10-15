# Image Resize Area - Area-based automatic resize

**Node Purpose:** `Image Resize Area` measures the input image area, derives the reference square side `reference = clamp(sqrt(width x height), min_sq_area, max_sq_area)`, and scales the image and mask to the aspect-preserving size whose area is about `reference^2`. Like `Image Resize Universal`, both output sides are then rounded up to a multiple of `divisible_by` in **every** branch, so the final resolution is always aligned. Oversized images are downscaled with `method`; undersized ones are upscaled with `method` or, when connected, first by an `upscale_model`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image batch (`B x H x W x C`); the first frame's `H` / `W` decide the target. Alpha is preserved. |
| `mask` | optional | MASK | - | - | Optional mask batch; must match the image size. Transformed with the same `fit` / `pad_color` rules. When absent, a default mask is generated. |
| `upscale_model` | optional | UPSCALE_MODEL | - | - | Optional upscaler (connect `Load Upscale Model`). Used only when the image is below `min_sq_area`; the model output is then resized to the target with `method`. Lazy: the loader is skipped entirely when not needed. |
| `min_sq_area` | - | INT | `1024` | `1-8192` | Lower bound of the reference square side; images with `sqrt(area) < min_sq_area` are upscaled. Target area = `reference^2`. |
| `max_sq_area` | - | INT | `2048` | `1-8192` | Upper bound of the reference square side; images with `sqrt(area) > max_sq_area` are downscaled. |
| `method` | - | COMBO | `lanczos` | `nearest` / `bilinear` / `lanczos` / `bicubic` / `hamming` / `box` | Resampling method used for the resize. |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | How the source fits the target; identical to `Image Resize Universal`. It is applied whenever the aligned target differs from the source size. |
| `pad_color` | - | STRING | `1.0` | grayscale / HEX / RGB / `edge` / `average` / `extend` / `mirror` | Fill strategy for `fit = pad`; identical to `Image Resize Universal`. |
| `divisible_by` | - | INT | `32` | `1-1024` | Both output sides are rounded up to a multiple of this value, exactly like `Image Resize Universal`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Resized image at the aspect-preserving target size (area about `reference^2`, both sides multiples of `divisible_by`), or the input unchanged when the aligned target equals the source size. |
| `mask` | MASK | Resized mask matching `image`, or a generated default mask when `mask` is not connected. |
| `resolution` | INT | The clamped, `divisible_by`-aligned reference square side `N`; the target area is about `N^2`. |

## Features

- Aspect-preserving sizing: `reference = clamp(sqrt(width x height), min_sq_area, max_sq_area)` normalises the pixel budget while the target keeps the source aspect ratio. The output is deliberately **not** forced square — e.g. a 4000x3000 input with `max_sq_area = 2048` becomes 2368x1792.
- Divisible alignment: every output side is rounded up to a multiple of `divisible_by`, matching `Image Resize Universal`. This also applies to images whose area is already inside the range.
- Three-branch behaviour:
  - `sqrt(area)` inside `[min_sq_area, max_sq_area]`: the target keeps the source size, and only `divisible_by` may enlarge it a little;
  - above `max_sq_area`: downscale with `method`;
  - below `min_sq_area`: upscale with `method`, or first run `upscale_model` and then normalise with `method`.
- Shared resize core: `fit` / `pad_color` / mask transforms reuse `Image Resize Universal`, so crop/pad/stretch, alpha handling and default masks behave identically.
- Model upscaling: equivalent to the official `Upscale Image (using Model)`, using tiled scaling with automatic tile reduction on out-of-memory.

## Mask Rules

- `mask` connected: resized with the same `fit` and sampler as the image.
- `mask` absent, `fit = pad`: target size, original region white (255), padded region black (0).
- `mask` absent, `fit = crop` / `stretch`: target size, fully white (255).
- Target equals the source size: a connected mask is returned unchanged; otherwise an all-white mask at the source size is generated.

## Padding Strategies (`pad_color`)

Same as `Image Resize Universal`: grayscale (`0.5`), HEX (`#FF0000`), RGB (`255,0,0` or `0.5,0.2,0.8`), `edge`, `average`, `extend`, `mirror`.

## Typical Usage

- Bounded normalisation: keep `sqrt(area)` between `min_sq_area` and `max_sq_area` and get an aspect-preserving, 32-aligned result for models.
- Quality upscaling: connect `Load Upscale Model`; it only runs for images below `min_sq_area`.
- Mask-synchronised workflow: feed image + mask and get both back at the same size.

## Notes & Tips

- Only the first frame's spatial size drives the target; the whole batch is resized together.
- Target size uses the same `sqrt(ratio) x reference` formula as `Image Resize Universal`'s `length_to_sq_area`, then rounds each side up to `divisible_by`.
- `min_sq_area > max_sq_area` is normalised by swapping the two bounds.
- If no multiple of `divisible_by` exists inside `[min_sq_area, max_sq_area]` the node no longer raises an error: it falls back to the closest integer reference side and still aligns the output sides.
- When the source size already equals the aligned target, the image/mask pass through untouched, so no needless resampling happens.
- `upscale_model` is lazy: the upstream `Load Upscale Model` is skipped entirely for images that do not need it.
