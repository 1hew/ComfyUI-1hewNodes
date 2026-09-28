# Image Alpha Join - Combine image and alpha

**Node Purpose:** `Image Alpha Join` combines optional `image` and `mask` inputs into a single 4-channel IMAGE output for producing RGBA images with alpha.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Optional input image, supports single image or batch; if it already contains alpha, only the RGB channels are used |
| `mask` | - | MASK | - | `0~1` | Optional alpha mask; becomes the output alpha channel when `image` is present, and the output is empty when `image` is empty |
| `invert_mask` | - | BOOLEAN | `true` | `true` / `false` | Whether to invert `mask` before writing it into alpha; enabled by default |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Combined 4-channel image (RGBA) |

## Features

- Empty `image` (unconnected or a zero-length batch): returns an empty output, even when `mask` is provided.
- Only `image`: fills alpha with full white so the whole image stays visible.
- Both `image` and `mask`: combines the RGB from `image` with alpha from `mask`; when `invert_mask=true`, the mask is inverted first.
- A zero-length `mask` batch is treated as "no mask" and yields a full-white alpha.
- Size mismatch: resizes `mask` to the `image` size with nearest-neighbor sampling.
- Batch support: when one side has a shorter batch, frames are reused in order.

## Typical Usage

- Rebuild RGBA images from a regular image plus a separate mask.
- Quickly add a full alpha channel when a downstream workflow expects 4-channel IMAGE input.
- Short-circuit a branch when the upstream image is empty instead of passing an invalid transparent placeholder downstream.

## Notes & Tips

- With an `image` input the output is always a 4-channel IMAGE; when `image` is empty the output is empty.
- If the `image` input already contains alpha, this node does not preserve that original alpha; it uses the external `mask` or the default full-white alpha instead.
- `invert_mask` is enabled by default, which is more convenient for workflows where white means "remove/make transparent"; disable it if your `mask` already represents the intended alpha directly.
