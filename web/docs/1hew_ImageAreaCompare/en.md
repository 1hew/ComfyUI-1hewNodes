# Image Area Compare - Compare Two Images' Pixel Area

**Node Purpose:** `Image Area Compare` compares the pixel area (`width × height`, the total pixel count) of two IMAGE inputs and outputs a boolean based on the selected operator. It is the image counterpart of `Int Number Compare`.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `a` | - | IMAGE | - | - | Left image; its `width × height` is the left value. |
| `operator` | - | COMBO | `==` | `==`, `!=`, `>`, `>=`, `<`, `<=` | Comparison operator. |
| `b` | - | IMAGE | - | - | Right image; its `width × height` is the right value. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bool` | BOOLEAN | `true` when the comparison is satisfied, otherwise `false`. |

## Features

- Pixel-area comparison: the compared values are each image's total pixel count (`width × height`), not its file size or subject area.
- Same operators as `Int Number Compare`: equal, not equal, greater than, greater than or equal, less than, and less than or equal.
- Batch-agnostic: every frame in an IMAGE batch shares the same dimensions, so the batch size does not change the area.
- Robust fallback: an empty or non-image input, or an exception, outputs `false`.

## Typical Usage

- Resolution gating: test whether one image is larger than another before choosing an upscale path.
- Branch control: connect the output to `Any Switch Bool` to route high- vs low-resolution images.

## Notes & Tips

- The area is computed from the tensor shape (`height × width`), so it is independent of color channels and alpha.
- Combine with `Image Minimum Area` to verify that an image reached the required minimum area.
