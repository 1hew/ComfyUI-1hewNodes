# String Ratio Doubao Seedream 5.0 Pro - Doubao Seedream 5.0 Pro Ratio Selector

**Node Purpose:** `String Ratio Doubao Seedream 5.0 Pro` outputs the nearest Doubao Seedream 5.0 Pro supported aspect ratio string for each input image. If no image is connected, it passes through the manually selected ratio. It is designed to pair with the 1hewOffice `Doubao Seedream 5.0 Pro` API node.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1:1` | `21:9` / `16:9` / `3:2` / `4:3` / `1:1` / `3:4` / `2:3` / `9:16` | Manual Doubao Seedream 5.0 Pro ratio preset used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest supported ratio per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Ratio string result; batched inputs are joined by newline in batch order. |

## Features

- Doubao Seedream 5.0 Pro ratio set: `21:9`, `16:9`, `3:2`, `4:3`, `1:1`, `3:4`, `2:3`, `9:16` (8 ratios).
- Image-driven inference: matches each input image to the nearest preset ratio in log-ratio space.
- Manual fallback: when no image is connected, returns the selected ratio directly.
- Batch-friendly: supports multi-image batches and returns one ratio per image line.

## Typical Usage

- Use before Doubao Seedream 5.0 Pro image nodes that require an aspect-ratio string: connect this node's `string` output to the API node's `aspect_ratio` input.
- Keep a fixed downstream ratio by leaving `image` empty and selecting the target ratio manually.
- Image-to-image / reference-image flows: connect the same reference image to both this node's `image` and the API node's `image_1` so the API node uses the reference ratio.

## Notes & Tips

- This node only outputs the ratio string; it does not resize images and does not decide the final pixels. Output pixels come from the API node's `resolution` (`1k` / `1.5k` / `2k`) combined with the ratio via the official size table.
- The ratio set is an exact match of `DoubaoSeedream50Pro.ASPECT_RATIO_OPTIONS` in the 1hewOffice `Doubao Seedream 5.0 Pro` API node minus `auto` (8 items, verified item by item). That API node belongs to 1hewOffice and its source is not in this repository.
- `auto` is intentionally not offered here: the API node infers it from the reference image, while this node emits a concrete ratio you can feed directly.
- That API node has **no `get_image_size` input**, so for text-to-image (no reference image) set the API node's `aspect_ratio` directly; this node is mainly useful when an image is available and you want the ratio snapped to the 8 supported tiers.
- The returned ratio is the nearest preset in log-ratio space, so extreme ratios (e.g. `3:1`, `1:3`) snap to the closest supported tier.
