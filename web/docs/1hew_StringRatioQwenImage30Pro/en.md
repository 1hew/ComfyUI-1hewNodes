# String Ratio Qwen Image 3.0 Pro - Qwen Image 3.0 Pro Ratio Selector

**Node Purpose:** `String Ratio Qwen Image 3.0 Pro` outputs the nearest Qwen Image 3.0 Pro supported aspect ratio string for each input image. If no image is connected, it passes through the manually selected ratio. It is designed to pair with the 1hewOffice `Qwen Image 3.0 Pro` API node.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1:1` | `3:1` / `21:9` / `2:1` / `16:9` / `3:2` / `4:3` / `5:4` / `1:1` / `4:5` / `3:4` / `2:3` / `9:16` / `1:2` / `9:21` / `1:3` | Manual Qwen Image 3.0 Pro ratio preset used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest supported ratio per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Ratio string result; batched inputs are joined by newline in batch order. |

## Features

- Qwen Image 3.0 Pro ratio set: `3:1`, `21:9`, `2:1`, `16:9`, `3:2`, `4:3`, `5:4`, `1:1`, `4:5`, `3:4`, `2:3`, `9:16`, `1:2`, `9:21`, `1:3` (15 ratios).
- Image-driven inference: matches each input image to the nearest preset ratio in log-ratio space.
- Manual fallback: when no image is connected, returns the selected ratio directly.
- Batch-friendly: supports multi-image batches and returns one ratio per image line.

## Typical Usage

- Use before Qwen Image 3.0 Pro image nodes that require an aspect-ratio string: connect this node's `string` output to the API node's `aspect_ratio` input.
- Keep a fixed downstream ratio by leaving `image` empty and selecting the target ratio manually.
- You can also connect the same reference image to the API node's `get_image_size` and let `aspect_ratio=auto` infer the ratio; this node is useful when you need the ratio string explicitly (logging, forwarding, or conditional logic).

## Notes & Tips

- This node only outputs the ratio string; it does not resize images. Output pixels come from the API node's `resolution` (`1k` / `2k`) combined with the ratio via the official size table.
- The ratio set is an exact match of `QwenImage30Pro.ASPECT_RATIO_OPTIONS` in the 1hewOffice `Qwen Image 3.0 Pro` API node minus `auto` / `dynamic` (15 items, verified item by item). That API node belongs to 1hewOffice and its source is not in this repository. It also matches the preset ratio table of `Image Resize Qwen Image 3.0 Pro` in this repository.
- `auto` / `dynamic` are intentionally not offered here: the API node infers them from the reference image or prompt, while this node emits a concrete ratio you can feed directly.
- The API node's image-edit path (`/v1/images/edits`) has its `size` ignored upstream (about 2K output following the reference ratio), so this node mainly affects **text-to-image**.
- The returned ratio is the nearest preset in log-ratio space; Qwen Image 3.0 Pro officially supports `1:8` ~ `8:1`, so ratios beyond the 15 tiers snap to the closest supported one.
