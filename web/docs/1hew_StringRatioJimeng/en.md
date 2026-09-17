# String Ratio Jimeng - Jimeng Ratio Selector

**Node Purpose:** `String Ratio Jimeng` outputs the nearest Jimeng supported aspect-ratio string for each input image. If no image is connected, it passes through the manually selected ratio.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `selection` | - | COMBO | `1:1` | `21:9` / `16:9` / `3:2` / `4:3` / `1:1` / `3:4` / `2:3` / `9:16` | Manual Jimeng ratio preset used when `image` is not connected. |
| `image` | optional | IMAGE | - | - | Input image batch used to infer the nearest supported ratio per frame. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | Ratio string result; batched inputs are joined by newline in batch order. |

## Features

- Jimeng ratio set: `21:9`, `16:9`, `3:2`, `4:3`, `1:1`, `3:4`, `2:3`, `9:16`.
- Image-driven inference: matches each input image to the nearest preset ratio in log-ratio space.
- Manual fallback: when no image is connected, returns the selected ratio directly.
- Batch-friendly: supports multi-image batches and returns one ratio per image line.

## Typical Usage

- Use before Jimeng image nodes that require an aspect-ratio string.
- Leave `image` empty when you want to force a fixed ratio string downstream.

## Notes & Tips

- This node only outputs the ratio string; it does not resize images.
- The returned ratio is the closest preset by logarithmic ratio difference.
